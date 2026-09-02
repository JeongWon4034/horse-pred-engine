"""학습 — 단계(S1~S4)를 같은 손실·같은 채점기로 돌려 LightGBM 표 옆에 놓는다.

    PYTHONUTF8=1 uv run python -m model.train linear             # S1, 73피처 (인기도 제외 — 주 비교축)
    PYTHONUTF8=1 uv run python -m model.train linear --market    # S1, 77피처 (배당 포함 — 게임 모드)
    kind ∈ linear | embed | history | attn

매 epoch valid 를 채점기로 재고 최고 epoch 의 예측을 experiments/pred/{kind}_{73|77}.parquet 에 남긴다.
마지막에 C.report 표와 LightGBM 재현 예측(model.baseline) 대비 paired bootstrap CI, 장부 한 줄을 찍는다.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # cuBLAS 결정성 — torch import 전에

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .data import Races, feature_cols, fit_encoder, flatten_scores, load, to_races
from .evaluate import compare, hits, ledger_line, load_pred, metrics, save_pred
from .losses import plackett_luce
from .models import LinearRanker
from .team import C

SEED = 20260901
STAGE = {"linear": "S1", "embed": "S2", "history": "S3", "attn": "S4"}
RUNS = Path(__file__).resolve().parents[2] / "experiments" / "runs"


def build(kind: str, enc, dev: str):
    """단계별 모델. weight decay 도 여기서 정한다."""
    if kind == "linear":
        return LinearRanker(enc.dim).to(dev), 1e-4
    raise NotImplementedError(f"{kind}: 아직 없는 단계")


@torch.no_grad()
def predict(model, races: Races, dev: str, batch: int = 1024) -> np.ndarray:
    """[R, MAX_FIELD] 점수. 패딩 슬롯은 -1e9."""
    model.eval()
    out = []
    for i in range(0, len(races), batch):
        x = torch.as_tensor(races.x[i:i + batch], device=dev)
        m = torch.as_tensor(races.mask[i:i + batch], device=dev)
        out.append(model(x, m).masked_fill(m == 0, -1e9).cpu().numpy())
    return np.concatenate(out)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[2]).strip()
    except Exception:
        return "—"


def run(kind: str, market: bool, epochs: int, bs: int, lr: float, topk: int, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():                 # attention 커널을 결정적 경로로 고정 (S4)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_feat = 77 if market else 73
    tag = f"{kind}_{n_feat}"

    t0 = time.time()
    tr_df, va_df = load("train"), load("valid")
    cols = feature_cols(exclude_pop=not market)
    assert len(cols) == n_feat, (len(cols), n_feat)
    enc = fit_encoder(tr_df, cols)
    tr, va = to_races(tr_df, enc), to_races(va_df, enc)
    print(f"[data] {time.time()-t0:.0f}s  피처 {n_feat}개 → 입력 {enc.dim}차원  "
          f"train {len(tr):,}경주  valid {len(va):,}경주  device={dev}  topk={topk}")

    model, wd = build(kind, enc, dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[{STAGE[kind]} {kind}] 파라미터 {n_param:,}  epochs {epochs}  bs {bs}  lr {lr}  wd {wd}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.as_tensor(tr.x, device=dev); M = torch.as_tensor(tr.mask, device=dev)
    O = torch.as_tensor(tr.order, device=dev)
    n = len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs), pct_start=0.25)

    best, best_scores, best_state = None, None, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)      # 경주 단위 셔플 — 경주 내부 순서는 그대로
        tot = cnt = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = plackett_luce(model(X[j], M[j]), M[j], O[j], topk=topk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(j); cnt += len(j)

        scores = flatten_scores(predict(model, va, dev), va)
        m = metrics(va_df, scores)
        better = best is None or (m["top1"], -m["logloss"]) > (best["top1"], -best["logloss"])
        if better:
            best, best_scores = {**m, "epoch": ep}, scores
            best_state = copy.deepcopy(model.state_dict())
        print(f"  ep{ep:>2}  loss {tot/cnt:.4f}   top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}  "
              f"logloss {m['logloss']:.4f}  ECE {m['ece']:.4f}{'  ←best' if better else ''}")

    print(f"\n[최고] ep{best['epoch']}  top1 {best['top1']:.2f}  top3 {best['top3']:.2f}  "
          f"logloss {best['logloss']:.4f}  ECE {best['ece']:.4f}  ({time.time()-t0:.0f}s)")
    print("  " + str(save_pred(va_df, best_scores, tag)))
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "kind": kind, "market": market, "epoch": best["epoch"],
                "names": enc.names}, RUNS / f"{tag}.pt")          # S5 game 채점용. gitignore

    name = f"DL {STAGE[kind]} {kind} {n_feat}피처"
    rows = {name: best_scores}
    lgb = load_pred(va_df, f"lgb_{n_feat}")
    if lgb is not None:
        rows[f"LightGBM {n_feat}피처 (재현)"] = lgb
    print()
    print(C.report(rows, va_df))
    if lgb is not None:
        d, lo, hi = compare(hits(va_df, best_scores), hits(va_df, lgb))
        verdict = "차이 있음" if lo > 0 or hi < 0 else "차이 없음 (CI 가 0 을 품음)"
        print(f"\n[DL − LGB] top-1 {d:+.2f}%p  95% CI [{lo:+.2f}, {hi:+.2f}]  → {verdict}")
    else:
        print("\n(experiments/pred/lgb_%d.parquet 없음 — `uv run python -m model.baseline` 먼저)" % n_feat)

    memo = f"{STAGE[kind]} {kind}, ep{best['epoch']}/{epochs}, PL topk={topk}, {dev}"
    print("\n[장부]")
    print(ledger_line(best, f"DL {STAGE[kind]} {kind}", n_feat, commit=git_commit(), seed=seed, memo=memo))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(STAGE))
    ap.add_argument("--market", action="store_true", help="77피처 (배당 포함). 없으면 73피처")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--topk", type=int, default=1, help="Plackett-Luce 깊이. 1 = LGB 와 같은 1착 logloss")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    run(a.kind, a.market, a.epochs, a.bs, a.lr, a.topk, a.seed)


if __name__ == "__main__":
    main()
