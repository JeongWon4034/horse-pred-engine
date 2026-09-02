"""학습 — 단계(S1~S4)를 같은 손실·같은 채점기로 돌려 LightGBM 표 옆에 놓는다.

    PYTHONUTF8=1 uv run python -m model.train linear             # S1, 73피처 (인기도 제외 — 주 비교축)
    PYTHONUTF8=1 uv run python -m model.train linear --market    # S1, 77피처 (배당 포함 — 게임 모드)
    PYTHONUTF8=1 uv run python -m model.train embed              # S2, 기수·조교사·부마 임베딩
    kind ∈ linear | embed | history | attn

매 epoch valid 를 채점기로 재고 최고 epoch 의 예측을 experiments/pred/{kind}_{73|77}.parquet 에,
가중치를 experiments/runs/{kind}_{73|77}.pt 에 남긴다.
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

from .categorical import EMBED_COLS, describe, encode_cats, fit_vocabs  # noqa: E402
from .data import Races, feature_cols, fit_encoder, flatten_scores, load, to_races  # noqa: E402
from .evaluate import compare, hits, ledger_line, load_pred, metrics, save_pred  # noqa: E402
from .losses import plackett_luce  # noqa: E402
from .history import FEATS as HIST_FEATS, history_for  # noqa: E402
from .models import EmbedRanker, HistoryRanker, LinearRanker, RaceTransformer  # noqa: E402
from .team import C  # noqa: E402

SEED = 20260901
STAGE = {"linear": "S1", "embed": "S2", "history": "S3", "attn": "S4"}
RUNS = Path(__file__).resolve().parents[2] / "experiments" / "runs"

# 단계별 기본 하이퍼파라미터. 선형은 민감하지 않고, 그 위는 과적합이 심해 정규화를 세게 건다.
DEFAULTS = {
    "linear":  dict(epochs=12, lr=3e-3, wd=1e-4, patience=0),
    "embed":   dict(epochs=20, lr=1e-3, wd=1e-2, patience=5),
    "history": dict(epochs=20, lr=1e-3, wd=1e-2, patience=5),
    "attn":    dict(epochs=20, lr=1e-3, wd=1e-2, patience=5),
}


def build(kind: str, enc, vocabs, dev: str):
    sizes = [v.size for v in vocabs.values()]
    if kind == "linear":
        return LinearRanker(enc.dim).to(dev)
    if kind == "embed":
        return EmbedRanker(enc.dim, sizes).to(dev)
    if kind == "history":
        return HistoryRanker(enc.dim, sizes, len(HIST_FEATS)).to(dev)
    if kind == "attn":
        return RaceTransformer(enc.dim, sizes, len(HIST_FEATS)).to(dev)
    raise NotImplementedError(f"{kind}: 아직 없는 단계")


def extras_for(kind: str, df, split: str, vocabs, hist_len: int = 10) -> dict[str, np.ndarray]:
    """단계별 추가 입력(행 단위). pack_races 가 경주 단위로 접는다."""
    out = {}
    if kind in ("embed", "history", "attn"):
        out["cat"] = encode_cats(df, vocabs)
    if kind in ("history", "attn"):
        out["hist"], out["hist_len"] = history_for(df, split, L=hist_len)
    return out


@torch.no_grad()
def predict(model, races: Races, T: dict[str, torch.Tensor], dev: str, batch: int = 1024) -> np.ndarray:
    """[R, MAX_FIELD] 점수. 패딩 슬롯은 -1e9."""
    model.eval()
    out = []
    for i in range(0, len(races), batch):
        x = torch.as_tensor(races.x[i:i + batch], device=dev)
        m = torch.as_tensor(races.mask[i:i + batch], device=dev)
        ex = {k: v[i:i + batch] for k, v in T.items()}
        out.append(model(x, m, **ex).masked_fill(m == 0, -1e9).cpu().numpy())
    return np.concatenate(out)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[2]).strip()
    except Exception:
        return "—"


def run(kind: str, market: bool, epochs: int, bs: int, lr: float, wd: float, patience: int,
        topk: int, seed: int, min_count: int, tag_suffix: str = "",
        drop_groups: tuple[str, ...] = (), hist_len: int = 10) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():                 # attention 커널을 결정적 경로로 고정 (S4)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_feat = 77 if market else 73
    tag = f"{kind}_{n_feat}{tag_suffix}"

    t0 = time.time()
    tr_df, va_df = load("train"), load("valid")
    cols = feature_cols(exclude_pop=not market)
    assert len(cols) == n_feat, (len(cols), n_feat)
    if drop_groups:                               # ablation — 예: F1 집계 피처를 빼고 GRU 이력만
        cols = [c for c in cols if c.split("_")[0] not in drop_groups]
        print(f"[ablation] {drop_groups} 제외 → 피처 {len(cols)}개")
    enc = fit_encoder(tr_df, cols)
    vocabs = fit_vocabs(tr_df, EMBED_COLS, min_count) if kind != "linear" else {}
    tr = to_races(tr_df, enc, extras_for(kind, tr_df, "train", vocabs, hist_len))
    va = to_races(va_df, enc, extras_for(kind, va_df, "valid", vocabs, hist_len))
    print(f"[data] {time.time()-t0:.0f}s  피처 {len(cols)}개 → 입력 {enc.dim}차원  "
          f"train {len(tr):,}경주  valid {len(va):,}경주  device={dev}  topk={topk}")
    if vocabs:
        print(f"[임베딩 어휘] train 기준, 등장 <{min_count}회 → <rare>")
        print(describe(vocabs, va_df))

    model = build(kind, enc, vocabs, dev)
    n_param = sum(p.numel() for p in model.parameters())
    pat = patience if patience else "없음"
    print(f"[{STAGE[kind]} {kind}] 파라미터 {n_param:,}  epochs {epochs}  bs {bs}  lr {lr}  wd {wd}  patience {pat}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.as_tensor(tr.x, device=dev); M = torch.as_tensor(tr.mask, device=dev)
    O = torch.as_tensor(tr.order, device=dev)
    T_tr = {k: torch.as_tensor(v, device=dev) for k, v in tr.extra.items()}
    T_va = {k: torch.as_tensor(v, device=dev) for k, v in va.extra.items()}
    n = len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs), pct_start=0.25)

    best, best_scores, best_state, stale = None, None, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)      # 경주 단위 셔플 — 경주 내부 순서는 그대로
        tot = cnt = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            ex = {k: v[j] for k, v in T_tr.items()}
            loss = plackett_luce(model(X[j], M[j], **ex), M[j], O[j], topk=topk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(j); cnt += len(j)

        scores = flatten_scores(predict(model, va, T_va, dev), va)
        m = metrics(va_df, scores)
        better = best is None or (m["top1"], -m["logloss"]) > (best["top1"], -best["logloss"])
        if better:
            best, best_scores, stale = {**m, "epoch": ep}, scores, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        flag = "  ←best" if better else ""
        print(f"  ep{ep:>2}  loss {tot/cnt:.4f}   top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}  "
              f"logloss {m['logloss']:.4f}  ECE {m['ece']:.4f}{flag}")
        if patience and stale >= patience:
            print(f"  (조기 중단: {patience} epoch 개선 없음)")
            break

    print(f"\n[최고] ep{best['epoch']}  top1 {best['top1']:.2f}  top3 {best['top3']:.2f}  "
          f"logloss {best['logloss']:.4f}  ECE {best['ece']:.4f}  ({time.time()-t0:.0f}s)")
    print("  " + str(save_pred(va_df, best_scores, tag)))
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "kind": kind, "market": market, "epoch": best["epoch"],
                "names": enc.names, "vocabs": {c: v.index for c, v in vocabs.items()}},
               RUNS / f"{tag}.pt")                                   # S5 game 채점용. gitignore

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
        print(f"\n(experiments/pred/lgb_{n_feat}.parquet 없음 — `uv run python -m model.baseline` 먼저)")

    memo = f"{STAGE[kind]} {kind}, ep{best['epoch']}/{epochs}, PL topk={topk}, {dev}"
    if drop_groups:
        memo += ", ablation -" + "/".join(drop_groups)
    if kind in ("history", "attn"):
        memo += f", 이력 L={hist_len}"
    print("\n[장부]")
    print(ledger_line(best, f"DL {STAGE[kind]} {kind}", n_feat, commit=git_commit(), seed=seed, memo=memo))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(STAGE))
    ap.add_argument("--market", action="store_true", help="77피처 (배당 포함). 없으면 73피처")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--wd", type=float)
    ap.add_argument("--patience", type=int, help="valid 개선 없는 epoch 수. 0 = 끝까지")
    ap.add_argument("--topk", type=int, default=1, help="Plackett-Luce 깊이. 1 = LGB 와 같은 1착 logloss")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-count", type=int, default=5, help="임베딩 어휘: 이 미만 등장은 <rare>")
    ap.add_argument("--tag", default="", help="예측·가중치 파일명 뒤에 붙일 접미사 (변형 실험 구분)")
    ap.add_argument("--drop-group", nargs="*", default=[], help="ablation: 제외할 피처 그룹 (예: F1)")
    ap.add_argument("--hist-len", type=int, default=10, help="S3/S4 이력 길이 (직전 출전 수)")
    a = ap.parse_args()
    d = DEFAULTS[a.kind]
    run(a.kind, a.market,
        a.epochs if a.epochs is not None else d["epochs"], a.bs,
        a.lr if a.lr is not None else d["lr"],
        a.wd if a.wd is not None else d["wd"],
        a.patience if a.patience is not None else d["patience"],
        a.topk, a.seed, a.min_count, a.tag, tuple(a.drop_group), a.hist_len)


if __name__ == "__main__":
    main()
