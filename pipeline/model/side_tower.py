"""사이드 실험 — S5 6축 헤드 (게임 모드, 77피처).

축(F1 전적 · F2 혈통 · F3 기수/조교사 · F4 조건 · F5 주행 스타일 · F6 인기도)마다 작은 MLP 를 두고,
학습 중 축 가중치를 Dirichlet(α=0.7) 에서 매 배치 새로 뽑는다. 유저가 슬라이더를 극단으로 밀어도
모델이 무너지지 않게 하려는 것. 평가는 균등 가중치(유저 기본값).

    PYTHONUTF8=1 uv run python -m model.side_tower            # 77피처
    PYTHONUTF8=1 uv run python -m model.side_tower --no-market  # 73피처 (F6 축 없음)

확인할 것: 축 단독 top-1 바닥(≥25 가 목표), 축 간 예측 상관(≤0.9), 균등 가중 vs 최고 단일 축.
기존 models.TowerRanker 를 그대로 쓴다(수정 없음). 평가 시 가중치를 주는 부분만 여기서 처리.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .data import feature_cols, fit_encoder, flatten_scores, load, to_races, tower_slices  # noqa: E402
from .evaluate import compare, hits, ledger_line, load_pred, metrics, save_pred  # noqa: E402
from .losses import plackett_luce  # noqa: E402
from .models import TowerRanker  # noqa: E402
from .team import C  # noqa: E402

SEED = 20260901


@torch.no_grad()
def predict(model, races, dev, w, batch=1024):
    model.eval(); out = []
    for i in range(0, len(races), batch):
        x = torch.as_tensor(races.x[i:i + batch], device=dev); m = torch.as_tensor(races.mask[i:i + batch], device=dev)
        out.append(model(x, m, w).masked_fill(m == 0, -1e9).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-market", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    market = not a.no_market; n_feat = 77 if market else 73
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    tr_df, va_df = load("train"), load("valid")
    cols = feature_cols(exclude_pop=not market)
    enc = fit_encoder(tr_df, cols); tr, va = to_races(tr_df, enc), to_races(va_df, enc)
    slices = tower_slices(enc)
    model = TowerRanker(slices, hidden=a.hidden, alpha=a.alpha).to(dev)
    A = len(model.axes); eq = torch.full((A,), 1 / A, device=dev)
    print(f"[S5 tower {n_feat}] 축 {model.axes}  파라미터 {sum(p.numel() for p in model.parameters()):,}  device={dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    X = torch.as_tensor(tr.x, device=dev); M = torch.as_tensor(tr.mask, device=dev); O = torch.as_tensor(tr.order, device=dev)
    n, bs = len(tr), 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((n + bs - 1) // bs), pct_start=0.25)
    best = best_scores = best_state = None; stale = 0
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(n, device=dev); tot = cnt = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = plackett_luce(model(X[j], M[j]), M[j], O[j], topk=3)      # 가중치 = Dirichlet 샘플, 3착까지
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(j); cnt += len(j)
        scores = flatten_scores(predict(model, va, dev, eq), va); m = metrics(va_df, scores)
        better = best is None or (m["top1"], -m["logloss"]) > (best["top1"], -best["logloss"])
        if better: best, best_scores, best_state, stale = {**m, "epoch": ep}, scores, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        print(f"  ep{ep:>2}  loss {tot/cnt:.4f}   top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}  logloss {m['logloss']:.4f}{'  ←best' if better else ''}")
        if stale >= 5: print("  (조기 중단)"); break
    model.load_state_dict(best_state)
    print(f"\n[최고] ep{best['epoch']}  top1 {best['top1']:.2f}  top3 {best['top3']:.2f}  logloss {best['logloss']:.4f}  ({time.time()-t0:.0f}s)")
    tag = f"side_tower_{n_feat}_s{a.seed}"
    save_pred(va_df, best_scores, tag)
    torch.save({"state": best_state, "axes": model.axes, "names": enc.names, "market": market}, f"../experiments/runs/{tag}.pt")

    print("\n[축 단독 — 그 축만 100, 나머지 0]")
    axis_scores = {}
    for k, ax in enumerate(model.axes):
        w = torch.zeros(A, device=dev); w[k] = 1.0
        s = flatten_scores(predict(model, va, dev, w), va); axis_scores[ax] = s; m = metrics(va_df, s)
        print(f"  {ax:<4} top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}  logloss {m['logloss']:.4f}")
    print("\n[축 간 예측 상관 (행 점수 피어슨)]")
    names = list(axis_scores); mat = np.corrcoef(np.stack([axis_scores[k] for k in names]))
    print("      " + "".join(f"{k:>7}" for k in names))
    for i, k in enumerate(names): print(f"  {k:<4}" + "".join(f"{mat[i,j]:7.2f}" for j in range(len(names))))
    print(f"  최대 비대각 상관 {np.abs(mat - np.eye(len(names))).max():.2f}")

    print("\n[극단 가중치 3종 — 유저 프리셋 흉내]")
    for name, w in (("혈통 100", {"F2": 1.0}), ("기수 70 전적 30", {"F3": .7, "F1": .3}), ("인기도 0 나머지 균등", {k: 1 / (A - 1) for k in model.axes if k != "F6"})):
        wt = torch.tensor([w.get(k, 0.0) for k in model.axes], device=dev)
        m = metrics(va_df, flatten_scores(predict(model, va, dev, wt), va)); print(f"  {name:<18} top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}")

    lgb = load_pred(va_df, f"lgb_{n_feat}")
    if lgb is not None:
        d = compare(hits(va_df, best_scores), hits(va_df, lgb)); print(f"\n[균등 가중 − LGB] {d[0]:+.2f}%p CI [{d[1]:+.2f}, {d[2]:+.2f}]")
    print("\n[장부]"); print(ledger_line(best, f"S5 tower 균등가중", n_feat, seed=a.seed, memo=f"6축 MLP({a.hidden}) Dirichlet α={a.alpha}, PL topk=3, ep{best['epoch']}"))


if __name__ == "__main__":
    main()
