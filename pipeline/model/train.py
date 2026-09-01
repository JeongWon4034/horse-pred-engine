"""학습 — 베이스라인과 타워 모델을 같은 손실로 비교한다.

    uv run python -m model.train linear      # 조건부 로지스틱 (Benter)
    uv run python -m model.train tower       # 6축 타워 + 랜덤 가중치
    uv run python -m model.train linear --no-market   # 인기도 축 제외 (주말 실시간용)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from .data import feature_columns, fit_encoder, load, to_races, tower_slices
from .evaluate import evaluate, market_baseline
from .losses import plackett_luce
from .models import LinearRanker, TowerRanker

SEED = 20260901


def run(kind: str, use_market: bool = True, epochs: int = 12, bs: int = 256, lr: float = 3e-3):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    tr_df, va_df = load("train"), load("valid")
    excl = () if use_market else ("F6",)
    cols = feature_columns(tr_df, exclude_groups=excl)

    enc = fit_encoder(tr_df, cols)
    tr, va = to_races(tr_df, enc), to_races(va_df, enc)
    print(f"[data] {time.time()-t0:.0f}s  피처 {enc.dim}개  "
          f"train {len(tr):,}경주  valid {len(va):,}경주  device={dev}")

    mk = market_baseline(va_df)
    print(f"[시장] 인기1위마  top1 {mk['top1']*100:.1f}%  top3 {mk['top3']*100:.1f}%  (n={mk['n']:,})")

    if kind == "linear":
        model = LinearRanker(enc.dim).to(dev)
        wd = 1e-4
    else:
        model = TowerRanker(tower_slices(enc)).to(dev)
        wd = 1e-3
        print(f"[tower] 축 {model.axes}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.as_tensor(tr.x); M = torch.as_tensor(tr.mask); O = torch.as_tensor(tr.order)
    n = len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs), pct_start=0.25)

    best = {"top1": 0}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)          # 경주 단위 셔플 — 경주 내부 순서는 유지된다
        tot = cnt = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            x, m, o = X[j].to(dev), M[j].to(dev), O[j].to(dev)
            loss = plackett_luce(model(x, m), m, o, topk=3)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(j); cnt += len(j)

        w = None
        if kind == "tower":       # 균등 가중치로 평가 (유저 기본값에 해당)
            w = torch.full((len(model.axes),), 1 / len(model.axes), device=dev)
        ev = evaluate(model, va, dev, w=w)
        flag = ""
        if ev["top1"] > best["top1"]:
            best = {**ev, "epoch": ep}; flag = "  ←best"
        print(f"  ep{ep:>2}  loss {tot/cnt:.4f}   "
              f"top1 {ev['top1']*100:5.2f}%  top3 {ev['top3']*100:5.2f}%  "
              f"ECE {ev['ece']:.4f}{flag}")

    print(f"\n[최고] ep{best['epoch']}  top1 {best['top1']*100:.2f}%  "
          f"top3 {best['top3']*100:.2f}%  ECE {best['ece']:.4f}")
    print(f"[시장 대비] top1 {(best['top1']-mk['top1'])*100:+.2f}%p  "
          f"top3 {(best['top3']-mk['top3'])*100:+.2f}%p")

    if kind == "tower":
        print("\n[축 단독 성능] 그 축만 100, 나머지 0")
        for a, ax in enumerate(model.axes):
            w = torch.zeros(len(model.axes), device=dev); w[a] = 1.0
            e = evaluate(model, va, dev, w=w)
            print(f"  {ax:<4} top1 {e['top1']*100:5.2f}%  top3 {e['top3']*100:5.2f}%")

    return model, enc


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "linear"
    run(kind, use_market="--no-market" not in sys.argv)
