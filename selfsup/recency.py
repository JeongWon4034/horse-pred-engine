"""P11 — 최근성 가중 학습. train 2010~2025 를 동등하게 학습하던 것을 최근 연도에 무게를 준다.

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.recency

EDA 가 드리프트(X_rating·F1_prize PSI, 2020~21 무배당 경주)를 찾았는데 학습은 전 연도 동등 가중이었다. valid 가 최신 구간이라
효과가 있으면 바로 보인다. LightGBM lambdarank 200r · seed 3 · 73피처, 가중치만 바꾼다.
  균등          w = 1
  선형 3배      2010 → 1, 2025 → 3 (선형)
  반감기 4년    w = 0.5^((2025−y)/4)
  최근 3년 ×3   2023~ 는 3, 그 전 1
  최근 5년만    2021~ 만 학습 (행 수 감소)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import data as D


def main(seeds=(0, 1, 2)):
    from model.evaluate import metrics, hits, compare, win_probs, race_logloss
    tr, va = D.load("train"), D.load("valid")
    cols = D.feature_cols()
    tr_e, va_e = D.C.encode(tr, va, cols)
    y = (tr["rcDate"] // 10000).to_numpy()
    schemes = {
        "균등": np.ones(len(tr)),
        "선형 1→3": 1 + 2 * (y - y.min()) / (y.max() - y.min()),
        "반감기 4년": 0.5 ** ((y.max() - y) / 4),
        "최근3년 ×3": np.where(y >= 2023, 3.0, 1.0),
        "최근5년만": np.where(y >= 2021, 1.0, 0.0),
    }
    base = None; rows = []
    for name, w in schemes.items():
        keep = w > 0
        t = tr_e[keep]; g = t.groupby("race_id", sort=False).size().to_numpy()
        ss = []
        for s in seeds:
            m = lgb.train(dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=s),
                          lgb.Dataset(t[cols], label=t["y_rel"], group=g, weight=w[keep]), num_boost_round=200)
            ss.append(m.predict(va_e[cols]))
        ms = [metrics(va, s) for s in ss]
        p = np.mean([win_probs(va, s) for s in ss], 0)
        if base is None:
            base = p
        d_, lo, hi = compare(hits(va, np.log(p)), hits(va, np.log(base)))
        rows.append((name, int(keep.sum()), np.mean([x["top1"] for x in ms]), np.std([x["top1"] for x in ms]),
                     np.mean([x["top3"] for x in ms]), race_logloss(va, np.log(p)), d_, lo, hi))
        print(f"{name:10s} 행 {int(keep.sum()):>7,}  top1 {rows[-1][2]:.2f} ±{rows[-1][3]:.2f}  top3 {rows[-1][4]:.2f}  "
              f"ll {rows[-1][5]:.4f}  균등 대비 {d_:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
