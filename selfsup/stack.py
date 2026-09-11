"""P12 — 스태킹 메타모델. "어느 경주에서 어느 모델을 믿을지" 를 학습한다.

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.stack

기저 모델 3개(S3 seed5 · 사전학습 seed5 · LGB73)의 valid 확률 위에, 경주 특성(두수·거리·경마장·각 모델의 확신도)을 붙여
작은 lambdarank 를 얹는다. train 에 대한 out-of-fold 예측이 없으므로 **valid 를 경주 단위 5-fold 로 교차적합**한다 —
4/5 에서 메타를 학습하고 1/5 를 예측, 다섯 조각을 모아 단순 평균(1:1:1)과 짝 비교. 반복 5회(fold seed) 평균.
정원 result_dl.md §3 의 '앙상블 가중치 교차 적합' 과 같은 원칙이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import data as D

BASES = {"s3": [f"history_73_L20_{s}" for s in ("s1", "s2", "s3", "s4", "s20260901")],
         "ssl": [f"selfsup_mask_base_ft_s{i}" for i in range(1, 6)],
         "lgb": ["lgb_73"]}


def entropy(p, rid):
    s = pd.Series(-p * np.log(p + 1e-12)).groupby(rid, sort=False).transform("sum")
    return s.to_numpy()


def main(n_rep=5):
    from model.evaluate import win_probs, load_pred, metrics, hits, compare, race_logloss
    va = D.load("valid"); rid = va["race_id"].to_numpy()
    P = {k: np.mean([win_probs(va, load_pred(va, n)) for n in names], 0) for k, names in BASES.items()}
    X = pd.DataFrame({f"lp_{k}": np.log(p + 1e-12) for k, p in P.items()})
    X["p_avg"] = np.log(np.mean(list(P.values()), 0) + 1e-12)
    X["dusu"] = va["X_dusu"].to_numpy(float); X["dist"] = va["X_rcDist"].to_numpy(float) / 1000
    X["meet"] = va["race_id"].str.split("_").str[1].astype(int).to_numpy()
    for k, p in P.items():
        X[f"ent_{k}"] = entropy(p, rid)                                       # 그 모델이 이 경주를 얼마나 혼전으로 보나
        X[f"top_{k}"] = pd.Series(p).groupby(rid, sort=False).transform("max").to_numpy()
    X["disagree"] = (X[["lp_s3", "lp_ssl", "lp_lgb"]].max(1) - X[["lp_s3", "lp_ssl", "lp_lgb"]].min(1)).to_numpy()
    y = va["y_rel"].to_numpy()
    races = np.unique(rid)
    avg = X["p_avg"].to_numpy()
    print(f"기저: S3 {metrics(va, X['lp_s3'])['top1']:.2f} · 사전학습 {metrics(va, X['lp_ssl'])['top1']:.2f} · LGB {metrics(va, X['lp_lgb'])['top1']:.2f} · 평균 {metrics(va, avg)['top1']:.2f} (ll {race_logloss(va, avg):.4f})")

    outs = []
    for rep in range(n_rep):
        rng = np.random.default_rng(20260901 + rep); perm = rng.permutation(races); folds = np.array_split(perm, 5)
        pred = np.zeros(len(va))
        for f in folds:
            te = np.isin(rid, f); trm = ~te
            g = pd.Series(rid[trm]).groupby(rid[trm], sort=False).size().reindex(pd.unique(rid[trm])).to_numpy()
            m = lgb.train(dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=4, min_data_in_leaf=50,
                               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=rep),
                          lgb.Dataset(X[trm], label=y[trm], group=g), num_boost_round=150)
            pred[te] = m.predict(X[te])
        mm = metrics(va, pred); d_, lo, hi = compare(hits(va, pred), hits(va, avg))
        outs.append((mm["top1"], mm["top3"], race_logloss(va, pred), d_, lo, hi))
        print(f"  rep{rep}: 스태킹 top1 {mm['top1']:.2f} top3 {mm['top3']:.2f} ll {outs[-1][2]:.4f}  평균 대비 {d_:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    o = np.array(outs)
    print(f"\n스태킹 5회 평균: top1 {o[:,0].mean():.2f} ±{o[:,0].std():.2f}  top3 {o[:,1].mean():.2f}  ll {o[:,2].mean():.4f}  "
          f"단순평균 대비 {o[:,3].mean():+.2f} (CI 평균 [{o[:,4].mean():+.2f}, {o[:,5].mean():+.2f}])")
    imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("메타 gain 상위:", imp.head(6).round(0).to_dict())


if __name__ == "__main__":
    main()
