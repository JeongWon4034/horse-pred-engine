"""P13 — 시장 증류. 확정배당(F6_mkt_prob)을 입력이 아니라 **학습 신호**로만 쓴다. 추론은 73피처 그대로.

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.mktdistill

"73피처로 시장이 어디에 돈을 거는지 배워서 가짜 배당을 만들고, 그걸 쓰자" 는 아이디어의 정직한 시험.
  A  LGB 73 lambdarank (y_rel)                     — 기준선
  B  LGB 73 회귀 → log F6_mkt_prob (가짜 배당 생성기). 경주 내 softmax 로 확률화, 그대로 1등 고르기
  C  A 에 B 의 예측(train 은 5겹 out-of-fold) 을 피처로 추가        — 스태킹
  D  A 확률 × B 확률 기하평균                                       — 앙상블
  E  lambdarank 라벨을 시장 순위(뒤집은 F6_mkt_rank) 로            — 순위 목적함수로 같은 것
비교는 전부 valid, seed 3, 짝 비교 CI 는 model.evaluate.compare.
OOF 겹은 경주 순서대로 5등분 — 셔플 없음, 같은 경주 행은 같은 겹.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import data as D

RANK = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
REG = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def groups(df):
    return df.groupby("race_id", sort=False).size().to_numpy()


def fit_rank(t, cols, label, seed, rounds=200):
    return lgb.train({**RANK, "seed": seed}, lgb.Dataset(t[cols], label=label, group=groups(t)), rounds)


def fit_reg(t, cols, label, seed, rounds=300):
    return lgb.train({**REG, "seed": seed}, lgb.Dataset(t[cols], label=label), rounds)


def oof_market(tr_e, cols, target, seed, k=5):
    """train 에 대한 out-of-fold 시장 예측. 경주 id 순서로 k 등분 — 셔플 없음."""
    rid = tr_e["race_id"].to_numpy()
    uniq = pd.unique(rid)
    fold_of = {r: i * k // len(uniq) for i, r in enumerate(uniq)}
    f = np.array([fold_of[r] for r in rid])
    out = np.empty(len(tr_e))
    for i in range(k):
        m = fit_reg(tr_e[f != i], cols, target[f != i], seed)
        out[f == i] = m.predict(tr_e[f == i][cols])
    return out


def main(seeds=(0, 1, 2)):
    from model.evaluate import metrics, hits, compare, win_probs, race_logloss, ece
    tr, va = D.load("train"), D.load("valid")
    cols = D.feature_cols()
    tr_e, va_e = D.C.encode(tr, va, cols)
    tr_e = tr_e.copy(); va_e = va_e.copy()
    log_mkt = np.log(tr["F6_mkt_prob"].clip(1e-4, 1).to_numpy())
    mkt_rel = (tr["F6_mkt_rank"].max() - tr["F6_mkt_rank"]).clip(lower=0).astype(int).to_numpy()

    res = {}      # name -> list of valid score arrays (seed 별)
    for s in seeds:
        print(f"seed {s}", flush=True)
        a = fit_rank(tr_e, cols, tr["y_rel"], s).predict(va_e[cols])
        b_model = fit_reg(tr_e, cols, log_mkt, s)
        b = b_model.predict(va_e[cols])
        oof = oof_market(tr_e, cols, log_mkt, s)
        tr_e["mkt_hat"] = oof; va_e["mkt_hat"] = b
        c = fit_rank(tr_e, cols + ["mkt_hat"], tr["y_rel"], s).predict(va_e[cols + ["mkt_hat"]])
        d = np.log(win_probs(va, a)) + np.log(win_probs(va, b))
        e = fit_rank(tr_e, cols, mkt_rel, s).predict(va_e[cols])
        for k, v in dict(A=a, B=b, C=c, D=d, E=e).items():
            res.setdefault(k, []).append(v)
        if s == seeds[0]:
            imp = pd.Series(b_model.feature_importance("gain"), index=cols).sort_values(ascending=False)
            print("  B 가짜배당 생성기 gain 상위:", ", ".join(f"{n} {v:.0f}" for n, v in imp.head(8).items()))
            imp_c = pd.Series(lgb.train({**RANK, "seed": s}, lgb.Dataset(tr_e[cols + ["mkt_hat"]], label=tr["y_rel"],
                                                                         group=groups(tr_e)), 200).feature_importance("gain"),
                              index=cols + ["mkt_hat"]).sort_values(ascending=False)
            print(f"  C 에서 mkt_hat gain 순위: {list(imp_c.index).index('mkt_hat') + 1} / {len(imp_c)}")
            # 시장 자체를 얼마나 맞히나 — 시장 1등마 = B 1등마 비율
            mk = hits(va, np.log(va["F6_mkt_prob"].clip(1e-4, 1).to_numpy()))
            bb_top = pd.DataFrame({"r": va["race_id"], "b": b, "m": va["F6_mkt_prob"]})
            agree = bb_top.groupby("r").apply(lambda g: g["b"].idxmax() == g["m"].idxmax()).mean()
            print(f"  B 의 1등 픽이 시장 1등마와 같은 경주: {agree * 100:.1f}%  (시장 top-1 {mk.mean() * 100:.1f})")

    names = {"A": "LGB73 기준선", "B": "가짜배당 단독", "C": "A+B피처(스태킹)", "D": "A×B 앙상블", "E": "시장순위 lambdarank"}
    base_p = np.mean([win_probs(va, s) for s in res["A"]], 0)
    print(f"\n{'':22s} top-1        top-3   logloss  ECE     A 대비 [CI]")
    for k, ss in res.items():
        ms = [metrics(va, s) for s in ss]
        p = np.mean([win_probs(va, s) for s in ss], 0)
        d_, lo, hi = compare(hits(va, np.log(p)), hits(va, np.log(base_p)))
        print(f"{k} {names[k]:18s} {np.mean([m['top1'] for m in ms]):5.2f} ±{np.std([m['top1'] for m in ms]):.2f}  "
              f"{np.mean([m['top3'] for m in ms]):5.2f}   {race_logloss(va, np.log(p)):.4f}  {ece(va, np.log(p)):.4f}  "
              f"{d_:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
