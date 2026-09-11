"""P9 — 말의 과거 인기도 (as-of). 당일 배당(F6, tier G)은 발주 전에 없지만, **과거 경주의 확정배당**은 공개 데이터다.
"시장이 늘 높게 보는 말"이라는 정보를 as-of 로 옮긴다. 배당 그 자체가 아니라 시장 평판의 지속분.

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.pastodds build
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.pastodds eval

as-of 4개 (전부 '그 경주 이전' 출전만):
  po_rank3   직전 3출전 인기순위 평균 (1=최고인기)
  po_prob5   직전 5출전 시장 내재확률(오버라운드 제거) 평균
  po_rank_last 직전 출전 인기순위
  po_n       계산에 쓰인 출전 수 (신뢰도)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data as D

LEDGER = D.PIPE / "data" / "raw" / "ledger_api_orig.csv"
OUT = D.RUNS / "pastodds.parquet"
FEATS = ["po_rank3", "po_prob5", "po_rank_last", "po_n"]


def build():
    d = pd.read_csv(LEDGER, dtype=str, low_memory=False, usecols=["rcDate", "meet", "rcNo", "hrNo", "winOdds"])
    meet = d["meet"].map({"서울": 1, "제주": 2, "부경": 3, "부산경남": 3, "1": 1, "2": 2, "3": 3})
    d["race_id"] = d["rcDate"].astype(str) + "_" + meet.astype("Int64").astype(str) + "_" + pd.to_numeric(d["rcNo"]).astype("Int64").astype(str)
    d["rcDate"] = pd.to_numeric(d["rcDate"], errors="coerce")
    odds = pd.to_numeric(d["winOdds"], errors="coerce")
    odds = odds.where((odds > 0) & (odds < 900))                      # 0·9999.9 는 결측
    inv = 1.0 / odds
    d["mkt_prob"] = inv / inv.groupby(d["race_id"]).transform("sum")
    d["mkt_rank"] = odds.groupby(d["race_id"]).rank(method="min")
    d = d.dropna(subset=["rcDate", "hrNo"]).sort_values(["hrNo", "rcDate", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("hrNo")
    d["po_rank3"] = g["mkt_rank"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    d["po_prob5"] = g["mkt_prob"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    d["po_rank_last"] = g["mkt_rank"].shift(1)
    d["po_n"] = g["mkt_rank"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).count())
    out = d[["race_id", "hrNo"] + FEATS].copy(); out["hrNo"] = out["hrNo"].astype(str)
    out.to_parquet(OUT, index=False)
    print(f"→ {OUT}  행 {len(out):,}  po_rank3 충전 {out['po_rank3'].notna().mean():.2f}")


def attach(df: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(OUT)
    df = df.copy(); df["hrNo"] = df["hrNo"].astype(str)
    m = df.merge(g, on=["race_id", "hrNo"], how="left"); assert len(m) == len(df); m.index = df.index
    return m


def eval_lgb(seeds=(0, 1, 2)):
    import lightgbm as lgb
    from model.evaluate import metrics, hits, compare, win_probs
    tr, va = attach(D.load("train")), attach(D.load("valid"))
    cols = D.feature_cols()
    tr_e, va_e = D.C.encode(tr, va, cols + FEATS)
    g = tr_e.groupby("race_id", sort=False).size().to_numpy()
    res = {}
    for name, cs in (("lgb73", cols), ("lgb73+pastodds", cols + FEATS)):
        ss = []
        for s in seeds:
            m = lgb.train(dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31,
                               min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                               verbose=-1, seed=s),
                          lgb.Dataset(tr_e[cs], label=tr_e["y_rel"], group=g), num_boost_round=200)
            ss.append(m.predict(va_e[cs]))
        res[name] = ss
        ms = [metrics(va, s) for s in ss]
        print(f"{name:15s} top1 {np.mean([x['top1'] for x in ms]):.2f} ±{np.std([x['top1'] for x in ms]):.2f}  "
              f"top3 {np.mean([x['top3'] for x in ms]):.2f}  ll {np.mean([x['logloss'] for x in ms]):.4f}")
        if name.endswith("pastodds"):
            imp = pd.Series(m.feature_importance("gain"), index=cs).sort_values(ascending=False)
            print("  과거인기도 gain 순위:", {f: int((imp.index == f).argmax()) + 1 for f in FEATS}, "/", len(cs))
    a = np.mean([win_probs(va, s) for s in res["lgb73+pastodds"]], axis=0)
    b = np.mean([win_probs(va, s) for s in res["lgb73"]], axis=0)
    d_, lo, hi = compare(hits(va, np.log(a)), hits(va, np.log(b)))
    print(f"짝 비교 (+pastodds − 기본): {d_:+.2f} [{lo:+.2f}, {hi:+.2f}] → {'차이 없음' if lo <= 0 <= hi else '유의'}")


if __name__ == "__main__":
    {"build": build, "eval": eval_lgb}[sys.argv[1]]()
