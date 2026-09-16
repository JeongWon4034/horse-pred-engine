"""P7 — 마체중(wgHr). 원장에 있는데 팀 73피처에 없는 열이다 (tier B: 발주 60~90분 전 확정).

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.bodyweight build      # 원장 → runs/bodyweight.parquet
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.bodyweight eval       # LGB73 vs LGB73+마체중3 (seed 3)

말별 as-of 로 셋을 만든다 — 전부 "그 경주 이전" 정보와 당일 계체값만 쓴다.
  bw        이번 경주 마체중 (kg)                       — 당일 계체, 발주 전 공개
  bw_diff   직전 출전 대비 증감 (kg)                     — 급격한 감량/증량 = 컨디션 신호
  bw_dev    최근 5출전 평균 대비 편차 (kg)                — 그 말의 '평소 체중' 기준
검증은 팀 기준선과 같은 자(LightGBM lambdarank 200r, valid, seed 3)로 먼저 한다. LGB 에서 안 오르면 DL 도 안 오른다.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data as D

LEDGER = D.PIPE / "data" / "raw" / "ledger_api_orig.csv"
BW = D.RUNS / "bodyweight.parquet"
FEATS = ["bw", "bw_diff", "bw_dev"]


def build():
    d = pd.read_csv(LEDGER, dtype=str, low_memory=False, usecols=["rcDate", "meet", "rcNo", "hrNo", "wgHr"])
    meet = d["meet"].map({"서울": 1, "제주": 2, "부경": 3, "부산경남": 3, "1": 1, "2": 2, "3": 3})
    d["race_id"] = d["rcDate"].astype(str) + "_" + meet.astype("Int64").astype(str) + "_" + pd.to_numeric(d["rcNo"]).astype("Int64").astype(str)
    d["rcDate"] = pd.to_numeric(d["rcDate"], errors="coerce")
    # 원장 표기 '461(+3)' = 마체중(괄호 안은 직전 대비 공식 증감). '0()' '()' 는 결측
    ext = d["wgHr"].astype(str).str.extract(r"^\s*(\d+)\s*\(\s*([+-]?\d+)?\s*\)")
    d["bw"] = pd.to_numeric(ext[0], errors="coerce").where(lambda s: s > 300)
    d["bw_diff_off"] = pd.to_numeric(ext[1], errors="coerce").where(d["bw"].notna())
    d = d.dropna(subset=["rcDate", "hrNo"]).sort_values(["hrNo", "rcDate", "race_id"], kind="stable")
    g = d.groupby("hrNo")["bw"]
    d["bw_diff"] = (d["bw"] - g.shift(1)).fillna(d["bw_diff_off"])                   # 직전 출전 대비 (없으면 공식 증감)
    d["bw_dev"] = d["bw"] - g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    out = d[["race_id", "hrNo", "bw", "bw_diff", "bw_dev"]].copy(); out["hrNo"] = out["hrNo"].astype(str)
    out.to_parquet(BW, index=False)
    print(f"→ {BW}  행 {len(out):,}  bw 충전 {out['bw'].notna().mean():.2f}  diff 충전 {out['bw_diff'].notna().mean():.2f}")


def attach(df: pd.DataFrame) -> pd.DataFrame:
    bw = pd.read_parquet(BW)
    df = df.copy(); df["hrNo"] = df["hrNo"].astype(str)
    m = df.merge(bw, on=["race_id", "hrNo"], how="left")
    assert len(m) == len(df)
    m.index = df.index
    return m


def eval_lgb(seeds=(0, 1, 2)):
    import lightgbm as lgb
    from model.evaluate import metrics, hits, compare
    tr, va = attach(D.load("train")), attach(D.load("valid"))
    cols = D.feature_cols()
    tr_e, va_e = D.C.encode(tr, va, cols + FEATS)
    g = tr_e.groupby("race_id", sort=False).size().to_numpy()
    res = {}
    for name, cs in (("lgb73", cols), ("lgb73+bw", cols + FEATS)):
        ss = []
        for s in seeds:
            m = lgb.train(dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31,
                               min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                               verbose=-1, seed=s),
                          lgb.Dataset(tr_e[cs], label=tr_e["y_rel"], group=g), num_boost_round=200)
            ss.append(m.predict(va_e[cs]))
        res[name] = ss
        ms = [metrics(va, s) for s in ss]
        print(f"{name:10s} top1 {np.mean([x['top1'] for x in ms]):.2f} ±{np.std([x['top1'] for x in ms]):.2f}  "
              f"top3 {np.mean([x['top3'] for x in ms]):.2f}  ll {np.mean([x['logloss'] for x in ms]):.4f}")
        if name == "lgb73+bw":
            imp = pd.Series(m.feature_importance("gain"), index=cs).sort_values(ascending=False)
            print("  마체중 3개 gain 순위:", {f: int((imp.index == f).argmax()) + 1 for f in FEATS}, "/", len(cs))
    a = np.mean([np.exp(D.E.win_probs(va, s)) for s in res["lgb73+bw"]], axis=0)
    b = np.mean([np.exp(D.E.win_probs(va, s)) for s in res["lgb73"]], axis=0)
    d, lo, hi = compare(hits(va, np.log(a)), hits(va, np.log(b)))
    print(f"짝 비교 (+bw − 기본, seed 평균 확률): {d:+.2f} [{lo:+.2f}, {hi:+.2f}] → {'차이 없음' if lo <= 0 <= hi else '유의'}")


if __name__ == "__main__":
    {"build": build, "eval": eval_lgb}[sys.argv[1]]()
