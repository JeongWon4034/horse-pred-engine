"""P8 — 장구(hrTool). 경주성적 API 필드인데 팀 피처에 없다. 2016년부터 채워져 있고 출발 전 행에도 들어 있다(실시간 가능).

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.gear build
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.gear eval

값 예: '망사눈, 승인재갈' · '계란형큰,망사' · '눈가면' · '-'(없음). 2010~2015 는 전부 '-' 라 결측(NaN) 으로 둔다.
as-of 피처 5개 — 전부 이번 출전표와 직전 출전 기록만 쓴다.
  g_n        장구 개수
  g_eye      눈 계열(눈가면·양눈가면·망사눈·망사·반가지) 착용 0/1     ← 블링커 계열
  g_bit      재갈 계열(승인재갈·계란형·…재갈) 0/1
  g_change   직전 출전과 장구 구성이 다르다 0/1                       ← 문헌의 '장구 변경'
  g_first_eye 이번에 처음 눈 계열 착용 0/1                            ← '첫 블링커'
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from . import data as D

LEDGER = D.PIPE / "data" / "raw" / "ledger_api_orig.csv"
OUT = D.RUNS / "gear.parquet"
FEATS = ["g_n", "g_eye", "g_bit", "g_change", "g_first_eye"]
EYE = re.compile(r"눈|망사|가지")
BIT = re.compile(r"재갈|계란")


def _items(s: str) -> frozenset:
    if not isinstance(s, str) or s.strip() in ("", "-"):
        return frozenset()
    return frozenset(x.strip() for x in s.split(",") if x.strip())


def build():
    d = pd.read_csv(LEDGER, dtype=str, low_memory=False, usecols=["rcDate", "meet", "rcNo", "hrNo", "hrTool"])
    meet = d["meet"].map({"서울": 1, "제주": 2, "부경": 3, "부산경남": 3, "1": 1, "2": 2, "3": 3})
    d["race_id"] = d["rcDate"].astype(str) + "_" + meet.astype("Int64").astype(str) + "_" + pd.to_numeric(d["rcNo"]).astype("Int64").astype(str)
    d["rcDate"] = pd.to_numeric(d["rcDate"], errors="coerce")
    d = d.dropna(subset=["rcDate", "hrNo"]).sort_values(["hrNo", "rcDate", "race_id"], kind="stable").reset_index(drop=True)
    known = d["rcDate"] >= 20160101                                  # 그 전은 필드가 비어 있다
    sets = d["hrTool"].map(_items)
    d["g_n"] = sets.map(len).astype(float)
    d["g_eye"] = sets.map(lambda s: float(any(EYE.search(x) for x in s)))
    d["g_bit"] = sets.map(lambda s: float(any(BIT.search(x) for x in s)))
    prev = sets.groupby(d["hrNo"]).shift(1)
    prev_known = known.groupby(d["hrNo"]).shift(1).fillna(False).astype(bool)
    d["g_change"] = np.where(prev_known & known, (sets != prev).astype(float), np.nan)
    prev_eye = prev.map(lambda s: any(EYE.search(x) for x in s) if isinstance(s, frozenset) else np.nan)
    d["g_first_eye"] = np.where(prev_known & known, ((d["g_eye"] == 1) & (prev_eye == False)).astype(float), np.nan)
    for c in ("g_n", "g_eye", "g_bit"):
        d.loc[~known, c] = np.nan
    out = d[["race_id", "hrNo"] + FEATS].copy(); out["hrNo"] = out["hrNo"].astype(str)
    out.to_parquet(OUT, index=False)
    k = out[out["g_n"].notna()]
    print(f"→ {OUT}  행 {len(out):,}  2016~ 충전 {len(k)/len(out):.2f} · 눈계열 {k['g_eye'].mean():.2f} · 재갈 {k['g_bit'].mean():.2f} · "
          f"변경 {k['g_change'].mean():.2f} · 첫눈 {k['g_first_eye'].mean():.3f}")


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
    for name, cs in (("lgb73", cols), ("lgb73+gear", cols + FEATS)):
        ss = []
        for s in seeds:
            m = lgb.train(dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31,
                               min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                               verbose=-1, seed=s),
                          lgb.Dataset(tr_e[cs], label=tr_e["y_rel"], group=g), num_boost_round=200)
            ss.append(m.predict(va_e[cs]))
        res[name] = ss
        ms = [metrics(va, s) for s in ss]
        print(f"{name:12s} top1 {np.mean([x['top1'] for x in ms]):.2f} ±{np.std([x['top1'] for x in ms]):.2f}  "
              f"top3 {np.mean([x['top3'] for x in ms]):.2f}  ll {np.mean([x['logloss'] for x in ms]):.4f}")
        if name.endswith("gear"):
            imp = pd.Series(m.feature_importance("gain"), index=cs).sort_values(ascending=False)
            print("  장구 gain 순위:", {f: int((imp.index == f).argmax()) + 1 for f in FEATS}, "/", len(cs))
    a = np.mean([win_probs(va, s) for s in res["lgb73+gear"]], axis=0)
    b = np.mean([win_probs(va, s) for s in res["lgb73"]], axis=0)
    d_, lo, hi = compare(hits(va, np.log(a)), hits(va, np.log(b)))
    print(f"짝 비교 (+gear − 기본): {d_:+.2f} [{lo:+.2f}, {hi:+.2f}] → {'차이 없음' if lo <= 0 <= hi else '유의'}")


if __name__ == "__main__":
    {"build": build, "eval": eval_lgb}[sys.argv[1]]()
