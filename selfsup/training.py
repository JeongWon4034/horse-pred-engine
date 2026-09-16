"""P10 — 조교(일별훈련) as-of 피처. 원장 pipeline/data/raw/training_api.csv (training_fetch.py).

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.training build
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.training eval

전문지가 보는 '경주 사이의 상태' 를 재는 유일한 공개 데이터다. 정원이 잰 F1_tr_sessions_28d 는 횟수뿐이었고 0 이었다.
여기서는 시간·강도(습보)·최근성을 넣는다. 전부 경주일 **이전** 훈련만 (엄격히 <).
  tr_n14 · tr_n28        14·28일 훈련 횟수
  tr_sec14               14일 훈련시간 합 (초)
  tr_fast14              14일 습보(run2) 횟수            ← 강도
  tr_fast_ratio28        28일 습보 / (구보+습보)
  tr_days_since          마지막 훈련 경과일
  tr_jk14                14일 중 기수 기승 횟수           ← 실전 조교
  tr_sec_trend           최근 14일 시간합 − 그 전 14일 시간합  ← 상승/하강
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data as D

RAW = D.PIPE / "data" / "raw" / "training_api.csv"
OUT = D.RUNS / "training.parquet"
FEATS = ["tr_n14", "tr_n28", "tr_sec14", "tr_fast14", "tr_fast_ratio28", "tr_days_since", "tr_jk14", "tr_sec_trend"]


def build():
    t = pd.read_csv(RAW, dtype=str, low_memory=False, usecols=["trDate", "hrNo", "prGubun", "run1Cnt", "run2Cnt", "trTerm"])
    t["day"] = pd.to_datetime(t["trDate"].str[:8], format="%Y%m%d", errors="coerce")
    for c in ("run1Cnt", "run2Cnt", "trTerm"):
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0)
    t["jk"] = t["prGubun"].astype(str).str.contains("기수").astype(int)
    t = t.dropna(subset=["day", "hrNo"])
    # 하루 여러 세션 → 일 단위 집계
    g = t.groupby(["hrNo", "day"], as_index=False).agg(n=("trTerm", "size"), sec=("trTerm", "sum"),
                                                        fast=("run2Cnt", "sum"), slow=("run1Cnt", "sum"), jk=("jk", "max"))
    g = g.sort_values(["hrNo", "day"]).reset_index(drop=True)
    print(f"조교 원장 {len(t):,}행 → 말·일 {len(g):,}행, 말 {g['hrNo'].nunique():,}두, {g['day'].min().date()} ~ {g['day'].max().date()}")

    # 경주 행(모든 분할 + 오늘 미출발)에 대해 as-of 창 집계
    races = []
    for sp in ("train", "valid", "test", "game"):
        races.append(pd.read_parquet(D.DATA / "dataset" / "model" / f"{sp}.parquet", columns=["race_id", "hrNo", "rcDate"]))
    live = D.PIPE / "data" / "build" / "dataset" / "v2" / "model" / "new.parquet"
    if live.exists():
        races.append(pd.read_parquet(live, columns=["race_id", "hrNo", "rcDate"]))
    r = pd.concat(races, ignore_index=True).drop_duplicates(["race_id", "hrNo"])
    r["hrNo"] = r["hrNo"].astype(str); r["day"] = pd.to_datetime(r["rcDate"].astype(str), format="%Y%m%d")
    r = r.sort_values(["hrNo", "day"]).reset_index(drop=True)

    out = []
    gi = {h: df for h, df in g.groupby("hrNo")}
    for h, rr in r.groupby("hrNo", sort=False):
        tg = gi.get(h)
        if tg is None:
            out.append(pd.DataFrame({"race_id": rr["race_id"], "hrNo": h}).assign(**{f: np.nan for f in FEATS})); continue
        days = tg["day"].to_numpy(); sec = tg["sec"].to_numpy(); fast = tg["fast"].to_numpy(); slow = tg["slow"].to_numpy(); jk = tg["jk"].to_numpy()
        csec = np.r_[0, np.cumsum(sec)]; cfast = np.r_[0, np.cumsum(fast)]; cslow = np.r_[0, np.cumsum(slow)]; cjk = np.r_[0, np.cumsum(jk)]
        rd = rr["day"].to_numpy()
        e = np.searchsorted(days, rd, side="left")                              # 경주일 이전(<) 훈련 수
        i14 = np.searchsorted(days, rd - np.timedelta64(14, "D"), side="left")
        i28 = np.searchsorted(days, rd - np.timedelta64(28, "D"), side="left")
        n14 = e - i14; n28 = e - i28
        sec14 = csec[e] - csec[i14]; sec_prev14 = csec[i14] - csec[i28]
        fast14 = cfast[e] - cfast[i14]; fast28 = cfast[e] - cfast[i28]; slow28 = cslow[e] - cslow[i28]
        jk14 = cjk[e] - cjk[i14]
        last = np.where(e > 0, (rd - days[np.clip(e - 1, 0, None)]).astype("timedelta64[D]").astype(float), np.nan)
        out.append(pd.DataFrame({"race_id": rr["race_id"].to_numpy(), "hrNo": h, "tr_n14": n14, "tr_n28": n28, "tr_sec14": sec14,
                                 "tr_fast14": fast14, "tr_fast_ratio28": np.where(fast28 + slow28 > 0, fast28 / np.maximum(fast28 + slow28, 1), np.nan),
                                 "tr_days_since": last, "tr_jk14": jk14, "tr_sec_trend": sec14 - sec_prev14}))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(OUT, index=False)
    k = res["tr_n28"].notna() & (res["tr_n28"] > 0)
    print(f"→ {OUT}  경주행 {len(res):,}  28일 내 훈련 있음 {k.mean():.2f}  n14 평균 {res.loc[k,'tr_n14'].mean():.1f}  습보비율 {res.loc[k,'tr_fast_ratio28'].mean():.2f}")


def attach(df: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(OUT)
    df = df.copy(); df["hrNo"] = df["hrNo"].astype(str)
    m = df.merge(g, on=["race_id", "hrNo"], how="left"); assert len(m) == len(df); m.index = df.index
    return m


def eval_lgb(seeds=(0, 1, 2)):
    import lightgbm as lgb
    from model.evaluate import metrics, hits, compare, win_probs, race_logloss
    tr, va = attach(D.load("train")), attach(D.load("valid"))
    cols = D.feature_cols()
    tr_e, va_e = D.C.encode(tr, va, cols + FEATS)
    g = tr_e.groupby("race_id", sort=False).size().to_numpy()
    res = {}
    for name, cs in (("lgb73", cols), ("lgb73+training", cols + FEATS)):
        ss = []
        for s in seeds:
            m = lgb.train(dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=s),
                          lgb.Dataset(tr_e[cs], label=tr_e["y_rel"], group=g), num_boost_round=200)
            ss.append(m.predict(va_e[cs]))
        res[name] = ss; ms = [metrics(va, s) for s in ss]
        print(f"{name:15s} top1 {np.mean([x['top1'] for x in ms]):.2f} ±{np.std([x['top1'] for x in ms]):.2f}  top3 {np.mean([x['top3'] for x in ms]):.2f}  ll {np.mean([x['logloss'] for x in ms]):.4f}")
        if name.endswith("training"):
            imp = pd.Series(m.feature_importance("gain"), index=cs).sort_values(ascending=False)
            print("  조교 gain 순위:", {f: int((imp.index == f).argmax()) + 1 for f in FEATS}, "/", len(cs))
    a = np.mean([win_probs(va, s) for s in res["lgb73+training"]], 0); b = np.mean([win_probs(va, s) for s in res["lgb73"]], 0)
    d_, lo, hi = compare(hits(va, np.log(a)), hits(va, np.log(b)))
    print(f"짝 비교 (+training − 기본): {d_:+.2f} [{lo:+.2f}, {hi:+.2f}] → {'차이 없음' if lo <= 0 <= hi else '유의'}   "
          f"logloss {race_logloss(va, np.log(a)):.4f} vs {race_logloss(va, np.log(b)):.4f}")


if __name__ == "__main__":
    {"build": build, "eval": eval_lgb}[sys.argv[1]]()
