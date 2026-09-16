"""P14 — 당일 앞 경주 신호. 같은 날·같은 경마장에서 **먼저 끝난 경주**의 결과만 보고 만든 as-of 피처.

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.dayflow

시장 우위의 정체가 "경주 당일 새로 생기는 정보" 라면(P13), 공공데이터 안에서 그 성격에 가장 가까운 건 그날 앞 경주다.
전문가가 실제로 보는 "오늘 트랙 편향" — 안쪽 게이트가 유리한 날인지, 선행마가 버티는 날인지, 인기마가 오는 날인지,
기수·조교사가 오늘 컨디션이 어떤지. 6경주 출발 전에 1~5경주는 이미 API 로 들어와 있으니 실시간에도 그대로 쓸 수 있다.
경주 1 은 전부 결측(LightGBM 이 처리). 크롤링 없음 — 이미 있는 원장뿐.

피처 (경주 n 에 대해 같은 (rcDate, meet) 의 rcNo < n 만):
  day_n_prev          앞 경주 수
  day_win_gate        앞 경주 우승마 X_gate_rel 평균 − 그 경주 평균   (+ 면 바깥이 이기는 날)
  day_gate_x          (X_gate_rel − 경주 평균) × day_win_gate          (편향과 내 게이트의 정렬)
  day_win_epos        앞 경주 우승마 F5_early_pos 평균 − 경주 평균     (− 면 선행이 버티는 날)
  day_epos_x          (F5_early_pos − 경주 평균) × day_win_epos
  day_fav_rate        앞 경주에서 인기 1위마 우승 비율                  (확정배당은 경주 직후 공개)
  day_speed           앞 경주 y_speed_fig 평균 − 100                    (트랙이 빠른 날)
  day_jk_n · day_jk_ordpct · day_jk_win   기수의 오늘 앞 경주 수 · 착순백분율 평균 · 우승 수
  day_tr_n · day_tr_ordpct                조교사의 오늘 앞 경주 수 · 착순백분율 평균
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import data as D

RANK = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
DAY = ["day_n_prev", "day_win_gate", "day_gate_x", "day_win_epos", "day_epos_x", "day_fav_rate", "day_speed",
       "day_jk_n", "day_jk_ordpct", "day_jk_win", "day_tr_n", "day_tr_ordpct"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """행 순서 유지. 같은 (rcDate, meet) 안에서 rcNo 오름차순으로 누적 — 자기 경주는 절대 포함하지 않는다."""
    d = df[["race_id", "rcDate", "meet", "rcNo", "jkNo", "trNo", "y_ord", "y_win", "y_speed_fig",
            "X_gate_rel", "X_dusu", "F5_early_pos", "F6_mkt_rank"]].copy()
    d["ordpct"] = (d["X_dusu"] - d["y_ord"]) / (d["X_dusu"] - 1).clip(lower=1)
    d["gate_c"] = d["X_gate_rel"] - d.groupby("race_id")["X_gate_rel"].transform("mean")
    d["epos_c"] = d["F5_early_pos"] - d.groupby("race_id")["F5_early_pos"].transform("mean")

    # 경주 단위 요약 (우승마 기준)
    win = d[d["y_win"] == 1].groupby("race_id").agg(win_gate=("gate_c", "mean"), win_epos=("epos_c", "mean"),
                                                    fav=("F6_mkt_rank", lambda s: float((s == 1).any())))
    race = d.groupby("race_id").agg(rcDate=("rcDate", "first"), meet=("meet", "first"), rcNo=("rcNo", "first"),
                                    speed=("y_speed_fig", "mean")).join(win)
    race = race.sort_values(["rcDate", "meet", "rcNo"])
    g = race.groupby(["rcDate", "meet"], sort=False)
    prev = pd.DataFrame(index=race.index)
    prev["day_n_prev"] = g.cumcount()
    for src, dst in [("win_gate", "day_win_gate"), ("win_epos", "day_win_epos"), ("fav", "day_fav_rate"), ("speed", "day_speed")]:
        s = g[src].apply(lambda x: x.shift().expanding().mean()).reset_index(level=[0, 1], drop=True)
        prev[dst] = s
    prev["day_speed"] = prev["day_speed"] - 100
    prev.loc[prev["day_n_prev"] == 0, DAY[1:7]] = np.nan
    out = d.join(prev, on="race_id")

    # 기수·조교사 오늘 앞 경주 — 사람×경주 단위로 묶은 뒤 같은 날 rcNo 순 누적. 같은 경주의 동반 출전마는 들어가지 않는다.
    for who, pre in [("jkNo", "day_jk"), ("trNo", "day_tr")]:
        pr = d.groupby(["rcDate", "meet", who, "rcNo"]).agg(n=("ordpct", "size"), pct=("ordpct", "sum"), w=("y_win", "sum"))
        pr = pr.sort_index()
        gg = pr.groupby(level=[0, 1, 2], sort=False)
        cum = pd.DataFrame({k: gg[k].cumsum() - pr[k] for k in ["n", "pct", "w"]})
        cum[f"{pre}_n"] = cum["n"]
        cum[f"{pre}_ordpct"] = cum["pct"] / cum["n"].replace(0, np.nan)
        cum[f"{pre}_win"] = cum["w"]
        keys = ["rcDate", "meet", who, "rcNo"]
        out = out.join(cum[[f"{pre}_n", f"{pre}_ordpct", f"{pre}_win"]], on=keys)
    out["day_gate_x"] = out["gate_c"] * out["day_win_gate"]
    out["day_epos_x"] = out["epos_c"] * out["day_win_epos"]
    return out[DAY]


def main(seeds=(0, 1, 2)):
    from model.evaluate import metrics, hits, compare, win_probs, race_logloss
    tr, va = D.load("train"), D.load("valid")
    cols = D.feature_cols()
    tr_e, va_e = D.C.encode(tr, va, cols)
    tr_e, va_e = tr_e.copy(), va_e.copy()
    ftr, fva = build(tr), build(va)
    for c in DAY:
        tr_e[c] = ftr[c].to_numpy(); va_e[c] = fva[c].to_numpy()
    print("피처 결측률(valid):", {c: round(float(fva[c].isna().mean()), 2) for c in DAY})
    print("day_fav_rate 평균:", round(float(fva["day_fav_rate"].mean()), 3), " day_win_gate 평균:", round(float(fva["day_win_gate"].mean()), 3))
    # 원시 신호 자체가 착순과 관련 있나 — valid 에서 상관
    yv = va["y_win"].to_numpy()
    for c in ["day_gate_x", "day_epos_x", "day_jk_ordpct", "day_tr_ordpct"]:
        m = fva[c].notna().to_numpy()
        print(f"  corr({c}, y_win) = {np.corrcoef(fva[c][m], yv[m])[0, 1]:+.4f}  n={int(m.sum())}")

    g = tr_e.groupby("race_id", sort=False).size().to_numpy()
    res = {"A LGB73": [], "B LGB73+day": []}
    for s in seeds:
        for name, cc in [("A LGB73", cols), ("B LGB73+day", cols + DAY)]:
            m = lgb.train({**RANK, "seed": s}, lgb.Dataset(tr_e[cc], label=tr["y_rel"], group=g), 200)
            res[name].append(m.predict(va_e[cc]))
            if s == seeds[0] and name.startswith("B"):
                imp = pd.Series(m.feature_importance("gain"), index=cc).sort_values(ascending=False)
                print("  day 피처 gain 순위:", {c: int(list(imp.index).index(c)) + 1 for c in DAY}, "/", len(cc))
    base = np.mean([win_probs(va, s) for s in res["A LGB73"]], 0)
    print(f"\n{'':14s} top-1        top-3   logloss  A 대비 [CI]")
    for name, ss in res.items():
        ms = [metrics(va, s) for s in ss]
        p = np.mean([win_probs(va, s) for s in ss], 0)
        d_, lo, hi = compare(hits(va, np.log(p)), hits(va, np.log(base)))
        print(f"{name:14s} {np.mean([x['top1'] for x in ms]):5.2f} ±{np.std([x['top1'] for x in ms]):.2f}  "
              f"{np.mean([x['top3'] for x in ms]):5.2f}   {race_logloss(va, np.log(p)):.4f}  {d_:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    # 경주 번호별: 앞 경주가 많을수록 효과가 커지나
    hb = hits(va, np.log(np.mean([win_probs(va, s) for s in res["B LGB73+day"]], 0)))
    ha = hits(va, np.log(base))
    rn = va.groupby("race_id", sort=False)["rcNo"].first().to_numpy()
    for lo_, hi_ in [(1, 1), (2, 4), (5, 8), (9, 20)]:
        m = (rn >= lo_) & (rn <= hi_)
        print(f"  경주 {lo_}~{hi_}: n={int(m.sum())}  A {ha[m].mean() * 100:.1f}  B {hb[m].mean() * 100:.1f}")


if __name__ == "__main__":
    main()
