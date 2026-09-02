"""사이드 실험 — game 3,700경주 단승 ROI. 모델별 · 베팅 전략별.

    PYTHONUTF8=1 uv run python -m model.side_roi

전략
  top1        경주마다 모델 1위마에 1단위 (기본 — 시장·LGB·S3 비교의 기준)
  edge>t      모델 확률 − 시장 암시 확률 이 t 이상인 경주에만 1위마 베팅 (안정형 프리셋 후보)
  ev>1        모델 확률 × 배당 이 가장 큰 말이 EV>1 일 때만 그 말에 베팅 (이변형 프리셋 후보)
시장 암시 확률 = 1/winOdds 를 경주 안에서 정규화 (공제율 제거).
game 은 여기서만 쓴다 — 학습·튜닝에 쓰지 않는다. 임계값 t 도 여기서 고르면 game 에 맞춘 낙관값이므로 표만 보이고 고르지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import load_game, market_scores, roi
from .evaluate import PRED_DIR, win_probs


def probs_of(e: pd.DataFrame, name: str) -> np.ndarray:
    s = pd.read_parquet(PRED_DIR / f"{name}.parquet").set_index(["race_id", "hrNo"])["score"]
    s = s.reindex(pd.MultiIndex.from_arrays([e["race_id"], e["hrNo"]]))
    assert not s.isna().any(), name
    return win_probs(e, s.to_numpy())


def implied(e: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / e["winOdds"].to_numpy(float)
    g = e["race_id"].to_numpy()
    out = np.full(len(e), np.nan)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]); ends = np.r_[starts[1:], len(g)]
    for a, b in zip(starts, ends):
        v = inv[a:b]
        if np.isfinite(v).all() and v.sum() > 0:
            out[a:b] = v / v.sum()
    return out


def roi_edge(e: pd.DataFrame, p: np.ndarray, t: float) -> dict:
    """1위마 베팅, 단 (모델 p − 시장 p) ≥ t 인 경주만."""
    d = e.assign(_p=p, _m=implied(e))
    pick = d.loc[d.groupby("race_id", sort=False)["_p"].idxmax()]
    pick = pick[(pick["_p"] - pick["_m"] >= t) & pick["winOdds"].notna()]
    return _summ(pick)


def roi_ev(e: pd.DataFrame, p: np.ndarray, t: float = 1.0) -> dict:
    """EV = p × 배당 최대 말, EV ≥ t 일 때만."""
    d = e.assign(_ev=p * e["winOdds"].to_numpy(float))
    d = d[d["_ev"].notna()]
    pick = d.loc[d.groupby("race_id", sort=False)["_ev"].idxmax()]
    pick = pick[pick["_ev"] >= t]
    return _summ(pick)


def _summ(pick: pd.DataFrame) -> dict:
    hit = pick["y_ord"].to_numpy() == 1
    n = len(pick); back = float((pick.loc[hit, "winOdds"]).sum())
    return {"n_bet": n, "hit_rate": hit.mean() * 100 if n else np.nan,
            "roi": (back / n - 1) * 100 if n else np.nan,
            "avg_odds_hit": float(pick.loc[hit, "winOdds"].mean()) if hit.any() else np.nan}


def line(name, r):
    return f"{name:<34}베팅 {r['n_bet']:>5}  적중 {r['hit_rate']:5.1f}%  ROI {r['roi']:+6.1f}%  적중배당 {r['avg_odds_hit']:.2f}"


def main() -> None:
    e = load_game()
    print(f"game {e['race_id'].nunique():,}경주 · 배당 결측 {e['winOdds'].isna().mean()*100:.1f}%\n")

    P = {"LightGBM 73": probs_of(e, "lgb_73_game"), "LightGBM 77 (배당 포함)": probs_of(e, "lgb_77_game")}
    l10 = [probs_of(e, n) for n in ["history_73_game", "history_73_s1_game", "history_73_s2_game",
                                    "history_73_s3_game", "history_73_s4_game"]]
    l20 = [probs_of(e, f"history_73_L20_s{s}_game") for s in ["20260901", 1, 2, 3, 4]]
    P["S3 L10 단일(seed 20260901)"] = l10[0]
    P["S3 L10 seed5 평균"] = np.mean(l10, 0)
    P["S3 L20 seed5 평균"] = np.mean(l20, 0)
    P["LGB73 + S3 L20 seed5 (5:5)"] = (P["LightGBM 73"] + P["S3 L20 seed5 평균"]) / 2

    print("[전략 top1 — 매 경주 모델 1위마]")
    mk = roi(e, market_scores(e)); print(line("시장 (인기 1위마)", mk))
    for k, p in P.items():
        print(line(k, roi(e, np.log(p))))

    print("\n[전략 edge — 모델 확률이 시장보다 t 이상 높을 때만 (73피처 모델만 의미 있음)]")
    for k in ["LightGBM 73", "S3 L20 seed5 평균", "LGB73 + S3 L20 seed5 (5:5)"]:
        for t in (0.0, 0.03, 0.06, 0.10):
            print(line(f"{k} t={t:.2f}", roi_edge(e, P[k], t)))

    print("\n[전략 ev — p×배당 최대 말, EV ≥ t]")
    for k in ["LightGBM 73", "S3 L20 seed5 평균", "LGB73 + S3 L20 seed5 (5:5)"]:
        for t in (1.0, 1.2, 1.5):
            print(line(f"{k} EV≥{t}", roi_ev(e, P[k], t)))


if __name__ == "__main__":
    main()
