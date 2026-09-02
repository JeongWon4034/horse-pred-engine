"""game 분할 ROI 백테스트 — 모델 1위마에 단승 1단위씩 걸었을 때.

game 3,700경주는 학습·검증·튜닝에 쓰지 않는다. 여기서만, 그것도 채점이 아니라
'이 모델대로 걸었으면 얼마가 남았나'를 보는 용도로만 읽는다.

    PYTHONUTF8=1 uv run python -m model.backtest                       # 시장 기준선(인기 1위마)만
    PYTHONUTF8=1 uv run python -m model.backtest --pred experiments/scores/game.parquet --col score

배당은 entries.parquet 의 winOdds (확정 단승 배당). 9999.9 특수값은 schema_v2.clean 이 NaN 으로 바꾸고,
배당이 없는 경주는 베팅에서 뺀다. payouts.csv 단승식과 대조해 두 값이 같은지도 확인한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .team import S, TEAM_DATASET

GAME = TEAM_DATASET / "game"


def load_game() -> pd.DataFrame:
    """game 경주의 출전표 + 확정 단승 배당. race_id 순서는 entries 그대로."""
    e = S.clean(pd.read_parquet(GAME / "entries.parquet"))
    return e[["race_id", "hrNo", "chulNo", "winOdds", "mkt_rank", "y_ord"]]


def win_payouts() -> pd.Series:
    """payouts.csv 의 단승식 배당. index = (race_id, chulNo)."""
    p = pd.read_csv(GAME / "payouts.csv")
    p = p[p["pool"] == "단승식"].copy()
    p["chulNo"] = p["combo"].astype(int)
    return p.set_index(["race_id", "chulNo"])["odds"]


def roi(entries: pd.DataFrame, scores: np.ndarray, stake: float = 1.0) -> dict:
    """경주마다 점수 최고 말 한 마리에 stake 를 건다.

    반환: 베팅 경주 수, 적중 수, 적중률, 투입, 회수, ROI(%), 적중 시 평균 배당
    """
    scores = np.asarray(scores, float)
    if np.isnan(scores).any():
        raise ValueError(f"점수에 NaN {int(np.isnan(scores).sum())}개 — 베팅 대상을 고를 수 없다")
    d = entries.assign(_s=scores)
    pick = d.loc[d.groupby("race_id", sort=False)["_s"].idxmax()]
    pick = pick[pick["winOdds"].notna()]              # 배당 없는 경주는 걸 수 없다
    hit = pick["y_ord"].to_numpy() == 1
    spent = stake * len(pick)
    back = float((pick.loc[hit, "winOdds"] * stake).sum())
    return {"n_bet": int(len(pick)), "n_hit": int(hit.sum()), "hit_rate": float(hit.mean() * 100),
            "spent": spent, "back": back, "roi": (back / spent - 1) * 100 if spent else float("nan"),
            "avg_odds_hit": float(pick.loc[hit, "winOdds"].mean()) if hit.any() else float("nan")}


def market_scores(entries: pd.DataFrame) -> np.ndarray:
    """인기 1위마 — 배당이 낮을수록 점수가 높다."""
    return -entries["mkt_rank"].to_numpy(float)


def fmt(name: str, r: dict) -> str:
    return (f"{name:<24}베팅 {r['n_bet']:>5}  적중 {r['n_hit']:>4} ({r['hit_rate']:5.1f}%)  "
            f"ROI {r['roi']:+6.1f}%  적중 평균배당 {r['avg_odds_hit']:.2f}")


def check_payouts(entries: pd.DataFrame) -> None:
    """entries.winOdds 와 payouts 단승식이 같은 값인지. 다르면 어느 쪽을 믿을지 정해야 한다."""
    p = win_payouts()
    w = entries.loc[entries["y_ord"] == 1].set_index(["race_id", "chulNo"])["winOdds"]
    both = pd.concat([w.rename("entries"), p.rename("payouts")], axis=1, join="inner").dropna()
    diff = (both["entries"] - both["payouts"]).abs()
    print(f"[대조] 1착마 단승 배당 entries vs payouts: 공통 {len(both):,}경주, "
          f"불일치(>0.05) {(diff > 0.05).sum():,}개, 최대 차 {diff.max():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, help="game 점수 parquet (race_id, hrNo, <col>)")
    ap.add_argument("--col", default="score")
    ap.add_argument("--name", default="모델")
    args = ap.parse_args()

    e = load_game()
    print(f"game {e['race_id'].nunique():,}경주 · {len(e):,}두 · 배당 결측 {e['winOdds'].isna().mean()*100:.1f}%")
    check_payouts(e)
    print()
    print(fmt("시장 (인기 1위마)", roi(e, market_scores(e))))

    if args.pred:
        s = pd.read_parquet(args.pred).set_index(["race_id", "hrNo"])[args.col]
        s = s.reindex(pd.MultiIndex.from_arrays([e["race_id"], e["hrNo"]]))
        if s.isna().any():
            raise SystemExit(f"{args.pred}: game 행과 맞지 않는 점수 {int(s.isna().sum())}개")
        print(fmt(args.name, roi(e, s.to_numpy())))


if __name__ == "__main__":
    main()
