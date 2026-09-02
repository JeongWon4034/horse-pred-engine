"""저장된 예측들을 한 표에 놓고 쪼개 본다 — 어디서 이기고 어디서 지는가.

    PYTHONUTF8=1 uv run python -m model.breakdown lgb_73 linear_73 embed_73 history_73 attn_73
    PYTHONUTF8=1 uv run python -m model.breakdown lgb_73 attn_73 --by entropy grade dist

experiments/pred/{name}.parquet (model.train / model.baseline 이 저장) 을 valid 에 맞춰 읽는다.
분해 축
  entropy  F6_field_entropy 3분위 — 혼전 경주에서만 앞서는지 (73 모델도 valid df 에 컬럼이 있어 가능)
  grade    X_grade 상위 등급 묶음
  dist     X_rcDist 구간 (단거리 ≤1200 / 중거리 ≤1700 / 장거리)
  field    출전 두수 (≤8 / 9~11 / 12+)
첫 번째 이름이 기준이고, 나머지는 기준 대비 paired bootstrap CI 를 같이 찍는다.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .evaluate import compare, hits, load_pred, metrics
from .team import C


def race_table(df: pd.DataFrame) -> pd.DataFrame:
    """경주 단위 속성 한 줄씩 (race_id 등장 순서 = hits() 순서)."""
    g = df.groupby("race_id", sort=False)
    t = pd.DataFrame({
        "entropy": g["F6_field_entropy"].first(),
        "grade": g["X_grade"].first().astype(str),
        "dist": g["X_rcDist"].first(),
        "n": g.size(),
    })
    t["entropy_bin"] = pd.qcut(t["entropy"].rank(method="first"), 3, labels=["뚜렷", "중간", "혼전"])
    t["dist_bin"] = pd.cut(t["dist"], [0, 1200, 1700, 9999], labels=["단거리≤1200", "중거리≤1700", "장거리>1700"])
    t["field_bin"] = pd.cut(t["n"], [0, 8, 11, 99], labels=["≤8두", "9~11두", "12두+"])
    top = t["grade"].value_counts().index[:6]
    t["grade_bin"] = np.where(t["grade"].isin(top), t["grade"], "기타")
    return t.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="experiments/pred 의 파일명(확장자 없이). 첫 번째가 기준")
    ap.add_argument("--by", nargs="*", default=["entropy"], choices=["entropy", "grade", "dist", "field"])
    a = ap.parse_args()

    va = C.load("valid")
    preds = {}
    for nm in a.names:
        s = load_pred(va, nm)
        if s is None:
            raise SystemExit(f"experiments/pred/{nm}.parquet 없음")
        preds[nm] = s
    base = a.names[0]
    H = {nm: hits(va, s) for nm, s in preds.items()}
    tbl = race_table(va)

    print(f"valid {len(tbl):,}경주 · 기준 = {base}\n")
    print(f"{'모델':<14}{'top-1':>7}{'top-3':>7}{'logloss':>9}{'ECE':>8}   vs {base} (paired 95% CI)")
    for nm, s in preds.items():
        m = metrics(va, s)
        line = f"{nm:<14}{m['top1']:>6.1f}%{m['top3']:>6.1f}%{m['logloss']:>9.4f}{m['ece']:>8.4f}"
        if nm != base:
            d, lo, hi = compare(H[nm], H[base])
            line += f"   {d:+.2f}%p [{lo:+.2f}, {hi:+.2f}]" + ("  ★" if lo > 0 or hi < 0 else "")
        print(line)

    for axis in a.by:
        col = f"{axis}_bin"
        print(f"\n[{axis}] top-1 (%) · 경주 수")
        cats = [c for c in tbl[col].astype(str).unique()] if axis == "grade" else list(tbl[col].cat.categories)
        head = f"{'구간':<14}{'n':>6}" + "".join(f"{nm:>14}" for nm in preds)
        print(head)
        for c in cats:
            m = (tbl[col].astype(str) == str(c)).to_numpy()
            if m.sum() == 0:
                continue
            row = f"{str(c):<14}{m.sum():>6}"
            for nm in preds:
                row += f"{H[nm][m].mean()*100:>13.1f}%"
            print(row)


if __name__ == "__main__":
    main()
