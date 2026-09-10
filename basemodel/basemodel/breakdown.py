# -*- coding: utf-8 -*-
"""분해표 — 어떤 경주에서 잘하고 못하는지, 그리고 **역배는 언제 나오는지**.

    uv run python -m basemodel.breakdown --ckpt artifacts/runs/axis_77_s20260901_final.pt

역배형(UPSET)의 근거가 여기 있다. "시장 계수를 음수로 놓으면 역배를 고른다"는 것만으로는
왜 그게 되는지 설명이 안 된다. 이 표는 **인기마가 지는 경주의 조건**을 실제로 세어
그 조건에서 역배형이 실제로 이득을 보는지 확인한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import presets as P
from .evaluate import est_odds
from .serve import Scorer, combine, load_game
from .team import ARTIFACTS, C


def race_frame(df: pd.DataFrame) -> pd.DataFrame:
    """경주 단위 표 — 조건 + '이변이었나'."""
    o = est_odds(df)
    d = df.assign(_o=o, _fav=(df["F6_mkt_rank"].to_numpy() == 1))
    g = d.groupby("race_id", sort=False)
    out = pd.DataFrame({
        "dusu": g["X_dusu"].first(),
        "dist": g["X_rcDist"].first(),
        "grade": g["X_grade"].first().astype(str),
        "entropy": g["F6_field_entropy"].first(),
        "fav_prob": g["F6_mkt_prob"].max(),
        # 이변 = 인기 1위마가 1착을 못 했다
        "fav_lost": g.apply(lambda x: int(not bool(((x["F6_mkt_rank"] == 1) & (x["y_win"] == 1)).any())),
                            include_groups=False),
        # 고배당 우승 = 1착마의 배당이 기준 이상
        "upset_win": g.apply(lambda x: int(bool(((x["_o"] >= cfg.UPSET_ODDS) & (x["y_win"] == 1)).any())),
                             include_groups=False),
        "win_odds": g.apply(lambda x: float(x.loc[x["y_win"] == 1, "_o"].max()) if (x["y_win"] == 1).any() else np.nan,
                            include_groups=False),
    })
    return out


def segment(rf: pd.DataFrame, col: str, bins) -> pd.Series:
    if isinstance(bins, int):
        return pd.qcut(rf[col], bins, duplicates="drop")
    return pd.cut(rf[col], bins)


def upset_conditions(rf: pd.DataFrame) -> str:
    """언제 이변이 나오나 — 조건별 '인기마 패배율'과 '고배당 우승률'."""
    lines = ["[이변은 언제 나오나 — 경주 조건별]",
             f"  {'조건':<22}{'경주':>7}{'인기마 패배':>12}{'고배당 우승':>12}{'1착 평균배당':>13}"]
    segs = {
        "출주두수": segment(rf, "dusu", [0, 7, 9, 11, 13, 99]),
        "거리(m)": segment(rf, "dist", [0, 1000, 1200, 1400, 1800, 9999]),
        "혼전도(엔트로피)": segment(rf, "entropy", 4),
        "1인기 확률": segment(rf, "fav_prob", [0, .18, .25, .35, 1.01]),
    }
    for name, s in segs.items():
        lines.append(f"  ── {name}")
        for k, sub in rf.groupby(s, observed=True):
            if len(sub) < 30:
                continue
            lines.append(f"  {str(k):<22}{len(sub):>7,}{sub['fav_lost'].mean() * 100:>11.1f}%"
                         f"{sub['upset_win'].mean() * 100:>11.1f}%{sub['win_odds'].mean():>13.1f}")
    return "\n".join(lines)


def preset_by_segment(df: pd.DataFrame, rf: pd.DataFrame, axis_rows: np.ndarray,
                      presets: dict, n_axes: int, which=("BASIC", "UPSET")) -> str:
    """같은 세그먼트에서 프리셋별 성적 — 역배형이 실제로 어디서 이득인지."""
    rid = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(rid)]
    win, odds = df["y_win"].to_numpy(float), est_odds(df)

    hit, hit_odds = {}, {}
    for name in which:
        if name not in presets:
            continue
        p = presets[name]
        s = combine(axis_rows, p["w"], p["m"], n_axes)
        idx = np.array([a + int(np.argmax(s[a:b])) for a, b in zip(starts, ends)])
        hit[name] = win[idx]
        hit_odds[name] = odds[idx]

    rf = rf.copy()
    for name in hit:
        rf[f"hit_{name}"] = hit[name]
        rf[f"odds_{name}"] = hit_odds[name]

    names = list(hit)
    head = "".join(f"{cfg.PRESETS[n]['label'] + ' 적중':>13}{'평균배당':>10}" for n in names)
    lines = ["", "[세그먼트별 프리셋 성적]", f"  {'조건':<22}{'경주':>7}{head}"]
    segs = {"혼전도(엔트로피)": segment(rf, "entropy", 4),
            "1인기 확률": segment(rf, "fav_prob", [0, .18, .25, .35, 1.01])}
    for sname, s in segs.items():
        lines.append(f"  ── {sname}")
        for k, sub in rf.groupby(s, observed=True):
            if len(sub) < 30:
                continue
            cells = "".join(f"{sub[f'hit_{n}'].mean() * 100:>12.1f}%{sub[f'odds_{n}'].mean():>10.1f}"
                            for n in names)
            lines.append(f"  {str(k):<22}{len(sub):>7,}{cells}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", default=str(ARTIFACTS / "export" / "meta.json"))
    ap.add_argument("--split", default="valid", choices=["valid", "game"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    sc = Scorer(a.ckpt, device=a.device)
    df = load_game() if a.split == "game" else C.load(a.split)
    ax = sc.axis_rows(df, a.split)
    rf = race_frame(df)
    print(f"[{a.split}] {len(rf):,}경주\n")
    print(upset_conditions(rf))

    meta = json.loads(Path(a.meta).read_text("utf-8")) if Path(a.meta).exists() else None
    if meta:
        presets = {k: {"w": np.array([v["groupWeights"][ax_] / 100 for ax_ in sc.axes]),
                       "m": v["market"]}
                   for k, v in meta["presets"].items()}
        print(preset_by_segment(df, rf, ax, presets, sc.n_axes,
                                which=tuple(k for k in ("BASIC", "UPSET", "ODDS") if k in presets)))
    else:
        print("\n(meta.json 이 없어 프리셋 비교는 생략 — basemodel.export 를 먼저 돌릴 것)")


if __name__ == "__main__":
    main()
