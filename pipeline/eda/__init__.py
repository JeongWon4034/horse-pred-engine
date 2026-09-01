"""데이터 탐색 도구.

    from eda import *
    tr = load("train")
    missing(tr)
    race(tr)                 # 아무 경주 하나 통째로
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"

GROUPS = ["X", "F1", "F2", "F3", "F4", "F5", "F6"]
GROUP_NAME = {
    "X": "공통", "F1": "최근성적·조교", "F2": "부모(육종가)", "F3": "기수",
    "F4": "거리·날씨", "F5": "주행", "F6": "인기도",
}
IDS = ["race_id", "row_id", "rcDate", "meet", "rcNo", "hrNo", "hrName", "jkNo", "trNo", "split"]
TARGETS = ["y_ord", "y_win", "y_plc", "y_rel", "y_speed_fig"]

# Windows 한글 깨짐 방지
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 80)


# ────────────────────────────── 로딩 ──────────────────────────────

def load(split: str = "train", cols: list[str] | None = None) -> pd.DataFrame:
    """split: train / valid / test / game"""
    df = pd.read_parquet(DATA / "model" / f"{split}.parquet", columns=cols)
    if "rcDate" in df.columns:
        df["yr"] = df.rcDate.astype(str).str[:4].astype(int)
    return df


def load_game(name: str) -> pd.DataFrame:
    """name: race_card / entries / payouts"""
    p = DATA / "game" / f"{name}.parquet"
    return pd.read_csv(DATA / "game" / f"{name}.csv") if not p.exists() else pd.read_parquet(p)


def load_passing() -> pd.DataFrame:
    return pd.read_parquet(DATA / "sim" / "passing.parquet")


def features(df: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    g = groups or GROUPS
    return [c for c in df.columns if c.split("_")[0] in g]


# ────────────────────────────── 요약 ──────────────────────────────

def overview() -> pd.DataFrame:
    rows = []
    for s in ["train", "valid", "test", "game"]:
        d = pd.read_parquet(DATA / "model" / f"{s}.parquet", columns=["race_id", "rcDate"])
        rows.append({"split": s, "행": len(d), "경주": d.race_id.nunique(),
                     "시작": d.rcDate.min(), "끝": d.rcDate.max()})
    return pd.DataFrame(rows).set_index("split")


def missing(df: pd.DataFrame, by_group: bool = False, min_rate: float = 0.0) -> pd.DataFrame:
    """결측률. by_group=True 면 그룹별 평균."""
    f = features(df)
    if by_group:
        rows = [{"그룹": g, "이름": GROUP_NAME[g], "피처수": len([c for c in f if c.startswith(g + "_")]),
                 "결측률": df[[c for c in f if c.startswith(g + "_")]].isna().mean().mean()}
                for g in GROUPS]
        return pd.DataFrame(rows).set_index("그룹")
    s = df[f].isna().mean().sort_values(ascending=False)
    s = s[s >= min_rate]
    return s.rename("결측률").to_frame()


def missing_by_year(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """연도별 결측률 — 시점 때문에 비는 컬럼을 찾을 때."""
    return df.groupby("yr")[cols].apply(lambda t: t.isna().mean()).round(3)


def describe(df: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    cols = features(df, [group] if group else None)
    num = df[cols].select_dtypes("number")
    d = num.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    d["결측률"] = df[num.columns].isna().mean()
    return d.round(3)


def cats(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 컬럼의 값 분포."""
    out = []
    for c in features(df):
        if df[c].dtype == object or str(df[c].dtype).startswith("category"):
            vc = df[c].value_counts()
            out.append({"컬럼": c, "고유값": len(vc), "결측률": round(df[c].isna().mean(), 3),
                        "상위": ", ".join(f"{k}({v:,})" for k, v in vc.head(4).items())})
    return pd.DataFrame(out).set_index("컬럼")


# ────────────────────────────── 경주 단위 ──────────────────────────────

def race(df: pd.DataFrame, race_id: str | None = None, group: str | None = None) -> pd.DataFrame:
    """경주 하나를 착순순으로. group 을 주면 그 축 피처만."""
    rid = race_id or df.race_id.iloc[-1]
    one = df[df.race_id == rid].sort_values("y_ord")
    cols = ["hrName", "y_ord"] + (features(one, [group]) if group else [])
    return one[cols] if group else one[["hrName", "y_ord", "y_win", "y_plc"] + features(one)]


def field_size(df: pd.DataFrame) -> pd.Series:
    return df.groupby("race_id", sort=False).size().describe()


# ────────────────────────────── 그림 ──────────────────────────────

def plot_missing_by_year(df: pd.DataFrame, cols: list[str], ax=None):
    m = missing_by_year(df, cols)
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    for c in cols:
        ax.plot(m.index, m[c] * 100, marker="o", ms=3, label=c)
    ax.set_ylabel("결측률 (%)"); ax.set_xlabel("연도"); ax.grid(alpha=.3)
    ax.legend(fontsize=8); ax.set_title("연도별 결측률")
    return ax


def plot_hist(df: pd.DataFrame, col: str, bins: int = 50, by_win: bool = True, ax=None):
    """분포. by_win=True 면 1착 / 비1착을 겹쳐 그린다."""
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    s = df[col].dropna()
    lo, hi = s.quantile([.01, .99])
    if by_win:
        ax.hist(df.loc[df.y_win == 0, col].dropna(), bins=bins, range=(lo, hi),
                density=True, alpha=.5, label="비1착")
        ax.hist(df.loc[df.y_win == 1, col].dropna(), bins=bins, range=(lo, hi),
                density=True, alpha=.5, label="1착")
        ax.legend()
    else:
        ax.hist(s, bins=bins, range=(lo, hi))
    ax.set_title(col); ax.grid(alpha=.3)
    return ax


def signal(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """피처 하나로 순위를 매겼을 때의 1착 적중률. 그 피처가 혼자 얼마나 세는지.

    한 경주가 통째로 결측인 피처(혈통 없는 경주 등)는 그 경주를 빼고 계산하고,
    몇 %의 경주에서 판단이 가능했는지를 `커버` 로 같이 보여준다.
    """
    cols = cols or [c for c in features(df) if df[c].dtype.kind in "fi"]
    n_race = df.race_id.nunique()
    base = df.y_win.mean()
    out = []

    for c in cols:
        for sign, label in ((1, "높을수록"), (-1, "낮을수록")):
            t = pd.DataFrame({"r": df.race_id, "v": df[c] * sign, "w": df.y_win}).dropna(subset=["v"])
            if t.empty:
                continue
            idx = t.groupby("r", sort=False).v.idxmax()
            out.append({"피처": c, "방향": label,
                        "top1": t.loc[idx, "w"].mean(),
                        "커버": t.r.nunique() / n_race})

    r = pd.DataFrame(out)
    if r.empty:
        return r
    best = r.loc[r.groupby("피처").top1.idxmax()].set_index("피처")
    best["무작위대비"] = best.top1 / base
    return best.sort_values("top1", ascending=False).round(4)
