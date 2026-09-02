"""평가 — 적중률은 팀 채점기(C.evaluate / C.report)가 재고, 여기서는 그 옆에 붙일 것만 만든다.

정확도만 보면 인기마만 찍는 모델이 1등이 되는데 공제율 때문에 실제로는 손해다.
그래서 확률의 품질(logloss·ECE·Brier)과 "차이가 우연인가"(paired bootstrap)를 같이 본다.

입력은 전부 (df, 행 단위 점수 배열). LightGBM 예측과 딥러닝 예측을 같은 함수에 넣기 위한 규약이다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .team import C

SEED = 20260901
PRED_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pred"


# ─────────────────────────── 경주 내 확률 ───────────────────────────
def win_probs(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """경주 내 softmax. 온도는 1 로 고정한다 — LightGBM 점수도 같은 식으로 확률화한다."""
    s = np.asarray(scores, np.float64)
    g = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    p = np.empty_like(s)
    for a, b in zip(starts, ends):
        e = np.exp(s[a:b] - s[a:b].max())
        p[a:b] = e / e.sum()
    return p


def race_logloss(df: pd.DataFrame, scores: np.ndarray) -> float:
    """1착마에 준 확률의 -log 평균. 경주당 하나 — 동착(1착 2두)은 확률을 합쳐 한 값으로 센다."""
    p = win_probs(df, scores)
    win = df["y_win"].to_numpy() == 1
    p_win = pd.Series(p * win).groupby(df["race_id"].to_numpy(), sort=False).sum()
    return float(-np.log(np.clip(p_win.to_numpy(), 1e-12, 1)).mean())


def brier(df: pd.DataFrame, scores: np.ndarray) -> float:
    p = win_probs(df, scores)
    y = df["y_win"].to_numpy(float)
    return float(((p - y) ** 2).mean())


def expected_calibration_error(p, y, bins: int = 15) -> float:
    """'41%라고 한 것들이 실제로 41% 맞았나' — 작을수록 좋다."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def ece(df: pd.DataFrame, scores: np.ndarray) -> float:
    return expected_calibration_error(win_probs(df, scores), df["y_win"].to_numpy(float))


# ─────────────────────────── 경주별 적중 · 짝 비교 ───────────────────────────
def hits(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """경주마다 점수 최고 말이 1착이었는지 0/1. 순서는 race_id 등장 순서."""
    scores = np.asarray(scores, float)
    if np.isnan(scores).any():
        raise ValueError(f"점수에 NaN {int(np.isnan(scores).sum())}개 — idxmax 가 조용히 틀린다")
    d = df[["race_id", "y_win"]].assign(_s=scores)
    pick = d.loc[d.groupby("race_id", sort=False)["_s"].idxmax()]
    return pick["y_win"].to_numpy(int)


def compare(hits_a: np.ndarray, hits_b: np.ndarray, n: int = 2000, seed: int = SEED):
    """같은 경주 묶음에서 a − b 의 top-1 차이(%p)와 paired bootstrap 95% CI.

    CI 가 0 을 품으면 우연 범위, 벗어나면 차이가 있다고 읽는다.
    """
    a = np.asarray(hits_a, float); b = np.asarray(hits_b, float)
    if len(a) != len(b):
        raise ValueError(f"경주 수 불일치: {len(a)} vs {len(b)}")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1) * 100
    return float(d.mean() * 100), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ─────────────────────────── 한 번에 ───────────────────────────
def metrics(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """top1/top3 는 채점기 값 그대로, 나머지는 위 함수. 장부 한 줄에 필요한 것 전부."""
    m = C.evaluate(df, scores)
    return {"top1": m["top1"], "top3": m["top3"], "n": m["n_races"], "se": m["se_top1"],
            "logloss": race_logloss(df, scores), "ece": ece(df, scores), "brier": brier(df, scores)}


def ledger_line(m: dict, model: str, feats: int, commit: str = "—", data: str = "f17ca36",
                seed: int | str = SEED, memo: str = "", date: str | None = None) -> str:
    """experiments/ledger.md 표 형식 한 줄."""
    date = date or pd.Timestamp.today().strftime("%Y-%m-%d")
    return (f"| {date} | {commit} | {data} | {model} | {feats} | {seed} | "
            f"{m['top1']:.1f} | {m['top3']:.1f} | {m['logloss']:.4f} | {m['ece']:.4f} | — | {memo} |")


# ─────────────────────────── 예측 저장 · 로드 ───────────────────────────
def save_pred(df: pd.DataFrame, scores: np.ndarray, name: str) -> Path:
    """experiments/pred/{name}.parquet — race_id · hrNo · score. hrNo 는 줄 맞추기용 꼬리표."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{name}.parquet"
    pd.DataFrame({"race_id": df["race_id"].to_numpy(), "hrNo": df["hrNo"].to_numpy(),
                  "score": np.asarray(scores, np.float32)}).to_parquet(out, index=False)
    return out


def load_pred(df: pd.DataFrame, name: str) -> np.ndarray | None:
    """저장된 예측을 df 행 순서에 맞춰 돌려준다. 없으면 None."""
    p = PRED_DIR / f"{name}.parquet"
    if not p.exists():
        return None
    saved = pd.read_parquet(p)
    key = pd.MultiIndex.from_arrays([df["race_id"], df["hrNo"]])
    s = saved.set_index(["race_id", "hrNo"])["score"].reindex(key)
    if s.isna().any():
        raise ValueError(f"{p.name}: df 와 (race_id, hrNo) 가 맞지 않는 행 {int(s.isna().sum())}개")
    return s.to_numpy()
