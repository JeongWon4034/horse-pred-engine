# -*- coding: utf-8 -*-
"""평가.

적중률(top-1/top-3)은 팀 채점기(C.evaluate)가 재고, 여기서는 그 옆에 붙일 것만 만든다.
정확도만 보면 인기마만 찍는 모델이 1등이 되는데 공제율 20% 때문에 실제로는 손해다.
그래서 확률 품질(logloss·ECE·Brier), 역배 성적, "차이가 우연인가"(paired bootstrap)를
같이 본다.

logloss·ECE·paired bootstrap 원본: 이정원 horse-pred-engine `pipeline/model/evaluate.py`.
역배·ROI 지표는 이 프로젝트에서 추가한 것.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .team import C

SEED = cfg.SEED

# 환급률 — EDA 실측(docs/eda/out/18_takeout.csv): 경주별 1/배당 합 중앙값 1.25 → 80%.
# 학습셋에는 확정배당이 없고 오버라운드가 제거된 F6_mkt_prob 만 있으므로 여기서 역산한다.
PAYOUT_RETURN = 0.80


def est_odds(df: pd.DataFrame) -> np.ndarray:
    """시장 내재확률에서 역산한 단승배당 추정치.

    학습셋에 확정배당 원값이 없고 경주 내 합=1 로 정규화된 확률만 있어서 역산한다.
    꼬리에서 1/p 가 터지므로(배당 없음 센티널이 섞인 말) `cfg.ODDS_CAP` 에서 자른다 —
    자르지 않으면 90배·270배짜리 한 번 적중이 ROI 를 +500% 로 만든다(실제로 그랬다).
    """
    p = df["F6_mkt_prob"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.where(p > 0, PAYOUT_RETURN / p, np.nan)
    return np.minimum(o, cfg.ODDS_CAP)


def race_znorm(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """결합 점수를 경주 내에서 평균 0 · 표준편차 1 로. 온도 T 를 가중치와 무관하게 만든다.

    축 점수는 각각 경주 내 z-score 지만, 가중합의 폭은 가중치가 얼마나 몰렸느냐에 따라
    달라진다(균등이면 좁고 한 축에 몰리면 넓다). 그래서 T 하나를 균등에서 맞추면
    극단 가중치에서 확률이 무너진다 — 실측으로 '실력 100' 에서 상위 5% 예측 76.4% vs
    실제 4.5% 였다. 폭을 먼저 1 로 맞추면 T 가 어느 가중치에서도 같은 뜻이 된다.

    잃는 것: '이 경주는 한 마리가 압도적'이라는 폭 정보. 축 점수가 이미 경주별로
    표준화돼 있어 그 정보는 대부분 분포의 모양에 남고, 폭에는 거의 없다.
    """
    s = np.asarray(raw, np.float64)
    g = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    out = np.empty_like(s)
    for a, b in zip(starts, ends):
        blk = s[a:b]
        out[a:b] = (blk - blk.mean()) / (blk.std() + 1e-6)
    return out


# ─────────────────────────── 경주 내 확률 ───────────────────────────
def win_probs(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """경주 내 softmax. 온도 1 고정 — LightGBM 점수도 같은 식으로 확률화한다."""
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
    """1착마에 준 확률의 -log 평균. 동착은 확률을 합쳐 한 값으로 센다."""
    p = win_probs(df, scores)
    win = df["y_win"].to_numpy() == 1
    p_win = pd.Series(p * win).groupby(df["race_id"].to_numpy(), sort=False).sum()
    return float(-np.log(np.clip(p_win.to_numpy(), 1e-12, 1)).mean())


def brier(df: pd.DataFrame, scores: np.ndarray) -> float:
    p = win_probs(df, scores)
    return float(((p - df["y_win"].to_numpy(float)) ** 2).mean())


def ece(df: pd.DataFrame, scores: np.ndarray, bins: int = 15) -> float:
    """'41%라고 한 것들이 실제로 41% 맞았나' — 작을수록 좋다."""
    p, y = win_probs(df, scores), df["y_win"].to_numpy(float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


# ─────────────────────────── 선택 · 역배 ───────────────────────────
def picks(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """경주마다 점수 최고 말 한 행. 순서는 race_id 등장 순서."""
    scores = np.asarray(scores, float)
    if np.isnan(scores).any():
        raise ValueError(f"점수에 NaN {int(np.isnan(scores).sum())}개 — idxmax 가 조용히 틀린다")
    d = df[["race_id", "y_win", "y_plc"]].assign(_s=scores, _o=est_odds(df))
    return d.loc[d.groupby("race_id", sort=False)["_s"].idxmax()]


def hits(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """경주마다 고른 말이 1착이었는지 0/1."""
    return picks(df, scores)["y_win"].to_numpy(int)


def upset_metrics(df: pd.DataFrame, scores: np.ndarray,
                  odds_floor: float = cfg.UPSET_ODDS,
                  odds_ceil: float = cfg.UPSET_ODDS_MAX) -> dict:
    """역배 성적.

    upset_hit  전체 경주 중 '배당 [floor, ceil] 인 말을 골라 그 말이 1착' 한 비율
               (schema.sql `is_upset_hit` = 리더보드 UPSET 축과 같은 방향)
    roi        고른 말에 매 경주 1 씩 단승 베팅했을 때 수익률. 역산 배당이라 근사치다.
    avg_odds   고른 말의 평균 배당 — 이 프리셋이 실제로 역배를 고르고 있는지 본다
    """
    p = picks(df, scores)
    win, odds = p["y_win"].to_numpy(float), p["_o"].to_numpy(float)
    long_shot = (odds >= odds_floor) & (odds <= odds_ceil)
    payout = np.where(win > 0, np.nan_to_num(odds, nan=0.0), 0.0)
    return {
        "upset_hit": float(np.mean(long_shot & (win > 0)) * 100),
        "long_shot_pick": float(np.mean(long_shot) * 100),
        "roi": float(payout.mean() - 1) * 100,
        "avg_odds": float(np.nanmean(odds)),
        "median_odds": float(np.nanmedian(odds)),
    }


def compare(hits_a: np.ndarray, hits_b: np.ndarray, n: int = 2000, seed: int = SEED):
    """같은 경주 묶음에서 a − b 의 top-1 차이(%p)와 paired bootstrap 95% CI.

    CI 가 0 을 품으면 우연 범위, 벗어나면 차이가 있다고 읽는다.
    """
    a, b = np.asarray(hits_a, float), np.asarray(hits_b, float)
    if len(a) != len(b):
        raise ValueError(f"경주 수 불일치: {len(a)} vs {len(b)}")
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1) * 100
    return float(d.mean() * 100), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ─────────────────────────── 한 번에 ───────────────────────────
def metrics(df: pd.DataFrame, scores: np.ndarray, with_upset: bool = False) -> dict:
    """top1/top3 는 팀 채점기 값 그대로, 나머지는 위 함수."""
    m = C.evaluate(df, scores)
    out = {"top1": m["top1"], "top3": m["top3"], "n": m["n_races"], "se": m["se_top1"],
           "logloss": race_logloss(df, scores), "ece": ece(df, scores), "brier": brier(df, scores)}
    if with_upset:
        out.update(upset_metrics(df, scores))
    return out


def axis_report(df: pd.DataFrame, axis_scores: dict[str, np.ndarray],
                slider_axes: list[str] | None = None) -> str:
    """축 단독 성적과 축 간 상관. 슬라이더가 실제로 다른 예측을 내는지 보는 표.

    합격선 두 개 (이정원 S5 기준):
      · 축 단독 top-1 >= 25  — 그 축만 100 으로 밀어도 쓸모가 있어야 한다
      · 축 간 상관 <= 0.9    — 넘으면 슬라이더를 움직여도 예측이 안 바뀐다

    ★ 합격선은 **슬라이더 축끼리만** 본다. 시장 축은 유저가 못 만지므로 다른 축과
      상관이 높아도 슬라이더 체감과 무관하다(오히려 시장이 실력을 반영하니 높은 게 자연스럽다).
      두 숫자를 같이 내서 어느 쪽 이야기인지 헷갈리지 않게 한다.
    """
    names = list(axis_scores)
    lines = ["[축 단독 — 그 축만 100, 나머지 0]",
             f"  {'축':<12}{'top-1':>8}{'top-3':>8}{'평균배당':>10}"]
    for k in names:
        m = C.evaluate(df, axis_scores[k])
        u = upset_metrics(df, axis_scores[k])
        lines.append(f"  {k:<12}{m['top1']:>7.1f}%{m['top3']:>7.1f}%{u['avg_odds']:>10.1f}")

    mat = np.corrcoef(np.stack([axis_scores[k] for k in names]))
    lines += ["", "[축 간 예측 상관 (행 점수 피어슨)]", "      " + "".join(f"{k[:6]:>8}" for k in names)]
    for i, k in enumerate(names):
        lines.append(f"  {k[:4]:<4}" + "".join(f"{mat[i, j]:8.2f}" for j in range(len(names))))

    sl = [k for k in names if k in (slider_axes or names)]
    idx = [names.index(k) for k in sl]
    sub = mat[np.ix_(idx, idx)]
    off_slider = np.abs(sub - np.eye(len(idx))).max() if len(idx) > 1 else 0.0
    off_all = np.abs(mat - np.eye(len(names))).max()
    verdict = "합격 (<=0.9)" if off_slider <= 0.9 else "★ 불합격 — 슬라이더가 죽는다"
    lines.append(f"  슬라이더 축끼리 최대 상관 {off_slider:.2f}   {verdict}")
    if len(sl) != len(names):
        lines.append(f"  (시장 축 포함하면 {off_all:.2f} — 합격선 판정에는 쓰지 않는다)")
    return "\n".join(lines)
