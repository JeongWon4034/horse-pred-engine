# -*- coding: utf-8 -*-
"""베이스 모델 5종의 시작 가중치를 **데이터에서 맞춘다**.

기능명세서 MP-01 은 "각 모델은 시작 가중치 분배만 다름"이라고만 정한다. 그 숫자를
손으로 고르면 근거를 댈 수 없으므로, 프리셋마다 목적함수를 정하고 학습 구간에서
최적 가중치를 찾는다.

    최종 점수 = m · s_MARKET + Σ_k w_k · s_k      Σ w_k = 1, w_k >= 0

  BASIC     제약 없음, 1착 적중 최대                     → "데이터가 고른 균형"
  HUMAN     JOCKEY 40 고정, 나머지 적합                  → 사람형
  MOMENTUM  CONDITION 40 고정, 나머지 적합               → 상승세형
  ODDS      m 을 크게, 나머지 적합                       → 배당형 (시장 추종)
  UPSET     m 을 음수로, **고배당 적중** 최대            → 역배형

## 왜 랜덤 탐색인가
목적함수가 둘 다 미분 불가능하다(1착 적중 = argmax, 고배당 적중 = 배당 임계값).
6차원 심플렉스는 디리클레 몇 천 개면 충분히 덮이고, 다섯 프리셋에 **같은 절차**를
쓸 수 있어 비교가 성립한다. 경사법으로 풀면 UPSET 만 다른 방법이 되어버린다.

## 어디서 맞추고 어디서 보고하나
**적합은 train, 보고는 valid.** valid 로 고르고 valid 로 점수를 내면 모의고사로 고르고
모의고사로 채점하는 것이라 부풀려진다(레퍼런스가 앙상블 비율에서 실측한 편향 +0.5%p).
train 은 신경망이 이미 본 구간이지만 여기서 맞추는 건 7개 숫자뿐이다.
`crossfit_report()` 가 valid 2-fold 교차적합으로 그 판단이 맞는지 다시 확인한다.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import config as cfg
from .evaluate import est_odds, upset_metrics
from .team import C

# 시장 계수 m 의 탐색 구간 — 프리셋 정책별. (하한, 상한) 균등 샘플.
MARKET_RANGE = {
    "fit+": (0.0, 0.8),      # 시장을 봐도 되지만 강제하지 않는다
    "fit-": (-0.8, 0.0),     # 역배 — 시장이 저평가한 말 쪽으로
    "high": (0.6, 2.0),      # 배당형 — 시장을 정면으로 따라간다
    "zero": (0.0, 0.0),      # 배당이 없는 조건(73피처)
}


def _rows(axis_rows: np.ndarray, w: np.ndarray, m: float, n_axes: int) -> np.ndarray:
    """행 단위 점수 — 화면(AI-06)이 하는 계산과 같은 식."""
    out = axis_rows[:, :n_axes] @ np.asarray(w, float)
    if axis_rows.shape[1] > n_axes:
        out = out + axis_rows[:, -1] * float(m)
    return out


class Objective:
    """경주를 [경주, 최대두수] 격자로 한 번 접어 두고 후보 수천 개를 한꺼번에 채점한다.

    경주마다 argmax 를 파이썬으로 돌면 36,586경주 × 후보 4,000개라 끝나지 않는다.
    축 점수를 [R, MAXF, A+1] 로 접어 두면 후보 채점이 행렬곱 한 번 + argmax 한 번이 된다.
    시장 계수 m 도 후보 벡터의 마지막 칸으로 같이 넣어 격자 탐색을 없앤다.
    """

    def __init__(self, df: pd.DataFrame, axis_rows: np.ndarray, n_axes: int):
        rid = df["race_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        ends = np.r_[starts[1:], len(rid)]
        sizes = ends - starts
        R, F = len(starts), int(sizes.max())
        row_race = np.repeat(np.arange(R), sizes)
        row_slot = np.arange(len(df)) - np.repeat(starts, sizes)

        self.n_axes = n_axes
        self.has_market = axis_rows.shape[1] > n_axes
        self.dim = n_axes + (1 if self.has_market else 0)
        self.n_races = R

        self.AX = np.zeros((R, F, self.dim), np.float32)
        self.AX[row_race, row_slot] = axis_rows[:, :self.dim]
        self.valid = np.zeros((R, F), bool)
        self.valid[row_race, row_slot] = True

        self.win = np.zeros((R, F), np.float32)
        self.win[row_race, row_slot] = df["y_win"].to_numpy(np.float32)
        self.odds = np.full((R, F), np.nan, np.float32)
        self.odds[row_race, row_slot] = est_odds(df).astype(np.float32)

        # 패딩 슬롯이 뽑히지 않도록 점수에 더할 큰 음수
        self.pad_penalty = np.where(self.valid, 0.0, -1e9).astype(np.float32)

    def _theta(self, w: np.ndarray, m: float) -> np.ndarray:
        return np.r_[np.asarray(w, np.float32), np.float32(m)] if self.has_market \
            else np.asarray(w, np.float32)

    def pick_slots(self, theta: np.ndarray) -> np.ndarray:
        """[R] 또는 [R, C] — 경주마다 점수 최고 슬롯."""
        s = self.AX @ theta.astype(np.float32)            # [R,F] 또는 [R,F,C]
        s = s + (self.pad_penalty[..., None] if s.ndim == 3 else self.pad_penalty)
        return s.argmax(1)

    def _from_slots(self, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.arange(self.n_races)
        if slots.ndim == 1:
            return self.win[r, slots], self.odds[r, slots]
        return self.win[r[:, None], slots], self.odds[r[:, None], slots]

    def evaluate(self, w: np.ndarray, m: float) -> dict:
        win, odds = self._from_slots(self.pick_slots(self._theta(w, m)))
        long_shot = (odds >= cfg.UPSET_ODDS) & (odds <= cfg.UPSET_ODDS_MAX)
        return {
            "top1": float(win.mean() * 100),
            "upset_hit": float(np.mean(long_shot & (win > 0)) * 100),
            "avg_odds": float(np.nanmean(odds)),
            "median_odds": float(np.nanmedian(odds)),
            "roi": float(np.where(win > 0, np.nan_to_num(odds), 0.0).mean() - 1) * 100,
        }

    def batch_objective(self, THETA: np.ndarray, key: str, chunk: int = 256) -> np.ndarray:
        """[C] — 후보별 목적함수 값. THETA 는 [dim, C]."""
        out = np.empty(THETA.shape[1], np.float64)
        for i in range(0, THETA.shape[1], chunk):
            sl = slice(i, i + chunk)
            win, odds = self._from_slots(self.pick_slots(THETA[:, sl]))
            out[sl] = (win.mean(0) if key == "top1"
                       else ((odds >= cfg.UPSET_ODDS) & (odds <= cfg.UPSET_ODDS_MAX)
                             & (win > 0)).mean(0))
        return out


def _candidates(axes: list[str], pin: dict[str, float], n: int, rng) -> np.ndarray:
    """제약을 만족하는 가중치 후보 [n, A]. pin 은 비율(0~1) 로 받는다."""
    A = len(axes)
    free = [i for i, ax in enumerate(axes) if ax not in pin]
    fixed = {axes.index(k): v for k, v in pin.items()}
    rest = 1.0 - sum(fixed.values())
    if rest < 0:
        raise ValueError(f"pin 합이 1 을 넘는다: {pin}")

    W = np.zeros((n, A), np.float64)
    for i, v in fixed.items():
        W[:, i] = v
    if free:
        # α 를 섞어 뽑는다 — 작은 α 는 한 축에 몰린 조합, 큰 α 는 고른 조합
        alphas = rng.choice([0.3, 0.7, 1.0, 3.0], size=n)
        d = np.stack([rng.dirichlet(np.full(len(free), a)) for a in alphas])
        W[:, free] = d * rest
    return W



def _fit_market_only(obj: Objective, axes: list[str], name: str, spec: dict,
                     w: np.ndarray, key: str, policy: str,
                     n_grid: int = 81, smooth: int = 9) -> dict:
    """가중치를 고정하고 시장 계수 m 하나만 맞춘다 — 격자 + 평활.

    고배당 적중은 1% 미만의 희귀사건이라 격자에서 최댓값 한 점을 그냥 집으면 잡음을 집는다
    (실측: 후보를 자유롭게 뒀을 때 교차적합에서 평균배당이 34.8 → 14.1 로 흔들렸다).
    m 이 1차원이라 곡선 전체를 그릴 수 있으므로, 이동평균으로 고른 뒤 봉우리를 집는다.
    `curve` 를 같이 돌려주어 근거를 눈으로 확인할 수 있게 한다.
    """
    lo, hi = MARKET_RANGE[policy]
    grid = np.linspace(lo, hi, n_grid) if hi > lo else np.array([lo])
    THETA = np.vstack([np.repeat(w[:, None], len(grid), axis=1), grid[None, :]]) \
        if obj.has_market else np.repeat(w[:, None], len(grid), axis=1)
    vals = obj.batch_objective(THETA.astype(np.float32), key)

    k = min(smooth, len(vals))
    if k > 1:
        pad = np.r_[np.repeat(vals[0], k // 2), vals, np.repeat(vals[-1], k // 2)]
        sm = np.convolve(pad, np.ones(k) / k, mode="valid")[:len(vals)]
    else:
        sm = vals
    m = float(grid[int(np.argmax(sm))])
    return {"name": name, "label": spec["label"], "desc": spec["desc"],
            "w": w.copy(), "m": m, "objective": spec["objective"],
            "fit": obj.evaluate(w, m),
            "curve": [{"m": round(float(g), 3), "raw": round(float(v) * 100, 3),
                       "smooth": round(float(s) * 100, 3)}
                      for g, v, s in zip(grid, vals, sm)]}


def fit_preset(obj: Objective, axes: list[str], name: str, spec: dict,
               n_samples: int = 4000, seed: int = cfg.SEED,
               allow_market: bool = True, inherited: np.ndarray | None = None,
               top_frac: float = 0.01) -> dict:
    """한 프리셋의 (w, m) 을 찾는다. 반환 w 는 합 1 비율.

    w 와 m 을 **같이** 뽑는다. 시장 계수를 격자로 훑고 그 안에서 다시 가중치를 훑으면
    후보가 곱해져 못 돈다. 후보 벡터 하나에 (w, m) 을 담아 한 번에 채점한다.
    """
    # ★ 파이썬 문자열 hash() 는 프로세스마다 달라진다(PYTHONHASHSEED). 실행할 때마다
    #   프리셋 가중치가 바뀌는 원인이었다. crc32 는 어디서 돌려도 같은 값이다.
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()) % 10_000)
    key = "top1" if spec["objective"] == "top1" else "upset_hit"
    policy = spec["market"] if (allow_market and obj.has_market) else "zero"

    if "inherit" in spec:
        if inherited is None:
            raise ValueError(f"{name} 은 {spec['inherit']} 의 가중치를 물려받는데 못 받았다")
        return _fit_market_only(obj, axes, name, spec, inherited, key, policy)
    else:
        W = _candidates(axes, {k: v / 100.0 for k, v in spec.get("pin", {}).items()},
                        n_samples, rng)                               # [n, A]
    lo, hi = MARKET_RANGE[policy]
    M = rng.uniform(lo, hi, size=n_samples) if hi > lo else np.full(n_samples, lo)
    THETA = np.vstack([W.T, M[None, :]]) if obj.has_market else W.T  # [dim, n]

    vals = obj.batch_objective(THETA.astype(np.float32), key)

    # 최댓값 하나를 집지 않고 **상위 top_frac 후보의 평균**을 쓴다.
    #   심플렉스 위에서 목적함수가 평평하다 — 꽤 다른 가중치가 오차 안에서 같은 점수를 낸다
    #   (실측: 기수 32 vs 기수 37 조합의 top-1 차이가 0.5%p, 표준오차 0.85%p 안).
    #   그 상태에서 argmax 한 점을 집으면 표본 잡음을 프리셋으로 굳히게 되고, 후보 집합이
    #   조금만 달라져도 화면에 뜨는 숫자가 바뀐다. 평균을 쓰면 안정적이고, 여러 좋은 답의
    #   중심이라 해석도 자연스럽다.
    n_top = max(int(len(vals) * top_frac), 1)
    top = np.argpartition(-vals, n_top - 1)[:n_top]
    w = W[top].mean(0)
    w = w / w.sum()
    for k, v in spec.get("pin", {}).items():        # 평균이 고정 칸을 흔들지 않게 되돌린다
        i = axes.index(k)
        rest = 1.0 - v / 100.0
        others = w.sum() - w[i]
        w = w * (rest / others if others > 0 else 0.0)
        w[i] = v / 100.0
    m = float(M[top].mean()) if obj.has_market else 0.0
    return {"name": name, "label": spec["label"], "desc": spec["desc"],
            "w": w, "m": m, "objective": spec["objective"],
            "n_top": n_top, "best_single": float(vals.max() * 100),
            "fit": obj.evaluate(w, m)}


def to_group_weights(w: np.ndarray, axes: list[str]) -> dict[str, int]:
    """비율 → 합이 정확히 100 인 정수 퍼센트. BE `pae_group_weight` 가 그렇게 저장한다.

    최대 잔차 배분(Hamilton 방식) — 반올림만 하면 합이 99 나 101 이 된다.
    """
    raw = np.asarray(w, float) * 100
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[:100 - base.sum()]:
        base[i] += 1
    return {ax: int(v) for ax, v in zip(axes, base)}


def fit_all(obj: Objective, axes: list[str], n_samples: int = 4000,
            seed: int = cfg.SEED, allow_market: bool = True,
            obj_eval: "Objective | None" = None) -> dict[str, dict]:
    """obj = 적합 구간(train), obj_eval = 평가 구간(valid).

    `fit_on: "eval"` 인 프리셋은 obj_eval 에서 맞춘다 — config.PRESETS 의 주석 참조.
    obj_eval 을 안 주면 전부 obj 에서 맞춘다(교차적합 계산이 그렇게 쓴다).
    """
    out = {}
    # inherit 하는 프리셋은 원본이 먼저 나와야 한다 — 의존 순서대로 정렬한다.
    order = sorted(cfg.PRESETS, key=lambda k: "inherit" in cfg.PRESETS[k])
    for name in order:
        spec = cfg.PRESETS[name]
        if not allow_market and name in cfg.REPLAY_ONLY_PRESETS:
            continue                      # 73피처 모드에는 배당형·역배형이 성립하지 않는다
        src = out.get(spec["inherit"], {}).get("w") if "inherit" in spec else None
        target = obj_eval if (spec.get("fit_on") == "eval" and obj_eval is not None) else obj
        p = fit_preset(target, axes, name, spec, n_samples, seed, allow_market, inherited=src)
        p["fit_on"] = "eval" if target is obj_eval else "train"
        p["groupWeights"] = to_group_weights(p["w"], axes)
        out[name] = p
    return {k: out[k] for k in cfg.PRESETS if k in out}


def report(presets: dict[str, dict], df: pd.DataFrame, axis_rows: np.ndarray,
           n_axes: int, axes: list[str]) -> str:
    """프리셋별 성적표. 적합 구간이 아니라 **보고 구간(valid)** 에서 잰 값."""
    lines = [f"  {'프리셋':<10}{'top-1':>8}{'top-3':>8}{'역배적중':>9}{'평균배당':>9}{'ROI':>8}   {'m':>6}  가중치"]
    for name, p in presets.items():
        s = _rows(axis_rows, p["w"], p["m"], n_axes)
        t = C.evaluate(df, s)
        u = upset_metrics(df, s)
        gw = "·".join(f"{cfg.AXIS_KO[ax]}{v}" for ax, v in p["groupWeights"].items() if v)
        lines.append(f"  {p['label']:<10}{t['top1']:>7.1f}%{t['top3']:>7.1f}%"
                     f"{u['upset_hit']:>8.1f}%{u['avg_odds']:>9.1f}{u['roi']:>7.1f}%"
                     f"{p['m']:>7.2f}  {gw}")
    return "\n".join(lines)


def crossfit_report(df: pd.DataFrame, axis_rows: np.ndarray, n_axes: int, axes: list[str],
                    n_samples: int = 2000, seed: int = cfg.SEED,
                    allow_market: bool = True) -> str:
    """valid 를 경주 단위로 반 갈라 한쪽에서 맞추고 다른 쪽에서 잰다.

    "train 에서 맞춘 가중치가 valid 에서도 통하나"를 valid 안에서 한 번 더 확인하는 것.
    train 적합값과 크게 다르면 적합이 신경망의 train 과적합을 타고 있다는 뜻이다.
    """
    rid = df["race_id"].to_numpy()
    uniq = pd.unique(rid)
    rng = np.random.default_rng(seed)
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    fold = np.isin(rid, list(half))

    acc: dict[str, list] = {}
    for a, b in ((fold, ~fold), (~fold, fold)):
        sub_fit, sub_ev = df[a].reset_index(drop=True), df[b].reset_index(drop=True)
        obj_fit = Objective(sub_fit, axis_rows[a], n_axes)
        obj_ev = Objective(sub_ev, axis_rows[b], n_axes)
        for name, p in fit_all(obj_fit, axes, n_samples, seed, allow_market).items():
            acc.setdefault(name, []).append(obj_ev.evaluate(p["w"], p["m"]))

    lines = [f"  {'프리셋':<10}{'top-1':>8}{'역배적중':>9}{'평균배당':>9}"]
    for name, ms in acc.items():
        t1 = np.mean([m["top1"] for m in ms])
        uh = np.mean([m["upset_hit"] for m in ms])
        od = np.mean([m["avg_odds"] for m in ms])
        lines.append(f"  {cfg.PRESETS[name]['label']:<10}{t1:>7.1f}%{uh:>8.1f}%{od:>9.1f}")
    return "\n".join(lines)
