# -*- coding: utf-8 -*-
"""시장 대비 유의성 검정.

    uv run python -m basemodel.significance --ckpt artifacts/runs/axis_77_s20260901_final.pt

## 무엇을 묻는가

"시장을 이겼나"만 보면 답은 거의 항상 '아니오'다. 시장 배당에는 수만 명의 판단과 우리가
못 보는 정보(당일 컨디션·마방 분위기·직전 조교)가 들어 있고, 게다가 우리가 가진 건
**확정배당(경주 후 값)** 이라 발주 직전 배당보다 약간 더 똑똑하다. 그래서 세 가지를 나눠 묻는다.

  ① 얼마나 지는가        시장 − 모델 차이와 95% 신뢰구간, McNemar 검정
  ② 정보를 더하는가       시장 단독 vs 시장 + 모델 — **조건부 로짓 우도비 검정**
                          Benter(1994) 가 홍콩에서 실제로 돈을 번 근거가 이 검정이다.
                          이기지 못해도 '시장이 모르는 것'을 갖고 있으면 값이 있다.
  ③ 확률이 정직한가       logloss · Brier · ECE — 순위가 아니라 확률의 품질

## 검정 방법

| 무엇 | 방법 | 왜 |
|---|---|---|
| 두 모델의 적중률 차이 | **McNemar 정확검정** | 같은 경주를 둘 다 맞히므로 짝지은 자료다. 독립 표본 검정은 틀린다 |
| 차이의 신뢰구간 | 경주 단위 **paired bootstrap** | 분포 가정 없이 구간을 낸다 |
| 정보 증분 | **우도비 검정** (χ²₁) | 시장 항만 있는 모형에 모델 항을 넣어 로그우도가 유의하게 오르는지 |
| 동률 | **1/k 기대값** | 인기 1위가 2두면 실제 베팅은 한 두만 고른다. valid 16경주(1.3%) 해당 |
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as cfg
from . import presets as P
from .evaluate import brier, ece, est_odds, race_logloss, win_probs
from .serve import Scorer, combine
from .team import ARTIFACTS, C

SEED = cfg.SEED


# ═══════════════════════════ 채점 (동률 1/k) ═══════════════════════════
def race_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rid = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    return starts, np.r_[starts[1:], len(rid)]


def hit_vectors(df: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """경주별 (1착 적중 기대값, 3착내 적중 기대값). 동률은 1/k 로 나눈다.

    팀 채점기(C.evaluate)는 idxmax 라 동률에서 첫 행을 집는다 — 임의 선택이라
    시장 기준선이 0.16%p 낮게 나온다. 유의성을 따지는 자리에서는 기대값을 쓴다.
    """
    s = np.asarray(scores, float)
    if np.isnan(s).any():
        raise ValueError(f"점수에 NaN {int(np.isnan(s).sum())}개")
    starts, ends = race_index(df)
    win, plc = df["y_win"].to_numpy(float), df["y_plc"].to_numpy(float)
    h1 = np.empty(len(starts))
    h3 = np.empty(len(starts))
    for i, (a, b) in enumerate(zip(starts, ends)):
        blk = s[a:b]
        top = blk >= blk.max() - 1e-12
        h1[i] = win[a:b][top].mean()
        h3[i] = plc[a:b][top].mean()
    return h1, h3


# ═══════════════════════════ 검정 ═══════════════════════════
def mcnemar(h_a: np.ndarray, h_b: np.ndarray) -> dict:
    """짝지은 이항 결과의 McNemar 정확검정.

    b = A 만 맞힌 경주, c = B 만 맞힌 경주. 둘이 같다는 귀무가설 아래
    b ~ Binom(b+c, 0.5). 양측 p 를 정확히 센다(정규근사 안 쓴다).

    동률(1/k) 때문에 적중이 0/1 이 아닐 수 있어 0.5 기준으로 이산화한다 —
    해당 경주가 1.3% 뿐이라 결론에 영향이 없고, 함께 내는 bootstrap 구간은 원값을 쓴다.
    """
    from scipy.stats import binomtest

    A = np.asarray(h_a) > 0.5
    B = np.asarray(h_b) > 0.5
    b = int((A & ~B).sum())
    c = int((~A & B).sum())
    n = b + c
    p = float(binomtest(b, n, 0.5).pvalue) if n else 1.0
    return {"a_only": b, "b_only": c, "discordant": n, "p": p}


def race_nll(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """경주별 −log(1착마에 준 확률). 평균이 race_logloss 다.

    적중률은 '어느 말을 골랐나'만 보고 나머지 확률은 버린다. logloss 는 확률 전체를 본다 —
    배당·연승 화면이 쓰는 것도 순위가 아니라 확률이라 이쪽이 서비스에 더 직접적이다.
    경주마다 값이 하나씩 나오므로 적중률과 **같은 방식으로 짝 검정**을 걸 수 있다.
    """
    p = win_probs(df, scores)
    win = df["y_win"].to_numpy() == 1
    s = pd.Series(p * win).groupby(df["race_id"].to_numpy(), sort=False).sum()
    return -np.log(np.clip(s.to_numpy(), 1e-12, 1))


def paired_bootstrap(h_a: np.ndarray, h_b: np.ndarray, n: int = 10000,
                     seed: int = SEED) -> tuple[float, float, float]:
    """(A−B) 평균 차이(%p)와 95% 신뢰구간. 경주 단위 재표집."""
    d = np.asarray(h_a, float) - np.asarray(h_b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1) * 100
    return float(d.mean() * 100), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def mde(h_a: np.ndarray, h_b: np.ndarray, power: float = 0.8, alpha: float = 0.05) -> float:
    """최소 검출 가능 효과(%p) — 이 표본으로 몇 %p 차이부터 잡아낼 수 있나.

    "차이가 없다"와 "차이를 못 잰다"를 구분하려면 이 숫자가 필요하다.

    ★ **짝지은 자료의 공식을 쓴다** — sd(차이)/√n 이지 sd(적중)·√(2/n) 이 아니다.
      두 모델이 같은 경주를 대부분 같이 맞히면 차이의 분산이 훨씬 작아서, 비짝 공식을
      쓰면 검출력을 크게 과소평가한다(여기서는 5.5%p vs 실제 2.1%p 로 2.6배 차이).
    """
    from scipy.stats import norm

    d = np.asarray(h_a, float) - np.asarray(h_b, float)
    sd = float(d.std(ddof=1))
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / np.sqrt(len(d)) * 100)


def required_n(d_ref: np.ndarray, target_pp: float, power: float = 0.8,
               alpha: float = 0.05) -> int:
    """target_pp(%p) 차이를 80% 확률로 잡아내려면 경주가 몇 개 필요한가.

    차이의 표준편차는 기존 짝 비교에서 추정한다. "차이가 없다"는 결론을 내리기 전에
    표본이 충분했는지 확인하는 용도이고, 주말 경주를 얼마나 모아야 하는지도 여기서 나온다.
    """
    from scipy.stats import norm

    sd = float(np.asarray(d_ref, float).std(ddof=1))
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z * sd / (target_pp / 100)) ** 2))


# ═══════════════════ 정보 증분 — 조건부 로짓 우도비 ═══════════════════
def _pl_loglik(df: pd.DataFrame, X: np.ndarray, beta: np.ndarray) -> float:
    starts, ends = race_index(df)
    eta = X @ beta
    win = df["y_win"].to_numpy() == 1
    ll = 0.0
    for a, b in zip(starts, ends):
        e = eta[a:b]
        e = e - e.max()
        p = np.exp(e) / np.exp(e).sum()
        ll += np.log(max(p[win[a:b]].sum(), 1e-12))
    return float(ll)


def fit_conditional_logit(df: pd.DataFrame, X: np.ndarray, iters: int = 400,
                          lr: float = 0.1, init: np.ndarray | None = None
                          ) -> tuple[np.ndarray, float]:
    """경주 내 조건부 로짓(= Plackett-Luce 1착) 최대우도. (계수, 로그우도).

    경주 블록마다 softmax 를 걸고 1착마의 로그확률을 최대화한다. 파라미터가 1~3개라
    풀배치 LBFGS 로 충분하다.

    ★ `init` — 중첩 모형을 잴 때 작은 모형의 해에서 출발한다(웜스타트). 안 하면 큰 모형이
      작은 모형보다 **낮은** 우도로 수렴하는 일이 생겨 우도비가 음수가 된다. 효과가 작을수록
      (계수 0.01 수준) 최적화 잡음이 신호보다 커서 실제로 그런 일이 일어났다 — 마체중
      검정에서 LR 이 전부 정확히 0.00 으로 나온 원인이다. 웜스타트는 ll1 >= ll0 을 보장한다.
    """
    # ★ 1착마가 없는 경주가 섞이면 조용히 망가진다 — 그 행의 logsumexp 가 마스크값(-1e30)이
    #   되어 로그우도가 -1e30 단위로 더해지고, 우도비가 전부 0 으로 나온다(실제로 겪었다).
    #   원인은 대개 상위 필터가 우승마만 지운 경우다. 여기서 막고 알린다.
    n_win = df.groupby("race_id", sort=False)["y_win"].transform("sum").to_numpy()
    if (n_win == 0).any():
        bad = df.loc[n_win == 0, "race_id"].nunique()
        raise ValueError(
            f"1착마가 없는 경주 {bad}개가 섞여 있다 — 조건부 로짓이 성립하지 않는다. "
            "말 단위로 거르지 말고 경주 단위로 걸러야 한다.")

    starts, ends = race_index(df)
    sizes = ends - starts
    R, F_ = len(starts), int(sizes.max())
    row_race = np.repeat(np.arange(R), sizes)
    row_slot = np.arange(len(df)) - np.repeat(starts, sizes)

    Xp = torch.zeros(R, F_, X.shape[1], dtype=torch.float64)
    Xp[row_race, row_slot] = torch.as_tensor(X, dtype=torch.float64)
    mask = torch.zeros(R, F_, dtype=torch.bool)
    mask[row_race, row_slot] = True
    winp = torch.zeros(R, F_, dtype=torch.float64)
    winp[row_race, row_slot] = torch.as_tensor(df["y_win"].to_numpy(float))

    b0 = torch.zeros(X.shape[1], dtype=torch.float64)
    if init is not None:
        b0[:len(init)] = torch.as_tensor(np.asarray(init, float), dtype=torch.float64)
    beta = b0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([beta], lr=lr, max_iter=iters, tolerance_grad=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        eta = (Xp @ beta).masked_fill(~mask, -1e30)
        logp = eta - torch.logsumexp(eta, 1, keepdim=True)
        ll = torch.logsumexp(logp.masked_fill(winp == 0, -1e30), 1).sum()
        (-ll).backward()
        return -ll

    opt.step(closure)
    with torch.no_grad():
        eta = (Xp @ beta).masked_fill(~mask, -1e30)
        logp = eta - torch.logsumexp(eta, 1, keepdim=True)
        ll = float(torch.logsumexp(logp.masked_fill(winp == 0, -1e30), 1).sum())
    return beta.detach().numpy(), ll


def incremental_test(df: pd.DataFrame, model_score: np.ndarray) -> dict:
    """시장 항만 있는 모형에 모델 점수를 넣어 로그우도가 유의하게 오르는지.

        H0:  eta = a·log(시장확률)
        H1:  eta = a·log(시장확률) + b·모델점수
        LR = 2(ll1 − ll0) ~ chi2(1)

    b 가 유의하면 **모델이 시장에 없는 정보를 갖고 있다**는 뜻이다. 적중률로 시장을
    못 이겨도 이게 유의하면 결합해서 쓸 값이 있다 — Benter(1994) 가 한 일이 정확히 이것.
    """
    from scipy.stats import chi2

    lp = np.log(np.clip(df["F6_mkt_prob"].to_numpy(float), 1e-9, None))
    s = np.asarray(model_score, float)
    s = (s - s.mean()) / (s.std() + 1e-12)

    b0, ll0 = fit_conditional_logit(df, lp[:, None])
    b1, ll1 = fit_conditional_logit(df, np.c_[lp, s], init=b0)   # 웜스타트 — ll1 >= ll0 보장
    lr = 2 * (ll1 - ll0)
    return {"ll_market": ll0, "ll_both": ll1, "LR": float(lr),
            "p": float(chi2.sf(max(lr, 0), 1)),
            "a_market": float(b1[0]), "b_model": float(b1[1]),
            "a_market_alone": float(b0[0])}


def benter_blend(ev_df: pd.DataFrame, ev_score: np.ndarray, seed: int = SEED) -> dict:
    """시장 + 모델 결합이 시장 단독보다 실제로 더 맞히는가.

    §3 의 우도비 검정은 "정보가 있나"를 묻고, 이건 **"그래서 더 맞히나"** 를 묻는다.

        eta = a·log(시장확률) + b·모델점수

    ★ **계수를 valid 2-fold 교차적합으로 맞춘다.** train 에서 맞추면 안 된다 — 모델이 train 으로
      학습됐으므로 train 점수는 in-sample 이고(LGB 는 train 45.97% vs valid 33.49% 로 과적합),
      그 위에서 고른 결합 계수는 모델을 과대평가한다. valid 에서 맞추고 valid 로 재는 것도
      안 된다(모의고사로 고르고 모의고사로 채점). 반씩 갈라 한쪽에서 맞추고 다른 쪽에서 잰다.

    Benter(1994) 가 홍콩에서 한 것이 이 2단계다. 펀더멘털 모델로 시장을 이기려 한 게
    아니라, 시장 배당에 자기 모델을 **더해서** 시장보다 나은 확률을 만들었다.
    """
    lp = np.log(np.clip(ev_df["F6_mkt_prob"].to_numpy(float), 1e-9, None))
    s = np.asarray(ev_score, float)

    rid = ev_df["race_id"].to_numpy()
    uniq = pd.unique(rid)
    half = set(np.random.default_rng(seed).choice(uniq, len(uniq) // 2, replace=False))
    fold = np.isin(rid, list(half))

    blended = np.empty(len(ev_df))
    coefs = []
    for fit, ev in ((fold, ~fold), (~fold, fold)):
        sub_fit = ev_df[fit].reset_index(drop=True)
        z = (s[fit] - s[fit].mean()) / (s[fit].std() + 1e-12)
        beta, _ = fit_conditional_logit(sub_fit, np.c_[lp[fit], z])
        coefs.append(beta)
        z_ev = (s[ev] - s[fit].mean()) / (s[fit].std() + 1e-12)   # fit 쪽 통계로 표준화
        blended[ev] = np.c_[lp[ev], z_ev] @ beta

    h_b1, h_b3 = hit_vectors(ev_df, blended)
    h_m1, h_m3 = hit_vectors(ev_df, lp)
    d1, lo1, hi1 = paired_bootstrap(h_b1, h_m1)
    d3, lo3, hi3 = paired_bootstrap(h_b3, h_m3)
    a_, b_ = np.mean(coefs, axis=0)
    return {
        "a": float(a_), "b": float(b_),
        "top1": h_b1.mean() * 100, "top3": h_b3.mean() * 100,
        "mkt_top1": h_m1.mean() * 100, "mkt_top3": h_m3.mean() * 100,
        "d1": d1, "ci1": (lo1, hi1), "p1": mcnemar(h_b1, h_m1)["p"],
        "d3": d3, "ci3": (lo3, hi3), "p3": mcnemar(h_b3, h_m3)["p"],
        "logloss": race_logloss(ev_df, blended), "mkt_logloss": race_logloss(ev_df, lp),
        "ece": ece(ev_df, blended), "mkt_ece": ece(ev_df, lp),
        "nll": paired_bootstrap(race_nll(ev_df, blended), race_nll(ev_df, lp)),
        "scores": blended,
    }


def period_split(df: pd.DataFrame, a: np.ndarray, b: np.ndarray,
                 name_a: str, name_b: str, k: int = 3) -> str:
    """시기를 k 등분해 차이가 기간에 걸쳐 유지되는지 — 한 시기의 우연인지 본다."""
    starts, _ = race_index(df)
    dates = df["rcDate"].to_numpy()[starts]
    h_a, _ = hit_vectors(df, a)
    h_b, _ = hit_vectors(df, b)
    edges = np.quantile(dates, np.linspace(0, 1, k + 1))
    lines = [f"  {'기간':<22}{'경주':>7}{name_a:>9}{name_b:>9}{'차이':>9}{'95% CI':>18}"]
    for i in range(k):
        lo_d, hi_d = edges[i], edges[i + 1]
        sel = (dates >= lo_d) & (dates <= hi_d) if i == k - 1 else (dates >= lo_d) & (dates < hi_d)
        if sel.sum() < 60:
            continue
        d, lo, hi = paired_bootstrap(h_a[sel], h_b[sel], n=4000)
        lines.append(f"  {int(lo_d)}~{int(hi_d):<11}{sel.sum():>7,}"
                     f"{h_a[sel].mean() * 100:>8.1f}%{h_b[sel].mean() * 100:>8.1f}%"
                     f"{d:>+9.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>18}")
    return "\n".join(lines)


# ═══════════════════════════ 표 ═══════════════════════════
def compare_table(df: pd.DataFrame, models: dict[str, np.ndarray], ref: str) -> str:
    """기준(ref) 대비 모든 모델의 top-1 / top-3 차이와 검정 결과."""
    H = {k: hit_vectors(df, v) for k, v in models.items()}
    r1, r3 = H[ref]
    n = len(r1)
    lines = [
        f"  기준 = {ref}   경주 {n:,}개",
        f"  {'모델':<26}{'top-1':>8}{'차이':>8}{'95% CI':>18}{'단독승/패':>10}{'p':>10}   판정",
    ]
    for k in models:
        h1, _ = H[k]
        d, lo, hi = paired_bootstrap(h1, r1)
        mc = mcnemar(h1, r1)
        verdict = ("기준" if k == ref else
                   "우세" if lo > 0 else "열세" if hi < 0 else "차이 없음")
        disc = f"{mc['a_only']}/{mc['b_only']}"
        lines.append(f"  {k:<26}{h1.mean() * 100:>7.2f}%{d:>+8.2f}"
                     f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}{disc:>10}{mc['p']:>10.4f}   {verdict}")
    lines += ["", f"  {'모델':<26}{'top-3':>8}{'차이':>8}{'95% CI':>18}{'단독승/패':>10}{'p':>10}   판정"]
    for k in models:
        _, h3 = H[k]
        d, lo, hi = paired_bootstrap(h3, r3)
        mc = mcnemar(h3, r3)
        verdict = ("기준" if k == ref else
                   "우세" if lo > 0 else "열세" if hi < 0 else "차이 없음")
        disc = f"{mc['a_only']}/{mc['b_only']}"
        lines.append(f"  {k:<26}{h3.mean() * 100:>7.2f}%{d:>+8.2f}"
                     f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}{disc:>10}{mc['p']:>10.4f}   {verdict}")
    ref_key = next(k for k in models if k != ref)
    m1 = mde(H[ref_key][0], r1)
    m3 = mde(H[ref_key][1], r3)
    lines += ["", "  [검정력] 이 표본으로 무엇을 잴 수 있나",
              f"    최소 검출 가능 차이(80% 검정력, α=.05, 짝 공식): top-1 ±{m1:.2f}%p · top-3 ±{m3:.2f}%p",
              "    → 이보다 작은 차이는 '없다'가 아니라 '이 표본으로는 못 잰다'는 뜻이다."]
    d_ref = H[ref_key][0] - r1
    for target in (1.0, 2.0, 3.0):
        need = required_n(d_ref, target)
        lines.append(f"    {target:.0f}%p 차이를 잡으려면 경주 {need:,}개 필요 "
                     f"(지금 {n:,}개 · 주말 40경주 기준 {max(need - n, 0) / 40:.0f}주 누적)")
    return "\n".join(lines)


def probability_table(df: pd.DataFrame, models: dict[str, np.ndarray]) -> str:
    """확률의 품질 — 순위가 아니라 '41%라 한 것이 41% 맞았나'."""
    lines = [f"  {'모델':<26}{'logloss':>10}{'Brier':>10}{'ECE':>9}"]
    for k, s in models.items():
        lines.append(f"  {k:<26}{race_logloss(df, s):>10.4f}{brier(df, s):>10.5f}{ece(df, s):>9.4f}")
    return "\n".join(lines)


def reliability(df: pd.DataFrame, scores: np.ndarray, bins: int = 8) -> str:
    """신뢰도 곡선 — 예측 확률 구간별 실제 승률."""
    p = win_probs(df, scores)
    y = df["y_win"].to_numpy(float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    out = [f"  {'예측 구간':<16}{'두수':>8}{'예측':>8}{'실제':>8}{'차이':>8}"]
    for b in range(bins):
        m = idx == b
        if m.sum() < 20:
            continue
        out.append(f"  {p[m].min() * 100:5.1f}~{p[m].max() * 100:5.1f}%{m.sum():>8,}"
                   f"{p[m].mean() * 100:>7.1f}%{y[m].mean() * 100:>7.1f}%"
                   f"{(p[m].mean() - y[m].mean()) * 100:>+8.1f}")
    return "\n".join(out)


def segment_table(df: pd.DataFrame, a: np.ndarray, b: np.ndarray,
                  name_a: str, name_b: str) -> str:
    """어디서 이기고 어디서 지나 — 조건별 짝 비교."""
    h_a, _ = hit_vectors(df, a)
    h_b, _ = hit_vectors(df, b)
    starts, _ = race_index(df)
    r = df.iloc[starts]
    seg = {
        "출주두수": pd.cut(r["X_dusu"], [0, 9, 11, 99]),
        "거리(m)": pd.cut(r["X_rcDist"], [0, 1200, 1400, 9999]),
        "혼전도": pd.qcut(r["F6_field_entropy"], 3, duplicates="drop"),
        "1인기 확률": pd.cut(r.groupby("race_id", sort=False)["F6_mkt_prob"].transform("max")
                         if "F6_mkt_prob" in r else r["F6_mkt_prob"], [0, .25, .35, 1.01]),
    }
    lines = [f"  {'조건':<24}{'경주':>7}{name_a:>10}{name_b:>10}{'차이':>9}{'95% CI':>18}"]
    for sname, s in seg.items():
        lines.append(f"  ── {sname}")
        for k, m in pd.Series(s.to_numpy(), index=np.arange(len(s))).groupby(s.to_numpy(),
                                                                            observed=True):
            sel = m.index.to_numpy()
            if len(sel) < 60:
                continue
            d, lo, hi = paired_bootstrap(h_a[sel], h_b[sel], n=4000)
            lines.append(f"  {str(k):<24}{len(sel):>7,}{h_a[sel].mean() * 100:>9.1f}%"
                         f"{h_b[sel].mean() * 100:>9.1f}%{d:>+9.1f}"
                         f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}")
    return "\n".join(lines)


# ═══════════════════════════ 진입점 ═══════════════════════════
def load_scores(ckpt: Path, df: pd.DataFrame, split: str, meta: dict | None,
                device: str = "cpu") -> tuple[dict[str, np.ndarray], Scorer, np.ndarray]:
    """체크포인트 하나에서 (프리셋별 행 점수, Scorer, 축 점수)."""
    sc = Scorer(ckpt, device=device)
    ax = sc.axis_rows(df, split)
    eq = np.full(sc.n_axes, 1 / sc.n_axes)
    out = {"AxisRanker 균등가중": combine(ax, eq, 0.0, sc.n_axes)}
    if meta:
        for k, p in meta["presets"].items():
            w = np.array([p["groupWeights"][a] / 100 for a in sc.axes])
            out[f"AxisRanker {p['label']}"] = combine(ax, w, p["market"], sc.n_axes)
    return out, sc, ax


def lgb_scores(df: pd.DataFrame, split: str) -> dict[str, np.ndarray]:
    """팀 LightGBM 기준선 (docs/model/baseline_lgbm.py 와 같은 설정).

    **별도 프로세스에서 돌려 parquet 로 캐시한다.** torch 와 lightgbm 이 각자 OpenMP 런타임을
    싣고 있어 한 프로세스에 같이 올리면 libomp 충돌로 죽는다. KMP_DUPLICATE_LIB_OK 로
    덮는 방법은 lightgbm 이 "조용히 틀린 값을 낼 수 있다"고 경고하는 우회라 쓰지 않는다.
    """
    import subprocess
    import sys

    cache = ARTIFACTS / "pred" / f"lgb_{split}.parquet"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"  (LightGBM 기준선 계산 중 — 별도 프로세스, 1회만: {cache.name})")
        subprocess.run([sys.executable, "-m", "basemodel.lgb_baseline",
                        "--split", split, "--out", str(cache)], check=True)
    saved = pd.read_parquet(cache)
    key = pd.MultiIndex.from_arrays([df["race_id"], df["hrNo"]])
    out = {}
    for col in ("LightGBM 77", "LightGBM 73"):
        s = saved.set_index(["race_id", "hrNo"])[col].reindex(key)
        if s.isna().any():
            raise ValueError(f"{cache.name}: df 와 (race_id, hrNo) 가 안 맞는 행 {int(s.isna().sum())}개")
        out[col] = s.to_numpy()
    return out


def seed_scores(pattern: str, df: pd.DataFrame, split: str, meta: dict | None,
                device: str = "cpu") -> dict[str, np.ndarray]:
    """시드별 체크포인트를 모아 개별 점수와 시드 평균(앙상블)을 만든다."""
    runs = sorted((ARTIFACTS / "runs").glob(pattern))
    if not runs:
        return {}
    per, axes = {}, []
    for r in runs:
        sc = Scorer(r, device=device)
        ax = sc.axis_rows(df, split)
        axes.append(ax)
        eq = np.full(sc.n_axes, 1 / sc.n_axes)
        per[r.stem] = combine(ax, eq, 0.0, sc.n_axes)
    n_ax = Scorer(runs[0], device=device).n_axes
    mean_ax = np.mean(axes, axis=0)
    per["__ensemble__"] = combine(mean_ax, np.full(n_ax, 1 / n_ax), 0.0, n_ax)
    return per


def main() -> None:
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="주 체크포인트 (77피처)")
    ap.add_argument("--meta", default=str(ARTIFACTS / "export" / "meta.json"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed-glob", default="axis_77_*_s[1-9].pt",
                    help="시드 분산을 잴 체크포인트 패턴")
    ap.add_argument("--no-lgb", action="store_true")
    a = ap.parse_args()

    df = C.load(a.split)
    meta = json.loads(Path(a.meta).read_text("utf-8")) if Path(a.meta).exists() else None
    n_races = df["race_id"].nunique()

    print("=" * 96)
    print(f"시장 대비 유의성 검정   [{a.split}]  {n_races:,}경주 {len(df):,}두")
    print("=" * 96)

    # ★ 시장 점수는 **로그확률**이다. 확률을 그대로 넣으면 win_probs 의 softmax 가 한 번 더
    #   걸려 거의 균등분포가 된다 — 실측으로 시장 logloss 가 2.21(무작위 2.38 근처)로
    #   나왔다. log 를 취해야 softmax(log p) = p 로 원래 확률이 복원된다.
    #   순위(top-1/top-3)는 단조변환이라 영향이 없고, 확률 품질 표만 바로잡힌다.
    market = np.log(np.clip(df["F6_mkt_prob"].to_numpy(float), 1e-9, None))
    models = {"시장 (배당 내재확률)": market}
    if not a.no_lgb:
        models.update(lgb_scores(df, a.split))
    dl, sc, ax = load_scores(Path(a.ckpt), df, a.split, meta, a.device)
    models.update(dl)
    models["무작위"] = np.random.default_rng(0).random(len(df))

    print("\n[1] 적중률 — 시장 기준 짝 비교 (동률 1/k 기대값)")
    print(compare_table(df, models, ref="시장 (배당 내재확률)"))

    print("\n[2] 확률 품질")
    print(probability_table(df, models))

    print("\n[3] 정보 증분 — 시장에 없는 정보를 갖고 있나 (조건부 로짓 우도비, χ²₁)")
    print(f"  {'모델':<26}{'LR':>10}{'p':>12}{'모델 계수 b':>14}   판정")
    for k, s in models.items():
        if k.startswith("시장"):
            continue
        r = incremental_test(df, s)
        verdict = ("정보 있음" if r["p"] < 0.01 else
                   "약한 근거" if r["p"] < 0.05 else "근거 없음")
        print(f"  {k:<26}{r['LR']:>10.2f}{r['p']:>12.3e}{r['b_model']:>14.3f}   {verdict}")

    print("\n[4] 시장 + 모델 결합 — 그래서 더 맞히나 (valid 2-fold 교차적합)")
    cands = {k: models[k] for k in ("AxisRanker 균등가중", "AxisRanker 기본형", "LightGBM 73",
                                    "LightGBM 77") if k in models}
    blends = {k: benter_blend(df, s) for k, s in cands.items()}
    ref = next(iter(blends.values()))

    def _ci(t2):
        return f"[{t2[0]:+.2f}, {t2[1]:+.2f}]"

    print(f"  {'':<28}{'top-1':>8}{'차이':>8}{'95% CI':>18}{'McNemar p':>11}{'logloss':>10}")
    print(f"  {'시장 단독':<28}{ref['mkt_top1']:>7.2f}%{'—':>8}{'—':>18}{'—':>11}{ref['mkt_logloss']:>10.4f}")
    for k, r in blends.items():
        print(f"  {'+ ' + k:<28}{r['top1']:>7.2f}%{r['d1']:>+8.2f}{_ci(r['ci1']):>18}"
              f"{r['p1']:>11.4f}{r['logloss']:>10.4f}")

    print(f"\n  logloss 짝 검정 — 확률의 품질 (음수 = 결합이 나음, %p 아니라 nats)")
    print(f"  {'':<28}{'logloss':>10}{'차이':>9}{'95% CI':>20}")
    print(f"  {'시장 단독':<28}{ref['mkt_logloss']:>10.4f}{'—':>9}{'—':>20}")
    for k, r in blends.items():
        d, lo, hi = r["nll"]
        mark = "  ← 유의" if hi < 0 else ""
        print(f"  {'+ ' + k:<28}{r['logloss']:>10.4f}{d / 100:>+9.4f}"
              f"{f'[{lo / 100:+.4f}, {hi / 100:+.4f}]':>20}{mark}")

    print(f"\n  {'':<28}{'top-3':>8}{'차이':>8}{'95% CI':>18}{'McNemar p':>11}{'ECE':>9}")
    print(f"  {'시장 단독':<28}{ref['mkt_top3']:>7.2f}%{'—':>8}{'—':>18}{'—':>11}{ref['mkt_ece']:>9.4f}")
    for k, r in blends.items():
        print(f"  {'+ ' + k:<28}{r['top3']:>7.2f}%{r['d3']:>+8.2f}{_ci(r['ci3']):>18}"
              f"{r['p3']:>11.4f}{r['ece']:>9.4f}")

    print("\n  결합 계수 (2-fold 평균)")
    for k, r in blends.items():
        print(f"    {k:<26} 시장 {r['a']:>6.3f}   모델 {r['b']:>6.3f}")

    print("\n[5] 시기 안정성 — 결합 모델 vs 시장 (한 시기의 우연인지)")
    best = max(blends, key=lambda k: blends[k]["d1"])
    print(period_split(df, blends[best]["scores"], market, "결합", "시장"))

    print("\n[6] 신뢰도 — AxisRanker 균등가중")
    print(reliability(df, models["AxisRanker 균등가중"]))
    print("\n    시장")
    print(reliability(df, market))

    print("\n[7] 세그먼트 — 시장 대비 어디서 이기고 지나 (기본형)")
    key = "AxisRanker 기본형" if "AxisRanker 기본형" in models else "AxisRanker 균등가중"
    print(segment_table(df, models[key], market, "모델", "시장"))

    seeds = seed_scores(a.seed_glob, df, a.split, meta, a.device)
    if seeds:
        print(f"\n[8] 시드 분산 — 같은 설정 다른 난수 ({a.seed_glob})")
        vals = [hit_vectors(df, v)[0].mean() * 100 for k, v in seeds.items()
                if k != "__ensemble__"]
        ens = hit_vectors(df, seeds["__ensemble__"])[0].mean() * 100
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        print(f"  개별 시드 {len(vals)}개 top-1: {np.mean(vals):.2f}% ± {sd:.2f} "
              f"(최소 {min(vals):.2f} · 최대 {max(vals):.2f})")
        print(f"  시드 평균 앙상블 top-1: {ens:.2f}%")
        d, lo, hi = paired_bootstrap(hit_vectors(df, seeds["__ensemble__"])[0],
                                     hit_vectors(df, market)[0])
        print(f"  앙상블 − 시장: {d:+.2f}%p  95% CI [{lo:+.2f}, {hi:+.2f}]")
        r = incremental_test(df, seeds["__ensemble__"])
        print(f"  앙상블 정보 증분: LR {r['LR']:.2f}  p {r['p']:.3e}")


if __name__ == "__main__":
    main()
