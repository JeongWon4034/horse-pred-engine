# -*- coding: utf-8 -*-
"""말별 과거 전적 시퀀스.

LightGBM 이 보는 `F1_ord_avg3` 같은 피처는 과거를 평균으로 뭉갠 값이라 순서가 사라진다.
"최근 3경주 평균 2등"은 1·2·3착이든 3·2·1착이든 같은 값이다. 여기서는 말마다 직전 L
출전을 시간순 그대로 텐서로 만들어 GRU 에 넣는다.

원본: 이정원 `pipeline/model/history.py` (S3). 그대로 계승한다.

⚠ **기여 범위는 좁다.** 정원님이 뒤에 돌린 이력 길이 곡선(`exp/hist-len-curve`, 2026-09-09)이
L=0 통제 실험으로 잰 결과:
  · **top-1 에는 기여하지 않는다** — L=0 대비 짝 비교가 L=5·10·20·30 전부 0 을 품고
    L=5·10·30 은 부호가 음수다. 이력을 비워도 적중률이 안 떨어진다.
  · **logloss 에는 기여한다** — LGB 대비 우위 0.0410 중 0.0246(60%)이 시퀀스 몫.
  · **이득의 대부분이 최근 5출전에서 나온다** — L=5 로 이미 1.8525, L=30 이 1.8493.
따라서 "과거를 순서대로, 길게 읽어 적중률을 올린다"는 서술은 쓰면 안 된다.
여기서 L=20 을 쓰는 건 정원님 채택값을 맞춰 비교를 성립시키려는 것이고,
**L=5 로 줄여도 거의 같을 가능성이 높다**(미측정 — 백로그).

시점 규칙 — 새지 않게
  · 이력에는 그 경주의 rcDate 보다 **엄격히 이전** 출전만 들어간다 (assert 로 강제).
  · 이력 풀은 train ∪ valid 원본. game·test 는 넣지 않는다 (assert 로 강제).
  · 이력 항목 값은 전부 그 과거 경주 자체의 결과다.
  · hrNo 는 과거 경주를 찍어오는 조인 키로만 쓴다. 텐서에는 안 들어간다.

알려진 구멍 — game 경주(2015~24 개최일의 15%)가 풀에 없어 그 날 출전이 이력에서 빠진다.
학습에 game 을 못 쓰게 한 팀 규칙을 지키느라 감수하는 비대칭이고, `describe_gap()` 이
크기를 잰다. 실제 운영(주말 실시간)에서는 게임풀 자체가 없어 이 구멍이 사라진다.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as cfg
from .team import S, TEAM_DATASET, ARTIFACTS

POOL_SPLITS = ("train", "valid")
POOL_COLS = ["hrNo", "rcDate", "race_id", "X_rcDist", "X_dusu", "X_wgBudam", "X_grade",
             "jkNo", "y_ord", "y_speed_fig"]
FEATS = ["ordpct", "win", "speed", "dist", "dist_gap", "days", "wg", "grade", "same_jk", "na"]
K = len(FEATS)
VERSION = 1              # 이력 항목 정의가 바뀌면 올린다 — 캐시 키에 들어간다
CACHE = ARTIFACTS / "hist"


def grade_level(g) -> float:
    """등급 문자열 → 숫자. 국1/혼1 = 1(최상) … 국6 = 6, OPEN = 0(그 위). 없으면 NaN."""
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return np.nan
    t = str(g)
    if "OPEN" in t.upper() or "오픈" in t:
        return 0.0
    m = re.search(r"\d", t)
    return float(m.group()) if m else np.nan


def _ymd_to_days(ymd: np.ndarray) -> np.ndarray:
    return (pd.to_datetime(pd.Series(ymd).astype(str), format="%Y%m%d").to_numpy()
            .astype("datetime64[D]").astype(np.int64))


def load_pool() -> pd.DataFrame:
    """train ∪ valid 원본(clean 만, usable 전 — 배당 없는 경주도 착순은 정상)."""
    frames = []
    for sp in POOL_SPLITS:
        df = pd.read_parquet(TEAM_DATASET / "model" / f"{sp}.parquet", columns=POOL_COLS)
        frames.append(S.clean(df))
    pool = pd.concat(frames, ignore_index=True)
    pool["hrNo"] = pool["hrNo"].astype(str)
    pool["jkNo"] = pool["jkNo"].astype(str)
    pool = pool.sort_values(["hrNo", "rcDate", "race_id"], kind="stable").reset_index(drop=True)

    for sp in ("game", "test"):
        ids = pd.read_parquet(TEAM_DATASET / "model" / f"{sp}.parquet", columns=["race_id"])["race_id"]
        assert not pool["race_id"].isin(set(ids)).any(), f"이력 풀에 {sp} 경주가 들어갔다"
    return pool


@dataclass
class HistStats:
    """train 이력 값으로만 잡은 표준화 상수."""
    speed_mu: float
    speed_sd: float
    wg_mu: float
    wg_sd: float

    def to_dict(self) -> dict:
        return asdict(self)


def fit_stats(pool: pd.DataFrame, train_max_date: int) -> HistStats:
    p = pool[pool["rcDate"] <= train_max_date]
    sp, wg = p["y_speed_fig"].dropna(), p["X_wgBudam"].dropna()
    return HistStats(float(sp.mean()), float(sp.std() or 1.0),
                     float(wg.mean()), float(wg.std() or 1.0))


def build(target: pd.DataFrame, pool: pd.DataFrame, stats: HistStats,
          L: int = cfg.HIST_LEN) -> tuple[np.ndarray, np.ndarray]:
    """target 각 행에 대해 (hist [N,L,K] 오른쪽 패딩·오래된 것부터, hist_len [N])."""
    # (말, 날짜) 합성 키로 정렬해 두고 이진 탐색 — 40만 행이라 groupby 는 안 쓴다
    hr_all = pd.concat([pool["hrNo"], target["hrNo"].astype(str)], ignore_index=True)
    codes, _ = pd.factorize(hr_all, sort=True)
    pool_code, tgt_code = codes[:len(pool)], codes[len(pool):]
    pool_key = pool_code.astype(np.int64) * 10**8 + pool["rcDate"].to_numpy(np.int64)
    assert (np.diff(pool_key) >= 0).all(), "풀 정렬이 깨졌다"
    tgt_date = target["rcDate"].to_numpy(np.int64)
    tgt_key = tgt_code.astype(np.int64) * 10**8 + tgt_date

    end = np.searchsorted(pool_key, tgt_key, side="left")        # 첫 인덱스 with key >= (말, 오늘)
    start = np.searchsorted(pool_key, tgt_code.astype(np.int64) * 10**8, side="left")
    n_hist = np.minimum(L, end - start)

    pos = np.arange(L)[None, :]
    valid = pos < n_hist[:, None]
    idx = (end[:, None] - n_hist[:, None] + pos).clip(0, len(pool) - 1)

    P = {c: pool[c].to_numpy() for c in POOL_COLS}
    pool_days = _ymd_to_days(P["rcDate"])
    tgt_days = _ymd_to_days(tgt_date)

    assert (P["rcDate"][idx][valid] < np.broadcast_to(tgt_date[:, None], idx.shape)[valid]).all(), \
        "이력에 현재 이후 경주가 섞였다"
    assert (pool_code[idx][valid] == np.broadcast_to(tgt_code[:, None], idx.shape)[valid]).all(), \
        "이력에 다른 말이 섞였다"

    def g(col):
        return P[col][idx]

    ord_ = g("y_ord").astype(float)
    dusu = g("X_dusu").astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ordpct = (ord_ - 1) / np.maximum(dusu - 1, 1)
    speed = g("y_speed_fig").astype(float)
    dist = g("X_rcDist").astype(float)
    wg = g("X_wgBudam").astype(float)
    grade = np.vectorize(grade_level, otypes=[float])(g("X_grade"))
    same_jk = (g("jkNo").astype(str)
               == np.broadcast_to(target["jkNo"].astype(str).to_numpy()[:, None], idx.shape))
    cur_dist = target["X_rcDist"].to_numpy(float)[:, None]
    days = (tgt_days[:, None] - pool_days[idx]).astype(float)

    na = np.isnan(ordpct) | np.isnan(speed) | np.isnan(grade) | np.isnan(dist) | np.isnan(cur_dist)
    feats = np.stack([
        np.nan_to_num(ordpct, nan=0.5),
        np.nan_to_num((ord_ == 1).astype(float)),
        np.nan_to_num((speed - stats.speed_mu) / stats.speed_sd, nan=0.0),
        dist / 1000.0,
        (cur_dist - dist) / 1000.0,
        np.log1p(np.maximum(days, 0)),
        np.nan_to_num((wg - stats.wg_mu) / stats.wg_sd, nan=0.0),
        np.nan_to_num(grade, nan=3.0) / 6.0,
        same_jk.astype(float),
        na.astype(float),
    ], axis=-1).astype(np.float32)
    feats = np.clip(np.nan_to_num(feats, nan=0.0), -8, 8)
    feats[~valid] = 0.0
    assert not np.isnan(feats).any()
    return feats, n_hist.astype(np.int64)


def _cache_key(target: pd.DataFrame, L: int) -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(target["race_id"], index=False).to_numpy().tobytes())
    for part in (len(target), L, VERSION):
        h.update(str(part).encode())
    return h.hexdigest()[:10]


def history_for(target: pd.DataFrame, split: str, pool: pd.DataFrame | None = None,
                stats: HistStats | None = None, L: int = cfg.HIST_LEN,
                verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """캐시를 거쳐 (hist, hist_len). target 은 C.load(split) 결과."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"hist_{split}_L{L}_{_cache_key(target, L)}.npz"
    if f.exists():
        z = np.load(f)
        hist, n = z["hist"], z["hist_len"]
    else:
        if pool is None:
            pool = load_pool()
        if stats is None:
            stats = fit_stats(pool, _stats_cutoff(target, pool, split))
        hist, n = build(target, pool, stats, L)
        np.savez_compressed(f, hist=hist, hist_len=n)
    if verbose:
        filled = hist[..., FEATS.index("na")][n[:, None] > np.arange(L)]
        print(f"[이력 {split}] 행 {len(n):,}  평균 {n.mean():.2f}/{L}  이력0 {np.mean(n == 0) * 100:.1f}%  "
              f"꽉참 {np.mean(n == L) * 100:.1f}%  결측플래그 {filled.mean() * 100:.1f}%")
    return hist, n


def _stats_cutoff(target: pd.DataFrame, pool: pd.DataFrame, split: str) -> int:
    """표준화 상수를 잡을 날짜 상한. train 이 아니면 target 시작 이전까지만 본다."""
    if split == "train":
        return int(target["rcDate"].max())
    before = pool.loc[pool["rcDate"] < int(target["rcDate"].min()), "rcDate"]
    return int(before.max()) if len(before) else int(pool["rcDate"].max())
