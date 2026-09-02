"""말별 과거 전적 시퀀스 (S3).

LightGBM 이 보는 F1_ord_avg3 같은 피처는 과거를 평균으로 뭉갠 값이라 순서와 추세가 사라진다.
여기서는 말마다 직전 L 출전을 시간순 그대로 텐서로 만들어 GRU 에 넣는다.

시점 규칙 — 새지 않게
  · 이력에는 그 경주의 rcDate 보다 **엄격히 이전**(rcDate < 현재) 출전만 들어간다.
  · 이력 풀은 train ∪ valid 의 원본 행(usable 필터 전 — 배당 없는 경주도 착순은 정상).
    game·test 행은 풀에 넣지 않는다. 둘 다 assert 로 못 박는다.
  · 이력 항목 값은 전부 그 과거 경주 자체의 결과(착순·스피드지수·거리·부담중량·등급·기수)다.
  · hrNo 는 과거 경주를 찍어 오는 조인 키로만 쓴다. 텐서에는 들어가지 않는다.

알려진 구멍 — game 경주(2015~24 개최일의 15%)가 풀에 없어 그 날 출전은 이력에서 빠진다.
F1_* 집계 피처는 빌더가 game 을 포함해 계산했으므로 둘은 비대칭이다. 장부에 명기한다.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .team import S, TEAM_DATASET

L = 10
POOL_SPLITS = ("train", "valid")
POOL_COLS = ["hrNo", "rcDate", "race_id", "X_rcDist", "X_dusu", "X_wgBudam", "X_grade",
             "jkNo", "y_ord", "y_speed_fig"]
FEATS = ["ordpct", "win", "speed", "dist", "dist_gap", "days", "wg", "grade", "same_jk", "na"]
K = len(FEATS)
CACHE = Path(__file__).resolve().parents[2] / "experiments" / "runs"


def grade_level(g) -> float:
    """등급 문자열 → 숫자. 국1/혼1 = 1(최상) … 국6 = 6, OPEN = 0(그 위). 없으면 NaN.

    이력 항목 안에서만 쓰는 정규화라 LGB 비교 조건(X_grade 원본 유지)과 무관하다.
    """
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return np.nan
    t = str(g)
    if "OPEN" in t.upper() or "오픈" in t:
        return 0.0
    m = re.search(r"\d", t)
    return float(m.group()) if m else np.nan


def _ymd_to_days(ymd: np.ndarray) -> np.ndarray:
    """YYYYMMDD 정수 → 1970-01-01 기준 일수."""
    return (pd.to_datetime(pd.Series(ymd).astype(str), format="%Y%m%d").to_numpy()
            .astype("datetime64[D]").astype(np.int64))


def load_pool() -> pd.DataFrame:
    """train ∪ valid 원본(clean 만, usable 전). hrNo·rcDate 순 정렬."""
    frames = []
    for sp in POOL_SPLITS:
        df = pd.read_parquet(TEAM_DATASET / "model" / f"{sp}.parquet", columns=POOL_COLS)
        frames.append(S.clean(df))
    pool = pd.concat(frames, ignore_index=True)
    pool["hrNo"] = pool["hrNo"].astype(str)
    pool["jkNo"] = pool["jkNo"].astype(str)
    pool = pool.sort_values(["hrNo", "rcDate", "race_id"], kind="stable").reset_index(drop=True)

    game = pd.read_parquet(TEAM_DATASET / "model" / "game.parquet", columns=["race_id"])["race_id"]
    test = pd.read_parquet(TEAM_DATASET / "model" / "test.parquet", columns=["race_id"])["race_id"]
    assert not pool["race_id"].isin(set(game)).any(), "이력 풀에 game 경주가 들어갔다"
    assert not pool["race_id"].isin(set(test)).any(), "이력 풀에 test 경주가 들어갔다"
    return pool


@dataclass
class HistStats:
    """train 이력 값으로만 잡은 표준화 상수."""
    speed_mu: float; speed_sd: float
    wg_mu: float; wg_sd: float


def fit_stats(pool: pd.DataFrame, train_max_date: int) -> HistStats:
    p = pool[pool["rcDate"] <= train_max_date]
    sp, wg = p["y_speed_fig"].dropna(), p["X_wgBudam"].dropna()
    return HistStats(float(sp.mean()), float(sp.std() or 1.0), float(wg.mean()), float(wg.std() or 1.0))


def build(target: pd.DataFrame, pool: pd.DataFrame, stats: HistStats, L: int = L
          ) -> tuple[np.ndarray, np.ndarray]:
    """target 의 각 행에 대해 (hist [N, L, K] float32 — 오른쪽 패딩, 오래된 것부터, hist_len [N])."""
    # ── 풀을 (말, 날짜) 합성 키로 정렬해 두고 이진 탐색 — 385k 행이라 groupby 는 쓰지 않는다
    hr_all = pd.concat([pool["hrNo"], target["hrNo"].astype(str)], ignore_index=True)
    codes, _ = pd.factorize(hr_all, sort=True)
    pool_code, tgt_code = codes[:len(pool)], codes[len(pool):]
    pool_key = pool_code.astype(np.int64) * 10**8 + pool["rcDate"].to_numpy(np.int64)
    assert (np.diff(pool_key) >= 0).all(), "풀 정렬이 깨졌다"
    tgt_date = target["rcDate"].to_numpy(np.int64)
    tgt_key = tgt_code.astype(np.int64) * 10**8 + tgt_date

    end = np.searchsorted(pool_key, tgt_key, side="left")           # 첫 인덱스 with key ≥ (말, 오늘)
    start = np.searchsorted(pool_key, tgt_code.astype(np.int64) * 10**8, side="left")
    n_hist = np.minimum(L, end - start)                              # 엄격히 이전 출전 수 (≤ L)
    N = len(target)

    # 각 행의 이력 인덱스: end-n_hist … end-1 (오래된 것 → 최근 것), 오른쪽 패딩
    pos = np.arange(L)[None, :]                                      # [1, L]
    valid = pos < n_hist[:, None]                                    # [N, L]
    idx = (end[:, None] - n_hist[:, None] + pos).clip(0, len(pool) - 1)

    P = {c: pool[c].to_numpy() for c in POOL_COLS}
    pool_days = _ymd_to_days(P["rcDate"])
    tgt_days = _ymd_to_days(tgt_date)

    # ── 시점 누수 검사: 유효 슬롯의 날짜는 전부 현재보다 이전, 말도 같아야 한다
    assert (P["rcDate"][idx][valid] < np.broadcast_to(tgt_date[:, None], idx.shape)[valid]).all(), \
        "이력에 현재 이후 경주가 섞였다"
    assert (pool_code[idx][valid] == np.broadcast_to(tgt_code[:, None], idx.shape)[valid]).all(), \
        "이력에 다른 말이 섞였다"

    def g(col):
        return P[col][idx]

    ord_ = g("y_ord").astype(float); dusu = g("X_dusu").astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ordpct = (ord_ - 1) / np.maximum(dusu - 1, 1)
    speed = g("y_speed_fig").astype(float)
    dist = g("X_rcDist").astype(float)
    wg = g("X_wgBudam").astype(float)
    grade = np.vectorize(grade_level, otypes=[float])(g("X_grade"))
    same_jk = (g("jkNo").astype(str) == np.broadcast_to(target["jkNo"].astype(str).to_numpy()[:, None], idx.shape))
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
    ], axis=-1).astype(np.float32)                                    # [N, L, K]
    feats = np.clip(np.nan_to_num(feats, nan=0.0), -8, 8)           # 남은 결측(거리 등)은 0 + na 플래그
    feats[~valid] = 0.0
    assert not np.isnan(feats).any()
    return feats, n_hist.astype(np.int64)


VERSION = 2   # 이력 항목 정의가 바뀌면 올린다 — 캐시 키에 들어간다


def _cache_key(target: pd.DataFrame, L: int) -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(target["race_id"], index=False).to_numpy().tobytes())
    h.update(str(len(target)).encode()); h.update(str(L).encode()); h.update(str(VERSION).encode())
    return h.hexdigest()[:10]


def history_for(target: pd.DataFrame, split: str, pool: pd.DataFrame | None = None,
                stats: HistStats | None = None, L: int = L, verbose: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
    """캐시를 거쳐 (hist, hist_len) 을 돌려준다. target 은 C.load(split) 결과."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"hist_{split}_L{L}_{_cache_key(target, L)}.npz"
    if f.exists():
        z = np.load(f)
        hist, n = z["hist"], z["hist_len"]
    else:
        if pool is None:
            pool = load_pool()
        if stats is None:
            stats = fit_stats(pool, int(target["rcDate"].max()) if split == "train"
                              else int(pool.loc[pool["rcDate"] < int(target["rcDate"].min()), "rcDate"].max()))
        hist, n = build(target, pool, stats, L)
        np.savez_compressed(f, hist=hist, hist_len=n)
    if verbose:
        print(f"[이력 {split}] 행 {len(n):,}  평균 이력 {n.mean():.2f}/{L}  이력 0개 {np.mean(n == 0)*100:.1f}%  "
              f"꽉 찬 행 {np.mean(n == L)*100:.1f}%  결측 플래그 {hist[..., FEATS.index('na')][n[:, None] > np.arange(L)].mean()*100:.1f}%  ({f.name})")
    return hist, n
