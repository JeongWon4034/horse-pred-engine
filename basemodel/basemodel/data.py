# -*- coding: utf-8 -*-
"""전처리 + 경주 단위 패딩.

경마는 '경주 안에서 누가 앞서나'의 문제라 행 단위가 아니라 경주 단위(최대 16두)로
패딩해 다룬다. 전처리 상수(중앙값·평균·표준편차·원핫 어휘)는 **train 에서만** 잡는다.

원본: 이정원 horse-pred-engine `pipeline/model/data.py`.
바꾼 곳 — 피처를 스키마 그룹(F1~F6)이 아니라 **화면 슬라이더 6축**으로 자른다(config.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config as cfg
from .team import C

# 임베딩으로 다루므로 원핫에서 뺀다 (categorical.py 가 맡는다).
HIGH_CARD = ["F2_sire_id"]

CORE = "CORE"            # 모든 타워가 공유하는 칸 (경주 내 상수 포함)
TOPK = 3                 # order 텐서에 담는 착순 깊이


def load(split: str) -> pd.DataFrame:
    return C.load(split)


def feature_cols(exclude_pop: bool) -> list[str]:
    """73개(인기도 제외, 주말 실시간) 또는 77개(인기도 포함, 게임 리플레이)."""
    return C.feature_cols(exclude_pop=exclude_pop)


def axis_of(col: str) -> str:
    """피처 이름 → 슬라이더 축 이름. 어느 축에도 없으면 CORE."""
    for ax, names in cfg.AXIS_FEATURES.items():
        if col in names:
            return CORE if col in cfg.RACE_CONSTANT else ax
    return CORE


@dataclass
class Encoder:
    numeric: list[str]
    onehot: dict[str, list]
    median: pd.Series
    mean: pd.Series
    std: pd.Series
    na_flag: list[str]
    axis_of: list[str]          # 최종 피처별 소속 축 (타워 경계)
    names: list[str]

    @property
    def dim(self) -> int:
        return len(self.names)


def fit_encoder(tr: pd.DataFrame, cols: list[str], na_flag_thresh: float = 0.2) -> Encoder:
    cats = [c for c in cols if not is_numeric_dtype(tr[c]) and c not in HIGH_CARD]
    numeric = [c for c in cols if is_numeric_dtype(tr[c])]

    onehot = {c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in cats}
    median = tr[numeric].median()
    na_rate = tr[numeric].isna().mean()
    na_flag = na_rate[na_rate > na_flag_thresh].index.tolist()

    filled = tr[numeric].fillna(median)
    mean, std = filled.mean(), filled.std().replace(0, 1.0)

    names, axes = [], []
    for c in numeric:
        names.append(c); axes.append(axis_of(c))
    for c in cats:
        for v in onehot[c]:
            names.append(f"{c}={v}"); axes.append(axis_of(c))
    for c in na_flag:
        # 결측 플래그는 "이 말은 이 정보가 없다" 라는 그 축의 정보다 — 같은 축에 둔다.
        names.append(f"{c}#na"); axes.append(axis_of(c))

    return Encoder(numeric, onehot, median, mean, std, na_flag, axes, names)


def transform(df: pd.DataFrame, enc: Encoder) -> np.ndarray:
    num = df[enc.numeric]
    isna = num.isna()
    z = ((num.fillna(enc.median) - enc.mean) / enc.std).to_numpy(np.float32)
    z = np.clip(z, -8, 8)                       # 극단값이 학습을 흔드는 것 방지

    parts = [z]
    for c, vals in enc.onehot.items():
        col = df[c].astype(str).to_numpy()      # 결측은 어느 칸에도 안 걸린다 (0 벡터)
        parts.append(np.stack([(col == v) for v in vals], 1).astype(np.float32))
    if enc.na_flag:
        parts.append(isna[enc.na_flag].to_numpy(np.float32))
    return np.concatenate(parts, 1)


def axis_slices(enc: Encoder) -> dict[str, np.ndarray]:
    """축별 피처 인덱스. 타워가 이걸로 입력을 자른다. 빈 축은 넣지 않는다."""
    a = np.array(enc.axis_of)
    out = {CORE: np.flatnonzero(a == CORE)}
    for ax in cfg.AXES + [cfg.MARKET]:
        idx = np.flatnonzero(a == ax)
        if len(idx):
            out[ax] = idx
    return out


@dataclass
class Races:
    """경주 단위로 패딩된 텐서 묶음."""
    x: np.ndarray          # [R, MAX_FIELD, D]
    mask: np.ndarray       # [R, MAX_FIELD]   1 = 실제 출주마
    order: np.ndarray      # [R, TOPK]        1~3착의 슬롯 인덱스, 없으면 -1
    race_id: np.ndarray    # [R]
    n: np.ndarray          # [R] 출주 두수
    row_race: np.ndarray   # [len(df)] 각 행이 속한 경주 인덱스
    row_slot: np.ndarray   # [len(df)] 각 행의 경주 내 슬롯
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.race_id)


def race_blocks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rid = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    return starts, np.r_[starts[1:], len(rid)]


def pack_races(df: pd.DataFrame, x_flat: np.ndarray,
               extra: dict[str, np.ndarray] | None = None) -> Races:
    order_col = df["y_ord"].to_numpy()
    starts, ends = race_blocks(df)
    sizes = ends - starts
    if sizes.max() > cfg.MAX_FIELD:
        raise ValueError(f"출주 두수 {sizes.max()} > MAX_FIELD {cfg.MAX_FIELD} — 잘리는 경주가 생긴다")

    R, D = len(starts), x_flat.shape[1]
    x = np.zeros((R, cfg.MAX_FIELD, D), np.float32)
    mask = np.zeros((R, cfg.MAX_FIELD), np.float32)
    order = np.full((R, TOPK), -1, np.int64)
    row_race = np.repeat(np.arange(R), sizes)
    row_slot = np.arange(len(df)) - np.repeat(starts, sizes)

    x[row_race, row_slot] = x_flat
    mask[row_race, row_slot] = 1.0
    for i, (s, e) in enumerate(zip(starts, ends)):
        ranks = order_col[s:e]
        for k in range(TOPK):
            hit = np.flatnonzero(ranks == k + 1)
            if len(hit):
                order[i, k] = hit[0]

    packed = {}
    for name, arr in (extra or {}).items():
        if len(arr) != len(df):
            raise ValueError(f"extra[{name}] 길이 {len(arr)} != df {len(df)}")
        full = np.zeros((R, cfg.MAX_FIELD) + arr.shape[1:], arr.dtype)
        full[row_race, row_slot] = arr
        packed[name] = full

    return Races(x, mask, order, df["race_id"].to_numpy()[starts], sizes, row_race, row_slot, packed)


def to_races(df: pd.DataFrame, enc: Encoder, extra: dict[str, np.ndarray] | None = None) -> Races:
    return pack_races(df, transform(df, enc), extra)


def flatten_scores(scores: np.ndarray, races: Races) -> np.ndarray:
    """모델 출력 [R, MAX_FIELD] → df 행 순서 [len(df)].

    딥러닝과 LightGBM 이 같은 채점 함수에 들어가는 유일한 다리.
    """
    return np.asarray(scores)[races.row_race, races.row_slot]
