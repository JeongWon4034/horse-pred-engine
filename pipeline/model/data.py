"""데이터 로딩 + 경주 단위 패딩 배치.

경마는 '경주 안에서 누가 앞서나'의 문제라, 행 단위가 아니라
경주 단위(최대 16두)로 패딩해서 다룬다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "dataset" / "model"

GROUPS = ["X", "F1", "F2", "F3", "F4", "F5", "F6"]

# 카디널리티가 낮아 원핫으로 펼칠 범주형
ONEHOT = ["X_sex", "X_prd_cty", "X_grade", "F4_weather", "F5_style"]
# 카디널리티가 높아 베이스라인에서는 제외 (신경망에서 임베딩으로 사용)
HIGH_CARD = ["F2_sire_id"]

MAX_FIELD = 16          # 실측 최대 출전 두수
TOPK = 3                # Plackett-Luce 에 쓸 착순 깊이


def load(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{split}.parquet")


def feature_columns(df: pd.DataFrame, exclude_groups: tuple[str, ...] = ()) -> list[str]:
    keep = [g for g in GROUPS if g not in exclude_groups]
    return [c for c in df.columns if c.split("_")[0] in keep]


@dataclass
class Encoder:
    """train 에서만 학습되는 전처리. valid/test 는 transform 만 한다."""
    numeric: list[str]
    onehot: dict[str, list]
    median: pd.Series
    mean: pd.Series
    std: pd.Series
    na_flag: list[str]          # 결측률이 높아 플래그를 따로 만드는 컬럼
    group_of: list[str]         # 최종 피처별 소속 그룹 (타워 경계용)
    names: list[str]

    @property
    def dim(self) -> int:
        return len(self.names)


def fit_encoder(tr: pd.DataFrame, cols: list[str], na_flag_thresh: float = 0.2) -> Encoder:
    cats = [c for c in cols if c in ONEHOT]
    numeric = [c for c in cols if c not in ONEHOT and c not in HIGH_CARD]

    onehot = {c: sorted(tr[c].dropna().unique().tolist()) for c in cats}
    median = tr[numeric].median()
    na_rate = tr[numeric].isna().mean()
    na_flag = na_rate[na_rate > na_flag_thresh].index.tolist()

    filled = tr[numeric].fillna(median)
    mean, std = filled.mean(), filled.std().replace(0, 1.0)

    names, group_of = [], []
    for c in numeric:
        names.append(c); group_of.append(c.split("_")[0])
    for c in cats:
        for v in onehot[c]:
            names.append(f"{c}={v}"); group_of.append(c.split("_")[0])
    for c in na_flag:
        names.append(f"{c}#na"); group_of.append(c.split("_")[0])

    return Encoder(numeric, onehot, median, mean, std, na_flag, group_of, names)


def transform(df: pd.DataFrame, enc: Encoder) -> np.ndarray:
    num = df[enc.numeric]
    isna = num.isna()
    z = ((num.fillna(enc.median) - enc.mean) / enc.std).to_numpy(np.float32)
    z = np.clip(z, -8, 8)                       # 극단값이 학습을 흔드는 것 방지

    parts = [z]
    for c, vals in enc.onehot.items():
        col = df[c].to_numpy()
        parts.append(np.stack([(col == v) for v in vals], 1).astype(np.float32))
    if enc.na_flag:
        parts.append(isna[enc.na_flag].to_numpy(np.float32))

    return np.concatenate(parts, 1)


@dataclass
class Races:
    """경주 단위로 패딩된 텐서 묶음."""
    x: np.ndarray          # [R, MAX_FIELD, D]
    mask: np.ndarray       # [R, MAX_FIELD]  1 = 실제 출전마
    order: np.ndarray      # [R, TOPK]       1~3착의 슬롯 인덱스, 없으면 -1
    race_id: np.ndarray    # [R]
    n: np.ndarray          # [R] 출전 두수

    def __len__(self) -> int:
        return len(self.race_id)


def to_races(df: pd.DataFrame, enc: Encoder) -> Races:
    x_flat = transform(df, enc)
    order_col = df["y_ord"].to_numpy()

    # race_id 는 연속 블록이라고 데이터셋 규약이 보장한다 (검증 완료)
    rid = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(rid)]

    R, D = len(starts), x_flat.shape[1]
    x = np.zeros((R, MAX_FIELD, D), np.float32)
    mask = np.zeros((R, MAX_FIELD), np.float32)
    order = np.full((R, TOPK), -1, np.int64)

    for i, (s, e) in enumerate(zip(starts, ends)):
        n = min(e - s, MAX_FIELD)
        x[i, :n] = x_flat[s : s + n]
        mask[i, :n] = 1.0
        ranks = order_col[s : s + n]
        for k in range(TOPK):
            hit = np.flatnonzero(ranks == k + 1)
            if len(hit):
                order[i, k] = hit[0]

    return Races(x, mask, order, rid[starts], (ends - starts).clip(max=MAX_FIELD))


def tower_slices(enc: Encoder) -> dict[str, np.ndarray]:
    """그룹별 피처 인덱스. 6타워 모델이 이걸로 입력을 자른다."""
    g = np.array(enc.group_of)
    return {name: np.flatnonzero(g == name) for name in GROUPS}
