"""데이터 로딩 + 경주 단위 패딩 배치.

데이터를 직접 읽지 않는다. 읽기·피처 선택·정렬 검사는 팀 공용 채점기(model.team.C)가
하고, 여기서는 그 결과를 경주 단위 텐서로 바꾸는 일만 한다. 그래야 LightGBM 과
같은 행, 같은 피처, 같은 자로 비교가 성립한다.

경마는 '경주 안에서 누가 앞서나'의 문제라, 행 단위가 아니라
경주 단위(최대 16두)로 패딩해서 다룬다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .team import C

GROUPS = ["X", "F1", "F2", "F3", "F4", "F5", "F6"]

# 카디널리티가 높아 원핫 대신 임베딩(S2)으로 다루는 범주형. S1 선형 모델에서는 뺀다.
HIGH_CARD = ["F2_sire_id"]

MAX_FIELD = 16          # 실측 최대 출전 두수
TOPK = 3                # order 텐서에 담는 착순 깊이 (손실은 topk 인자로 따로 자른다)


def load(split: str) -> pd.DataFrame:
    """clean + usable + 정렬 검사가 끝난 분할. game 은 채점기가 거부한다."""
    return C.load(split)


def feature_cols(exclude_pop: bool) -> list[str]:
    """73개(인기도 제외, 실시간) 또는 77개(인기도 포함, 게임)."""
    return C.feature_cols(exclude_pop=exclude_pop)


def categorical_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """수치가 아닌 피처. 스키마가 바뀌어도 하드코딩을 고칠 필요가 없다."""
    return [c for c in cols if not is_numeric_dtype(df[c])]


@dataclass
class Encoder:
    """train 에서만 학습되는 전처리. valid 는 transform 만 한다."""
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
    cats = [c for c in categorical_cols(tr, cols) if c not in HIGH_CARD]
    numeric = [c for c in cols if is_numeric_dtype(tr[c])]

    onehot = {c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in cats}
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
        col = df[c].astype(str).to_numpy()      # 결측은 NaN 으로 남아 어느 원핫에도 안 걸린다 (0 벡터)
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
    row_race: np.ndarray   # [len(df)] 각 행이 속한 경주 인덱스
    row_slot: np.ndarray   # [len(df)] 각 행의 경주 내 슬롯
    extra: dict[str, np.ndarray] = field(default_factory=dict)   # 단계별 추가 입력 [R, MAX_FIELD, ...]

    def __len__(self) -> int:
        return len(self.race_id)


def race_blocks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """race_id 연속 블록의 (시작, 끝) 인덱스. C.load 가 정렬을 보장한다."""
    rid = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(rid)]
    return starts, ends


def to_races(df: pd.DataFrame, enc: Encoder, extra: dict[str, np.ndarray] | None = None) -> Races:
    """extra 는 행 단위 배열 {이름: [len(df), ...]} — S2 범주형 인덱스, S3 이력 텐서 등."""
    x_flat = transform(df, enc)
    return pack_races(df, x_flat, extra)


def pack_races(df: pd.DataFrame, x_flat: np.ndarray,
               extra: dict[str, np.ndarray] | None = None) -> Races:
    """행 단위 피처 행렬 [len(df), D] 을 경주 단위 [R, MAX_FIELD, D] 로 접는다. extra 도 같은 규칙."""
    order_col = df["y_ord"].to_numpy()
    starts, ends = race_blocks(df)
    sizes = ends - starts
    if sizes.max() > MAX_FIELD:
        raise ValueError(f"출전 두수 {sizes.max()} > MAX_FIELD {MAX_FIELD} — 잘리는 경주가 생긴다")

    R, D = len(starts), x_flat.shape[1]
    x = np.zeros((R, MAX_FIELD, D), np.float32)
    mask = np.zeros((R, MAX_FIELD), np.float32)
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
            raise ValueError(f"extra[{name}] 길이 {len(arr)} ≠ df {len(df)}")
        full = np.zeros((R, MAX_FIELD) + arr.shape[1:], arr.dtype)
        full[row_race, row_slot] = arr
        packed[name] = full

    return Races(x, mask, order, df["race_id"].to_numpy()[starts], sizes, row_race, row_slot, packed)


def flatten_scores(scores: np.ndarray, races: Races) -> np.ndarray:
    """모델 출력 [R, MAX_FIELD] → df 행 순서 점수 [len(df)].

    채점기(C.report / C.evaluate)는 행 단위 점수 배열을 받는다. 딥러닝과 LightGBM 이
    같은 함수에 들어가는 유일한 다리가 이 변환이다.
    """
    return np.asarray(scores)[races.row_race, races.row_slot]


def tower_slices(enc: Encoder) -> dict[str, np.ndarray]:
    """그룹별 피처 인덱스. 6타워 모델이 이걸로 입력을 자른다."""
    g = np.array(enc.group_of)
    return {name: np.flatnonzero(g == name) for name in GROUPS}
