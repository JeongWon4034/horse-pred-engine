"""데이터 — pipeline/ 의 채점기·인코더·경주 패딩을 그대로 쓴다. 고치지 않는다.

여기서 새로 하는 일은 둘뿐이다.
  1. 2004ext(train 만 2004~) 를 같은 clean/usable/정렬 규칙으로 읽는다.
  2. 인코딩된 열 벡터의 어느 구간이 원래 어느 피처였는지(슬롯) 를 돌려준다 — 마스크·손상은
     "원본 피처 단위" 로 해야 한다. 원핫 한 칸만 가리면 나머지 칸이 답을 흘린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

from model.team import C, S, DATA                      # noqa: E402
from model.data import (Encoder, fit_encoder, transform, pack_races,   # noqa: E402
                        flatten_scores, Races, HIGH_CARD, MAX_FIELD)
from model import evaluate as E                        # noqa: E402

EXT = DATA / "dataset-2004ext" / "model"
RUNS = ROOT / "selfsup" / "runs"
RUNS.mkdir(exist_ok=True)


def load(split: str, ext: bool = False) -> pd.DataFrame:
    """ext=True 면 2004ext 폴더. valid/test 는 두 폴더가 같은 race_id 집합이라 어느 쪽이든 같다.
    game 은 여기서도 거부한다 — 라벨 없이도 안 본다."""
    if not ext:
        return C.load(split)
    if split == "game":
        raise ValueError("game 은 사전학습 풀에도 넣지 않는다")
    if not EXT.exists():
        raise FileNotFoundError(f"{EXT} 없음 → bash selfsup/sync_ext.sh")
    df = S.usable(S.clean(pd.read_parquet(EXT / f"{split}.parquet")))
    C._assert_sorted(df)
    return df


def feature_cols() -> list[str]:
    return C.feature_cols(exclude_pop=True)          # 73. 주 비교축


def slots(enc: Encoder) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """인코딩 열 → 원본 피처 번호. (열별 슬롯 id [D], [(피처명, 열 인덱스들)]).
    na_flag 열은 그 수치 피처와 같은 슬롯에 묶는다 — 결측 플래그가 값을 흘리지 않게."""
    names = enc.names
    feat_of = []
    for n in names:
        if "#na" in n:
            feat_of.append(n.split("#")[0])
        elif "=" in n:
            feat_of.append(n.split("=")[0])
        else:
            feat_of.append(n)
    uniq = list(dict.fromkeys(feat_of))
    idx = {f: i for i, f in enumerate(uniq)}
    slot = np.array([idx[f] for f in feat_of])
    groups = [(f, np.flatnonzero(slot == idx[f])) for f in uniq]
    return slot, groups


def numeric_slots(enc: Encoder, groups) -> list[int]:
    numeric = set(enc.numeric)
    return [i for i, (f, _) in enumerate(groups) if f in numeric]


def cat_slots(enc: Encoder, groups) -> list[int]:
    cats = set(enc.onehot)
    return [i for i, (f, _) in enumerate(groups) if f in cats]
