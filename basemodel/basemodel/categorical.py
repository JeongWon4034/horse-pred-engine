# -*- coding: utf-8 -*-
"""고카디널리티 범주형 → 임베딩 인덱스.

대상은 기수 jkNo · 조교사 trNo · 부마 F2_sire_id 셋.
말 ID(hrNo) 는 넣지 않는다 — 말 이름을 외우는 모델이 되어 새 말에서 무너진다.

어휘는 train 에서만 만든다.  0 = <pad> · 1 = <unk> · 2 = <rare> · 3~ 실제 값

원본: 이정원 horse-pred-engine `pipeline/model/categorical.py` (S2 단계). 그대로 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EMBED_COLS = ["jkNo", "trNo", "F2_sire_id"]
PAD, UNK, RARE = 0, 1, 2
MISSING = {"", "-", "nan", "None", "<NA>"}


def _as_str(s: pd.Series) -> pd.Series:
    out = s.astype("object").where(s.notna(), None)
    return out.map(lambda v: None if v is None or str(v).strip() in MISSING else str(v))


@dataclass
class Vocab:
    col: str
    index: dict[str, int]
    n_rare: int

    @property
    def size(self) -> int:
        return len(self.index) + 3

    def encode(self, s: pd.Series) -> np.ndarray:
        vals = _as_str(s).to_numpy()
        out = np.full(len(vals), UNK, np.int64)
        for i, v in enumerate(vals):
            if v is not None:
                out[i] = self.index.get(v, UNK)
        return out


def fit_vocab(tr: pd.DataFrame, col: str, min_count: int = 5) -> Vocab:
    """train 행 빈도 기준. min_count 미만은 <rare> 하나로 접는다."""
    counts = _as_str(tr[col]).dropna().value_counts()
    keep = counts[counts >= min_count].index.tolist()
    rare = counts[counts < min_count].index.tolist()
    index = {v: i + 3 for i, v in enumerate(sorted(keep))}
    for v in rare:
        index[v] = RARE
    return Vocab(col, index, len(rare))


def fit_vocabs(tr: pd.DataFrame, cols: list[str] = EMBED_COLS, min_count: int = 5) -> dict[str, Vocab]:
    return {c: fit_vocab(tr, c, min_count) for c in cols}


def encode_cats(df: pd.DataFrame, vocabs: dict[str, Vocab]) -> np.ndarray:
    """[len(df), len(vocabs)] int64 — 열 순서는 vocabs 키 순서."""
    return np.stack([v.encode(df[c]) for c, v in vocabs.items()], 1)


def describe(vocabs: dict[str, Vocab], va: pd.DataFrame | None = None) -> str:
    lines = []
    for c, v in vocabs.items():
        line = f"  {c:<12} 어휘 {v.size:>6,} (<rare> 로 접힘 {v.n_rare:,})"
        if va is not None:
            enc = v.encode(va[c])
            line += f"   valid <unk> {np.mean(enc == UNK) * 100:4.1f}%  <rare> {np.mean(enc == RARE) * 100:4.1f}%"
        lines.append(line)
    return "\n".join(lines)
