"""고카디널리티 범주형 → 임베딩 인덱스 (S2).

대상은 기수 jkNo · 조교사 trNo · 부마 F2_sire_id 셋이다.
말 ID(hrNo) 는 넣지 않는다 — 말 이름을 외우는 모델이 되어 새 말이 나오면 무너진다.

어휘는 train 에서만 만든다.
    0 = <pad>   패딩 슬롯
    1 = <unk>   train 에 없던 값, 결측, '-' (데이터의 결측 토큰)
    2 = <rare>  train 등장 횟수 < min_count
    3~          실제 값
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EMBED_COLS = ["jkNo", "trNo", "F2_sire_id"]
PAD, UNK, RARE = 0, 1, 2
MISSING = {"", "-", "nan", "None", "<NA>"}


def _as_str(s: pd.Series) -> pd.Series:
    """결측·결측 토큰을 전부 None 으로 통일한 문자열 시리즈."""
    out = s.astype("object").where(s.notna(), None)
    return out.map(lambda v: None if v is None or str(v).strip() in MISSING else str(v))


@dataclass
class Vocab:
    col: str
    index: dict[str, int]          # 값 → 3 이상 인덱스
    n_rare: int                    # <rare> 로 접힌 고유값 수

    @property
    def size(self) -> int:
        return len(self.index) + 3

    def encode(self, s: pd.Series) -> np.ndarray:
        vals = _as_str(s)
        out = np.full(len(vals), UNK, np.int64)
        for i, v in enumerate(vals.to_numpy()):
            if v is None:
                continue
            out[i] = self.index.get(v, UNK)
        return out


def fit_vocab(tr: pd.DataFrame, col: str, min_count: int = 5) -> Vocab:
    """train 행 빈도 기준. min_count 미만은 <rare> 하나로 접는다(인덱스에는 안 넣고 encode 때 RARE 로)."""
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
    """[len(df), len(vocabs)] int64 — 열 순서는 vocabs 의 키 순서."""
    return np.stack([v.encode(df[c]) for c, v in vocabs.items()], 1)


def describe(vocabs: dict[str, Vocab], va: pd.DataFrame | None = None) -> str:
    lines = []
    for c, v in vocabs.items():
        line = f"  {c:<12} 어휘 {v.size:>6,} (<rare> 로 접힌 값 {v.n_rare:,})"
        if va is not None:
            enc = v.encode(va[c])
            line += f"  valid: <unk> {np.mean(enc == UNK)*100:4.1f}%  <rare> {np.mean(enc == RARE)*100:4.1f}%"
        lines.append(line)
    return "\n".join(lines)
