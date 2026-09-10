# -*- coding: utf-8 -*-
"""저장된 체크포인트로 새 데이터에 축 점수를 매긴다 — 재학습 없음.

학습(train.py)과 배포(export.py)가 같은 코드로 점수를 내게 하는 자리.
서비스에서 모델이 도는 곳은 여기 하나뿐이고, 유저가 슬라이더를 만질 때는
안 돈다(AI-06 — 화면이 곱셈·덧셈만 한다).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import categorical as cat_mod
from . import config as cfg
from . import data as D
from . import history as H
from .model import AxisRanker
from .team import S, TEAM_DATASET


class Scorer:
    """체크포인트 하나 = Scorer 하나. `axis_rows(df)` 가 행 단위 축 점수를 돌려준다."""

    def __init__(self, ckpt: Path, device: str = "cpu"):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        with open(Path(str(ckpt).replace(".pt", ".enc.pkl")), "rb") as f:
            side = pickle.load(f)
        self.enc = side["encoder"]
        self.vocabs = side["vocabs"]
        self.hist_stats = H.HistStats(**side["hist_stats"]) if side["hist_stats"] else None
        self.market = blob["market"]
        self.k_hist = blob["k_hist"]
        self.device = device
        self.model = AxisRanker(blob["slices"], blob["vocab_sizes"], k_hist=self.k_hist)
        self.model.load_state_dict(blob["state"])
        self.model.to(device).eval()
        self.axes = self.model.axes
        self.columns = self.model.score_columns
        self._pool = None

    @property
    def n_axes(self) -> int:
        return len(self.axes)

    def _hist(self, df: pd.DataFrame, split: str):
        if not self.k_hist:
            return None, None
        if self._pool is None:
            self._pool = H.load_pool()
        return H.history_for(df, split, pool=self._pool, stats=self.hist_stats, verbose=False)

    @torch.no_grad()
    def axis_rows(self, df: pd.DataFrame, split: str, batch: int = 512) -> np.ndarray:
        """[len(df), A(+1)] — 행 단위 축 점수 (경주 내 z-score)."""
        extra = {"cat": cat_mod.encode_cats(df, self.vocabs)}
        if self.k_hist:
            hist, hlen = self._hist(df, split)
            extra["hist"], extra["hist_len"] = hist, hlen
        races = D.to_races(df, self.enc, extra)

        chunks = []
        for i in range(0, len(races), batch):
            sl = slice(i, i + batch)
            t = lambda a: torch.as_tensor(a[sl], device=self.device)      # noqa: E731
            kw = {}
            if "cat" in races.extra:
                kw["cat"] = t(races.extra["cat"])
            if "hist" in races.extra:
                kw["hist"] = t(races.extra["hist"])
                kw["hist_len"] = t(races.extra["hist_len"])
            chunks.append(self.model.axis_scores(t(races.x), t(races.mask), **kw).cpu().numpy())
        s = np.concatenate(chunks)
        return s[races.row_race, races.row_slot]


def load_game(drop_flat_odds: bool = True) -> pd.DataFrame:
    """게임풀(3,700경주) 원본. 팀 채점기는 game 을 막아 두었으므로 여기서 직접 읽는다.

    막아 둔 이유는 **학습·튜닝 금지**이지 서비스 금지가 아니다(schema_v2 §분할).
    유저가 플레이할 경주라 점수를 미리 매겨 DB 에 넣어야 한다(기능명세서 AI-05).
    정제는 학습과 똑같이 clean → usable 을 거친다.
    """
    df = pd.read_parquet(TEAM_DATASET / "model" / "game.parquet")
    df = S.usable(S.clean(df), drop_flat_odds=drop_flat_odds)
    codes = pd.factorize(df["race_id"], sort=False)[0]
    if (np.diff(codes) < 0).any():
        raise RuntimeError("game race_id 정렬이 깨졌다")
    return df.reset_index(drop=True)


def combine(axis_rows: np.ndarray, w: np.ndarray, m: float, n_axes: int) -> np.ndarray:
    """화면(AI-06)이 하는 계산과 같은 식. 여기가 기준 구현이다."""
    out = axis_rows[:, :n_axes] @ np.asarray(w, float)
    if axis_rows.shape[1] > n_axes:
        out = out + axis_rows[:, -1] * float(m)
    return out
