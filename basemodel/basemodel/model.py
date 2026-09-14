# -*- coding: utf-8 -*-
"""AxisRanker — 6축 타워 + 시장 축, Plackett-Luce 로 함께 학습.

## 왜 이 구조인가

유저가 슬라이더를 움직여도 **재학습 없이** 예측이 바뀌어야 한다(AI-06). 그러려면
축별 점수를 미리 확정해 두고 화면에서는 곱셈·덧셈만 해야 한다.

    최종 점수 = m · s_MARKET + Σ_k w_k · s_k        Σ w_k = 100 (유저), m = 프리셋 상수

축을 따로따로 학습하면 축마다 점수 단위가 달라져(어떤 건 확률, 어떤 건 로그오즈)
`w1·s1 + w2·s2` 가 무의미해진다. 그래서 **한 네트워크 안에서 6갈래로 함께** 학습하고,
각 축 점수를 **경주 내 z-score** 로 내보낸다(스코어 계약).

## 부품마다 소속을 정한 이유

이정원 S5 의 타워는 축마다 `[축 피처 + 공용 X]` 만 봤다. 여기서는 S3 의 **전적 GRU** 를
얹는데(적중률이 아니라 확률 품질에 기여한다 — `history.py` 주석 참조),
그걸 공유 표현에 통째로 넣으면 모든
타워가 같은 것을 보게 되어 축 간 상관이 올라가고 슬라이더가 죽는다. 그래서 소속을 나눈다.

| 부품 | 소속 | 이유 |
|---|---|---|
| 경주 내 상수 + 공통 X (43칸) | 공유 trunk | 어느 축이든 "이 경주가 어떤 경주인지" 는 필요 |
| 기수·조교사 임베딩 | JOCKEY 타워 | 기수 정체성은 기수 축의 정보다 |
| 부마 임베딩 | ABILITY 타워 | 혈통은 실력 축에 넣었다 (config.AXIS_FEATURES) |
| 전적 GRU | **축마다 다른 선형 읽기** | 같은 이력에서 축마다 다른 것을 읽는다 |

## 학습 중 가중치를 랜덤으로 뽑는 이유

그냥 만들면 학습 때 본 적 없는 가중치 조합에서 무너진다. 매 배치 `w` 를 디리클레에서,
`m` 을 균등분포에서 새로 뽑아 쓴다. 효과 셋:
  1. 가끔 한 축에 몰리니 **각 타워가 혼자서도 쓸모 있어야** 한다 (프리셋이 실제로 동작)
  2. 어떤 조합이 와도 안 무너진다 → 유저 커스텀이 안전
  3. 드롭아웃 같은 정규화 효과 → 데이터가 적은 상황에 유리
`m` 을 **음수까지** 뽑는 것이 역배형(UPSET)이 성립하는 이유다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import config as cfg
from .data import CORE


def _mlp(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, d_out),
    )


def race_zscore(s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """경주 안에서 평균 0 · 표준편차 1. 다른 경주의 100점과 이 경주의 100점을 같게 만들지 않는다."""
    n = mask.sum(1, keepdim=True).clamp(min=1)
    mu = (s * mask).sum(1, keepdim=True) / n
    var = (((s - mu) ** 2) * mask).sum(1, keepdim=True) / n
    return ((s - mu) / (var.sqrt() + 1e-5)) * mask


class AxisRanker(nn.Module):
    """6축(+시장) 타워 랭커.

    slices       {축 이름: 입력 피처 인덱스}. data.axis_slices() 결과. CORE 는 공유.
    vocab_sizes  [기수, 조교사, 부마] 어휘 크기 (categorical.EMBED_COLS 순서)
    k_hist       이력 항목 수 (history.K). 0 이면 GRU 를 만들지 않는다.
    """

    # 임베딩을 어느 타워가 가져가는지. categorical.EMBED_COLS 순서와 맞춘다.
    EMB_OWNER = {"jkNo": "JOCKEY", "trNo": "JOCKEY", "F2_sire_id": "ABILITY"}

    def __init__(self, slices: dict[str, np.ndarray], vocab_sizes: list[int], k_hist: int,
                 emb: int = cfg.EMB_DIM, gru: int = cfg.GRU_DIM,
                 tower_hidden: int = cfg.TOWER_HIDDEN, trunk_hidden: int = cfg.TRUNK_HIDDEN,
                 hist_read: int = 16, dropout: float = cfg.DROPOUT):
        super().__init__()
        self.axes = [a for a in cfg.AXES if a in slices]
        self.has_market = cfg.MARKET in slices
        self.emb_cols = list(self.EMB_OWNER)
        self.k_hist = k_hist
        self.dropout_p = dropout

        self.register_buffer("core_idx", torch.as_tensor(slices[CORE], dtype=torch.long))
        for name in self.axes + ([cfg.MARKET] if self.has_market else []):
            self.register_buffer(f"idx_{name}", torch.as_tensor(slices[name], dtype=torch.long))

        # ── 공유 trunk — 경주 상황
        self.trunk = nn.Sequential(
            nn.Linear(len(slices[CORE]), trunk_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden // 2), nn.GELU(),
        )
        d_trunk = trunk_hidden // 2

        # ── 범주형 임베딩 — 소속 타워에만 들어간다
        self.embs = nn.ModuleList([nn.Embedding(v, emb, padding_idx=0) for v in vocab_sizes])
        self.emb_drop = nn.Dropout(dropout)
        d_emb_extra = {ax: emb * sum(1 for c, o in self.EMB_OWNER.items() if o == ax)
                       for ax in self.axes}

        # ── 전적 GRU — 하나를 공유하되 축마다 다른 선형으로 읽는다
        if k_hist:
            self.gru = nn.GRU(k_hist, gru, batch_first=True)
            self.gru_drop = nn.Dropout(dropout)
            self.hist_read = nn.ModuleDict({
                ax: nn.Sequential(nn.Linear(gru + 1, hist_read), nn.GELU()) for ax in self.axes
            })
            d_hist = hist_read
        else:
            self.gru = None
            d_hist = 0

        # ── 축 타워
        self.towers = nn.ModuleDict({
            ax: _mlp(len(slices[ax]) + d_trunk + d_hist + d_emb_extra[ax],
                     tower_hidden, 1, dropout)
            for ax in self.axes
        })
        # ── 시장 타워 (슬라이더 아님). 이력을 주지 않는다 — 배당은 그 자체로 완결된 정보다.
        if self.has_market:
            self.market_tower = _mlp(len(slices[cfg.MARKET]) + d_trunk, tower_hidden, 1, dropout)

    # ────────────────────────────────────────────────────────────────
    @property
    def score_columns(self) -> list[str]:
        """내보내는 점수 열 순서. axis_scores() 마지막 축과 같다."""
        return self.axes + ([cfg.MARKET] if self.has_market else [])

    def _seq_state(self, hist, hist_len, mask):
        """[B,N,gru+1] — GRU 마지막 유효 hidden 과 has_hist 플래그.

        pack_padded_sequence 는 정렬 비용과 cuDNN 비결정 때문에 쓰지 않는다.
        """
        B, N, L, _ = hist.shape
        flat = hist.reshape(B * N, L, -1)
        n = hist_len.reshape(B * N)
        real = (mask.reshape(B * N) > 0) & (n > 0)
        h = hist.new_zeros(B * N, self.gru.hidden_size)
        if real.any():
            out, _ = self.gru(flat[real])
            last = (n[real] - 1).clamp(min=0)
            h[real] = out.gather(1, last.view(-1, 1, 1).expand(-1, 1, out.shape[-1])).squeeze(1)
        has = (n > 0).float().unsqueeze(-1)
        return torch.cat([self.gru_drop(h), has], -1).reshape(B, N, -1)

    def axis_scores(self, x, mask, cat=None, hist=None, hist_len=None) -> torch.Tensor:
        """[B, N, A(+1)] — 축별 경주 내 z-score. 마지막 열이 시장(있을 때)."""
        core = self.trunk(x.index_select(-1, self.core_idx))
        emb = {}
        if cat is not None:
            for i, c in enumerate(self.emb_cols):
                emb.setdefault(self.EMB_OWNER[c], []).append(self.embs[i](cat[..., i]))
        seq = self._seq_state(hist, hist_len, mask) if (self.gru is not None and hist is not None) else None

        out = []
        for ax in self.axes:
            parts = [x.index_select(-1, getattr(self, f"idx_{ax}")), core]
            if seq is not None:
                parts.append(self.hist_read[ax](seq))
            if ax in emb:
                parts.append(self.emb_drop(torch.cat(emb[ax], -1)))
            out.append(race_zscore(self.towers[ax](torch.cat(parts, -1)).squeeze(-1), mask))
        if self.has_market:
            mkt = torch.cat([x.index_select(-1, getattr(self, f"idx_{cfg.MARKET}")), core], -1)
            out.append(race_zscore(self.market_tower(mkt).squeeze(-1), mask))
        return torch.stack(out, dim=-1)

    # ────────────────────────────────────────────────────────────────
    def sample_weights(self, B: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """디리클레 가중치 w [B,A] 와 시장 계수 m [B]. m 은 음수도 나온다(역배)."""
        a = torch.full((B, len(self.axes)), cfg.DIRICHLET_ALPHA, device=device)
        w = torch.distributions.Dirichlet(a).sample()
        lo, hi = cfg.MARKET_RANGE
        m = torch.rand(B, device=device) * (hi - lo) + lo if self.has_market \
            else torch.zeros(B, device=device)
        return w, m

    def combine(self, s: torch.Tensor, mask: torch.Tensor,
                w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """축 점수 [B,N,A(+1)] + 가중치 → 최종 점수 [B,N]."""
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(s.shape[0], -1)
        if m.dim() == 0:
            m = m.expand(s.shape[0])
        total = (s[..., :len(self.axes)] * w.unsqueeze(1)).sum(-1)
        if self.has_market:
            total = total + s[..., -1] * m.unsqueeze(1)
        return total * mask

    def forward(self, x, mask, cat=None, hist=None, hist_len=None,
                w: torch.Tensor | None = None, m: torch.Tensor | None = None):
        s = self.axis_scores(x, mask, cat, hist, hist_len)
        if w is None:
            w, m_s = self.sample_weights(x.shape[0], x.device)
            m = m_s if m is None else m
        if m is None:
            m = torch.zeros(x.shape[0], device=x.device)
        return self.combine(s, mask, w, m)
