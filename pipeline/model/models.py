"""랭킹 모델.

두 가지를 같은 손실함수(Plackett-Luce)로 학습해 비교한다.
  LinearRanker  = 조건부 로지스틱 회귀. Benter 방식. 베이스라인.
  TowerRanker   = 6축 타워 + 학습 중 가중치 랜덤 샘플링. 우리 차별점.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LinearRanker(nn.Module):
    """점수 = x·w. 경주 내 softmax 를 씌우면 조건부 로지스틱 회귀가 된다."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.w.weight)

    def forward(self, x, mask):                     # x [B,N,D]
        return self.w(x).squeeze(-1) * mask


class TowerRanker(nn.Module):
    """축마다 작은 MLP 를 두고, 축 점수를 유저 가중치로 결합한다.

    핵심: 학습 중에 가중치 w 를 디리클레 분포에서 매 배치 새로 뽑는다.
    그래야 유저가 극단값(예: 혈통 100, 나머지 0)을 넣어도 무너지지 않는다.
    """

    def __init__(self, slices: dict[str, np.ndarray], hidden: int = 64,
                 core: str = "X", alpha: float = 0.7):
        super().__init__()
        self.core_idx = torch.as_tensor(slices[core], dtype=torch.long)
        self.axes = [k for k in slices if k != core and len(slices[k]) > 0]
        self.idx = {k: torch.as_tensor(slices[k], dtype=torch.long) for k in self.axes}
        self.alpha = alpha

        n_core = len(self.core_idx)
        self.towers = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(len(self.idx[k]) + n_core, hidden), nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
            for k in self.axes
        })

    def to(self, *a, **kw):                          # 인덱스 텐서도 같이 옮긴다
        super().to(*a, **kw)
        self.core_idx = self.core_idx.to(*a, **kw)
        self.idx = {k: v.to(*a, **kw) for k, v in self.idx.items()}
        return self

    def axis_scores(self, x, mask):
        """[B, N, A] — 축별 점수. 경주 내 z-score 로 정규화해 척도를 통일한다."""
        core = x.index_select(-1, self.core_idx)
        out = []
        for k in self.axes:
            h = torch.cat([x.index_select(-1, self.idx[k]), core], dim=-1)
            s = self.towers[k](h).squeeze(-1)                       # [B,N]
            # 경주 안에서 표준화 — 다른 경주의 100점과 이 경주의 100점을 같게 만들지 않는다
            n = mask.sum(1, keepdim=True).clamp(min=1)
            mu = (s * mask).sum(1, keepdim=True) / n
            var = (((s - mu) ** 2) * mask).sum(1, keepdim=True) / n
            out.append(((s - mu) / (var.sqrt() + 1e-5)) * mask)
        return torch.stack(out, dim=-1)

    def sample_weights(self, B: int, device) -> torch.Tensor:
        """디리클레에서 가중치를 뽑는다. alpha<1 이면 한 축에 몰린 조합이 자주 나온다."""
        a = torch.full((B, len(self.axes)), self.alpha, device=device)
        return torch.distributions.Dirichlet(a).sample()

    def forward(self, x, mask, w: torch.Tensor | None = None):
        s = self.axis_scores(x, mask)                               # [B,N,A]
        if w is None:
            w = self.sample_weights(x.shape[0], x.device)
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(x.shape[0], -1)
        return (s * w.unsqueeze(1)).sum(-1) * mask
