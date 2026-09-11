"""몸통 하나, 머리 셋.

Body      : 인코딩된 행 [.., D] → 표현 [.., d]. P0~P4 전부 이걸 공유한다. 소형 MLP.
RankHead  : 표현 → 점수 1개 (미세조정 · 선형 탐침)
MaskHeads : 표현 → (원본 복원 [D], 어느 슬롯이 가려졌나 [n_slot])   ← VIME 식
ProjHead  : 표현 → 대조학습용 벡터                                    ← SCARF 식
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Body(nn.Module):
    def __init__(self, d_in: int, d: int = 64, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, d), nn.LayerNorm(d),
        )
        self.d = d

    def forward(self, x):
        return self.net(x)


class RankHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)


class MaskHeads(nn.Module):
    """가린 값 복원 + 가린 위치 맞히기."""
    def __init__(self, d: int, d_in: int, n_slot: int):
        super().__init__()
        self.recon = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d_in))
        self.which = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, n_slot))

    def forward(self, h):
        return self.recon(h), self.which(h)


class ProjHead(nn.Module):
    def __init__(self, d: int, out: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))

    def forward(self, h):
        return F.normalize(self.net(h), dim=-1)


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """z_a[i] 의 짝은 z_b[i]. 배치 안 나머지 전부가 negative. 양방향 평균."""
    logits = z_a @ z_b.t() / tau                        # [B, B]
    target = torch.arange(len(z_a), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
