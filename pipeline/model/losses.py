"""Plackett-Luce listwise 손실.

'1등 뽑고 → 걔 빼고 → 2등 뽑고 → 걔 빼고 → 3등 뽑고'를 그대로 확률로 쓴다.
softmax 가 경주 안에서만 걸리기 때문에 확률 합이 자동으로 1이 되고,
같은 경주 말들끼리 경쟁 관계가 생긴다.
"""
from __future__ import annotations

import torch

NEG = -1e9


def plackett_luce(scores: torch.Tensor, mask: torch.Tensor,
                  order: torch.Tensor, topk: int | None = None) -> torch.Tensor:
    """
    scores [B, N]  말별 점수
    mask   [B, N]  1 = 실제 출전마
    order  [B, K]  1~K착의 슬롯 인덱스 (-1 = 없음)
    topk        몇 착까지 학습에 쓸지. None 이면 order 전체
    """
    K = order.shape[1] if topk is None else min(topk, order.shape[1])
    avail = mask.clone()
    total = scores.new_zeros(())
    count = scores.new_zeros(())

    for k in range(K):
        idx = order[:, k]                                  # [B]
        valid = idx >= 0
        if not valid.any():
            continue

        masked = scores.masked_fill(avail == 0, NEG)
        logZ = torch.logsumexp(masked, dim=1)              # [B]
        safe = idx.clamp(min=0).unsqueeze(1)
        chosen = scores.gather(1, safe).squeeze(1)         # [B]

        total = total + ((logZ - chosen) * valid).sum()
        count = count + valid.sum()

        # 뽑힌 말을 후보에서 제거 (없는 경우는 0번 슬롯을 건드리지 않도록 valid 로 막음)
        avail = avail.scatter(1, safe, (~valid).float().unsqueeze(1) * avail.gather(1, safe))

    return total / count.clamp(min=1)


def win_probabilities(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """경주 내 1착 확률. 학습이 아니라 서빙/평가용."""
    return torch.softmax(scores.masked_fill(mask == 0, NEG), dim=1) * mask
