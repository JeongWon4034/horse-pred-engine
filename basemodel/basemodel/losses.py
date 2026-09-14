# -*- coding: utf-8 -*-
"""Plackett-Luce listwise 손실.

'1등 뽑고 → 걔 빼고 → 2등 뽑고 → 걔 빼고 → 3등 뽑고'을 그대로 확률로 쓴다.
softmax 가 **경주 안에서만** 걸리므로 확률 합이 자동으로 1이 되고, 같은 경주 말들끼리
경쟁 관계가 생긴다(한 마리 오르면 나머지 내려감). 이 형태가 조건부 로지스틱 회귀
= Plackett-Luce = ListNet 의 공통 뼈대이고, 제일 단순한 버전이 Benter(1994) 의 그것이다.

원본: 이정원 horse-pred-engine `pipeline/model/losses.py`.
"""
from __future__ import annotations

import torch

NEG = -1e9


def plackett_luce(scores: torch.Tensor, mask: torch.Tensor,
                  order: torch.Tensor, topk: int | None = None) -> torch.Tensor:
    """scores [B,N] · mask [B,N] 1=실제 출주마 · order [B,K] 1~K착 슬롯 인덱스(-1=없음)."""
    K = order.shape[1] if topk is None else min(topk, order.shape[1])
    avail = mask.clone()
    total = scores.new_zeros(())
    count = scores.new_zeros(())

    for k in range(K):
        idx = order[:, k]
        valid = idx >= 0
        if not valid.any():
            continue
        masked = scores.masked_fill(avail == 0, NEG)
        logZ = torch.logsumexp(masked, dim=1)
        safe = idx.clamp(min=0).unsqueeze(1)
        chosen = scores.gather(1, safe).squeeze(1)
        total = total + ((logZ - chosen) * valid).sum()
        count = count + valid.sum()
        # 뽑힌 말을 후보에서 제거. 없는 경우(valid=False)는 0번 슬롯을 건드리지 않는다.
        avail = avail.scatter(1, safe, (~valid).float().unsqueeze(1) * avail.gather(1, safe))

    return total / count.clamp(min=1)


def win_probabilities(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """경주 내 1착 확률. 학습이 아니라 서빙·평가용."""
    return torch.softmax(scores.masked_fill(mask == 0, NEG), dim=1) * mask


def axis_decorrelation(axis_scores: torch.Tensor, mask: torch.Tensor,
                       n_axes: int | None = None) -> torch.Tensor:
    """축 점수끼리 얼마나 같은 말을 가리키는지 — 슬라이더가 죽지 않게 누르는 벌점.

    축이 전부 같은 점수를 내면 유저가 비율을 바꿔도 순위가 안 바뀐다. 그 상태를
    막으려고 축 간 상관의 제곱 평균을 손실에 더한다.

    축 점수는 이미 **경주 내 z-score**(평균 0·분산 1)라 상관계수가 곱의 평균과 같다.
    그래서 마스크된 칸의 s_i·s_j 를 평균내면 그대로 상관이 된다.

    ⚠ 완전 무상관을 목표로 하지 않는다. 좋은 말은 대개 여러 축에서 같이 좋고,
      그건 데이터의 사실이다. 계수를 크게 주면 정확도를 잃는다 — 실측으로 고른다.
    """
    s = axis_scores if n_axes is None else axis_scores[..., :n_axes]
    m = mask.unsqueeze(-1)
    n = m.sum((0, 1)).clamp(min=1)                          # 유효 칸 수
    sm = s * m
    A = sm.shape[-1]
    corr = torch.einsum("bni,bnj->ij", sm, sm) / n          # [A,A]
    off = corr - torch.diag_embed(torch.diagonal(corr))
    return (off ** 2).sum() / max(A * (A - 1), 1)
