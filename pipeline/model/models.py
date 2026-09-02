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

    def forward(self, x, mask, **_):                # x [B,N,D]  (추가 입력은 무시)
        return self.w(x).squeeze(-1) * mask


def mlp_head(d_in: int, hidden: int = 128, dropout: float = 0.2) -> nn.Sequential:
    """S2~S4 공용 점수 헤드: d_in → hidden → hidden/2 → 1."""
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )


class EmbedRanker(nn.Module):
    """S2 — 수치 피처 + 범주형(기수·조교사·부마) 임베딩 → MLP.

    cat [B,N,C] 의 열 순서는 vocab_sizes 의 순서와 같다. 0 은 <pad>, 1 <unk>, 2 <rare>.
    말 ID 는 여기 들어오지 않는다(categorical.EMBED_COLS 참조).
    """

    def __init__(self, dim: int, vocab_sizes: list[int], emb: int = 16,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, emb, padding_idx=0) for v in vocab_sizes])
        self.emb_drop = nn.Dropout(dropout)
        self.head = mlp_head(dim + emb * len(vocab_sizes), hidden, dropout)

    def encode(self, x, cat):
        """[B,N,D + C·emb] — 뒤 단계(S3·S4)가 같은 표현 위에 쌓는다."""
        e = [emb(cat[..., i]) for i, emb in enumerate(self.embs)]
        return torch.cat([x, self.emb_drop(torch.cat(e, -1))], -1)

    def forward(self, x, mask, cat, **_):
        return self.head(self.encode(x, cat)).squeeze(-1) * mask


class HistoryRanker(nn.Module):
    """S3 — S2 표현 + 말별 과거 전적 GRU.

    hist [B,N,L,K] 는 오른쪽 패딩(오래된 것부터, hist_len 개가 유효). 실제 말만 골라 GRU 에 한 번에
    넣고 마지막 유효 스텝의 hidden 을 꺼낸다. 이력이 0개인 말은 0 벡터 + has_hist=0.
    pack_padded_sequence 는 정렬 비용과 cuDNN 비결정 때문에 쓰지 않는다.
    """

    def __init__(self, dim: int, vocab_sizes: list[int], k_hist: int, emb: int = 16,
                 gru: int = 64, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.base = EmbedRanker(dim, vocab_sizes, emb, hidden, dropout)
        self.gru = nn.GRU(k_hist, gru, batch_first=True)
        self.gru_drop = nn.Dropout(dropout)
        self.d_out = dim + emb * len(vocab_sizes) + gru + 1
        self.head = mlp_head(self.d_out, hidden, dropout)

    def seq_state(self, hist, hist_len, mask):
        """[B,N,gru+1] — GRU 마지막 유효 hidden 과 has_hist 플래그."""
        B, N, L, K = hist.shape
        flat = hist.reshape(B * N, L, K)
        n = hist_len.reshape(B * N)
        real = (mask.reshape(B * N) > 0) & (n > 0)
        h = hist.new_zeros(B * N, self.gru.hidden_size)
        if real.any():
            out, _ = self.gru(flat[real])                                  # [M, L, gru]
            last = (n[real] - 1).clamp(min=0)
            h[real] = out.gather(1, last.view(-1, 1, 1).expand(-1, 1, out.shape[-1])).squeeze(1)
        has = (n > 0).float().unsqueeze(-1)
        return torch.cat([self.gru_drop(h), has], -1).reshape(B, N, -1)

    def encode(self, x, mask, cat, hist, hist_len):
        return torch.cat([self.base.encode(x, cat), self.seq_state(hist, hist_len, mask)], -1)

    def forward(self, x, mask, cat, hist, hist_len, **_):
        return self.head(self.encode(x, mask, cat, hist, hist_len)).squeeze(-1) * mask


class RaceTransformer(nn.Module):
    """S4 — S3 표현 위에 경주 내 self-attention. 같은 경주 말들이 서로를 본다.

    positional encoding 은 없다 (출전 번호·게이트는 이미 X_chulNo·X_gate_rel 로 들어가 있고,
    경주 안 순서는 의미가 없다). 패딩 슬롯은 src_key_padding_mask 로 가린다.
    """

    def __init__(self, dim: int, vocab_sizes: list[int], k_hist: int, d_model: int = 128,
                 nhead: int = 4, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.enc = HistoryRanker(dim, vocab_sizes, k_hist, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(self.enc.d_out, d_model), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x, mask, cat, hist, hist_len, **_):
        h = self.proj(self.enc.encode(x, mask, cat, hist, hist_len))     # [B,N,d]
        h = self.attn(h, src_key_padding_mask=(mask == 0))
        return self.out(self.norm(h)).squeeze(-1) * mask


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
