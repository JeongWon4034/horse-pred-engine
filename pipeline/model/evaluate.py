"""평가 — 적중률만 보지 않는다.

정확도만 보면 인기마만 찍는 모델이 1등이 되는데,
공제율 때문에 실제로는 손해다. 그래서 캘리브레이션을 같이 본다.
"""
from __future__ import annotations

import numpy as np
import torch

from .losses import win_probabilities


@torch.no_grad()
def evaluate(model, races, device, w=None, batch=512) -> dict:
    model.eval()
    hit1 = hit3 = n = 0
    probs, wins = [], []

    for i in range(0, len(races), batch):
        x = torch.as_tensor(races.x[i:i + batch], device=device)
        m = torch.as_tensor(races.mask[i:i + batch], device=device)
        o = torch.as_tensor(races.order[i:i + batch], device=device)

        s = model(x, m, w) if w is not None or hasattr(model, "axes") else model(x, m)
        s = s.masked_fill(m == 0, -1e9)

        true1 = o[:, 0]
        ok = true1 >= 0
        pred1 = s.argmax(1)
        top3 = s.topk(min(3, s.shape[1]), dim=1).indices

        hit1 += ((pred1 == true1) & ok).sum().item()
        hit3 += ((top3 == true1.unsqueeze(1)).any(1) & ok).sum().item()
        n += ok.sum().item()

        p = win_probabilities(s, m)
        probs.append(p[m > 0].cpu().numpy())
        y = torch.zeros_like(m)
        y.scatter_(1, true1.clamp(min=0).unsqueeze(1), ok.float().unsqueeze(1))
        wins.append(y[m > 0].cpu().numpy())

    p = np.concatenate(probs); y = np.concatenate(wins)
    return {"top1": hit1 / max(n, 1), "top3": hit3 / max(n, 1), "n": n,
            "ece": expected_calibration_error(p, y), "brier": float(((p - y) ** 2).mean())}


def expected_calibration_error(p, y, bins=15) -> float:
    """'41%라고 한 것들이 실제로 41% 맞았나' — 작을수록 좋다."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def market_baseline(df) -> dict:
    """인기 1위마를 그냥 찍었을 때. 우리가 넘어야 할 선."""
    d = df[["race_id", "y_ord", "F6_mkt_rank"]].dropna()
    g = d.groupby("race_id", sort=False)
    fav = g.apply(lambda t: t.loc[t.F6_mkt_rank.idxmin(), "y_ord"], include_groups=False)
    return {"top1": float((fav == 1).mean()), "top3": float((fav <= 3).mean()), "n": len(fav)}
