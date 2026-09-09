"""사이드 실험 — S5 6축 타워의 F1 전적 축에 S3 의 이력 GRU 를 넣는다.

    PYTHONUTF8=1 uv run python -m model.side_tower_hist              # 77피처 (게임 모드)
    PYTHONUTF8=1 uv run python -m model.side_tower_hist --no-market  # 73피처
    PYTHONUTF8=1 uv run python -m model.side_tower_hist --no-hist    # GRU 없이 = 기존 S5 재현(대조군)

**왜.** 채택 모델 S3(`HistoryRanker`)는 임베딩 + GRU 를 쓰는데, 유저 슬라이더가 붙는
S5(`TowerRanker`)는 수치·원핫 피처만 쓴다. 둘이 다른 모델이라 **베이스 모델의 강점이
유저에게 전달되지 않는다.** 77피처 균등가중이 LGB 대비 −2.31%p [−4.47, −0.24] 로
이 프로젝트에서 유일하게 "유의하게 못함"이 붙은 결과인 것도 여기서 온다고 본다.

사이드 장부 §4 마지막 줄의 "S3 이력 GRU 를 F1 축 입력으로 넣는 변형"이 이 실험이다.

**어디에 넣나.** GRU 출력(64차원 + has_hist 1칸)을 **F1 전적 축에만** 붙인다. F1 은 말의
과거 성적 집계 피처 축이고 GRU 가 읽는 원본 시퀀스와 정보원이 같다. 다른 축(혈통·기수·
조건·구간·인기도)은 건드리지 않는다 — 유저가 "전적 0" 으로 밀면 시퀀스도 같이 빠져야
슬라이더 의미가 유지된다.

기존 `models.py` 는 수정하지 않는다. `TowerRanker` 를 상속해 F1 타워만 교체한다.
`seq_state` 는 `HistoryRanker` 의 것을 그대로 호출한다 — S3 와 계산이 한 글자도 다르지
않아야 비교가 성립한다.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from .data import feature_cols, fit_encoder, flatten_scores, load, to_races, tower_slices  # noqa: E402
from .evaluate import compare, hits, ledger_line, load_pred, metrics, save_pred  # noqa: E402
from .history import FEATS as HIST_FEATS, history_for  # noqa: E402
from .losses import plackett_luce  # noqa: E402
from .models import HistoryRanker, TowerRanker  # noqa: E402
from .team import C  # noqa: E402

SEED = 20260901
HIST_AXIS = "F1"


class TowerHistRanker(TowerRanker):
    """TowerRanker + F1 축에 이력 GRU. 나머지 축은 부모 그대로."""

    def __init__(self, slices: dict[str, np.ndarray], k_hist: int, hidden: int = 64,
                 core: str = "X", alpha: float = 0.7, gru: int = 64, dropout: float = 0.1):
        super().__init__(slices, hidden, core, alpha)
        if HIST_AXIS not in self.axes:
            raise ValueError(f"{HIST_AXIS} 축이 없다 — 축 목록 {self.axes}")
        self.gru = nn.GRU(k_hist, gru, batch_first=True)
        self.gru_drop = nn.Dropout(dropout)
        # F1 타워만 입력 차원을 늘려 새로 만든다. 층 구성은 부모와 동일하게 유지한다.
        n_in = len(self.idx[HIST_AXIS]) + len(self.core_idx) + gru + 1
        self.towers[HIST_AXIS] = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def axis_scores(self, x, mask, hist=None, hist_len=None):
        """[B, N, A] — 부모와 같되 F1 입력에 GRU 상태를 이어 붙인다."""
        core = x.index_select(-1, self.core_idx)
        # S3 와 동일한 계산을 보장하려고 HistoryRanker 의 메서드를 그대로 부른다
        # (self.gru · self.gru_drop 만 쓰므로 이름이 맞으면 그대로 동작한다).
        seq = HistoryRanker.seq_state(self, hist, hist_len, mask) if hist is not None else None
        out = []
        for k in self.axes:
            h = torch.cat([x.index_select(-1, self.idx[k]), core], dim=-1)
            if k == HIST_AXIS and seq is not None:
                h = torch.cat([h, seq], dim=-1)
            s = self.towers[k](h).squeeze(-1)
            # 경주 안에서 표준화 — 부모와 같은 식
            n = mask.sum(1, keepdim=True).clamp(min=1)
            mu = (s * mask).sum(1, keepdim=True) / n
            var = (((s - mu) ** 2) * mask).sum(1, keepdim=True) / n
            out.append(((s - mu) / (var.sqrt() + 1e-5)) * mask)
        return torch.stack(out, dim=-1)

    def forward(self, x, mask, w=None, hist=None, hist_len=None):
        s = self.axis_scores(x, mask, hist, hist_len)
        if w is None:
            w = self.sample_weights(x.shape[0], x.device)
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(x.shape[0], -1)
        return (s * w.unsqueeze(1)).sum(-1) * mask


@torch.no_grad()
def predict(model, races, T, dev, w, batch=1024):
    model.eval(); out = []
    for i in range(0, len(races), batch):
        x = torch.as_tensor(races.x[i:i + batch], device=dev)
        m = torch.as_tensor(races.mask[i:i + batch], device=dev)
        ex = {k: v[i:i + batch] for k, v in T.items()}
        out.append(model(x, m, w, **ex).masked_fill(m == 0, -1e9).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-market", action="store_true", help="73피처 (F6 인기도 축 없음)")
    ap.add_argument("--no-hist", action="store_true", help="GRU 없이 = 기존 S5 재현(대조군)")
    ap.add_argument("--hist-len", type=int, default=20, help="이력 길이. L 곡선 결과상 5 이상은 같다")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gru", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--select", choices=["logloss", "top1"], default="logloss",
                    help="epoch 선택 기준. top1 은 기존 side_tower.py 규칙 (ep1 이 뽑힌다)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    market = not a.no_market
    n_feat = 77 if market else 73
    use_hist = not a.no_hist

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    tr_df, va_df = load("train"), load("valid")
    cols = feature_cols(exclude_pop=not market)
    enc = fit_encoder(tr_df, cols)

    ex_tr, ex_va = {}, {}
    if use_hist:
        ex_tr["hist"], ex_tr["hist_len"] = history_for(tr_df, "train", L=a.hist_len)
        ex_va["hist"], ex_va["hist_len"] = history_for(va_df, "valid", L=a.hist_len)
    tr, va = to_races(tr_df, enc, ex_tr), to_races(va_df, enc, ex_va)
    slices = tower_slices(enc)

    if use_hist:
        model = TowerHistRanker(slices, len(HIST_FEATS), hidden=a.hidden,
                                alpha=a.alpha, gru=a.gru).to(dev)
    else:
        model = TowerRanker(slices, hidden=a.hidden, alpha=a.alpha).to(dev)
    A = len(model.axes); eq = torch.full((A,), 1 / A, device=dev)
    kind = f"tower+GRU(L={a.hist_len})" if use_hist else "tower (대조군)"
    print(f"[S5 {kind} {n_feat}] 축 {model.axes}  "
          f"파라미터 {sum(p.numel() for p in model.parameters()):,}  device={dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    X = torch.as_tensor(tr.x, device=dev); M = torch.as_tensor(tr.mask, device=dev)
    O = torch.as_tensor(tr.order, device=dev)
    T_tr = {k: torch.as_tensor(v, device=dev) for k, v in tr.extra.items()}
    T_va = {k: torch.as_tensor(v, device=dev) for k, v in va.extra.items()}
    n, bs = len(tr), 256
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * ((n + bs - 1) // bs), pct_start=0.25)

    # ── epoch 선택 — 기존 side_tower.py 는 (top1, -logloss) 순인데 그러면 ep1 이 뽑힌다.
    #    균등가중 top-1 이 초기 epoch 에 크게 튀기 때문이다(가중치가 Dirichlet 랜덤이라
    #    축 타워가 덜 학습된 상태에서도 균등가중 점수가 우연히 높게 나온다). 실제로 ep1
    #    모델은 축 단독 성능이 장부값보다 낮다(F2 25.7→17.9, F4 23.1→19.5).
    #    여기서는 logloss 로 고르고, top-1 로 골랐으면 어디였는지도 같이 찍는다.
    #    S3(train.py)는 (top1, -logloss)로도 ep6~14 가 뽑히므로 이 문제는 타워 전용이다.
    def key(m: dict):
        return (-m["logloss"], m["top1"]) if a.select == "logloss" else (m["top1"], -m["logloss"])

    best = best_scores = best_state = None; stale = 0
    alt = None                                     # 반대 기준으로 골랐다면 어디였나
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(n, device=dev); tot = cnt = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            ex = {k: v[j] for k, v in T_tr.items()}
            # 가중치는 Dirichlet 샘플(w=None), 손실은 3착까지 — 기존 S5 와 동일 조건
            loss = plackett_luce(model(X[j], M[j], None, **ex), M[j], O[j], topk=3)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(j); cnt += len(j)
        scores = flatten_scores(predict(model, va, T_va, dev, eq), va)
        m = {**metrics(va_df, scores), "epoch": ep}
        if alt is None or (m["top1"], -m["logloss"]) > (alt["top1"], -alt["logloss"]):
            alt = m
        better = best is None or key(m) > key(best)
        if better:
            best, best_scores, best_state, stale = m, scores, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        print(f"  ep{ep:>2}  loss {tot/cnt:.4f}   top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}  "
              f"logloss {m['logloss']:.4f}{'  ←best' if better else ''}")
        if stale >= 5:
            print("  (조기 중단)"); break

    model.load_state_dict(best_state)
    print(f"\n[최고 — {a.select} 기준] ep{best['epoch']}  top1 {best['top1']:.2f}  "
          f"top3 {best['top3']:.2f}  logloss {best['logloss']:.4f}  ({time.time()-t0:.0f}s)")
    if alt["epoch"] != best["epoch"]:
        print(f"  (top-1 기준으로 골랐다면 ep{alt['epoch']}  top1 {alt['top1']:.2f}  "
              f"logloss {alt['logloss']:.4f} — 기존 side_tower.py 의 규칙)")
    tag = f"side_towerhist_{n_feat}{'_nohist' if not use_hist else ''}_s{a.seed}{a.tag}"
    print("  " + str(save_pred(va_df, best_scores, tag)))

    print("\n[축 단독 — 그 축만 100, 나머지 0]")
    axis_scores = {}
    for k, ax in enumerate(model.axes):
        w = torch.zeros(A, device=dev); w[k] = 1.0
        s = flatten_scores(predict(model, va, T_va, dev, w), va)
        axis_scores[ax] = s; m = metrics(va_df, s)
        mark = "  ← GRU 들어간 축" if (ax == HIST_AXIS and use_hist) else ""
        print(f"  {ax:<4} top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}  logloss {m['logloss']:.4f}{mark}")

    print("\n[축 간 예측 상관]")
    names = list(axis_scores); mat = np.corrcoef(np.stack([axis_scores[k] for k in names]))
    print("      " + "".join(f"{k:>7}" for k in names))
    for i, k in enumerate(names):
        print(f"  {k:<4}" + "".join(f"{mat[i,j]:7.2f}" for j in range(len(names))))
    print(f"  최대 비대각 상관 {np.abs(mat - np.eye(len(names))).max():.2f}  (목표 ≤ 0.9)")

    print("\n[극단 가중치 — 유저 프리셋 흉내]")
    presets = [("혈통 100", {"F2": 1.0}), ("기수 70 전적 30", {"F3": .7, "F1": .3}),
               ("전적 100", {"F1": 1.0})]
    if "F6" in model.axes:
        presets.append(("인기도 0 나머지 균등", {k: 1 / (A - 1) for k in model.axes if k != "F6"}))
    for name, w in presets:
        wt = torch.tensor([w.get(k, 0.0) for k in model.axes], device=dev)
        m = metrics(va_df, flatten_scores(predict(model, va, T_va, dev, wt), va))
        print(f"  {name:<20} top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}")

    lgb = load_pred(va_df, f"lgb_{n_feat}")
    if lgb is not None:
        d = compare(hits(va_df, best_scores), hits(va_df, lgb))
        verdict = "차이 없음" if d[1] <= 0 <= d[2] else ("유의하게 나음" if d[0] > 0 else "유의하게 못함")
        print(f"\n[균등 가중 − LGB] {d[0]:+.2f}%p CI [{d[1]:+.2f}, {d[2]:+.2f}] → {verdict}")

    prev = load_pred(va_df, f"side_towerhist_{n_feat}_nohist_s{a.seed}")
    if use_hist and prev is not None:
        d = compare(hits(va_df, best_scores), hits(va_df, prev))
        print(f"[균등 가중 − 대조군(GRU 없음)] {d[0]:+.2f}%p CI [{d[1]:+.2f}, {d[2]:+.2f}]")

    print("\n[장부]")
    print(ledger_line(best, f"S5 tower{'+GRU' if use_hist else ''} 균등가중", n_feat, seed=a.seed,
                      memo=f"6축 MLP({a.hidden})"
                           + (f" + F1 축에 이력 GRU({a.gru}) L={a.hist_len}" if use_hist else "")
                           + f", Dirichlet α={a.alpha}, PL topk=3, ep{best['epoch']}"))


if __name__ == "__main__":
    main()
