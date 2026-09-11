"""P5-1 — 사전학습 몸통 ⊕ 정원 이력 GRU (S3 의 seq_state 를 그대로 빌린다).

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.hybrid --init mask_base.pt --seed 1   # 사전학습 몸통
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.hybrid --seed 1                       # 대조군 (랜덤 초기화)

앙상블(P5-2)에서 두 이득이 겹쳤다. 그래도 "한 모델 안에서" 합치면 다른지 한 번은 봐야 한다.
구조: Body(x) [64] ⊕ GRU(직전 L=20 출전) [64+1] → MLP → 점수. 정원 S3 와 다른 점은 S2 임베딩 자리에
사전학습 몸통이 들어간다는 것뿐이다.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from .models import Body
from .run import seed_all, device, git_commit, _pl


class Hybrid(nn.Module):
    def __init__(self, body: Body, k_hist: int, gru: int = 64, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        from model.models import HistoryRanker, mlp_head
        self.body = body
        # seq_state 만 빌린다 — HistoryRanker 를 통째로 만들고 GRU 부분만 쓴다
        self._hr = HistoryRanker(dim=1, vocab_sizes=[], k_hist=k_hist, gru=gru, hidden=hidden, dropout=dropout)
        self.head = mlp_head(body.d + gru + 1, hidden, dropout)

    def forward(self, x, mask, hist, hist_len):
        h = torch.cat([self.body(x), self._hr.seq_state(hist, hist_len, mask)], -1)
        return self.head(h).squeeze(-1) * mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hist-len", type=int, default=20); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3); ap.add_argument("--tag", default=None)
    ap.add_argument("--pace", action="store_true", help="이력 텐서에 페이스 5열을 붙인다 (P6)")
    a = ap.parse_args()

    from model.history import history_for, FEATS as HIST_FEATS
    dev = device(); seed_all(a.seed)
    tr = D.load("train"); va = D.load("valid"); cols = D.feature_cols()
    if a.init:
        # 09-11 에 만든 ckpt 는 args 에 __main__.pretrain 함수 참조가 들어 있다 — 이 모듈이 __main__ 이면 못 푼다
        import sys
        from . import run as _run
        sys.modules["__main__"].pretrain = _run.pretrain
    ck = torch.load(D.RUNS / a.init, weights_only=False) if a.init else None
    enc = ck["enc"] if ck else D.fit_encoder(tr, cols)

    def tensors(df, split):
        h, n = history_for(df, split, L=a.hist_len)
        if a.pace:
            from .pace import history_pace_for
            h = np.concatenate([h, history_pace_for(df, a.hist_len)], axis=-1)
        r = D.pack_races(df, D.transform(df, enc), {"hist": h, "hist_len": n})
        T = {"x": torch.from_numpy(r.x), "mask": torch.from_numpy(r.mask), "order": torch.from_numpy(r.order),
             "hist": torch.from_numpy(r.extra["hist"]).float(), "hist_len": torch.from_numpy(r.extra["hist_len"]).long()}
        return r, {k: v.to(dev) for k, v in T.items()}

    rtr, Ttr = tensors(tr, "train"); rva, Tva = tensors(va, "valid")
    body = Body(enc.dim)
    if ck:
        body.load_state_dict(ck["body"])
    k_hist = len(HIST_FEATS) + (5 if a.pace else 0)
    model = Hybrid(body, k_hist).to(dev)
    lr = a.lr or (3e-4 if ck else 1e-3)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.wd)
    R = len(rtr); steps = a.epochs * ((R + a.bs - 1) // a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    pl = _pl()
    tag = a.tag or ("hybrid_" + ("pre" if ck else "rand") + ("_pace" if a.pace else ""))
    print(f"[hybrid {tag}] init={a.init} L={a.hist_len} Din={enc.dim} lr={lr} params={sum(p.numel() for p in params):,} {dev}")

    def predict():
        model.eval(); out = []
        with torch.no_grad():
            for b in range(0, len(Tva["x"]), 512):
                sl = slice(b, b + 512)
                out.append(model(Tva["x"][sl], Tva["mask"][sl], Tva["hist"][sl], Tva["hist_len"][sl]).cpu())
        return D.flatten_scores(torch.cat(out).numpy(), rva)

    best = None; stale = 0; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(R, device=dev)
        for b in range(0, R, a.bs):
            i = perm[b:b + a.bs]
            s = model(Ttr["x"][i], Ttr["mask"][i], Ttr["hist"][i], Ttr["hist_len"][i])
            loss = pl(s, Ttr["mask"][i], Ttr["order"][i], topk=a.topk)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        scores = predict(); m = D.E.metrics(va, scores); flag = ""
        if best is None or m["logloss"] < best["logloss"]:
            best = {**m, "epoch": ep}; best_scores = scores; stale = 0; flag = " *"
        else:
            stale += 1
        print(f"  ep{ep:02d} top1 {m['top1']:.2f} top3 {m['top3']:.2f} ll {m['logloss']:.4f} {time.time() - t0:.0f}s{flag}")
        if a.patience and stale >= a.patience:
            break

    name = f"{tag}_s{a.seed}"
    D.E.save_pred(va, best_scores, f"selfsup_{name}")
    rec = {"name": name, "seed": a.seed, "pool": "base", "feats": len(cols), "init": a.init, "freeze": False,
           "commit": git_commit(), "data": D.DATA.joinpath("DATA_COMMIT").read_text().strip(), **best}
    with open(D.RUNS / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[최고] ep{best['epoch']} top1 {best['top1']:.2f} top3 {best['top3']:.2f} ll {best['logloss']:.4f}")


if __name__ == "__main__":
    main()
