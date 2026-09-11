"""selfsup 실행 진입점. 레포 루트에서:

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run finetune --seed 1                       # P0
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run pretrain --obj mask  --pool ext         # P1 ①
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run finetune --init runs/mask_ext.pt --seed 1  # P1 ②
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run finetune --init runs/mask_ext.pt --freeze  # 선형 탐침
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run compare --a p0 --b lgb_73               # 짝 비교

채점은 전부 pipeline/model/evaluate.py (= 팀 C.evaluate + 정원 logloss·CI). 여기서 새로 재지 않는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import data as D
from .models import Body, RankHead, MaskHeads, ProjHead, info_nce

PL = None  # 지연 임포트 (pipeline 경로가 data 에서 잡힌다)


def _pl():
    global PL
    if PL is None:
        from model.losses import plackett_luce
        PL = plackett_luce
    return PL


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=D.ROOT).decode().strip()
    except Exception:
        return "—"


def seed_all(seed: int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────── 사전학습 ───────────────────────────
def pretrain(a):
    dev = device(); seed_all(a.seed)
    pool = D.load("train", ext=(a.pool == "ext"))
    cols = D.feature_cols()
    if a.drop_group:
        cols = [c for c in cols if c.split("_")[0] not in a.drop_group]
    enc = D.fit_encoder(pool, cols)
    X = torch.from_numpy(D.transform(pool, enc))                      # [N, Din]
    slot, groups = D.slots(enc)
    n_slot = len(groups)
    slot_t = torch.from_numpy(slot).to(dev)                            # [Din]
    num_cols = torch.zeros(X.shape[1], dtype=torch.bool)
    for i in D.numeric_slots(enc, groups):
        num_cols[groups[i][1]] = True
    num_cols = num_cols.to(dev)

    # 경주 단위 배치 — 같은 경주 말들이 한 배치에 들어가 negative 가 되게 (SCARF 에서 중요)
    rid = pool["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]]); ends = np.r_[starts[1:], len(rid)]
    R = len(starts)

    body = Body(X.shape[1], a.d, a.hidden, a.dropout).to(dev)
    head = (MaskHeads(a.d, X.shape[1], n_slot) if a.obj == "mask" else ProjHead(a.d)).to(dev)
    params = list(body.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((R + a.races - 1) // a.races)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    X = X.to(dev)

    print(f"[pretrain {a.obj}] pool={a.pool} rows={len(pool):,} races={R:,} Din={X.shape[1]} slots={n_slot} "
          f"params={sum(p.numel() for p in params):,} {dev}")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        body.train(); head.train()
        perm = np.random.permutation(R); tot = n = 0
        for b in range(0, R, a.races):
            rows = np.concatenate([np.arange(starts[i], ends[i]) for i in perm[b:b + a.races]])
            x = X[rows]                                                # [B, Din]
            B = len(x)
            # 슬롯 단위 마스크 → 열 단위로 펼친다
            slot_mask = (torch.rand(B, n_slot, device=dev) < a.p_mask)          # [B, n_slot]
            col_mask = slot_mask.gather(1, slot_t.expand(B, -1))                  # [B, Din]
            if a.obj == "mask":
                x_in = x.masked_fill(col_mask, 0.0)
                recon, which = head(body(x_in))
                l_num = F.mse_loss(recon[col_mask & num_cols], x[col_mask & num_cols])
                l_cat = F.binary_cross_entropy_with_logits(recon[col_mask & ~num_cols], x[col_mask & ~num_cols])
                l_which = F.binary_cross_entropy_with_logits(which, slot_mask.float())
                loss = l_num + l_cat + l_which
            else:
                # SCARF: 가릴 자리를 같은 열의 다른 행 값으로 바꾼다
                donor = torch.randint(0, B, (B, x.shape[1]), device=dev)
                x_cor = torch.where(col_mask, x.gather(0, donor), x)
                loss = info_nce(head(body(x)), head(body(x_cor)), a.tau)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
            tot += loss.item() * B; n += B
        print(f"  ep{ep:02d} loss {tot / n:.4f}  {time.time() - t0:.0f}s")

    out = D.RUNS / (a.out or f"{a.obj}_{a.pool}.pt")
    torch.save({"body": body.state_dict(), "enc": enc, "cols": cols, "args": vars(a),
                "commit": git_commit(), "data": D.DATA.joinpath("DATA_COMMIT").read_text().strip()}, out)
    print(f"→ {out}")


# ─────────────────────────── 미세조정 · 선형 탐침 ───────────────────────────
def to_tensors(df, enc, dev):
    r = D.pack_races(df, D.transform(df, enc))
    T = {"x": torch.from_numpy(r.x), "mask": torch.from_numpy(r.mask), "order": torch.from_numpy(r.order)}
    return r, {k: v.to(dev) for k, v in T.items()}


def predict(body, head, T, dev, bs=1024):
    body.eval(); head.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(T["x"]), bs):
            out.append(head(body(T["x"][b:b + bs])).cpu())
    return torch.cat(out).numpy()


def finetune(a):
    dev = device(); seed_all(a.seed)
    tr = D.load("train", ext=(a.pool == "ext")); va = D.load("valid")
    cols = D.feature_cols()
    if a.drop_group:
        cols = [c for c in cols if c.split("_")[0] not in a.drop_group]

    ck = torch.load(D.RUNS / a.init, weights_only=False) if a.init else None
    if ck is not None:
        enc = ck["enc"]
        if sorted(ck["cols"]) != sorted(cols):
            raise ValueError("--init ckpt 의 피처와 --drop-group 이 안 맞는다 — 사전학습도 같은 --drop-group 으로")
    else:
        enc = D.fit_encoder(tr, cols)
    rtr, Ttr = to_tensors(tr, enc, dev); rva, Tva = to_tensors(va, enc, dev)

    body = Body(enc.dim, a.d, a.hidden, a.dropout).to(dev); head = RankHead(a.d).to(dev)
    if ck is not None:
        body.load_state_dict(ck["body"])
    if a.freeze:
        for p in body.parameters():
            p.requires_grad_(False)
    params = [p for p in list(body.parameters()) + list(head.parameters()) if p.requires_grad]
    lr = a.lr if a.lr else (3e-3 if a.freeze else (3e-4 if ck is not None else 1e-3))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.wd)
    R = len(rtr); steps = a.epochs * ((R + a.bs - 1) // a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    pl = _pl()

    tag = a.tag or ("p0" if ck is None else Path(a.init).stem + ("_probe" if a.freeze else "_ft"))
    print(f"[finetune {tag}] pool={a.pool} feats={len(cols)} Din={enc.dim} init={a.init} freeze={a.freeze} "
          f"lr={lr} params={sum(p.numel() for p in params):,} {dev}")
    best = None; stale = 0; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        body.train(not a.freeze); head.train()
        perm = torch.randperm(R, device=dev)
        for b in range(0, R, a.bs):
            i = perm[b:b + a.bs]
            s = head(body(Ttr["x"][i]))
            loss = pl(s, Ttr["mask"][i], Ttr["order"][i], topk=a.topk)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        scores = D.flatten_scores(predict(body, head, Tva, dev), rva)
        m = D.E.metrics(va, scores)
        flag = ""
        if best is None or m["logloss"] < best["logloss"]:           # epoch 선택은 valid logloss
            best = {**m, "epoch": ep}; best_scores = scores; stale = 0; flag = " *"
        else:
            stale += 1
        print(f"  ep{ep:02d} top1 {m['top1']:.2f} top3 {m['top3']:.2f} ll {m['logloss']:.4f} {time.time() - t0:.0f}s{flag}")
        if a.patience and stale >= a.patience:
            break

    name = f"{tag}_s{a.seed}"
    D.E.save_pred(va, best_scores, f"selfsup_{name}")
    rec = {"name": name, "seed": a.seed, "pool": a.pool, "feats": len(cols), "init": a.init, "freeze": a.freeze,
           "commit": git_commit(), "data": D.DATA.joinpath("DATA_COMMIT").read_text().strip(), **best}
    with open(D.RUNS / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[최고] ep{best['epoch']} top1 {best['top1']:.2f} top3 {best['top3']:.2f} "
          f"ll {best['logloss']:.4f} ece {best['ece']:.4f}  → runs/results.jsonl")


# ─────────────────────────── 요약 · 짝 비교 ───────────────────────────
def summary(a):
    rows = [json.loads(l) for l in open(D.RUNS / "results.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["tag"] = df["name"].str.replace(r"_s\d+$", "", regex=True)
    g = df.groupby("tag").agg(n=("seed", "size"), top1=("top1", "mean"), top1_sd=("top1", "std"),
                              top3=("top3", "mean"), ll=("logloss", "mean"), ll_sd=("logloss", "std"),
                              ep=("epoch", "mean"))
    print(g.round(4).to_string())


def compare(a):
    """tag a 의 seed 평균 확률 vs b. b 는 selfsup tag 또는 experiments/pred 이름(lgb_73)."""
    va = D.load("valid")

    def scores_of(tag):
        preds = sorted(D.E.PRED_DIR.glob(f"selfsup_{tag}_s*.parquet"))
        if preds:
            p = np.mean([D.E.win_probs(va, D.E.load_pred(va, q.stem)) for q in preds], axis=0)
            return np.log(p + 1e-12), len(preds)
        return D.E.load_pred(va, tag), 1

    sa, na = scores_of(a.a); sb, nb = scores_of(a.b)
    ma, mb = D.E.metrics(va, sa), D.E.metrics(va, sb)
    d, lo, hi = D.E.compare(D.E.hits(va, sa), D.E.hits(va, sb))
    verdict = "차이 없음" if lo <= 0 <= hi else ("이김" if d > 0 else "짐")
    print(f"{a.a} (seed {na}개 평균)  top1 {ma['top1']:.2f}  ll {ma['logloss']:.4f}")
    print(f"{a.b} (seed {nb}개 평균)  top1 {mb['top1']:.2f}  ll {mb['logloss']:.4f}")
    print(f"top-1 차이 {d:+.2f}%p  95% CI [{lo:+.2f}, {hi:+.2f}]  → {verdict}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain")
    p.add_argument("--obj", choices=["mask", "scarf"], required=True)
    p.add_argument("--pool", choices=["base", "ext"], default="ext")
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--races", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--p-mask", type=float, default=None); p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d", type=int, default=64); p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--drop-group", nargs="*", default=[], help="ablation (P4): 예 F1")
    p.set_defaults(fn=pretrain)

    f = sub.add_parser("finetune")
    f.add_argument("--init", default=None, help="runs/ 안의 사전학습 ckpt")
    f.add_argument("--freeze", action="store_true", help="몸통 동결 = 선형 탐침")
    f.add_argument("--pool", choices=["base", "ext"], default="base")
    f.add_argument("--drop-group", nargs="*", default=[], help="ablation (P4): 예 F1")
    f.add_argument("--epochs", type=int, default=20); f.add_argument("--bs", type=int, default=256)
    f.add_argument("--lr", type=float, default=None); f.add_argument("--wd", type=float, default=1e-2)
    f.add_argument("--patience", type=int, default=5); f.add_argument("--topk", type=int, default=3)
    f.add_argument("--d", type=int, default=64); f.add_argument("--hidden", type=int, default=256)
    f.add_argument("--dropout", type=float, default=0.2); f.add_argument("--seed", type=int, default=1)
    f.add_argument("--tag", default=None)
    f.set_defaults(fn=finetune)

    s = sub.add_parser("summary"); s.set_defaults(fn=summary)
    c = sub.add_parser("compare"); c.add_argument("--a", required=True); c.add_argument("--b", default="lgb_73")
    c.set_defaults(fn=compare)

    a = ap.parse_args()
    if a.cmd == "pretrain" and a.p_mask is None:
        a.p_mask = 0.3 if a.obj == "mask" else 0.6
    a.fn(a)


if __name__ == "__main__":
    main()
