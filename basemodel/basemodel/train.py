# -*- coding: utf-8 -*-
"""AxisRanker 학습 CLI.

    uv run python -m basemodel.train                 # 77피처 (게임 리플레이, 시장 축 포함)
    uv run python -m basemodel.train --no-market     # 73피처 (주말 실시간, 시장 축 없음)
    uv run python -m basemodel.train --seed 1 --tag _s1

검증 중 표기하는 top-1 은 **균등 가중치 · m=0** 기준이다. 유저가 아무것도 안 건드린
상태이자, 프리셋 적합(presets.py) 전의 맨몸 성능이다. 체크포인트도 이 값으로 고른다 —
특정 프리셋에 맞춰 고르면 나머지 프리셋이 손해를 본다.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time

import numpy as np
import torch

from . import config as cfg
from . import categorical as cat_mod
from . import data as D
from . import history as H
from .evaluate import axis_report, metrics
from .losses import axis_decorrelation, plackett_luce
from .model import AxisRanker
from .team import ARTIFACTS, C


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_split(df, enc, vocabs, split: str, use_hist: bool, pool=None, stats=None):
    extra = {"cat": cat_mod.encode_cats(df, vocabs)}
    if use_hist:
        hist, hlen = H.history_for(df, split, pool=pool, stats=stats)
        extra["hist"], extra["hist_len"] = hist, hlen
    return D.to_races(df, enc, extra)


def batch_tensors(races: D.Races, idx, device):
    t = lambda a: torch.as_tensor(a[idx], device=device)          # noqa: E731
    out = {"x": t(races.x), "mask": t(races.mask)}
    if "cat" in races.extra:
        out["cat"] = t(races.extra["cat"])
    if "hist" in races.extra:
        out["hist"] = t(races.extra["hist"])
        out["hist_len"] = t(races.extra["hist_len"])
    return out


@torch.no_grad()
def predict_axes(model, races: D.Races, device, batch: int = 512) -> np.ndarray:
    """[len(rows), A(+1)] — 행 단위 축 점수."""
    model.eval()
    chunks = []
    for i in range(0, len(races), batch):
        b = batch_tensors(races, slice(i, i + batch), device)
        chunks.append(model.axis_scores(b["x"], b["mask"], b.get("cat"),
                                        b.get("hist"), b.get("hist_len")).cpu().numpy())
    s = np.concatenate(chunks)                                     # [R, N, A]
    return s[races.row_race, races.row_slot]


def combine_rows(axis_rows: np.ndarray, w: np.ndarray, m: float = 0.0,
                 n_axes: int | None = None) -> np.ndarray:
    """행 단위 축 점수 + 가중치 → 행 점수. 화면(AI-06)이 하는 계산과 같은 식."""
    n_axes = n_axes if n_axes is not None else len(w)
    out = axis_rows[:, :n_axes] @ np.asarray(w, float)
    if axis_rows.shape[1] > n_axes:
        out = out + axis_rows[:, -1] * m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-market", action="store_true", help="73피처 — 주말 실시간 조건")
    ap.add_argument("--no-hist", action="store_true", help="전적 GRU 제거 (기여 측정용)")
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--batch", type=int, default=cfg.BATCH)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--wd", type=float, default=cfg.WEIGHT_DECAY)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--select", default="logloss", choices=["logloss", "top1"],
                    help="체크포인트 선택 기준 (기본 logloss — top1 은 잡음이 커 조기중단을 오작동시킨다)")
    ap.add_argument("--patience", type=int, default=cfg.PATIENCE)
    ap.add_argument("--decorr", type=float, default=cfg.DECORR_LAMBDA,
                    help="축 간 상관 벌점 계수. 0 이면 끈다")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    market = not a.no_market
    n_feat = 77 if market else 73
    use_hist = not a.no_hist
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = pick_device(a.device)
    t0 = time.time()

    # ── 데이터 ──────────────────────────────────────────────────────
    tr_df, va_df = C.load("train"), C.load("valid")
    cols = D.feature_cols(exclude_pop=not market)
    enc = D.fit_encoder(tr_df, cols)
    vocabs = cat_mod.fit_vocabs(tr_df)
    print(f"[데이터] train {len(tr_df):,}행 / {tr_df['race_id'].nunique():,}경주   "
          f"valid {len(va_df):,}행 / {va_df['race_id'].nunique():,}경주   피처 {n_feat} → 입력 {enc.dim}칸")
    print(cat_mod.describe(vocabs, va_df))

    pool = H.load_pool() if use_hist else None
    stats = H.fit_stats(pool, int(tr_df["rcDate"].max())) if use_hist else None
    tr = build_split(tr_df, enc, vocabs, "train", use_hist, pool, stats)
    va = build_split(va_df, enc, vocabs, "valid", use_hist, pool, stats)

    slices = D.axis_slices(enc)
    model = AxisRanker(slices, [v.size for v in vocabs.values()],
                       k_hist=H.K if use_hist else 0).to(dev)
    A = len(model.axes)
    eq = torch.full((A,), 1 / A, device=dev)
    zero_m = torch.zeros((), device=dev)
    print(f"[모델] 축 {model.score_columns}  파라미터 {sum(p.numel() for p in model.parameters()):,}  "
          f"device={dev}  이력 {'L=' + str(cfg.HIST_LEN) if use_hist else '없음'}")
    for k, v in slices.items():
        print(f"       {k:<12}{len(v):>4}칸")

    # ── 학습 ────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n, bs = len(tr), a.batch
    steps = a.epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.25)

    best = best_state = best_axes = None
    stale = 0
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n).numpy()
        tot = cnt = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            b = batch_tensors(tr, j, dev)
            order = torch.as_tensor(tr.order[j], device=dev)
            # 축 점수를 먼저 얻는다 — 결합에도 쓰고 탈상관 벌점에도 쓴다
            s_ax = model.axis_scores(b["x"], b["mask"], b.get("cat"), b.get("hist"), b.get("hist_len"))
            w_s, m_s = model.sample_weights(len(j), dev)      # 매 배치 새 가중치 (극단 조합 포함)
            zero = torch.zeros_like(m_s)

            # 손실 둘을 더한다 — 서빙에서 실제로 쓰는 두 조건 그대로.
            #   ① m=0  기본형·사람형·상승세형, 그리고 주말 실시간 전부
            #   ② m~U  배당형·역배형
            # ①이 없으면 시장 타워가 손실을 혼자 줄여버리고 나머지 여섯 축이 무임승차한다.
            # 실측으로 그랬다 — ① 없이 학습하니 축 단독 top-1 이 30%대에서 8~25% 로
            # 무너졌고 실력 축은 무작위(10.2%)보다 낮아졌다.
            loss = plackett_luce(model.combine(s_ax, b["mask"], w_s, zero),
                                 b["mask"], order, topk=cfg.PL_TOPK)
            if model.has_market:
                loss = loss + plackett_luce(model.combine(s_ax, b["mask"], w_s, m_s),
                                            b["mask"], order, topk=cfg.PL_TOPK)
            if a.decorr > 0:
                loss = loss + a.decorr * axis_decorrelation(s_ax, b["mask"], A)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
            cnt += len(j)

        ax_rows = predict_axes(model, va, dev)
        m = metrics(va_df, combine_rows(ax_rows, eq.cpu().numpy(), 0.0, A))
        # 선택 기준. top-1 은 1,254경주에서 표준오차 ±0.85%p 라 잡음으로 최고 epoch 이
        # 흔들리고, OneCycle 의 저학습률 구간에 닿기 전에 조기 중단이 걸린다.
        # logloss 는 경주마다 값이 나와 훨씬 안정적이라 기본으로 쓴다.
        key = (lambda d: -d["logloss"]) if a.select == "logloss" else (lambda d: (d["top1"], -d["logloss"]))
        better = best is None or key(m) > key(best)
        if better:
            best = {**m, "epoch": ep}
            best_state = copy.deepcopy(model.state_dict())
            best_axes = ax_rows
            stale = 0
        else:
            stale += 1
        print(f"  ep{ep:>2}  loss {tot / cnt:.4f}   top1 {m['top1']:5.2f}  top3 {m['top3']:5.2f}  "
              f"logloss {m['logloss']:.4f}{'  <- best' if better else ''}")
        if stale >= a.patience:
            print("  (조기 중단)")
            break

    model.load_state_dict(best_state)
    print(f"\n[최고] ep{best['epoch']}  top1 {best['top1']:.2f}  top3 {best['top3']:.2f}  "
          f"logloss {best['logloss']:.4f}   ({time.time() - t0:.0f}s)")

    # ── 축 진단 ─────────────────────────────────────────────────────
    cols_ax = model.score_columns
    alone = {}
    for k, ax in enumerate(cols_ax):
        w = np.zeros(A)
        if ax == cfg.MARKET:
            alone[ax] = best_axes[:, -1]
        else:
            w[k] = 1.0
            alone[ax] = combine_rows(best_axes, w, 0.0, A)
    print()
    print(axis_report(va_df, alone, slider_axes=model.axes))
    print()
    print(C.report({f"AxisRanker {n_feat} (균등, m=0)": combine_rows(best_axes, eq.cpu().numpy(), 0.0, A)}, va_df))

    # ── 저장 ────────────────────────────────────────────────────────
    tag = f"axis_{n_feat}{'_nohist' if not use_hist else ''}_s{a.seed}{a.tag}"
    out = ARTIFACTS / "runs"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "axes": model.axes, "market": market,
                "slices": {k: v for k, v in slices.items()},
                "vocab_sizes": [v.size for v in vocabs.values()],
                "k_hist": H.K if use_hist else 0, "seed": a.seed, "decorr": a.decorr}, out / f"{tag}.pt")
    with open(out / f"{tag}.enc.pkl", "wb") as f:
        pickle.dump({"encoder": enc, "vocabs": vocabs,
                     "hist_stats": stats.to_dict() if stats else None}, f)
    np.save(out / f"{tag}.valid_axes.npy", best_axes)
    with open(out / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "n_feat": n_feat, "use_hist": use_hist, "seed": a.seed,
                   "best": best, "axes": cols_ax, "decorr": a.decorr, "select": a.select}, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out / tag}.pt / .enc.pkl / .valid_axes.npy / .json")


if __name__ == "__main__":
    main()
