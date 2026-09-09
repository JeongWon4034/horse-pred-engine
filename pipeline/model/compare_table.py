"""단계별 비교표 — 저장된 예측을 모아 한 표로 찍는다. 학습하지 않는다.

    PYTHONUTF8=1 uv run python -m model.compare_table

`model.train <kind> --seed S --tag _sS` 로 만들어 둔 `experiments/pred/*.parquet` 를 읽어
seed 개별 평균±sd, seed 확률 평균, LGB 대비 짝 비교 CI 를 낸다.

**왜 필요한가.** 팀 `common.py` 의 `load()` 가 `usable()` 에서 `clean()` → `usable()` 로
바뀌면서(`39eeb26`, 2026-09-09 머지) 기준선과 DL 입력 차원이 같이 움직였다
(153/157 → 138/142). 그래서 장부의 기존 비교표는 구 커밋(`f17ca36`) 값이고, 새로 잰
LightGBM(33.49)과 같은 표에 놓을 수 없다. 이 스크립트가 한 커밋 기준으로 표를 다시 만든다.

판정 문구는 짝 비교 CI 로만 쓴다 — `C.report()` 가 찍는 절대 표준오차는 마지막 행(무작위)의
값이라 판정에 쓸 수 없다(장부 "팀 전달" 항목 참조).
"""
from __future__ import annotations

import numpy as np

from .evaluate import compare, ece, hits, load_pred, race_logloss, win_probs
from .team import C

SEEDS = ["20260901", 1, 2, 3, 4]

# 표에 올릴 단계 — (표시 이름, 예측 파일 접두어)
STAGES = [
    ("S1 linear",            "linear_73_s"),
    ("S2 embed",             "embed_73_s"),
    ("S3 history L=10",      "history_73_L10_s"),
    ("S3 history L=20",      "history_73_L20_s"),
    ("S3 history L=0",       "history_73_L0_s"),
    ("S4 attn",              "attn_73_s"),
]


def collect(va, prefix: str):
    out = []
    for s in SEEDS:
        p = load_pred(va, f"{prefix}{s}")
        if p is not None:
            out.append(p)
    return out


def probs_avg(va, preds):
    return np.mean([win_probs(va, p) for p in preds], axis=0)


def main() -> None:
    va = C.load("valid")
    lgb = load_pred(va, "lgb_73")
    if lgb is None:
        raise SystemExit("lgb_73 예측이 없다 — model.baseline 을 먼저 돌린다")
    h_lgb = hits(va, lgb)
    ll_lgb = race_logloss(va, lgb)
    mkt = C.evaluate(va, C.market_scores(va)) if hasattr(C, "market_scores") else None

    print(f"valid {va['race_id'].nunique():,}경주 · 73피처 (인기도 제외, 주 비교축)")
    if mkt:
        print(f"시장 (인기 1위마, 참고)   top-1 {mkt['top1']:5.2f}%  top-3 {mkt['top3']:5.2f}%")
    print(f"LightGBM 73 (기준)       top-1 {h_lgb.mean()*100:5.2f}%  logloss {ll_lgb:.4f}\n")

    rows = []
    for name, prefix in STAGES:
        preds = collect(va, prefix)
        if not preds:
            continue
        t1 = np.array([hits(va, p).mean() * 100 for p in preds])
        ll = np.array([race_logloss(va, p) for p in preds])
        pa = probs_avg(va, preds)
        sa = np.log(np.clip(pa, 1e-12, 1))
        h = hits(va, sa)
        d, lo, hi = compare(h, h_lgb)
        rows.append(dict(name=name, n=len(preds), t1=t1, ll=ll, preds=preds,
                         t1_avg=h.mean() * 100, ll_avg=race_logloss(va, sa),
                         ece_avg=ece(va, sa), d=d, lo=lo, hi=hi))

    print("[단계별 — seed 개별 평균±sd / seed 확률 평균]")
    print(f"{'모델':<20}{'n':>3}{'top-1 개별':>14}{'logloss 개별':>17}"
          f"{'평균 top-1':>12}{'평균 logloss':>14}{'ECE':>8}   LGB 대비 (paired 95% CI)")
    for r in rows:
        sd1 = r["t1"].std(ddof=1) if r["n"] > 1 else 0.0
        sdl = r["ll"].std(ddof=1) if r["n"] > 1 else 0.0
        print(f"{r['name']:<20}{r['n']:>3}{r['t1'].mean():>9.2f} ±{sd1:.2f}"
              f"{r['ll'].mean():>11.4f} ±{sdl:.4f}{r['t1_avg']:>11.2f}%{r['ll_avg']:>14.4f}"
              f"{r['ece_avg']:>8.4f}   {r['d']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"
              f"  {'차이 없음' if r['lo'] <= 0 <= r['hi'] else '유의'}")

    # ── 앙상블 — LGB + S3 L20 seed 평균
    s3 = collect(va, "history_73_L20_s")
    if s3:
        p_s3 = probs_avg(va, s3)
        p_lgb = win_probs(va, lgb)
        print("\n[앙상블 w·S3(L20 seed평균) + (1−w)·LGB — valid 전체이므로 낙관값]")
        print(f"{'w':>5}{'top-1':>9}{'logloss':>10}   LGB 대비 (paired 95% CI)")
        for w in (0.0, 0.3, 0.5, 0.7, 0.86, 1.0):
            p = w * p_s3 + (1 - w) * p_lgb
            s = np.log(np.clip(p, 1e-12, 1))
            h = hits(va, s); d, lo, hi = compare(h, h_lgb)
            mark = "  ← side_ensemble 교차적합 값" if w == 0.86 else ""
            print(f"{w:>5.2f}{h.mean()*100:>8.2f}%{race_logloss(va, s):>10.4f}"
                  f"   {d:+.2f} [{lo:+.2f}, {hi:+.2f}]{mark}")

    # ── 앙상블 w 교차 적합 — valid 전체로 고르면 낙관값이라 경주를 반씩 갈라 정한다.
    #    side_ensemble.py 와 같은 방식이지만 그쪽은 L10 seed 5개를 요구해 여기서 다시 짠다.
    if s3:
        rid = va["race_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        race_of_row = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rid)]))
        grid = np.round(np.linspace(0, 1, 11), 2)
        P = {float(w): w * p_s3 + (1 - w) * p_lgb for w in grid}
        rng = np.random.default_rng(20260901)
        picked, out_t1, out_ll, base_t1 = [], [], [], []
        for _ in range(20):
            perm = rng.permutation(len(starts))
            half = np.zeros(len(starts), bool); half[perm[: len(perm) // 2]] = True
            A = half[race_of_row]
            for fit, ev in ((A, ~A), (~A, A)):
                w = min(grid, key=lambda t: race_logloss(va[fit], np.log(P[float(t)][fit])))
                s = np.log(P[float(w)][ev])
                picked.append(float(w))
                out_t1.append(hits(va[ev], s).mean() * 100)
                out_ll.append(race_logloss(va[ev], s))
                base_t1.append(hits(va[ev], lgb[ev]).mean() * 100)
        print(f"\n[앙상블 w 교차 적합 — 20회 × 2분할, logloss 최소로 선택]")
        print(f"  선택된 w 평균 {np.mean(picked):.2f} (범위 {min(picked):.1f}~{max(picked):.1f})")
        print(f"  홀드아웃 top-1 {np.mean(out_t1):.2f}  vs 같은 홀드아웃 LGB {np.mean(base_t1):.2f}"
              f"  → {np.mean(out_t1) - np.mean(base_t1):+.2f}%p")
        print(f"  홀드아웃 logloss {np.mean(out_ll):.4f}")

    # ── S3 대비 S4 — "S4 는 S3 위에서 얻는 것이 없다" 판정의 재확인
    s4 = collect(va, "attn_73_s")
    if s3 and s4:
        h3 = hits(va, np.log(np.clip(probs_avg(va, s3), 1e-12, 1)))
        h4 = hits(va, np.log(np.clip(probs_avg(va, s4), 1e-12, 1)))
        d, lo, hi = compare(h4, h3)
        ll3 = race_logloss(va, np.log(np.clip(probs_avg(va, s3), 1e-12, 1)))
        ll4 = race_logloss(va, np.log(np.clip(probs_avg(va, s4), 1e-12, 1)))
        print(f"\n[S4 − S3(L20)] top-1 {d:+.2f}%p [{lo:+.2f}, {hi:+.2f}]   "
              f"logloss {ll4 - ll3:+.4f}  ({ll4:.4f} vs {ll3:.4f})")


if __name__ == "__main__":
    main()
