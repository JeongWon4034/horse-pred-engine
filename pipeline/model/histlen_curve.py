"""이력 길이 곡선 — 저장된 예측을 모아 L 별 표와 짝 비교 CI 를 찍는다.

    PYTHONUTF8=1 uv run python -m model.histlen_curve

학습은 하지 않는다. `model.train history --hist-len L --seed S --tag _L{L}_s{S}` 로 만들어 둔
`experiments/pred/history_73_L{L}_s{S}.parquet` 를 읽는다.

**왜 이 실험이 필요한가.** S2 → S3 비교는 GRU 모듈과 MLP 입력 차원이 같이 바뀌므로 시퀀스의
순수 기여분이 아니다. L=0 은 구조를 그대로 두고 이력만 비운다 — 딱 하나만 다른 통제 실험이다.
(L=0 이면 n_hist 가 0 이라 GRU 가 아예 실행되지 않고 has_hist 플래그도 0 이 된다.)

판정은 두 축을 따로 본다. top-1 은 짝 비교 CI, logloss 는 seed 평균.
"""
from __future__ import annotations

import numpy as np

from .evaluate import compare, ece, hits, load_pred, race_logloss, win_probs
from .team import C

LENGTHS = [0, 5, 10, 20, 30]
SEEDS = ["20260901", 1, 2, 3, 4]


def collect(va, L: int) -> tuple[list[np.ndarray], list[str]]:
    """L 에 대해 있는 예측만 모은다. 없는 seed 는 조용히 건너뛴다."""
    out, got = [], []
    for s in SEEDS:
        p = load_pred(va, f"history_73_L{L}_s{s}")
        if p is not None:
            out.append(p); got.append(str(s))
    return out, got


def main() -> None:
    va = C.load("valid")
    lgb = load_pred(va, "lgb_73")
    if lgb is None:
        raise SystemExit("lgb_73 예측이 없다 — model.baseline 을 먼저 돌린다")
    h_lgb = hits(va, lgb)
    ll_lgb = race_logloss(va, lgb)

    print(f"valid {va['race_id'].nunique():,}경주 · top-1 SE ±{C.evaluate(va, lgb)['se_top1']:.2f}%p")
    print(f"LightGBM 73 기준 — top-1 {h_lgb.mean()*100:.2f}%  logloss {ll_lgb:.4f}\n")

    rows = {}
    for L in LENGTHS:
        preds, seeds = collect(va, L)
        if not preds:
            continue
        t1 = np.array([hits(va, p).mean() * 100 for p in preds])
        ll = np.array([race_logloss(va, p) for p in preds])
        # seed 확률 평균 — 장부의 "seed N개 평균" 행과 같은 정의
        p_avg = np.mean([win_probs(va, p) for p in preds], axis=0)
        s_avg = np.log(np.clip(p_avg, 1e-12, 1))
        rows[L] = dict(n=len(preds), seeds=seeds, t1=t1, ll=ll,
                       t1_avg=hits(va, s_avg).mean() * 100,
                       ll_avg=race_logloss(va, s_avg), ece_avg=ece(va, s_avg),
                       preds=preds)

    print("[L 별 — seed 개별 평균±sd, 그리고 seed 확률 평균]")
    print(f"{'L':>4}{'n':>3}{'top-1 (평균±sd)':>18}{'logloss (평균±sd)':>21}"
          f"{'seed평균 top-1':>15}{'seed평균 logloss':>17}{'ECE':>8}")
    for L, r in rows.items():
        print(f"{L:>4}{r['n']:>3}"
              f"{r['t1'].mean():>12.2f} ±{r['t1'].std(ddof=1) if r['n'] > 1 else 0:.2f}"
              f"{r['ll'].mean():>15.4f} ±{r['ll'].std(ddof=1) if r['n'] > 1 else 0:.4f}"
              f"{r['t1_avg']:>14.2f}%{r['ll_avg']:>17.4f}{r['ece_avg']:>8.4f}")

    if 0 not in rows:
        raise SystemExit("\nL=0 이 없다 — 이 실험의 기준점이다. 먼저 돌린다.")

    base = rows[0]
    print(f"\n[L=0 대비 — 짝 비교, seed 평균 예측끼리]")
    print(f"{'L':>4}{'top-1 차':>10}{'95% CI':>20}{'logloss 차':>13}   판정")
    h0 = hits(va, np.log(np.clip(np.mean([win_probs(va, p) for p in base['preds']], 0), 1e-12, 1)))
    for L, r in rows.items():
        if L == 0:
            continue
        pa = np.log(np.clip(np.mean([win_probs(va, p) for p in r["preds"]], 0), 1e-12, 1))
        d, lo, hi = compare(hits(va, pa), h0)
        dll = r["ll_avg"] - base["ll_avg"]
        verdict = "차이 없음" if lo <= 0 <= hi else "차이 있음"
        print(f"{L:>4}{d:>+10.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}{dll:>+13.4f}   {verdict}")

    print(f"\n[LightGBM 대비 logloss — 시퀀스가 우위를 만드는가]")
    print(f"{'L':>4}{'logloss':>10}{'vs LGB':>10}   읽기")
    for L, r in rows.items():
        d = r["ll_avg"] - ll_lgb
        note = "LGB 보다 나쁘다" if d > 0 else "LGB 보다 낫다"
        print(f"{L:>4}{r['ll_avg']:>10.4f}{d:>+10.4f}   {note}")

    print("\n[요약]")
    best_ll = min(rows, key=lambda L: rows[L]["ll_avg"])
    print(f"  logloss 최소는 L={best_ll} ({rows[best_ll]['ll_avg']:.4f})")
    if rows[0]["ll_avg"] > ll_lgb:
        print(f"  L=0 의 logloss {rows[0]['ll_avg']:.4f} 는 LightGBM {ll_lgb:.4f} 보다 나쁘다")
        print("  → S3 의 logloss 우위는 전적 시퀀스에서 나온 것이다. 이력을 비우면 사라진다.")
    short = [L for L in rows if L and L <= 5]
    if short:
        L5 = short[0]
        gap = rows[L5]["ll_avg"] - rows[max(rows)]["ll_avg"]
        print(f"  L={L5} 와 L={max(rows)} 의 logloss 차 {gap:+.4f}"
              f" → 이득의 대부분이 최근 {L5}출전에서 나온다")


if __name__ == "__main__":
    main()
