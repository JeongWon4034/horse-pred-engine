"""사이드 실험 — 앙상블 가중치와 seed 개수. valid 를 경주 단위로 반씩 갈라 교차 적합한다.

    PYTHONUTF8=1 uv run python -m model.side_ensemble

가중치를 valid 전체로 고르면 그 valid 숫자는 낙관값이다. 그래서 A 반으로 w 를 고르고 B 반에서 재고, 반대로도 한 뒤 합친다.
seed 개수 곡선은 seed 부분집합의 확률 평균으로 만든다.
"""
from __future__ import annotations

import itertools

import numpy as np

from .evaluate import compare, hits, load_pred, metrics, win_probs
from .team import C

SEED = 20260901


def main() -> None:
    va = C.load("valid")
    lgb = win_probs(va, load_pred(va, "lgb_73"))
    l20 = [win_probs(va, load_pred(va, f"history_73_L20_s{s}")) for s in ["20260901", 1, 2, 3, 4]]
    l10 = [win_probs(va, load_pred(va, n)) for n in
           ["history_73", "history_73_s1", "history_73_s2", "history_73_s3", "history_73_s4"]]
    h_lgb = hits(va, np.log(lgb))

    def m(p):
        r = metrics(va, np.log(p)); return r["top1"], r["logloss"]

    print("[seed 개수 곡선 — S3 L20 확률 평균, 부분집합 전체 평균]")
    print(f"{'seeds':>5}{'top-1':>8}{'logloss':>9}   (부분집합 수)")
    for k in range(1, 6):
        subs = list(itertools.combinations(range(5), k))
        vals = np.array([m(np.mean([l20[i] for i in c], 0)) for c in subs])
        print(f"{k:>5}{vals[:,0].mean():>8.2f}{vals[:,1].mean():>9.4f}   ({len(subs)})")
    t1, ll = m(np.mean(l20 + l10, 0)); print(f"{'10':>5}{t1:>8.2f}{ll:>9.4f}   (L20 5 + L10 5)")

    s3 = np.mean(l20, 0)
    print("\n[가중치 w·S3 + (1−w)·LGB — valid 전체 (낙관값)]")
    for w in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
        p = w * s3 + (1 - w) * lgb; r = metrics(va, np.log(p)); d = compare(hits(va, np.log(p)), h_lgb)
        print(f"  w={w:.1f}  top1 {r['top1']:5.2f}  top3 {r['top3']:5.2f}  logloss {r['logloss']:.4f}  vs LGB {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]")

    print("\n[교차 적합 — 경주를 반으로 갈라 한쪽에서 w 선택(logloss 최소), 다른 쪽에서 측정. 20회 반복 평균]")
    rid = va["race_id"].to_numpy(); starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    race_of_row = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rid)]))
    rng = np.random.default_rng(SEED); grid = np.linspace(0, 1, 11)
    picked, out_top1, out_ll, lgb_top1 = [], [], [], []
    for _ in range(20):
        perm = rng.permutation(len(starts)); half = set(perm[: len(perm) // 2])
        A = np.isin(race_of_row, list(half)); B = ~A
        for fit, ev in ((A, B), (B, A)):
            best_w = min(grid, key=lambda w: metrics(va[fit], np.log(w * s3[fit] + (1 - w) * lgb[fit]))["logloss"])
            r = metrics(va[ev], np.log(best_w * s3[ev] + (1 - best_w) * lgb[ev]))
            picked.append(best_w); out_top1.append(r["top1"]); out_ll.append(r["logloss"])
            lgb_top1.append(metrics(va[ev], np.log(lgb[ev]))["top1"])
    print(f"  선택된 w 평균 {np.mean(picked):.2f} (범위 {min(picked):.1f}~{max(picked):.1f})")
    print(f"  홀드아웃 top-1 {np.mean(out_top1):.2f}  logloss {np.mean(out_ll):.4f}   같은 홀드아웃 LGB top-1 {np.mean(lgb_top1):.2f}  → 차이 {np.mean(out_top1)-np.mean(lgb_top1):+.2f}%p")


if __name__ == "__main__":
    main()
