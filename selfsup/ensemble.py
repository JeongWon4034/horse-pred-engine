"""P5-2 — 사전학습 몸통 확률 + 정원 S3 확률 앙상블. 코드 없이 parquet 두 묶음으로 잰다.

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.ensemble

이쪽(selfsup)은 top-1, S3 는 logloss 에서 앞선다. 둘이 독립이면 섞었을 때 둘 다 남아야 한다.
가중치 w 는 valid 로 고르면 낙관값이 되므로 **경주 단위로 반 갈라 한쪽에서 고르고 다른 쪽에서 잰다**
(정원 result_dl.md §3 '앙상블 가중치 교차 적합' 과 같은 절차, 20회).
"""
from __future__ import annotations

import numpy as np

from . import data as D

SSL = "selfsup_mask_base_ft"
S3 = "history_73_L20"
S3_SEEDS = ["s1", "s2", "s3", "s4", "s20260901"]
W = np.round(np.arange(0.0, 1.01, 0.1), 2)


def seed_avg_probs(va, names):
    return np.mean([D.E.win_probs(va, D.E.load_pred(va, n)) for n in names], axis=0)


def main():
    va = D.load("valid")
    p_ssl = seed_avg_probs(va, [f"{SSL}_s{i}" for i in range(1, 6)])
    p_s3 = seed_avg_probs(va, [f"{S3}_{s}" for s in S3_SEEDS])
    p_lgb = D.E.win_probs(va, D.E.load_pred(va, "lgb_73"))

    def mix(w):                       # w = selfsup 비중. 확률 평균 → log 를 점수로
        return np.log(w * p_ssl + (1 - w) * p_s3 + 1e-12)

    print("== 전체 valid 에서 w 격자 (참고용 — 낙관값)")
    print(f"{'w':>5} {'top1':>7} {'top3':>7} {'logloss':>8}")
    for w in W:
        m = D.E.metrics(va, mix(w))
        print(f"{w:>5.1f} {m['top1']:>7.2f} {m['top3']:>7.2f} {m['logloss']:>8.4f}")

    # 교차 적합 — 경주를 반으로 갈라 A 에서 w 선택(logloss 최소), B 에서 측정. 20회
    rid = va["race_id"].to_numpy()
    races = np.unique(rid)
    rng = np.random.default_rng(20260901)
    picked, hold_top1, hold_ll, hold_top1_ssl, hold_top1_s3, hold_top1_lgb = [], [], [], [], [], []
    for _ in range(20):
        perm = rng.permutation(races)
        A = set(perm[: len(races) // 2])
        mA = np.isin(rid, list(A)); mB = ~mA
        vaA, vaB = va[mA].reset_index(drop=True), va[mB].reset_index(drop=True)
        best_w = min(W, key=lambda w: D.E.race_logloss(vaA, mix(w)[mA]))
        picked.append(best_w)
        mB_ = D.E.metrics(vaB, mix(best_w)[mB])
        hold_top1.append(mB_["top1"]); hold_ll.append(mB_["logloss"])
        hold_top1_ssl.append(D.E.metrics(vaB, np.log(p_ssl[mB]))["top1"])
        hold_top1_s3.append(D.E.metrics(vaB, np.log(p_s3[mB]))["top1"])
        hold_top1_lgb.append(D.E.metrics(vaB, np.log(p_lgb[mB]))["top1"])

    print("\n== 교차 적합 20회 (A 에서 w 선택 · B 에서 측정)")
    print(f"선택된 w: 평균 {np.mean(picked):.2f}  분포 {dict(zip(*np.unique(picked, return_counts=True)))}")
    print(f"홀드아웃 top-1  앙상블 {np.mean(hold_top1):.2f} ±{np.std(hold_top1):.2f}"
          f"  | selfsup {np.mean(hold_top1_ssl):.2f}  S3 {np.mean(hold_top1_s3):.2f}  LGB {np.mean(hold_top1_lgb):.2f}")
    print(f"홀드아웃 logloss 앙상블 {np.mean(hold_ll):.4f} ±{np.std(hold_ll):.4f}")

    # 고정 w=0.5 의 전체 valid 짝 비교 (w 를 안 고르면 낙관이 아니다)
    print("\n== w=0.5 고정, 전체 valid 짝 비교")
    s_mix = mix(0.5)
    h_mix = D.E.hits(va, s_mix)
    for name, s in [("LGB", np.log(p_lgb)), ("S3 5seed", np.log(p_s3)), ("selfsup 5seed", np.log(p_ssl))]:
        d, lo, hi = D.E.compare(h_mix, D.E.hits(va, s))
        m = D.E.metrics(va, s)
        print(f"  vs {name:<14} top-1 {d:+.2f} [{lo:+.2f}, {hi:+.2f}]   (상대 top1 {m['top1']:.2f} ll {m['logloss']:.4f})")
    m = D.E.metrics(va, s_mix)
    print(f"  앙상블 w=0.5: top1 {m['top1']:.2f} top3 {m['top3']:.2f} logloss {m['logloss']:.4f} ece {m['ece']:.4f}")


if __name__ == "__main__":
    main()
