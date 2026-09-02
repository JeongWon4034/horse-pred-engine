"""LightGBM 기준선 재현 — 팀 baseline_lgbm.py 를 그대로 돌리고 예측만 저장한다.

    PYTHONUTF8=1 uv run python -m model.baseline

팀 표(valid): 77피처 38.9 / 68.1, 73피처 34.0 / 63.2. ±0.3 안에 안 들면 데이터 사본 문제다.
저장한 예측(experiments/pred/lgb_{73,77}.parquet)은 이후 딥러닝과 경주별 짝 비교에 쓴다.
"""
from __future__ import annotations

import sys
import time

from .team import C                # data/model 을 sys.path 에 넣는다 — baseline_lgbm 보다 먼저
from .evaluate import ledger_line, metrics, save_pred

import baseline_lgbm as B  # noqa: E402

EXPECTED = {77: (38.9, 68.1), 73: (34.0, 63.2)}
TOL = 0.3


def main() -> int:
    tr, va = C.load("train"), C.load("valid")
    print(f"학습 {len(tr):,}두 / 평가 {len(va):,}두 · {va['race_id'].nunique():,}경주 [valid]\n")

    scores, ok = {}, True
    for n_feat, exclude_pop in ((77, False), (73, True)):
        t0 = time.time()
        _, s, cols = B.train_and_score(tr, va, exclude_pop)
        assert len(cols) == n_feat, (len(cols), n_feat)
        name = f"LightGBM {n_feat}피처" + (" (인기도 제외)" if exclude_pop else "")
        scores[name] = s
        m = metrics(va, s)
        e1, e3 = EXPECTED[n_feat]
        d1, d3 = m["top1"] - e1, m["top3"] - e3
        good = abs(d1) <= TOL and abs(d3) <= TOL
        ok &= good
        print(f"[{name}] {time.time()-t0:.0f}s  top1 {m['top1']:.2f} ({d1:+.2f})  top3 {m['top3']:.2f} ({d3:+.2f})  "
              f"logloss {m['logloss']:.4f}  ECE {m['ece']:.4f}  {'OK' if good else 'MISMATCH'}")
        print("  " + str(save_pred(va, s, f"lgb_{n_feat}")))
        print("  " + ledger_line(m, f"LightGBM lambdarank 200r (재현)", n_feat, seed=0,
                                 memo="팀 baseline_lgbm.py 그대로, 예측 experiments/pred/lgb_%d" % n_feat))

    print()
    print(C.report(scores, va))
    if not ok:
        print("\n팀 표와 ±0.3 밖 — 데이터 사본 확인 후 다시 돌릴 것", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
