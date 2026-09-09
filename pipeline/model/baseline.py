"""LightGBM 기준선 재현 — 팀 baseline_lgbm.py 를 그대로 돌리고 예측만 저장한다.

    PYTHONUTF8=1 uv run python -m model.baseline

팀 표(valid): 77피처 39.1 / 68.6, 73피처 33.5 / 63.3. ±0.3 안에 안 들면 데이터 사본 문제다.
seed=0 · num_threads=4 로 결정적이라 소수 4자리까지 재현된다. 어긋나면 값이 아니라 사본을 의심한다.

기대값은 데이터 커밋 39eeb26(clean() 이 develop 에 머지된 지점) 이후 기준이다.
develop 끝은 서비스 쪽 머지로도 움직이므로, docs/dataset·docs/model 이 그대로면 값은 같다. 그 앞(f17ca36)까지는 77피처 38.9 / 68.1,
73피처 34.0 / 63.2 였는데, 팀 common.py 의 load() 가 usable() 만 부르다가
clean() → usable() 로 바뀌면서 옮겨갔다 (X_grade 표기 통합 41→26레벨,
F2_sire_id 의 '-' → 결측, 배당 9999.9 → NaN). 데이터 파일 자체는 그대로다.
팀 측 seed 3개 평균은 77피처 39.26 (±0.26) / 73피처 33.79 (±0.47) — 아래 값은 이 파이프라인의 seed=0 재현값.
저장한 예측(experiments/pred/lgb_{73,77}.parquet)은 이후 딥러닝과 경주별 짝 비교에 쓴다.
"""
from __future__ import annotations

import sys
import time

from .team import C                # data/model 을 sys.path 에 넣는다 — baseline_lgbm 보다 먼저
from .evaluate import ledger_line, metrics, save_pred

import baseline_lgbm as B  # noqa: E402

EXPECTED = {77: (39.1, 68.6), 73: (33.5, 63.3)}   # 데이터 커밋 39eeb26 · seed=0 재현값
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
