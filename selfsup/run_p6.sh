#!/usr/bin/env bash
# P6 페이스 곡선 — 정원 이력 10열 + 페이스 5열 (대조군 hybrid_rand 는 P5 에서 이미 5 seed)
set -euo pipefail
cd "$(dirname "$0")/.."
R="PYTHONUTF8=1 uv run --project pipeline python"
eval $R -m selfsup.pace build
for s in 1 2 3 4 5; do eval $R -m selfsup.hybrid --pace --seed $s --tag hybrid_rand_pace; done
eval $R -m selfsup.run summary
eval $R -m selfsup.run compare --a hybrid_rand_pace --b hybrid_rand
eval $R -m selfsup.run compare --a hybrid_rand_pace --b lgb_73
