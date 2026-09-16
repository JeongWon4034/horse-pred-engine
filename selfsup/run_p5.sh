#!/usr/bin/env bash
# P5-1 하이브리드 — 사전학습 몸통 ⊕ 이력 GRU vs 랜덤 몸통 ⊕ 이력 GRU
set -euo pipefail
cd "$(dirname "$0")/.."
R="PYTHONUTF8=1 uv run --project pipeline python -m selfsup.hybrid"
for s in 1 2 3 4 5; do eval $R --seed $s --tag hybrid_rand; done
for s in 1 2 3 4 5; do eval $R --init mask_base.pt --seed $s --tag hybrid_pre; done
PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run summary
