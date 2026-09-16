#!/usr/bin/env bash
# P3 통제 — 무엇이 기여했나
#  a) 2010~ 풀 마스크 사전학습 → 2010~ 미세조정      (사전학습 자체)
#  c) 사전학습 없이 2004~ 라벨로 지도학습             (옛 데이터 자체)
set -euo pipefail
cd "$(dirname "$0")/.."
R="PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run"
eval $R pretrain --obj mask --pool base --out mask_base.pt
for s in 1 2 3 4 5; do eval $R finetune --init mask_base.pt --seed $s --tag mask_base_ft; done
for s in 1 2 3 4 5; do eval $R finetune --pool ext --seed $s --tag p0_ext; done
eval $R summary
