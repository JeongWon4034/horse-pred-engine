#!/usr/bin/env bash
# P4 손피처 대체 — F1 집계 17개 제거(56피처). 사전학습 있음/없음 비교
set -euo pipefail
cd "$(dirname "$0")/.."
R="PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run"
for s in 1 2 3 4 5; do eval $R finetune --drop-group F1 --seed $s --tag p0_noF1; done
eval $R pretrain --obj mask --pool base --drop-group F1 --out mask_base_noF1.pt
for s in 1 2 3 4 5; do eval $R finetune --init mask_base_noF1.pt --drop-group F1 --seed $s --tag mask_base_noF1_ft; done
eval $R summary
