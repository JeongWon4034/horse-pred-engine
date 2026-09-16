#!/usr/bin/env bash
# P0 대조군 5 seed → P1 마스크 사전학습(ext) → P2 대조 사전학습(ext) → 각각 미세조정 5 seed + 선형 탐침 1회
set -euo pipefail
cd "$(dirname "$0")/.."
R="PYTHONUTF8=1 uv run --project pipeline python -m selfsup.run"
for s in 1 2 3 4 5; do eval $R finetune --seed $s --tag p0; done
eval $R pretrain --obj mask  --pool ext --out mask_ext.pt
eval $R pretrain --obj scarf --pool ext --out scarf_ext.pt
for o in mask scarf; do
  eval $R finetune --init ${o}_ext.pt --freeze --seed 1 --epochs 8 --tag ${o}_ext_probe
  for s in 1 2 3 4 5; do eval $R finetune --init ${o}_ext.pt --seed $s --tag ${o}_ext_ft; done
done
eval $R summary
