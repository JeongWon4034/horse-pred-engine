#!/bin/bash
# 탈상관 계수 λ 탐색 — 정확도·확률품질과 축 분리의 교환비를 잰다.
# 결과는 README §5 "탈상관 벌점 λ" 표. 12 epoch · 73피처 · 균등 가중 기준.
cd "$(dirname "$0")"

# Windows(cp949) 콘솔에서 한글 출력이 깨지는 것을 막는다. 맥·리눅스에서는 무해하다.
export PYTHONUTF8=1
for L in 0 0.15 0.25 0.5 2.0; do
  echo "=========== decorr=$L ==========="
  uv run python -m basemodel.train --no-market --epochs 12 --decorr "$L" --tag "_d$L" 2>&1 | tail -32
done
