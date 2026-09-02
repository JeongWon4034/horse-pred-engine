#!/usr/bin/env bash
# 사이드 실험 — S3 구조·학습 하이퍼파라미터 탐색. 구성마다 seed 2개(20260901, 1), 73피처, L=20.
# 결과는 experiments/runs/logs/side_<시각>/ 에 남고, 요약은 side_summary.txt.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1
LOG="../experiments/runs/logs/side_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG"

declare -a CFG=(
  "base|"
  "gru32|--gru 32"
  "gru128|--gru 128"
  "drop01|--dropout 0.1"
  "drop035|--dropout 0.35"
  "emb8|--emb 8"
  "emb32|--emb 32"
  "hid256|--hidden 256"
  "lr5e-4|--lr 5e-4"
  "lr2e-3|--lr 2e-3"
  "wd1e-3|--wd 1e-3"
  "wd5e-2|--wd 5e-2"
  "bs128|--bs 128"
  "bs512|--bs 512"
  "topk3|--topk 3"
  "rare20|--min-count 20"
  "L5|--hist-len 5"
  "big|--gru 128 --hidden 256 --dropout 0.3"
)
for c in "${CFG[@]}"; do
  name="${c%%|*}"; args="${c#*|}"
  for sd in 20260901 1; do
    echo "[$(date +%H:%M:%S)] ▶ $name seed $sd"
    uv run python -m model.side_train --name "$name" --seed "$sd" $args > "$LOG/${name}_s$sd.log" 2>&1
    grep -E "최고\]" "$LOG/${name}_s$sd.log" | sed "s/^/    /"
  done
done
{
  echo "구성 | seed | top-1 | top-3 | logloss | ECE | ep"
  for f in "$LOG"/*.log; do
    b="$(basename "$f" .log)"
    grep -E "최고\]" "$f" | sed -E "s/.*ep([0-9]+)  top1 ([0-9.]+)  top3 ([0-9.]+)  logloss ([0-9.]+)  ECE ([0-9.]+).*/$b | \2 | \3 | \4 | \5 | \1/"
  done
} > "$LOG/side_summary.txt"
echo "끝 → $LOG/side_summary.txt"
