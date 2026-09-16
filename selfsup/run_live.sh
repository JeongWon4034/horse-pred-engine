#!/usr/bin/env bash
# 발주 직전 한 방: 오늘 행 갱신 → 마체중 재빌드 → 팀 빌더 → 예측.  bash selfsup/run_live.sh 20260911 3
set -euo pipefail
cd "$(dirname "$0")/.."
DAY=$1; MEET=$2
R="PYTHONUTF8=1 uv run --project pipeline python"
eval $R -m selfsup.live refresh $DAY
eval $R -m selfsup.bodyweight build
eval $R -m selfsup.training build
(cd pipeline && TEAM_REPO=/c/git/S15P21A304 bash build_live.sh > data/build_live_log.txt 2>&1)
eval $R -m selfsup.live predict $DAY $MEET
