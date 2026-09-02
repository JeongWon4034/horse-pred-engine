#!/usr/bin/env bash
# 딥러닝 단계 전체를 사람 없이 끝까지 돌린다.
#
#   cd pipeline && bash run_stages.sh              # 기준선 재현 → S1 → S2 → S3 → S4 → 분해표
#   bash run_stages.sh S3 S4                        # 일부 단계만
#   SEEDS="20260901 1 2" bash run_stages.sh S3      # seed 여러 개 (73피처만 반복)
#
# 로그는 experiments/runs/logs/<날짜시각>/ 에 단계별로 남고, 마지막에 장부 형식 줄만 모아 ledger_lines.md 로 뽑는다.
# 예측·가중치는 model.train 이 experiments/pred, experiments/runs 에 저장한다 (gitignore).
# 데이터·하네스가 바뀌었으면 먼저 bash sync_dataset.sh (기준 커밋을 장부에 적을 것).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1

STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(BASE S1 S2 S3 S4 REPORT)
SEEDS="${SEEDS:-20260901}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="../experiments/runs/logs/$TS"; mkdir -p "$LOG"
declare -A KIND=([S1]=linear [S2]=embed [S3]=history [S4]=attn)

run() {  # run <이름> <명령...>  — 실패해도 다음 단계로 간다. 종료 코드는 로그 끝에.
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] ▶ $name: $*"
  "$@" > "$LOG/$name.log" 2>&1; local rc=$?
  echo "[exit $rc]" >> "$LOG/$name.log"
  [ $rc -eq 0 ] && grep -E "최고\]|\[DL − LGB\]|OK|MISMATCH" "$LOG/$name.log" | sed 's/^/    /' \
                 || echo "    ✗ 실패 (exit $rc) — $LOG/$name.log"
}

for st in "${STAGES[@]}"; do
  case "$st" in
    BASE)   run baseline uv run python -m model.baseline
            run backtest uv run python -m model.backtest ;;
    S1|S2|S3|S4)
            k="${KIND[$st]}"
            run "${st}_73" uv run python -m model.train "$k"
            run "${st}_77" uv run python -m model.train "$k" --market
            for sd in $SEEDS; do [ "$sd" = 20260901 ] && continue
              run "${st}_73_s$sd" uv run python -m model.train "$k" --seed "$sd" --tag "_s$sd"; done
            [ "$st" = S3 ] && run S3_73_noF1 uv run python -m model.train history --drop-group F1 --tag _noF1
            [ "$st" = S4 ] && run S4_73_top3 uv run python -m model.train attn --topk 3 --tag _top3 ;;
    REPORT) run breakdown uv run python -m model.breakdown lgb_73 linear_73 embed_73 history_73 attn_73 --by entropy dist field
            cat "$LOG/breakdown.log" ;;
    *)      echo "모르는 단계: $st (BASE S1 S2 S3 S4 REPORT)";;
  esac
done

grep -h "^| 20" "$LOG"/*.log > "$LOG/ledger_lines.md" 2>/dev/null
echo
echo "끝. 로그: $LOG   장부 줄: $LOG/ledger_lines.md  ($(wc -l < "$LOG/ledger_lines.md") 줄 — experiments/ledger.md 에 붙일 것)"
