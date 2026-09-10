#!/bin/bash
# 전체 재현 — 학습 → 배포 산출물 → 검정 → 분해표 → 성능 → 허깅페이스 폴더.
# 전부 CPU/MPS 로 돌고 약 25분 걸린다(시드 5개 포함).
set -e
cd "$(dirname "$0")"

# Windows(cp949) 콘솔에서 한글 출력이 깨지는 것을 막는다. 맥·리눅스에서는 무해하다.
export PYTHONUTF8=1
CK77=artifacts/runs/axis_77_s20260901_final.pt
CK73=artifacts/runs/axis_73_s20260901_final.pt

uv run python -m basemodel.train             --epochs 30 --tag _final   # 77 게임 리플레이
uv run python -m basemodel.train --no-market --epochs 30 --tag _final   # 73 주말 실시간
for S in 1 2 3 4 5; do                                                  # 시드 분산 (검정 전제)
  uv run python -m basemodel.train --epochs 30 --seed "$S" --tag "_s$S"
done

uv run python -m basemodel.export --ckpt "$CK77"
uv run python -m basemodel.export --ckpt "$CK73" --out artifacts/export73

uv run python -m basemodel.significance --ckpt "$CK77" | tee artifacts/logs/significance.log
uv run python -m basemodel.breakdown    --ckpt "$CK77"
uv run python -m basemodel.bench        --ckpt "$CK77"
uv run python -m basemodel.hf           --ckpt "$CK77" --verify
