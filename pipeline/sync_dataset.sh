#!/usr/bin/env bash
# 팀 레포(develop)에서 데이터셋과 모델 비교 하네스를 다시 가져온다.
# 팀원이 데이터나 docs/model 을 갱신했을 때 이것만 돌리면 된다.
#
#   data/dataset/   ← docs/dataset  (parquet + schema_v2.py 등 빌드 코드)
#   data/model/     ← docs/model    (common.py, baseline_lgbm.py, model_dl_template.py)
#   data/DATA_COMMIT ← 팀 develop 의 짧은 해시. 실험 장부에 적을 기준 커밋이다.
#
# common.py 는 자기 옆 폴더(../dataset)에서 schema_v2 를 찾으므로
# 팀 레포와 같은 상대 위치에 두면 수정 없이 그대로 동작한다.
set -euo pipefail

TEAM="${TEAM_REPO:-/c/Users/SSAFY/Desktop/S15P21A304}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/data"

echo "→ 팀 레포 fetch: $TEAM"
git -C "$TEAM" fetch origin develop --quiet

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "→ docs/dataset, docs/model 추출"
git -C "$TEAM" archive origin/develop docs/dataset docs/model | tar -x -C "$TMP"

mkdir -p "$DST"
rm -rf "$DST/dataset" "$DST/model"
mv "$TMP/docs/dataset" "$DST/dataset"
mv "$TMP/docs/model"   "$DST/model"

# 기준 커밋을 파일로 남긴다 — 장부의 "데이터 기준 커밋" 칸이 이걸 읽는다.
git -C "$TEAM" rev-parse --short origin/develop > "$DST/DATA_COMMIT"

echo "→ 완료"
du -sh "$DST/dataset" "$DST/model"
git -C "$TEAM" log origin/develop -1 --format='   기준 커밋: %h %s'
