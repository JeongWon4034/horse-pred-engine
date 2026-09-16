#!/usr/bin/env bash
# 팀 레포(develop)의 docs/dataset-2004ext 를 pipeline/data/dataset-2004ext 로 가져온다.
# pipeline/sync_dataset.sh 와 같은 방식. parquet 은 git 에 올리지 않는다 (pipeline/data 는 gitignore).
set -euo pipefail

TEAM="${TEAM_REPO:-/c/Users/SSAFY/Desktop/S15P21A304}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/../pipeline/data/dataset-2004ext"

git -C "$TEAM" fetch origin develop --quiet
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git -C "$TEAM" archive origin/develop docs/dataset-2004ext | tar -x -C "$TMP"
rm -rf "$DST"; mv "$TMP/docs/dataset-2004ext" "$DST"
du -sh "$DST"
git -C "$TEAM" log origin/develop -1 --format='   기준 커밋: %h %s' -- docs/dataset-2004ext
