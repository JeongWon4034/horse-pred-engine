#!/usr/bin/env bash
# 팀 레포(develop)에서 데이터셋만 다시 가져온다.
# 팀원이 데이터를 갱신했을 때 이것만 돌리면 된다.
set -euo pipefail

TEAM="${TEAM_REPO:-/c/Users/SSAFY/Desktop/S15P21A304}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/data/dataset"

echo "→ 팀 레포 fetch: $TEAM"
git -C "$TEAM" fetch origin develop --quiet

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "→ docs/dataset 추출"
git -C "$TEAM" archive origin/develop docs/dataset | tar -x -C "$TMP"

rm -rf "$DST"
mkdir -p "$(dirname "$DST")"
mv "$TMP/docs/dataset" "$DST"

echo "→ 완료"
du -sh "$DST"
git -C "$TEAM" log origin/develop -1 --format='   기준 커밋: %h %s'
