#!/usr/bin/env bash
# 실경기 예측용 데이터셋 빌드 — 팀 빌더를 우리 원장으로 돌린다.
#
#   cd pipeline && bash build_live.sh
#
# 하는 일
#   1. 팀 레포(origin/develop)의 빌더 코드를 data/build/ 로 복사
#   2. 우리 원장(ingest.ledger 산출물)을 팀 빌더가 기대하는 이름·형식으로 배치
#   3. 혈통 JSON → aux_pedigree.csv
#   4. 빌더에 패치 2개 적용 (아래 §패치)
#   5. build_v2.py model 실행 → data/build/dataset/v2/model/{train,valid,test,game,new}.parquet
#
# 패치 2개 — 둘 다 "팀 원본과 조건을 맞추기 위한 것"이지 개선이 아니다
#   ① 분할 경계를 날짜로 고정
#      팀 빌더는 "최신 날짜 기준 상대 구간"으로 나눈다. 우리 원장은 팀보다 최신
#      데이터가 더 있어서 그대로 두면 경계가 일주일씩 밀리고, 예측 대상인 9월
#      경주가 test 안으로 빨려든다. 팀 parquet 과 같은 날짜로 못 박고 그 뒤 경주는
#      "new" 로 따로 뺀다.
#   ② 등급 정규화 해제
#      팀이 커밋한 parquet 은 정규화 전 표기('국5'/'국5등급' 분리)이고 우리 모델도
#      그 조건으로 학습됐다. 여기서 통합하면 원핫 어휘가 어긋나 학습 때와 다른
#      입력이 된다. (팀은 common.load() 의 clean() 에서 나중에 통합한다)
#
# 검증 (2026-09-07, exp/live-race)
#   행 수: train 385,514 / valid 13,252 / test 7,894 — 팀 parquet 과 정확히 일치
#   피처: 공통 65개 중 59개 완전 일치. 나머지는 조교 2칸(API 미승인)·
#         혈통 3칸(현역마만 커버)·주행습성 1칸(원인 미상)
set -euo pipefail

TEAM="${TEAM_REPO:-$HOME/Desktop/S15P21A304}"
B=data/build

[ -f data/raw/ledger_api.csv ] || { echo "원장이 없다. 먼저: uv run python -m ingest.ledger"; exit 1; }
[ -d "$TEAM/.git" ] || { echo "팀 레포를 못 찾았다: $TEAM  (TEAM_REPO=경로 로 지정)"; exit 1; }

mkdir -p "$B/data/raw" "$B/dataset/v2/model" "$B/dataset/v2/game" "$B/dataset/v2/sim"
for f in schema_v2.py aux_join.py ebv_join.py build_v2.py validate_v2.py; do
  git -C "$TEAM" show origin/develop:docs/dataset/$f > "$B/$f"
done
git -C "$TEAM" show origin/develop:docs/dataset/game_holdout.json > "$B/dataset/v2/game_holdout.json"

# 경마장 표기: 우리 원장 '부경' → 팀 코드가 아는 '부산경남'
PYTHONUTF8=1 uv run python - <<'PY'
import pandas as pd
d = pd.read_csv("data/raw/ledger_api.csv", dtype=str, low_memory=False)
d["meet"] = d["meet"].replace({"부경": "부산경남"})
d.to_csv("data/build/data/raw/ledger_2010_2026.csv", index=False, encoding="utf-8-sig")
print(f"원장 {len(d):,}행 배치")
PY

# 혈통: API8_2 벌크 → 팀 빌더가 읽는 aux_pedigree.csv
if [ -f data/raw/pedigree_api.json ]; then
PYTHONUTF8=1 uv run python - <<'PY'
import json, csv, io
ped = json.load(open("data/raw/pedigree_api.json", encoding="utf-8"))
with io.open("data/build/data/raw/aux_pedigree.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["hrNo","faHrNo","faHrName","moHrNo","moHrName",
                                      "damsireNo","damsireName","src"])
    w.writeheader()
    for r in ped:
        hn = str(r.get("hrNo","")).strip()
        if hn:
            w.writerow({"hrNo":hn, "faHrNo":str(r.get("faHrNo","")).strip(),
                        "faHrName":str(r.get("faHrName","")).strip(),
                        "moHrNo":str(r.get("moHrNo","")).strip(),
                        "moHrName":str(r.get("moHrName","")).strip(),
                        "damsireNo":"", "damsireName":"", "src":"API8_2"})
print(f"혈통 {len(ped):,}두")
PY
else
  echo "혈통 JSON 없음 — F2_sire_* 는 결측으로 간다"
fi

# 패치 2개
PYTHONUTF8=1 uv run python - <<'PY'
import io
p = "data/build/build_v2.py"
s = io.open(p, encoding="utf-8").read()

old = '''    rest = df[~is_game]
    last = int(rest["_ordinal"].max())
    t0 = last - S.TEST_WEEKS * 7
    v0 = t0 - S.VALID_WEEKS * 7
    df.loc[~is_game & (df["_ordinal"] > t0), "split"] = "test"
    df.loc[~is_game & (df["_ordinal"] > v0) & (df["_ordinal"] <= t0), "split"] = "valid"'''
new = '''    # [패치①] 분할 경계를 팀 parquet 과 같은 날짜로 고정. 그 뒤 경주는 "new".
    d = df["rcDate"].astype("Int64")
    df.loc[~is_game & (d > 20251109) & (d <= 20260510), "split"] = "valid"
    df.loc[~is_game & (d > 20260510) & (d <= 20260830), "split"] = "test"
    df.loc[~is_game & (d > 20260830), "split"] = "new"'''
assert old in s, "패치① 대상을 못 찾았다 — 팀 빌더가 바뀌었나"
s = s.replace(old, new)

old2 = '    df["rank"] = df["rank"].map(S.normalize_grade)'
new2 = '    # [패치②] 등급 정규화 해제 — 모델이 학습된 조건(정규화 전)과 맞춘다\n' \
       '    # df["rank"] = df["rank"].map(S.normalize_grade)'
assert old2 in s, "패치② 대상을 못 찾았다"
s = s.replace(old2, new2)

s = s.replace('    for sp in ("train", "valid", "test", "game"):',
              '    for sp in ("train", "valid", "test", "game", "new"):')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("패치 2개 적용")
PY

cd "$B" && PYTHONUTF8=1 uv run --project ../.. python build_v2.py model
