# selfsup/ — 라벨 없는 사전학습 · 새 데이터 · 출발 전 예측 (김민지)

`pipeline/`(정원)이 "어떤 구조가 기여하는가"를 재는 곳이라면, 이 폴더는 **학습 신호·새 데이터·서빙**을 시험하는 곳이다.
`pipeline/` 코드는 고치지 않고 임포트만 한다 (인코더·경주 패딩·PL 손실·채점기).

| 문서 | 내용 |
|---|---|
| `plan.md` | 계획 (P0~P5) |
| `ledger.md` | **실험 장부 전부** — 12개 실험, 9경주 출발 전 예측, 채점기 SE 정정 |
| `references.md` | 참고문헌과 우리 결과의 대응 |

---

## 0. 결과 한 눈에 (valid 1,254경주 · 73피처 · LGB 기준선 33.49)

| 시도 | 데이터 | top-1 | 판정 |
|---|---|---:|---|
| **마스크 사전학습 → 미세조정** (P1~P3) | train 38만 행, 라벨 없이 | **34.77** (+1.36 [+0.24, +2.55] vs 대조군) | **팀 첫 유의 차이. S3 와 같은 자리** |
| 2004~2009 추가 (P3) | 2004ext | 차이 없음 | 데이터 양은 답이 아니다 |
| 페이스 곡선 · 마체중 · 장구 · 과거 인기도 · 조교 (P6~P10) | 원장 · 조교 API 644만 행 | 전부 차이 없음 (마체중·인기도는 logloss −0.01) | 공개 데이터로 1등 적중은 더 못 올린다 |
| 최근 3년 ×3 가중 (P11) | 가중치만 | +0.88 [−0.08, +1.91] | 경계선 |
| S3 앙상블 · 하이브리드 · 스태킹 (P5, P12) | | 차이 없음 / −0.96 | 1:1:1 평균 유지 |
| **출발 전 실경기 예측** (09-11, 9경주) | API 출마표 | 1등 2/9 · 복승 6/9 | 팀 첫 출발 전 예측 |

---

## 1. 맥북(또는 새 PC)에서 처음부터

### ① 클론 · 환경

```bash
git clone https://github.com/JeongWon4034/horse-pred-engine.git && cd horse-pred-engine
git checkout exp/ssl
git clone https://lab.ssafy.com/s15-blockchain-pred-sub1/S15P21A304.git ~/S15P21A304   # 팀 GitLab (데이터 원본)
cd pipeline
uv sync                    # 윈도우/리눅스 (CUDA 13 torch)
uv sync --no-sources       # ★ 맥: pyproject 의 torch CUDA 인덱스를 무시하고 PyPI torch(CPU/MPS). uv.lock 이 바뀌면 커밋하지 말 것
```

이 폴더의 모든 명령은 **레포 루트에서** `uv run --project pipeline python -m selfsup.<모듈>` 로 돈다. 맥은 `PYTHONUTF8=1` 이 필요 없다.

### ② 팀 데이터 사본 (깃에 없음)

```bash
cd pipeline && TEAM_REPO=~/S15P21A304 bash sync_dataset.sh       # docs/dataset, docs/model → pipeline/data/
cd .. && TEAM_REPO=~/S15P21A304 bash selfsup/sync_ext.sh         # docs/dataset-2004ext → pipeline/data/dataset-2004ext
uv run --project pipeline python -c "from model.team import C; print(C.report({}, C.load('valid')))"   # 시장 39.3 / 69.3 이면 정상
```

### ③ 원장·조교·산출물 — 릴리스에서 받기 (재수집 대신)

```bash
gh release download selfsup-data-20260911 -D /tmp/rel          # 4개 파일, 약 190MB
gunzip -c /tmp/rel/ledger_api.csv.gz   > pipeline/data/raw/ledger_api.csv      # 경주성적 원장 2010~2026-09-11 (457,202행)
cp pipeline/data/raw/ledger_api.csv pipeline/data/raw/ledger_api_orig.csv      # live.py prep 이 원본 사본을 본다
gunzip -c /tmp/rel/training_api.csv.gz > pipeline/data/raw/training_api.csv    # 조교 2010~2026 (6,440,172행)
tar xzf /tmp/rel/selfsup_runs.tgz -C selfsup                                    # 파생 parquet · ckpt · results.jsonl · 09-11 예측
tar xzf /tmp/rel/experiments_pred_runs.tgz -C experiments                       # LGB·S3 valid 예측, S3 ckpt, selfsup 예측
```

릴리스가 아직 없으면(윈도우 노트북의 `selfsup/runs/bundle/` 에 파일이 있다) 레포 소유 계정으로 한 번 올린다:
```bash
gh release create selfsup-data-20260911 --target exp/ssl --prerelease \
  --title "selfsup 데이터 묶음 (2026-09-11)" --notes "exp/ssl 재현용. 전부 한국마사회 공공데이터(이용허락 제한 없음) 또는 파생물" \
  selfsup/runs/bundle/ledger_api.csv.gz selfsup/runs/bundle/training_api.csv.gz \
  selfsup/runs/bundle/selfsup_runs.tgz selfsup/runs/bundle/experiments_pred_runs.tgz
```

직접 다시 받으려면 (공공데이터포털 키 필요 — `pipeline/.env` 에 `KRA_API_KEY_ENCODED=키` 한 줄. **키는 커밋·채팅에 남기지 말 것**):
```bash
cd pipeline && uv run python -m ingest.ledger 2010 2026        # 경주성적 (15063979 활용신청) — 7분
cd .. && uv run --project pipeline python -m selfsup.training_fetch   # 조교 (15058782 활용신청) — 70분
```

---

## 2. 실험 재현

```bash
R="uv run --project pipeline python -m"
$R selfsup.run finetune --seed 1 --tag p0                          # P0 대조군 (사전학습 없음), 40초
$R selfsup.run pretrain --obj mask --pool base --out mask_base.pt  # P3-a 마스크 사전학습, 6분 (GPU) / 15분 (CPU)
$R selfsup.run finetune --init mask_base.pt --seed 1               # 미세조정
$R selfsup.run summary                                             # tag 별 seed 평균
$R selfsup.run compare --a mask_base_ft --b lgb_73                 # paired bootstrap 95% CI
bash selfsup/run_p0_p2.sh / run_p3.sh / run_p4.sh / run_p5.sh / run_p6.sh   # 단계별 일괄

$R selfsup.bodyweight eval   # P7 마체중        $R selfsup.gear eval      # P8 장구
$R selfsup.pastodds eval     # P9 과거 인기도    $R selfsup.training eval  # P10 조교
$R selfsup.recency           # P11 최근성 가중   $R selfsup.stack          # P12 스태킹
$R selfsup.ensemble          # P5-2 S3 앙상블    $R selfsup.hybrid --init mask_base.pt --seed 1   # P5-1
```

`eval` 류는 원장에서 파생 parquet 을 먼저 만든다 (`… build`). 릴리스 묶음을 풀었으면 이미 있다.

## 3. 출발 전 예측 (주말)

경주성적 API(API214_1)는 **미출발 경주의 출마표 행**(착순 빈칸)을 준다. 그걸로 경주 전에 예측한다.

```bash
bash selfsup/run_live.sh 20260913 1     # 서울   (meet 1 서울 · 2 제주 · 3 부산경남)
bash selfsup/run_live.sh 20260913 3     # 부경
```
한 줄이 오늘 행 재수집(날씨·마체중·끝난 경주 반영) → 마체중·조교 파생 → 팀 빌더 → S3·사전학습·LGB → 1:1:1 평균 → `selfsup/runs/live/pred_날짜_m경마장_HHMM.md` 를 만든다 (약 4~5분, 경주 10분 전에 돌린다).
출발 시각은 `race.kra.co.kr` 메인 표. 결과는 출발 8~13분 뒤 같은 API에 오른다.

**LGB 입력**: 73 + 마체중 3 + 조교 8, 최근 3년 ×3 가중. **S3**: 정원 `history_73_L20_s*` ckpt 5개 (이력 풀은 실시간 원장 빌드). **사전학습**: `mask_base.pt` 에서 그 자리 미세조정.

## 4. 규칙 (docs/10-팀원-시작하기.md 와 같다)

- game 은 학습·검증·사전학습 풀 어디에도 안 넣는다. test 는 열지 않는다. 비교는 valid, seed ≥ 3, paired bootstrap CI 가 0 을 품으면 "차이 없음"
- 채점은 `model.evaluate` (= 팀 `C.evaluate` + 정원 logloss·CI). 자체 채점기를 만들지 않는다
- `report()` 의 "±0.85%p" 는 마지막 행(무작위)의 SE — 34% 모델은 ±1.33%p (장부 참조)
- `.env` · 원장 CSV · parquet · ckpt · 예측 파일은 커밋하지 않는다 (전부 gitignore)
- 실험 1회 = 장부 한 줄. 결과가 나빠도 그대로 적는다
