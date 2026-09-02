# 경마 예측 대결 서비스 — 모델 실험 레포

> **한 줄 정의**
> 경마 예측을 분류가 아니라 **랭킹** 문제로 풀고, 유저가 6개 관점의 비중을 실시간으로 바꿔도 **재학습 없이** 동작하는 예측 대결 서비스.

팀 레포: `S15P21A304` (SSAFY GitLab, gitflow). 여기는 **모델 실험 전용** — 데이터·EDA·서비스는 팀 레포, 코드·실험은 여기서 하고 익은 것만 팀 레포로 옮긴다.

---

## 현재 상태 (2026-09-02)

| 항목 | 상태 |
|---|---|
| 데이터 | ✅ 팀 데이터셋 schema v2.2.0, 기준 커밋 `f17ca36` (train 36,586 / valid 1,254 / test 802 / game 3,700 경주) |
| 기준선 | ✅ 팀 LightGBM(도연) 재현 — valid 77피처 38.9 / 73피처 34.0, 팀 표와 ±0.05 |
| 딥러닝 4단계 | ✅ S1 선형 → S2 임베딩 → S3 전적 GRU → S4 경주 내 attention, 전부 같은 채점기로 측정 |
| 판정 | top-1 은 어느 단계도 LGB 와 통계적 차이 없음(선 35.7 미달). **S3 만 logloss 에서 10회 중 10회 우위.** 최종 후보 = S3 seed 평균 + LGB 앙상블 |
| 사이드 실험 | ✅ game ROI(시기 혼재로 부풀려짐 확인), S3 하이퍼 탐색(기본이 안정점), 앙상블 교차적합(+0.8%p), S5 6축 타워 |
| GPU | RTX 4070 8GB, torch 2.14 cu130. 한 번 학습 20~80초 |

### 결과 한눈에 (valid 1,254경주 · 73피처 = 배당 없는 실전 조건 · top-1 표준오차 ±0.85%p)

| 모델 | top-1 | top-3 | logloss | LGB 대비 top-1 (95% CI) |
|---|---:|---:|---:|---|
| 시장 (인기 1위마, 참고) | 39.3 | 69.3 | — | — |
| **LightGBM 73** (기준) | 34.0 | 63.2 | 1.885 | — |
| S1 linear | 33.6 | 64.0 | 1.902 | −0.4 [−2.1, +1.3] |
| S2 embed | 34.0 | 64.0 | 1.885 | +0.0 [−1.9, +1.8] |
| **S3 history** (L=20, seed 5 평균) | 34.8 | 64.3 | **1.846** | +0.8 [−1.2, +2.8] |
| S4 attn | 34.4 | 63.0 | 1.902 | +0.5 [−1.6, +2.7] |
| LGB + S3 (5:5) | 35.3 | 64.0 | 1.852 | +1.4 [−0.1, +2.8] (valid 로 고른 낙관값, 교차적합 시 +0.8) |

- 딥러닝은 사전학습 모델의 파인튜닝이 아니다. 구조를 직접 정하고 0 에서 이 데이터로만 학습했다.
- test 는 팀이 1회 열었고 다시 열지 않는다. game 은 학습·튜닝에 쓰지 않는다. 말 ID 는 입력에 넣지 않는다.
- 전체 기록: [experiments/ledger.md](experiments/ledger.md) (본 실험) · [experiments/side/ledger_side.md](experiments/side/ledger_side.md) (사이드)

---

## 실행

```bash
cd pipeline
bash sync_dataset.sh                      # 팀 origin/develop 의 docs/dataset · docs/model 사본 갱신 (기준 커밋 출력)
PYTHONUTF8=1 uv run python -c "from model.team import C; print(C.report({}, C.load('valid')))"   # 시장 39.3/69.3 이면 정상

PYTHONUTF8=1 uv run python -m model.baseline           # LightGBM 재현 → experiments/pred/lgb_{73,77}.parquet
PYTHONUTF8=1 uv run python -m model.train history      # S3, 73피처. --market 이면 77. kind ∈ linear|embed|history|attn
PYTHONUTF8=1 uv run python -m model.train history --hist-len 20 --seed 1 --tag _L20_s1
PYTHONUTF8=1 uv run python -m model.breakdown lgb_73 history_73   # 혼전도·등급·거리·두수별 분해 + 짝 비교
bash run_stages.sh                                     # 기준선 → S1 → S2 → S3 → S4 → 분해표, 무인 실행

PYTHONUTF8=1 uv run python -m model.score --split game --lgb --ckpt history_73_L20_s1   # 저장 가중치로 game 점수
PYTHONUTF8=1 uv run python -m model.backtest --pred ../experiments/pred/history_73_L20_s1_game.parquet   # 단승 ROI
```

콘솔은 항상 `PYTHONUTF8=1` (Windows cp949 에서 채점기 출력이 깨진다). 산출물(`experiments/pred`, `experiments/runs`)은 gitignore.

## 구조

```
pipeline/
  data/            팀 사본 (gitignore) — dataset/ (parquet, schema_v2.py), model/ (common.py 채점기, baseline_lgbm.py)
  model/
    team.py        팀 채점기 진입점. 데이터 읽기·피처 선택·채점은 전부 여기(C.load / C.feature_cols / C.report)로만
    data.py        경주 단위 패딩 텐서 [R,16,D] ↔ 행 점수 변환(flatten_scores)
    categorical.py 기수·조교사·부마 임베딩 어휘 (train 에서 fit, <rare>/<unk>)
    history.py     말별 직전 L 출전 시퀀스 [N,L,10]. rcDate < 현재 만, game·test 제외 (assert)
    models.py      LinearRanker · EmbedRanker · HistoryRanker(GRU) · RaceTransformer · TowerRanker(6축)
    losses.py      Plackett-Luce (경주 내 softmax, topk)
    evaluate.py    logloss · ECE · Brier · 경주별 적중 · paired bootstrap · 예측 저장/로드
    train.py       S1~S4 공용 학습 CLI (결정성 설정, best epoch 체크포인트, 장부 줄 출력)
    baseline.py    팀 LightGBM 그대로 재현 + 예측 저장
    backtest.py    game 단승 ROI
    score.py       저장 가중치로 다른 분할 점수 (재학습 없음)
    breakdown.py   예측 분해표
    side_*.py      사이드 실험 (구조 탐색·앙상블 교차적합·S5 타워·ROI 전략)
  run_stages.sh    전 단계 무인 실행
  side_sweep.sh    S3 하이퍼파라미터 탐색
experiments/
  ledger.md        본 실험 장부 — 실험 1회 = 한 줄, 종합표, 판정, 다음 할 일
  side/            사이드 장부
```

## 브랜치

- `main` — 항상 돌아가는 상태
- `feat/dl-stages` — 본 실험 (Phase 0 → S1~S4, 장부). 결과 확정
- `exp/side` — `feat/dl-stages` 위의 사이드 실험. 본 코드 수정 없이 새 파일만. 죽으면 브랜치째 삭제

## 문서

| 파일 | 내용 |
|---|---|
| [01-기획.md](01-기획.md) | 앱 성격, 개인화 논거, 삭제한 기능, 페르소나 |
| [02-모델.md](02-모델.md) | 랭킹 학습, 단계별 구조, 왜 딥러닝인지 |
| [03-데이터.md](03-데이터.md) | 분할 전략, 피처, API 현황 |
| [04-로드맵.md](04-로드맵.md) | 주차별 계획, 인원 배분 |
| [05-DL-실행계획.md](05-DL-실행계획.md) | 딥러닝 4단계 실행 계획 (인수인계 문서) |

## 데이터 규칙 (깨면 비교가 무효)

- 정렬 유지 — `race_id` 연속 블록. 셔플은 **경주끼리만**
- game 3,700경주는 학습·검증·튜닝 금지. ROI 백테스트 전용. **game ROI 는 train 과 시기가 겹쳐 낙관적** — 서비스 표기는 valid 값으로
- test 는 최종 1회(이미 개봉). 비교는 valid
- `winOdds ≥ 900` 은 결측(9999.9 센티널). 채점기의 `usable()` 이 배당 없는 경주를 뺀다
- `C.load` 는 `schema_v2.clean()` 을 호출하지 않는다(등급 표기 미통합, 부마 `'-'` 잔존). LGB 와 조건을 맞추기 위해 DL 쪽도 그대로 둔다

## 열려 있는 것

- S5 6축: 균등 가중 정확도가 LGB 보다 2.3%p 낮음 → 종합 헤드 분리 변형
- 장거리(>1700m, 152경주)에서 S3 열세 — 표본 부족인지 확인 필요
- 더 얻으려면 구조가 아니라 **경주 데이터 양** (valid 확장이 측정 오차를 줄이는 유일한 길)
