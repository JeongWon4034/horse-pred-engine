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

- 딥러닝은 사전학습 모델의 파인튜닝이 아니다. 구조를 직접 정하고 0 에서 이 데이터로만 학습했다 (아래 §모델 기반).
- test 는 팀이 1회 열었고 다시 열지 않는다. game 은 학습·튜닝에 쓰지 않는다. 말 ID 는 입력에 넣지 않는다.
- 전체 기록: [experiments/ledger.md](experiments/ledger.md) (본 실험) · [experiments/side/ledger_side.md](experiments/side/ledger_side.md) (사이드)

### 모델 기반 — 무엇으로 만들었나

사전학습 가중치는 없다. **부품과 방법론은 전부 기존 것**이고, 우리가 한 건 경마 데이터에 맞춘 입력 설계·조립·검증이다.

| 항목 | 값 |
|---|---|
| 프레임워크 | **PyTorch 2.14** (CUDA 13), 학습 RTX 4070 Laptop 8GB |
| 기준선 | LightGBM 4.7 (팀 코드 그대로) |
| 방법론 | **Benter 조건부 로지스틱 회귀**(1994, 경마 예측 고전) = S1. 손실은 **Plackett–Luce**(1975, 순위 확률 = ListMLE) |
| 최적화 | AdamW + OneCycle, grad clip 1.0, seed 고정·deterministic |

| 단계 | 쌓은 부품 (전부 PyTorch 표준 레이어) | 역할 | 파라미터 |
|---|---|---|---:|
| S1 linear | `nn.Linear` 1개 | 153가중치 곱셈 (조건부 로지스틱) | 153 |
| S2 embed | + `nn.Embedding` 3개 + MLP 3층 | 기수·조교사·부마 ID → 16차원 | 59k |
| S3 history | + `nn.GRU` 1개 | 말별 과거 20출전을 순서대로 읽음 | 116k |
| S4 attn | + `nn.TransformerEncoder` 2층 | 같은 경주 말들이 서로를 봄 | 416k |
| S5 tower | MLP 3층 × 6축 + Dirichlet 랜덤 가중치 | 슬라이더용 6과목 점수 | — |

### 자주 나온 질문 — 답의 핵심

전문은 [06-모델-쉬운설명.md](docs/06-모델-쉬운설명.md).

| 질문 | 답 |
|---|---|
| 허깅페이스 파인튜닝이 낫지 않나 | 이 데이터(말당 숫자 77개 표)엔 옮겨올 지식을 가진 모델이 없다. 지금 모델은 10만~40만 파라미터, 학습 1분 |
| "valid 로 고른 값"이 뭔가 | 모의고사로 고르고 모의고사로 점수 냄. 정직한 값은 35.3 이 아니라 교차적합 34.8 |
| 모델을 쉽게 설명하면 | 성적표 77칸 × 중요도 = 점수, 경주 안에서 높은 순. 중요도는 36,586경주에서 "1등 말 확률 올리기"로 학습됨 |
| 검증이 맞나 | train/valid 날짜 겹침 0, 피처 갱신 9,848쌍 100% 정상. 약점은 valid 3중 사용·DL test 미개봉·±0.85%p 오차 |
| 데이터 품질은 | 결측이 랜덤이 아니라 이유가 셋(2016 이전 육종가 없음·첫 조합·데뷔전). 채우지 않고 "비었음" 플래그 |
| 베이스 모델은 | PyTorch 표준 레이어 + Benter 조건부 로지스틱 + Plackett–Luce + GRU. 사전학습 가중치 없음 |

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
docs/              기획·모델·데이터·로드맵·DL 실행계획·쉬운 설명 (01~06, 아래 §문서)
experiments/
  ledger.md        본 실험 장부 — 실험 1회 = 한 줄, 종합표, 판정, 다음 할 일
  side/            사이드 장부
pipeline/
  sync_dataset.sh  팀 레포 데이터·채점기 가져오기 (기준 커밋 출력)
  ingest/          KRA 공공 API 수집기
  eda/             탐색 도구 + HTML 리포트 생성
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
```

## 브랜치

- `main` — 항상 돌아가는 상태
- `feat/dl-stages` — 본 실험 (Phase 0 → S1~S4, 장부). 결과 확정
- `exp/side` — `feat/dl-stages` 위의 사이드 실험. 본 코드 수정 없이 새 파일만. 죽으면 브랜치째 삭제

## 문서

| 파일 | 내용 |
|---|---|
| [01-기획.md](docs/01-기획.md) | 앱 성격, 개인화 논거, 삭제한 기능, 페르소나 |
| [02-모델.md](docs/02-모델.md) | 랭킹 학습, 단계별 구조, 왜 딥러닝인지 |
| [03-데이터.md](docs/03-데이터.md) | 분할 전략, 피처, API 현황 |
| [04-로드맵.md](docs/04-로드맵.md) | 주차별 계획, 인원 배분 |
| [05-DL-실행계획.md](docs/05-DL-실행계획.md) | 딥러닝 4단계 실행 계획 (인수인계 문서) |
| [09-딥러닝-티가-안난다.md](docs/09-딥러닝-티가-안난다.md) | 컨설턴트 피드백 대응 — 사실 확인, 이미 있는 근거, 채울 실험 4개 |
| [08-모델-작업정리.md](docs/08-모델-작업정리.md) | **작업 정리 — 된 것·안 된 것 12가지, 검증 방법, 팀에 넘길 것, 다음** |
| [07-모델-진행현황.md](docs/07-모델-진행현황.md) | 팀 공유용 현황 — 결과표, 팀에 알릴 것 4가지, 서비스 가능 여부, 다음 |
| [06-모델-쉬운설명.md](docs/06-모델-쉬운설명.md) | 비전공자용 Q&A — 왜 직접 만들었나, 피처와 가중치, 검증, 데이터 품질, 코드 6단계, PyTorch 부품 |

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
