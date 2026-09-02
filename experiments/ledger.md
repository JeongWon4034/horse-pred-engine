# 실험 장부

한 실험 = 한 줄. 같은 자(팀 하네스 `data/model/common.py` → `C.report()`)로 잰 것만 적는다.
비교는 valid. test 는 이미 1회 개봉됨(팀 `docs/model/README.md` §3) — 다시 열지 않는다.

- 데이터 기준 커밋: `f17ca36` (팀 origin/develop, schema v2.2.0, 2026-09-02) — `bash pipeline/sync_dataset.sh` 출력
- seed: 20260901 (LightGBM 재현은 팀 값 seed=0 그대로)
- 장치: RTX 4070 Laptop 8GB, torch 2.14.0+cu130. DL 은 `CUBLAS_WORKSPACE_CONFIG=:4096:8` + deterministic 모드
- 콘솔: `PYTHONUTF8=1` (Windows cp949 에서 `C.report()` 출력이 깨진다)
- 전처리 조건: `C.load()` 는 `usable()`(배당 없는 경주 제외)만 적용하고 `schema_v2.clean()` 은 부르지 않는다. 그래서 `X_grade` 41레벨·`F2_sire_id` 의 `'-'` 가 그대로다. LightGBM 과 같은 조건을 유지하려고 DL 쪽도 clean 을 끼워 넣지 않는다 (임베딩 사전에서 `'-'`·결측 → `<unk>` 로만 흡수)
- logloss / ECE / Brier: 행 점수를 경주 내 softmax(온도 1)로 확률화해 계산. 시장 행의 값은 `-mkt_rank` 를 점수로 쓴 것이라 척도 의미가 없어 적지 않는다
- epoch 선택: valid top-1 최고 epoch (LightGBM 의 200라운드도 valid 로 정한 값 — 조건 대칭). valid 숫자는 그만큼 낙관적이다
- 결정성 기준: S1(선형)은 같은 명령 2회 top-1 차 < 0.1%p. S2 이후는 임베딩 backward 가 비결정이라 seed 3개 평균±sd 를 단계당 1회 적는다
- game ROI: `model.backtest` — 시장(인기 1위마) 단승 1단위 ROI **−20.7%** (3,687경주, 적중 37.3%, 적중 평균배당 2.12). 팀 EDA −21.4% 와 일치. 모델 ROI 는 S5 에서

## 넘어야 할 선 (팀 측정, valid 1,254경주, top-1 SE ±0.85%p)

| 모델 | 피처 | top-1 | top-3 |
|---|---|---:|---:|
| 시장 (인기 1위마) | — | 39.3 | 69.3 |
| LightGBM lambdarank 200r | 77 (인기도 포함) | 38.9 | 68.1 |
| LightGBM lambdarank 200r | 73 (인기도 제외) | 34.0 | 63.2 |
| 무작위 | — | 10.2 | 30.8 |

"이겼다"는 73피처 valid top-1 ≥ 35.7 (34.0 + 2×SE) 일 때만.

## 기록

| 날짜 | 커밋 | 데이터 | 모델 | 피처 | seed | top-1 | top-3 | logloss | ECE | game ROI | 메모 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-09-02 | — | f17ca36 | 시장 (하네스 재현) | — | — | 39.3 | 69.3 | — | — | — | `C.report({}, valid)` 가 팀 표와 일치 |
| 2026-09-02 | — | f17ca36 | LightGBM lambdarank 200r (재현) | 77 | 0 | 38.9 | 68.1 | 1.7417 | 0.0092 | — | 팀 baseline_lgbm.py 그대로, 팀 표와 +0.02/+0.00. 예측 experiments/pred/lgb_77 |
| 2026-09-02 | — | f17ca36 | LightGBM lambdarank 200r (재현) | 73 | 0 | 34.0 | 63.2 | 1.8846 | 0.0156 | — | 팀 표와 −0.03/+0.04. 예측 experiments/pred/lgb_73 — DL 짝 비교 기준 |
| 2026-09-02 | f73df97 | f17ca36 | DL S1 linear | 73 | 20260901 | 33.6 | 64.0 | 1.9016 | 0.0069 | — | 수치 z-score+원핫 5개(입력 153차원), PL topk=1, ep6/12, cuda 11s. LGB 대비 −0.40%p CI [−2.07, +1.28] → 차이 없음. 재실행 2회 완전 동일 |
| 2026-09-02 | f73df97 | f17ca36 | DL S1 linear | 77 | 20260901 | 39.6 | 69.6 | 1.7621 | 0.0116 | — | 입력 157차원, ep4/12. LGB 대비 +0.64%p CI [−0.88, +2.07] → 차이 없음. 시장 39.3 과도 SE 안 |
| 2026-09-02 | 3db4df2 | f17ca36 | DL S2 embed | 73 | 20260901 | 34.0 | 64.0 | 1.8853 | 0.0064 | — | jkNo·trNo·F2_sire_id 임베딩 16d(어휘 389/242/937, <5회→rare, valid sire unk 14.4%) + MLP 128-64, dropout 0.2, wd 1e-2, ep10/20. LGB 대비 +0.00%p CI [−1.91, +1.83] → 차이 없음. seed 3개(20260901/1/2) top-1 33.97/33.41/34.29 = 33.9±0.4. 팀 README 의 "범주형 native 악화(29.2)"는 임베딩에서 재현 안 됨 |
| 2026-09-02 | 3db4df2 | f17ca36 | DL S2 embed | 77 | 20260901 | 39.7 | 69.8 | 1.7435 | 0.0089 | — | ep3/20 (이후 과적합). LGB 대비 +0.80%p CI [−0.72, +2.31] → 차이 없음 |
| 2026-09-02 | 58963e8 | f17ca36 | DL S3 history | 73 | 20260901 | 35.0 | 64.0 | 1.8639 | 0.0060 | — | S2 + 말별 직전 10출전 GRU(64) 이력 10항목, 파라미터 116k, ep8/20 조기중단. 이력 풀 = clean(train∪valid) usable 전, game·test 제외(assert). 평균 이력 6.6/10, 이력 0개 7.4%(game 구멍 포함). LGB 대비 +1.04%p CI [−0.96, +3.11] → 차이 없음. seed 3개 35.01/34.21/34.29 = 34.5±0.4. **logloss 는 3 seed 모두 LGB 1.8846 보다 낮음(1.854~1.877)** |
| 2026-09-02 | 58963e8 | f17ca36 | DL S3 history ablation −F1 | 56 | 20260901 | 35.1 | 63.5 | 1.8585 | 0.0044 | — | F1 집계 피처 17개 전부 제거, GRU 이력만. top-1·logloss 가 S3 본체와 같음 → 시퀀스가 평균 뭉개기(F1_ord_avg3 등)를 대체 가능. 단 F1 은 game 포함 계산, GRU 이력은 game 제외라 비대칭 |
| 2026-09-02 | 58963e8 | f17ca36 | DL S3 history | 77 | 20260901 | 39.4 | 68.8 | 1.7332 | 0.0044 | — | ep8/20. LGB 대비 +0.48%p CI [−1.04, +2.00] → 차이 없음. logloss 1.7332 < LGB 1.7417 |
