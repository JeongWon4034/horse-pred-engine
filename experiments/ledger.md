# 실험 장부

한 실험 = 한 줄. 같은 자(팀 하네스 `data/model/common.py` → `C.report()`)로 잰 것만 적는다.
비교는 valid. test 는 이미 1회 개봉됨(팀 `docs/model/README.md` §3) — 다시 열지 않는다.

- 데이터 기준 커밋: `f17ca36` (팀 origin/develop, schema v2.2.0, 2026-09-02) — `bash pipeline/sync_dataset.sh` 출력
- seed: 20260901
- 콘솔: `PYTHONUTF8=1` (Windows cp949 에서 `C.report()` 출력이 깨진다)

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
