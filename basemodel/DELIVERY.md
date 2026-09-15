# 인도 명세 — AI 가 무엇을, 어떤 형식으로, 누구에게 준다

`AI-CONTRACT.md` 가 **계산식**이라면 이 문서는 **인터페이스**다.
누가 무엇을 언제 만들지, 파일이 어떤 경로로 건너가는지를 적는다.
발행 2026-09-15 · 담당 이건모

## 0. 결론 — 물건은 셋이다

| # | 무엇 | 형식 | 받는 쪽 | 상태 |
|---|---|---|---|---|
| **1** | 지표 마스터 | `feature_seed.sql` (41행) | 백엔드 | **완성** |
| **2** | 게임풀 점수표 | `entry_feature_score` 행 · SQL/CSV 배치 | 백엔드 | 산출기 완성, 게임풀 미실행 |
| **3** | 실시간 점수표 | `POST /ai/score-tables` | 백엔드 → AI 서버 | **명세만. 라우트 없음** |

**엔드포인트는 하나만 새로 만들면 된다.** 나머지는 파일로 건넌다.

## 1. 왜 대부분을 배치로 하나

제품에 경주가 들어오는 길은 둘이고, 성질이 다르다.

| | 경주 수 | 언제 정해지나 | 그래서 |
|---|---:|---|---|
| **게임풀 (리플레이)** | 3,700 (고정) | 배포 전에 다 안다 | **배치 1회.** 엔드포인트 불필요 |
| **주말 실시간** | 주당 30~50 | 출마표 발표 후 | 엔드포인트 필요 |

게임풀에 HTTP 를 쓰면 3,700번 왕복하고 그 결과를 다시 DB 에 넣는다 — 같은 일을
두 번 한다. 배포 시점에 SQL 한 번이면 끝난다.

> 명세서 §11-1 에 `POST /ai/score-tables/bulk` (`202 + jobId`) 가 있는데,
> **MVP 에서는 안 만든다.** 잡 큐·폴링을 붙일 값이 없다 — 배포 때 한 번 돌리는 일이다.

## 2. 현재 어긋남 — 이걸 먼저 알아야 한다

| 층 | 무엇이 있나 |
|---|---|
| 명세서 §11 | AI 엔드포인트 **10개** 계획 |
| `ai/app/` (도연·박민기) | `/health` `/races/generate` `/predict` `/predict/conditional` `/backtest` `/live/board` `/ai/bots/config` `/ai/bots/bets` |
| BE `HttpAiClient` | `/ai/bots/bets` · `/ai/bots/config` **둘만 호출** |

**`/ai/score-tables` 는 어디에도 없다.** 명세서에 적혀 있을 뿐이다.
그리고 구현된 `/predict` 계열은 명세서 §11 경로와 이름이 다르다 — 정리가 필요하지만
MVP 를 막지는 않는다(BE 가 부르지 않는다).

## 3. 물건 1 — 지표 마스터 (완성)

`feature_group` 6행 + `feature` 41행. `AI-CONTRACT.md` §6 참조.

```sql
-- feature 41행 = ax_* 6 (is_displayed=FALSE, 계산용)
--              + 표시 지표 35 (is_displayed=TRUE, 화면용)
```

**백엔드 할 일** — `ReferenceDataSeeder` 보다 먼저 이 SQL 을 넣는다.
시더는 feature 테이블에 행이 하나라도 있으면 임시 픽스처를 건너뛴다
(`ReferenceDataSeeder.java:179`). 순서만 지키면 코드 수정이 필요 없다.

## 4. 물건 2 — 게임풀 점수표 (배치)

경주당 `출전마 수 × 41` 행.

```sql
INSERT INTO entry_feature_score
  (race_entry_id, feature_id, raw_value, score, raw_prob, model_version, computed_at)
VALUES (…);
```

| 칸 | 값 |
|---|---|
| `score` | 0~100. 축 6칸은 타워 출력의 경주 내 순위 백분위, 표시 35칸은 지표값의 순위 백분위 |
| `raw_value` | 표시 지표의 원본값. **축 6칸은 NULL** |
| `raw_prob` | 균등 가중치일 때 1착 확률. 경주 안에서 합 1.0 |
| `model_version` | `2026-09-15-axis73` |

**아직 안 한 것** — 게임풀 3,700경주 전체 산출. 산출기는 완성돼 valid 로 검증까지 됐고
(`tools/emit_score_table.py`), 게임풀에 돌리는 것만 남았다. **`game` 은 학습·검증에 쓰면
안 되지만 점수 산출은 추론이라 규칙에 걸리지 않는다** — 그래도 돌리기 전에 팀 확인을 받겠다.

## 5. 물건 3 — 실시간 엔드포인트 (명세만)

### `POST /ai/score-tables`

명세서 §11-1 형식 그대로다. 아래는 **실제 산출물에서 뽑은 값**이다.

```jsonc
// 요청 — 결과 필드 금지
{ "raceKey": "SEOUL-20260912-04", "kind": "WEEKLY",
  "distance": 1200, "track": "WET", "weather": "RAIN", "grade": "국5",
  "entries": [
    { "entryId": 9001, "gateNo": 1, "hrNo": "0412345", "jkNo": "080123", "trNo": "070045",
      "burdenWeight": 55.0, "horseWeight": 478, "horseWeightDiff": 2,
      "restDays": 21, "raceDate": "2026-09-12" }
  ] }
```

```jsonc
// 응답
{ "raceKey": "JEJU-20251114-01",
  "modelVersion": "2026-09-15-axis73",
  "featureKeys": [
    "ax_condition","ax_speed","ax_running","ax_jockey","ax_environment","ax_ability",
    "starts_life","win_rate_life","ord_avg3","recent5_avg_rank", "…35개"
  ],
  "scores": [
    { "entryId": "20251114_2_1_1",
      "score":    [30.0, 40.0, 50.0, 40.0, 80.0, 40.0, 80.0, 50.0, "…"],
      "rawValue": [null, null, null, null, null, null, 0.0, null, "…"],
      "rawProb":  0.0457 }
  ],
  "computedAt": "2026-09-15T11:00:00" }
```

- `featureKeys[0:6]` 이 **항상** `ax_*` 여섯이고 이 순서다. 계산은 이것만 쓴다
- `score` 는 `featureKeys` 와 같은 길이·같은 순서. 소수 1자리(`DECIMAL(5,1)`)
- `rawValue` 는 `is_displayed` 인 지표만 채우고 축 6칸은 `null`
- `rawProb` 는 경주 안에서 합 1.0

### 누가 만드나

AI 서버(`ai/app/`)는 도연님 것이다. **내가 라우트를 직접 넣지 않는다.**
대신 **순수 함수 하나**를 제공한다 — 프레임을 받아 위 응답 dict 를 돌려주는 것.
`tools/emit_score_table.py` 가 그 로직이고, 서버에 붙이는 건 라우트 한 줄이다.

```python
# ai/app/main.py 에 추가할 모양
@app.post("/ai/score-tables")
def score_tables(req: ScoreTableRequest):
    return build_score_table(req)        # basemodel 쪽에서 제공
```

### 호출 시점 (명세서 §11-1 그대로)

출마표 발표·정보 변경 시(WEEKLY), 방 생성 시 캐시 미스(POOL).
**스케줄러는 백엔드에만 둔다** — AI 는 요청을 받아 계산만 하고 역호출하지 않는다.

## 6. MVP 에서 **안 만드는 것** — 명세서 §11 의 나머지

| 경로 | 왜 안 하나 |
|---|---|
| `/ai/score-tables/bulk` | 게임풀은 배치로 넣는다(§1). 잡 큐를 붙일 값이 없다 |
| `/ai/score-tables/{raceKey}` GET | 점수는 DB 에 있다. BE 가 AI 에 되물을 이유가 없다 |
| `/ai/models` | 관리자 화면용. `model_version` 문자열로 충분하다 |
| `/ai/features/reference` | `ref_win_rate_bp` 는 시드에 이미 박아 넣었다. 재학습 때 SQL 로 갱신한다 |
| `/ai/backtest-dataset` | 브라우저 백테스트 기능이 MVP 범위인지 미확정 |
| `/ai/replays` `/ai/resimulate` `/ai/advice` `/ai/report` | 게임 리플레이·리포트. **내 담당이 아니다** |

**이 표는 제안이다.** 기획에서 범위가 다르면 알려주면 바꾼다.

## 7. 프런트에 주는 것

**파일은 없다.** 프런트가 받는 건 백엔드 API 응답이고, 바뀌는 건 하나다:

`GET /races/{raceId}/card` 의 `featureKeys` 앞 6칸이 `ax_*` 가 된다.
그 여섯에 `groupWeights` 를 곱하고 softmax 를 걸면 끝이다 — 교체할 `deckScores`
코드는 `AI-CONTRACT.md` §5 에 그대로 적어 뒀다.

트레이딩 카드 스탯 막대(`BettingBoard.jsx:11 STAT_AXES`)는 **한 글자도 안 바뀐다.**
지표 코드를 프런트가 쓰는 이름에 맞췄다.

## 8. 파일이 건너가는 경로 — **아직 안 정했다**

지금 산출물은 전부 **개인 레포 `mo-gun/hourse` 의 `out/`** 에 있다.
팀이 이걸 가져갈 방법이 정해지지 않았다. 후보:

1. 팀 GitLab `docs/ai/` 에 MR — 시드 SQL·마스터 CSV만. 가장 단순하다
2. 팀 GitLab `ai/model/` 에 MR — 산출기까지 같이. 도연님 폴더라 협의 필요
3. 정원님 GitHub `basemodel/` 에 두고 팀은 문서 링크만 — 지금 상태

**1번을 제안한다.** 백엔드가 실제로 필요한 건 SQL 두 개뿐이고, 산출기는 모델 레포에
있는 편이 맞다. **팀 레포에 올리기 전에 확인을 받겠다.**

## 9. 받는 쪽 체크리스트

### 백엔드
- [ ] `feature_seed.sql` 적용 (시더보다 먼저)
- [ ] `PredictionService.java:57` `exp(s/10.0)` → `exp(s*0.0445)`
- [ ] `EntryFeatureScoreAxisProvider` 그룹 평균 → `ax_*` 6행 직접 읽기
- [ ] `entry_feature_score` 배치 적재 경로 (게임풀)
- [ ] `HttpAiClient` 에 `score-tables` 메서드 (실시간 붙일 때)

### 프런트
- [ ] `rules.js deckScores` 를 `groupWeights` + softmax 로 교체
- [ ] min-max 재척도 제거 (`Math.round(20 + …*75)`)
- [ ] `BettingBoard.jsx:117-119` 선형 몫 → softmax 확률
- [ ] 픽스처의 레거시 `weights` 제거

### AI (건모)
- [x] 지표 마스터 · 시드 SQL
- [x] 축 점수 산출기 + 왕복 검증
- [x] 온도 실측
- [ ] 게임풀 3,700경주 산출 (팀 확인 후)
- [ ] `build_score_table()` 을 도연님 서버에 붙일 수 있는 함수로 정리

## 10. 정하지 못한 것 — 기획·팀 확인이 필요하다

- **파일 인도 경로** (§8) — 1·2·3 중 어느 것인가
- **게임풀 산출을 지금 돌려도 되나** — `game` 은 학습 금지지만 추론 산출은 다른 이야기다
- **`/ai/score-tables` 를 누가 언제 붙이나** — 도연님 서버, 내 함수. 일정 미정
- **`/ai/backtest-dataset`(브라우저 백테스트)이 MVP 인가**
- 실시간 경로는 **73피처**라 배당형·역배형 프리셋이 자동으로 빠진다.
  화면에서 그 둘을 어떻게 보여줄지 미정(숨김/비활성/리플레이 전용 표시)
