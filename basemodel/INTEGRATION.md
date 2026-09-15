# basemodel ↔ 제품 연동 — 지금 어디가 끊겨 있고 무엇을 채워야 하나

모델 이야기는 [README.md](README.md) 에 있다. 이 문서는 **모델이 제품에 붙는 지점만** 다룬다.
전부 2026-09-14 에 세 레포 코드를 직접 읽고 적었다. 문장마다 근거 파일·줄을 달았으니,
바뀌었으면 이 문서가 아니라 코드를 믿어라.

## 0. 레포 세 개 — 무엇이 어디에 있나

| | 주소 | 안에 있는 것 |
|---|---|---|
| **모델** | GitHub `JeongWon4034/horse-pred-engine` | `pipeline/`·`selfsup/` (정원) · `basemodel/`·`arena/` (건모) |
| **제품** | GitLab `s15-blockchain-pred-sub1/S15P21A304` (`develop`) | `front/`·`back/`·`ai/` (도연) · `docs/명세서` |
| 개인 | GitHub `mo-gun/hourse` | 원장 빌더 · `raceday.py` · 실시간 예측 기록 |

아래 경로에서 `front/`·`back/`·`ai/`·`docs/` 로 시작하면 **제품 레포**, 그 밖은 이 레포다.

## 1. 계약 — 문서에 적힌 것

```
점수 = Σ( groupWeights[k] × ax_k ) / 100          k ∈ 6축
확률 = softmax( 점수 × T )  , 경주 안에서 정규화     T = 0.0445
```

출처: `docs/명세서/schema.sql:168` (`pae_feature_weight` 주석), `back/…/deck/service/GetDeckMetaService.java:32`,
`back/…/ai/BotConfigDefaults.java:16`.

6축은 `feature_group.code` = `CONDITION · SPEED · RUNNING · JOCKEY · ENVIRONMENT · ABILITY`
(`docs/명세서/schema.sql:86`). 프런트 슬라이더 정의도 같다 (`front/src/mock/fixtures.js:106`).

## 2. AI 가 내보낼 것은 API 가 아니라 **행(row)** 이다

이게 제일 중요한 확인이다. 백엔드가 AI 서버로 나가는 접점은 둘뿐이고
(`back/…/ai/client/HttpAiClient.java:37,47` — `/ai/bots/bets`, `/ai/bots/config`),
**둘 다 봇 베팅용이다.** 축 점수는 HTTP 로 받지 않는다. 테이블에서 읽는다:

```
entry_feature_score(race_entry_id, feature_id, score 0~100, raw_prob, model_version)
                                                    docs/명세서/schema.sql:303
```

`back/…/ai/score/EntryFeatureScoreAxisProvider.java` 주석이 그대로 말한다 —
*"축 점수 테이블 적재(AI-05)가 끝나면 이 클래스만 바꾸면 된다."*
그리고 점수가 없으면 이렇게 끝난다:

```java
log.error("축 점수 없음: 출전마 {}두 중 {}두만 6축 완비 (AI-05 적재 전). 봇 베팅 건너뜀", …)
```

**지금 봇 베팅이 안 돈다. 그 구멍이 AI 쪽 몫이다.**

## 3. 끊긴 곳 넷 — 코드로 확인했다

### ① 슬라이더가 점수에 닿지 않는다 (프런트·백엔드 양쪽 다)

슬라이더는 `groupWeights`(6그룹)를 고친다 (`front/src/deck/DeckSliders.jsx:27`).
그런데 점수를 만드는 두 곳은 **둘 다 다른 필드를 본다**:

| 계산하는 곳 | 읽는 가중치 | 실제로 일어나는 일 |
|---|---|---|
| `front/src/room/rules.js:34` `deckScores` | `deck.weights` (8개 항목) | `DeckDto` 에 `weights` 가 **없다** (`back/…/deck/dto/DeckDto.java:22` 는 `groupWeights` 뿐) → 첫 줄 가드에서 `{}` 반환 → 화면 점수 전부 `-` |
| `back/…/room/service/PredictionService.java:40` | `pae_feature_weight` | 이 테이블에 **쓰는 코드가 없다.** `DeckWeightStore` 는 `pae_group_weight` 만 저장한다 → `weights.isEmpty()` → 전 항목 **균등 가중** |

목 데이터가 도는 건 픽스처가 레거시 `weights` 를 아직 들고 있어서다. 코드가 스스로 그렇게 적어 놨다 —
`front/src/mock/mockApi.js:400` *"픽스처에 남아 있는 항목 단위 weights 는 room/rules.js deckScores 전용 레거시다"*,
`front/src/mock/fixtures.js:92` *"deprecated"*.

> **빠진 함수는 정확히 하나다: 그룹 가중치 → 항목 가중치 분배.**
> `pae_group_weight`(스키마 158줄)와 `pae_feature_weight`(168줄)가 둘 다 설계돼 있는데
> 사이를 잇는 코드가 없다.

### ② 온도가 세 값이다

| 곳 | 값 |
|---|---|
| `back/…/ai/BotConfigDefaults.java:16` · 명세 | `T = 0.0445` |
| `back/…/room/service/PredictionService.java:57` | `Math.exp(score / 10.0)` → **T = 0.1** |
| 프런트 | **softmax 자체가 없다** |

`GET /decks/features` 가 `temperature: 0.0445` 를 내려주지만 (`front/src/deck/useDeckMeta.js`),
프런트에서 그 값을 쓰는 코드가 없다.

### ③ 프런트가 점수를 경주 안에서 재척도한다 — 확률 보정이 사라진다

```js
out[r.no] = Math.round(20 + ((r.v - min) / (max - min)) * 75)   // rules.js:47
```

1등마는 **항상 95**, 꼴찌는 **항상 20** 이 된다. 모델이 얼마나 확신하는지가 지워진다.
베팅 화면은 여기에 한 번 더 얹는다 — `front/src/room/BettingBoard.jsx:117-119` 은
`score / Σscore` 라는 선형 몫을 확률로 쓴다. softmax 가 아니다.

압도적 1강이 있는 경주와 8두가 고른 경주가 **같은 점수판**으로 보인다.

### ④ `ai/app/engine.py` 만 옛 축 어휘를 쓴다

`AXES = ["form","blood","jockey","condition","stamina","market"]` — 스키마·FE·BE·도연 `ai/model`·
`basemodel` 다섯 곳이 6축으로 일치하는데 여기만 다르다. 죽은 코드로 보이지만 확인이 필요하다.

## 4. 축 이름 번역표는 이미 백엔드에 있다

내가 만들자고 했던 번역표가 이미 있었다 — `back/…/ai/AxisCodes.java`:

```
ENVIRONMENT ↔ ENV        ABILITY ↔ RATING        나머지 넷은 같은 이름
```

**그러니 새로 만들 것이 없다.** `basemodel` 이 `ENV`·`RATING` 으로 내보내면 경계에서 알아서 바뀐다.

정원님 `pipeline/model/side_tower.py` 의 `F1..F6` 은 **다른 층**이다 — 데이터 출처별 묶음이지
화면 슬라이더가 아니다. 맞출 대상이 아니고, 맞추면 안 된다.
(`F6` 인기도를 여섯 중 하나로 두면 시장 신호가 희석된다 — 타워 35.33 vs 배당형 37.88, +2.55 p=0.014. README §5)

## 5. 모델 합치기 — 정원님 방법을 쓴다. 다만 기대이득이 낮다

정원님 `selfsup/` 가 이미 답을 냈다:

- `ensemble.py` — **확률** 평균 `log(w·p_A + (1−w)·p_B)`. 점수 평균이 아니다
- 가중치는 **교차적합**으로 고른다 (경주를 반 갈라 A 에서 w 선택, B 에서 측정, ×20). valid 로 고르면 낙관값이 나온다
- `stack.py` — 스태킹 메타모델은 **기각**. 단순 평균 대비 −0.96 [−2.23, +0.29]
- 원장 판정: *"1,254경주로는 규칙이 안 잡히고 과적합 손실이 더 크다. 1:1:1 평균 유지."*

그런데 **합쳐도 거의 안 바뀐다**는 증거가 이미 있다. 9/11 실시간 9경주에서
정원님 1:1:1 앙상블과 내 75피처 LGB 가 **8경주에서 같은 말을 골랐고 적중도 2/9 로 같았다.**
`arena/` 에서 계열 16종을 붙였을 때 예측 상관 중앙값 0.94 였던 것과 같은 그림이다.

→ 결론: **앙상블 한 벌만 적재한다.** `entry_feature_score` 가 경주당 한 벌인 것과도 맞는다.
스키마를 바꿀 이유가 없다.

## 6. 시장(배당) 축 — 실시간에서 뺀다

세 사람이 따로 확인하고 같은 결론에 닿았다:

- 건모 — 배당 API 5종 · 웹 경로 138개 · JS 번들 9개 · 시행일 1,168회 관측 → 배당은 **발주 T−5 ~ +1분**에만 존재
- 정원 — `selfsup/ledger.md` P13 *"배당 API 5종 전부 확정만"*, 가짜 배당 증류도 −1.75 로 실패
- 도연 — `ai/model` 에 배당 축이 아예 없다

프리셋 `ODDS`·`UPSET` 의 `groupWeights` 가 빈 맵인 것도 같은 이유다 (`front/src/mock/fixtures.js:118`,
`back/…/deck/dto/PresetsDto.java:9`). **리플레이(77피처)는 배당을 쓰고, 실시간(73피처)은 안 쓴다.**

## 7. 그래서 무엇을 할 것인가

의존 순서대로. 위가 막히면 아래가 못 간다.

| | 할 일 | 누가 | 막고 있는 것 |
|---|---|---|---|
| **1** | `feature` 마스터 확정 — 항목 몇 개를, 어느 그룹에 둘지. `basemodel/config.py::AXIS_FEATURES` 가 이미 피처→축 대응을 들고 있다 | 건모 + 도연 | 2·3 전부 |
| **2** | `entry_feature_score` 적재기 — 경주별로 6축 점수를 0~100 으로 환산해 행으로 쓴다 (AI-05) | 건모 | 봇 베팅 · 경주 카드 |
| **3** | 그룹 가중치 → 항목 가중치 분배 함수 (§3-①) | 백엔드 | 슬라이더 전체 |
| **4** | 온도 한 값으로 통일 — `T=0.0445` 인지 `0.1` 인지 실측으로 고른다 (§3-②) | 건모 측정 → 백엔드 반영 | 확률 표시 |
| **5** | 프런트 `deckScores` 를 `groupWeights` + softmax 로 교체, min-max 재척도 제거 (§3-②③) | 프런트 | 점수·확률 화면 |
| **6** | `ai/app/engine.py` 축 어휘 정리 또는 삭제 (§3-④) | 도연 | 없음 (정리) |

**1·2·4 가 내 몫이다.** 3·5 는 남의 코드라 손대지 않고, 이 문서를 근거로 넘긴다.

> **갱신 2026-09-15** — 1·2·4 가 끝났다. 측정 결과와 구현 명세는
> [AI-CONTRACT.md](AI-CONTRACT.md) 로 옮겼다. 요약: 온도 `0.0445` 는 그대로 두고,
> 축 점수는 타워가 직접 내며(지표 평균은 4.6%p 손해), 지표 마스터는 41행이다.

### 온도를 실측으로 고르는 법 (4번)

`T` 는 취향이 아니라 측정값이다. valid 1,254경주에서 `logloss(softmax(score × T))` 를 최소화하는
`T` 를 찾으면 된다. `arena/arena/evaluate.py::logloss_calibrated` 가 이미 2겹 교차적합으로
온도를 맞추는 코드를 들고 있으니 그대로 쓴다. 0.0445 와 0.1 중 무엇이 맞는지는 **한 번 돌리면 끝난다.**

## 8. 아직 확인 못 한 것 — 단정하지 않는다

- `BotConfigDefaults.BOTS` 9봇 × 6가중치의 **배열 순서**가 `AI_AXES` 와 같은지 눈으로만 봤다. 테스트로 확인해야 한다
- 실제 서버를 띄워 `GET /races/{id}/card` 응답을 받아 본 적이 없다. §3-① 은 DTO 와 서비스 코드에서 읽은 결론이다
- `entry_feature_score.score` 의 *"경주 내 순위 기반 환산"* (스키마 307줄) 이 정확히 어떤 환산인지 정의가 없다. 1번과 같이 정해야 한다
- 기수 변경 시 `F3_jk_*` 를 다시 못 만든다 (프레임에 그 시점 누적 상태가 없다). 지금은 경고만 띄운다
- 2·3착 확률의 λ 보정이 아직 없다 (도연 README §6-1)
