# AI ↔ 제품 계약 — 백엔드·프런트가 그대로 구현할 명세

[INTEGRATION.md](INTEGRATION.md) 가 "지금 어디가 끊겼나"라면, 이 문서는 **"무엇을 만들면
되나"** 다. 숫자는 전부 valid 1,254경주 실측이고 재현 코드 경로를 같이 적었다.
발행 2026-09-15 · 담당 이건모

## 0. 한 장 요약

```
① AI 가 경주마다 41칸을 채운다      entry_feature_score
      ax_* 6칸   슬라이더가 곱하는 축 점수 (0~100)  ← 계산의 전부
      표시 35칸  화면 표시 전용 (스탯 막대·원본값)

② 점수    raw = Σ( groupWeights[k] × ax_k ) / 100
③ 확률    p   = softmax( raw × 0.0445 ),  경주 안에서
```

바꿔야 할 것은 셋뿐이다 — 백엔드 둘(§4), 프런트 하나(§5).

## 1. 6축 — 정본은 `feature_group.code`

`CONDITION · SPEED · RUNNING · JOCKEY · ENVIRONMENT · ABILITY`
(`docs/명세서/schema.sql:86`). 스키마·프런트·백엔드·도연 `ai/model`·`basemodel` 이 이미 일치한다.

AI 서버 쪽 이름이 둘 다르다(`ENVIRONMENT↔ENV`, `ABILITY↔RATING`). 변환표는 **이미
백엔드에 있다** — `back/…/ai/AxisCodes.java`. 새로 만들 것이 없다.

`pipeline/model/side_tower.py` 의 `F1..F6` 은 **다른 층**이다 — 데이터 출처별 묶음이지
화면 슬라이더가 아니다. 맞출 대상이 아니다.

## 2. 온도 T = 0.0445 — 그대로 둔다

`BotConfigDefaults.TEMPERATURE` 를 고칠 필요가 없다. 다만 **조건이 있다**:
축 점수를 **경주 내 순위 백분위 0~100** 으로 낼 때만 이 값이 맞는다(§3).

### 왜 그 값인지 — 독립 측정 다섯 번

| 축 점수를 무엇으로 만들었나 | 실측 최적 T | 0.0445 의 손해 |
|---|---:|---:|
| 지표 순위 평균 | 0.0408 | +0.0035 |
| 축마다 대표 지표 1개 | 0.0447 | +0.0014 |
| 지표 순위 중앙값 | 0.0444 | +0.0011 |
| LGB 축 랭커 6개 | 0.0411 | — |
| **6축 타워 (채택)** | **0.0478** | **+0.0022** |

전혀 다른 방식인데 같은 자리에 떨어진다. **순위 환산을 거치면 점수 폭이 모델과 무관하게
고정되기 때문**이다. 모델을 갈아도 이 상수를 다시 안 고쳐도 된다.

### `0.0445` 와 타워 내부의 `1.436` 은 같은 온도다

타워는 축 점수를 경주 내 z-score(표준편차 1.00)로 내고 그 자에서는 T=1.436 이다.
순위 환산하면 표준편차가 28.7 이 되므로 `1.436 / 28.7 ≈ 0.050` — 실측 0.0478 과 같은 자리다.
**어느 자에도 안 맞는 값은 `0.1` 하나뿐이다**(§4-1).

재현: `tools/fit_temperature*.py` · `tools/roundtrip_tower.py` (mo-gun/hourse)

## 3. 축 점수 — 모델이 낸다. 지표를 평균하지 않는다

```
축 점수 = 6축 타워 출력의 **경주 내 순위 백분위 × 100**
```

`schema.sql:307` 의 "0~100. 경주 내 순위 기반 환산" 을 그대로 지킨 형태다.

### 왜 지표 평균이면 안 되나 — 같은 valid, 같은 채점기

| 축 점수를 만드는 법 | top-1 | top-3 |
|---|---:|---:|
| 지표값 순위 환산 → 그룹 평균 *(현재 BE 구현)* | **28.3%** | 56.9% |
| LGB 축 랭커 6개 | 31.0% | 60.5% |
| **6축 타워 → 순위 환산 (채택)** | **32.9%** | 62.7% |
| *(상한) LGB 73피처 통짜* | *33.5%* | *63.3%* |
| *(참고) 시장 인기 1위마* | *39.3%* | *69.3%* |

**지표 평균은 4.6%p 를 버린다** — top-1 표준오차 ±0.85%p 의 5배다.

순위 환산 자체의 비용은 33.2% → 32.9% 로 표준오차 안이다. 그 값에 스키마의 0~100 범위
준수와 T 고정(§2)을 얻는다.

## 4. 백엔드가 고칠 것 — 두 곳

### 4-1. `room/service/PredictionService.java:57`

```java
exp[i] = Math.exp(raw.get(i).score().doubleValue() / 10.0);   // T = 0.1  ← 틀렸다
exp[i] = Math.exp(raw.get(i).score().doubleValue() * 0.0445); // ← 이걸로
```

`0.1` 은 §2 의 다섯 가지 읽기 **전부에서** 틀린 값이다. 채택안 기준 logloss 손해 **+0.40**.

### 4-2. `ai/score/EntryFeatureScoreAxisProvider.fromFeatureScores`

지금은 그룹 안 지표 점수를 **산술 평균**한다(`sc[0] / sc[1]`). 그러면 §3 표의 첫 줄이
되어 top-1 이 32.9% → 28.3% 로 떨어진다.

**`code` 가 `ax_` 로 시작하는 여섯 행을 그대로 읽으면 된다.** 그 클래스 주석이 이미
예고한 변경이다 — *"축 점수 테이블 적재(AI-05)가 끝나면 이 클래스만 바꾸면 된다."*

### 4-3. 고칠 필요가 **없는** 것

`BotConfigDefaults.TEMPERATURE = 0.0445` · `AxisCodes` · `feature.is_displayed` ·
`pae_group_weight` / `pae_feature_weight` 스키마. 전부 이미 맞다.

## 5. 프런트가 고칠 것 — 한 곳

`front/src/room/rules.js:34` `deckScores` 가 `deck.weights`(8개 항목, 레거시)를 읽는데
`DeckDto` 에는 `groupWeights` 만 있다. 서버 응답에는 그 필드가 없어 가드에서 `{}` 가
반환되고 **화면 점수가 전부 `-` 가 된다**. 목데이터가 도는 건 픽스처에 레거시가 남아서다
(`mock/mockApi.js:400` 이 스스로 "레거시"라고 적어 놓았다).

```js
export function deckScores(card, deck, temperature = 0.0445) {
  const w = deck?.groupWeights; if (!card?.scores || !w) return {}
  const keys = card.featureKeys
  const ax = ['CONDITION','SPEED','RUNNING','JOCKEY','ENVIRONMENT','ABILITY']
    .map((g) => keys.indexOf('ax_' + g.toLowerCase()))
  const raw = card.horses.map((h, i) =>
    ax.reduce((s, j, k) => s + (card.scores[i]?.[j] ?? 50) *
      (w[['CONDITION','SPEED','RUNNING','JOCKEY','ENVIRONMENT','ABILITY'][k]] ?? 0), 0) / 100)
  const m = Math.max(...raw)                        // 경주 내 softmax
  const e = raw.map((r) => Math.exp((r - m) * temperature))
  const t = e.reduce((a, b) => a + b, 0)
  const out = {}
  card.horses.forEach((h, i) => { out[h.no] = Math.round(e[i] / t * 1000) / 10 })  // %
  return out
}
```

**지금의 min-max 재척도는 빼야 한다.** `Math.round(20 + (v-min)/(max-min) * 75)` 는
1등마를 항상 95, 꼴찌를 항상 20 으로 만들어 "1강이 확실한 경주"와 "다 비슷한 경주"를
화면에서 같게 만든다. `BettingBoard.jsx:117-119` 의 `score / Σscore` 도 softmax 로 바뀐다.

트레이딩 카드 스탯 막대(`BettingBoard.jsx:11 STAT_AXES`)는 **한 글자도 안 고쳐도 된다** —
지표 코드를 프런트가 쓰는 이름에 맞췄다(§6).

## 6. 지표 마스터 — 41행

명세가 없어서(백엔드 세 곳이 "55개"를 기다리는데 목록이 레포 어디에도 없다) AI 에서 낸다.

| | 개수 | `is_displayed` | 쓰임 |
|---|---:|---|---|
| `ax_condition` … `ax_ability` | 6 | FALSE | **점수 계산** |
| 표시 지표 | 35 | TRUE | 스탯 막대 · 출전표 원본값 |

프런트가 하드코딩한 이름 여섯을 그대로 받았다 — `speed_index` · `early_pace` ·
`late_pace` · `jockey_win_rate` · `wet_track_rate` · `recent5_avg_rank`.
`horse_place_rate` · `trainer_place_rate` 는 **복승률인데 데이터에 복승률이 없어** 받지
않았다(승률뿐이다). 둘 다 제거 예정인 레거시 픽스처에만 있어 화면은 안 깨진다.

각 지표에 `direction`(train 에서 측정)과 `ref_win_rate_bp`(valid 실측 — 이 지표 하나로만
1등을 뽑았을 때 적중률)를 채웠다. `ref_profit_bp` 는 **비워 뒀다** — 단독 수익률은
확정배당이 필요한데 valid 파케이에 `winOdds` 가 없다.

넘기는 파일 (mo-gun/hourse `out/`)
- `feature_seed.sql` — `feature_group` 6 + `feature` 41 INSERT
- `entry_feature_score_sample.sql` — 적재 예시
- `feature_master.csv` / `.json` — 축·방향·가용시점·단독적중률

> `ReferenceDataSeeder` 는 feature 테이블에 행이 하나라도 있으면 픽스처를 건너뛴다.
> 이 SQL 을 먼저 넣으면 임시 픽스처 5개 대신 41행이 쓰인다.

## 7. 실시간과 리플레이는 피처 수가 다르다

| | 피처 | 배당 축 | 프리셋 |
|---|---:|---|---|
| 게임 리플레이 (과거 경주) | 77 | 쓴다 | 5종 전부 |
| **주말 실시간** | **73** | **못 쓴다** | 배당형·역배형 자동 제외 |

배당은 발주 **T−5 ~ +1분**에만 존재한다. 세 사람이 따로 확인했다 — 건모(API 5종·웹 138경로·
JS 번들 9개·시행일 1,168회 관측), 정원(`selfsup/ledger.md` P13, 가짜 배당 증류도 −1.75 로
실패), 도연(배당 축 없음). 프리셋 `ODDS`·`UPSET` 의 `groupWeights` 가 빈 맵인 것도 같은 이유다.

## 8. 모델 셋을 합치는 문제 — 합쳐도 거의 안 바뀐다

9/11 실시간 9경주에서 정원님 1:1:1 앙상블과 건모 75피처 LGB 가 **8경주에서 같은 말**을
골랐고 적중도 2/9 로 같았다. `arena/` 의 계열 16종 예측 상관 중앙값 0.94 와 같은 그림이다.

합친다면 정원님 방법을 쓴다 — 확률 평균 `log(w·p_A + (1−w)·p_B)`, 가중치는 교차적합,
스태킹은 기각(−0.96 [−2.23, +0.29]), 결론 1:1:1. 다만 **앙상블 한 벌만 적재**하면 되고
이는 `entry_feature_score` 가 경주당 한 벌인 스키마와도 맞는다.

## 9. 아직 안 정한 것 — 단정하지 않는다

- **그룹 → 항목 가중치 분배**(`pae_feature_weight`). 그룹 안 비중을 적합하면 logloss 는
  −0.0167 로 개선되지만 top-1 은 +0.6%p 로 표준오차(±1.27%p) 안이다. **근거가 약해 보류**한다.
  축 점수를 타워가 직접 주는 지금 구조에서는 이 테이블이 비어 있어도 동작한다
- **λ 보정** — 2·3착 확률이 낙관 편향이다(도연 `ai/model/README.md` §6-1)
- **기수 변경** 시 `F3_jk_*` 를 다시 못 만든다. 프레임에 그 시점 누적 상태가 없어 경고만 띄운다
- `BotConfigDefaults.BOTS` 9봇 × 6가중치의 **배열 순서**가 `AI_AXES` 와 같은지 눈으로만 봤다
- 실제 서버를 띄워 `GET /races/{id}/card` 응답을 받아 본 적이 없다 — §5 는 DTO·서비스 코드에서 읽은 결론이다
