# 참고문헌 노트 — 경마 예측·베팅 시장 (김민지, 2026-09-11)

> 팀 `docs/dataset/schema_v2.py` 의 LIT 인용 + 이 축(selfsup·결합확률)에서 근거로 쓴 것.
> **전부 기억으로 정리한 요약이다. 발표·보고서에 인용하기 전에 원문(제목·연도·저널·수치)을 확인할 것.** 확인 안 된 항목은 ⚠ 표시.
> 우리 실험 결과와 어디서 맞닿는지를 같이 적는다 — 그게 이 노트의 용도다.

---

## 0. 한 줄 요약

40년 문헌의 결론은 셋이고, 우리 팀이 2주 동안 따로 확인한 것과 같다.

1. **공개 데이터(펀더멘털) 모델은 시장(배당)을 못 이긴다.** 어느 나라, 어느 모델이든.
2. 우위는 **① 펀더멘털 + 배당 결합**(작게), **② 조합 마권(연승·복승·삼복승)의 변환 오차**(더 크게)에서 난다.
3. 딥러닝·ML 논문들도 1 을 뒤집지 못했다. 적중률 이득은 오차 안, 수익은 결합에서.

---

## 1. 펀더멘털 모델의 출발 — 경쟁 구조

| 문헌 | 핵심 | 우리와의 관계 |
|---|---|---|
| **Bolton, R. N. & Chapman, R. G. (1986).** "Searching for positive returns at the track: A multinomial logit model for handicapping horse races." *Management Science* 32(8). | 경주를 **말끼리 경쟁하는 한 단위**로 보고 조건부(다항) 로짓으로 학습. 독립 분류가 아니라 경주 내 정규화된 확률. | 팀 lambdarank · Plackett-Luce 손실의 조상. 정원 S1(조건부 로지스틱)이 이것. `schema_v2` LIT "BC" |
| **Lessmann, S., Sung, M.-C. & Johnson, J. E. V. (2009).** "Identifying winners of competitive events: A SVM-based classification model for horserace prediction." *EJOR* 196. ⚠연도 | SVM 으로 승자 분류. 단독 ML 은 조건부 로짓과 큰 차이 없음. | "표 데이터에서 ML 이 로짓/GBDT 를 크게 못 이긴다" — 정원 S1~S4 · 내 P0~P5 와 같은 결론 |
| **Lessmann, S., Sung, M.-C. & Johnson, J. E. V. (2010).** "Alternative methods of predicting competitive events: An application in horserace betting markets." *IJF* 26. ⚠ | **경쟁 구조를 반영한 모델(경주 내 상대화)이 독립 분류보다 낫고**, 이득은 적중률보다 수익(결합)에서 난다. | 팀 "경주 내 정규화 24개" 의 근거. 건모 README "이진분류 전환 30.8% < 랭킹" 과 일치. `schema_v2` LIT "LS" |
| Pudaruth, S. et al. (2013). "Horse racing prediction at the Champ de Mars…" ⚠ | 모리셔스 데이터. 기수·경험·배당·게이트·거리 적성·중량·레이팅 가중 확률. | 팀 X_/F3/F4 피처의 출처("P4"). 시장을 못 넘음 |
| Borowski, P. & Chlebus, M. (2021). ⚠ | 폴란드 3,782경주, XGBoost 등. **과거 획득 상금이 최중요 피처.** | `F1_prize_*` 의 근거("B7"). v2.2.0 에서 `F1_prize_life` 는 연도 드리프트로 비활성 — 문헌과 데이터가 갈린 예 |

---

## 2. 시장은 효율적이다 — 이기려 하지 말 것

| 문헌 | 핵심 | 우리와의 관계 |
|---|---|---|
| **Benter, W. (1994).** "Computer-based horse race handicapping and wagering systems: A report." In *Efficiency of Racetrack Betting Markets* (Hausch, Lo, Ziemba eds.). | 홍콩에서 수년간 실제 수익. 비결은 모델이 아니라 **펀더멘털 확률과 공개 배당을 2단계 로짓으로 결합**하는 것. 배당 계수가 ~1 — 시장이 거의 다 알고, 펀더멘털은 그 위에 조금 얹는다. 실제 수익은 조합 pool 에서. | 내 §7-4 "Benter 2단계 기각"(펀더멘털 0.171 / 시장 0.895)은 **문헌 그대로** — 펀더멘털 몫이 작다는 뜻이지 결합이 틀렸다는 뜻이 아니다. 배당이 있어야 성립. `schema_v2` LIT "BT" |
| **Asch, P., Malkiel, B. G. & Quandt, R. E. (1982).** "Racetrack betting and informed behavior." *J. Financial Economics* 10. | **마감 직전 배당 변동에 정보가 있다** — 정보 있는 돈(smart money)은 늦게 들어온다. | 배당을 쓴다면 발주 직전 값이어야 하는 이유. API 확정배당이 곧 그것 |
| **Sung, M.-C., Johnson, J. E. V. & Bruce, A. C. (2005).** "Searching for semi-strong form inefficiency in the UK racetrack betting market." ⚠ | favourite–longshot bias: 비인기마는 과대평가, 인기마는 약간 과소평가. | 건모 EDA "내재확률 2% 미만 구간만 15% 부풀림"과 같은 현상 |
| Tondapu, N. (2024). ⚠ (건모 README 인용) | 100만 경주 규모로 경마 시장이 전통 금융시장보다 효율적임을 보임. | 팀 방향 "시장을 이기려 하지 말고 결합·틈새" 의 근거 |
| Thaler, R. & Ziemba, W. (1988). "Anomalies: Parimutuel betting markets." *J. Economic Perspectives* 2(2). | 위 편향들의 개관. | 교육용 서비스라면 유저에게 이 편향을 알려주는 것 자체가 콘텐츠 |

---

## 3. 조합 마권의 변환 오차 — 우리가 실제로 이긴 자리

| 문헌 | 핵심 | 우리와의 관계 |
|---|---|---|
| **Harville, D. A. (1973).** "Assigning probabilities to the outcomes of multi-entry competitions." *JASA* 68. | 1등 확률에서 2·3착 조합 확률을 뽑는 공식. 업계 표준. 잠재 실력이 Gumbel 분포라는 가정과 동치(Plackett-Luce). | 내 M0 기준선(3.9678). 도연 `probs.py` 의 7승식 전개도 이 공식 |
| **Henery, R. J. (1981).** "Permutation probabilities as models for horse races." *JRSS-B* 43. | Normal 잠재분포. "Harville 은 인기마 place 확률을 과대평가한다." | 내 M0 편향표(인기 1위마 +10.1%p)가 이것. 격자에서 Henery > Harville 확인 |
| **Stern, H. (1990).** "Models for distributions on permutations." *JASA* 85. | Gamma 잠재분포. Harville 은 shape=1 특수 경우. | 격자에서 t/Gamma 계열은 Logistic 에 진다 |
| Lo, V. S. Y. & Bacon-Shone, J. (1994). ⚠ | 거듭제곱 할인 근사 (λ). | 내 M1 λ 할인 (1, .7, .8) |
| **Hausch, D. B., Ziemba, W. T. & Rubinstein, M. (1981).** "Efficiency of the market for racetrack betting." *Management Science* 27(12). — "Dr. Z system" | **단승 시장은 효율적이지만 연승·복승(place/show) 시장은 Harville 로 변환하면 비효율이 보인다** → 거기서 수익. | **정확히 내 M5 결과.** 연승·복연승 절사 ROI +14~27%, 삼복승·삼쌍승 logloss 유의 우위. 1981년 발견을 신경망(M2)으로 확장한 셈 |
| Kelly, J. L. (1956). "A new interpretation of information rate." | 최적 베팅 비율. | 서비스엔 미적용(교육용). 백테스트 ROI 해석 시 참고 |

---

## 4. 딥러닝·시퀀스·최근

| 문헌 | 핵심 | 우리와의 관계 |
|---|---|---|
| 정준형 외 (2024). ⚠ 국내 LTR | Learning-to-rank. Shapley 상위에 출발훈련·질병진단·훈련기록. | 조교 API(15058782) 없이도 정원 실험에서 차이 0.00%p — 데이터 시기·정제가 달라서일 수 있음 |
| So Yubin 외 (2025). ⚠ 국내 LTR + 웹서비스 | 최근성적/통산 평균착순/부담중량/마령 4계열. | 팀 F1·X_ 피처 근거("K3") |
| 최혜민 외 (2015). ⚠ 서울경마 우승마 예측 | 과거 우승 경력·기수 우승 경력이 핵심. | F1·F3 근거("K1") |
| Yoon et al. / arXiv 계열 "deep learning for horse racing" ⚠ (제목·저자 미확인) | 시퀀스·어텐션 시도. 대체로 GBDT 대비 소폭 또는 무차이. | 정원 S3(GRU)·S4(Transformer) 결과와 같은 방향 |
| **자기지도 표 데이터**: Yoon et al. (2020) VIME · Bahri et al. (2022) SCARF · Somepalli et al. (2021) SAINT | 라벨 없이 마스크 복원·대조 사전학습. 라벨 적을 때 GBDT 와의 격차를 줄인다. | 내 P1~P3 의 방법. 이 데이터에선 +1.4%p, S3 와 같은 자리 |

---

## 5. 우리 실험과 문헌의 대응 — 발표용 한 문단

> 1986년 Bolton–Chapman 이후 경마 예측 문헌은 "공개 데이터로는 시장을 못 이긴다"로 수렴했고, Benter(1994)는
> 그 위에 배당을 결합해야 수익이 난다고 했다. 우리 팀은 같은 데이터로 모델 구조(정원, S1~S4), 데이터 양(건모, 2004ext),
> 학습 신호(민지, 자기지도), 새 피처(페이스·마체중·장구)를 각각 시험했고 전부 그 결론을 재현했다 — 73피처 top-1 천장 ~35%,
> 시장 39%. 반면 Hausch–Ziemba(1981)가 지목한 **연승·복승 시장의 Harville 변환 오차**는 신경망(M2)으로 인기 1위마 +10.1%p
> 편향을 +1.5%p 로 줄였고, 조합이 커질수록 우위가 커졌다(삼쌍승 logloss −0.087). 즉 "1등 맞히기로 시장 이기기"는
> 실패를 재현했고, "조합 확률의 변환 오차"는 45년 된 발견을 데이터로 확장했다.

---

## 6. 발주 전 배당 — 데이터 소스 현황 (2026-09-11 확인)

| 소스 | 발주 전 | 비고 |
|---|---|---|
| 공공데이터 API214_1 (경주성적) | ✗ | 확정배당만. 단 **출마표·장구·마체중은 경주 전에 준다** |
| 공공데이터 15058559 (확정배당율 통합) | ✗ | 경주 후 |
| `race.kra.co.kr` · `todayrace.kra.co.kr` | ✗ | 공개 정보 페이지에 발매 중 배당 없음 |
| 렛츠런 앱 · `derbyon.kra.co.kr` | ○ | 성인인증·로그인. 내부 JSON 을 뜯으면 스크래핑 — 약관·계정 리스크 |
| 서드파티(race-note 등) | △ | 남의 스크래핑을 다시 긁는 것. 지속성 미지수 |
| **말의 과거 인기도** (as-of, 직전 N경주 배당 순위 평균) | **○ 공개 데이터** | 배당 자체는 아니지만 "시장이 늘 높게 보는 말" 정보. **미시험** — P9 후보 |
