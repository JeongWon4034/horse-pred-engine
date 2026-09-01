"""KRA 오픈API 데이터셋 카탈로그.

KRA 엔드포인트는 두 가지 명명 체계가 섞여 있다.
    구식:  /B551015/API23_1/entryRaceHorse_1        (APIxx_x / camelCase_x)
    신식:  /B551015/racedetailresult/getracedetailresult   (소문자 / get소문자)

파라미터 이름도 데이터셋마다 다르다 (rc_date / rcDate / rc_month ...).
probe.py 가 후보를 전부 대입해보고 살아있는 조합을 알려준다.
"""

BASE = "https://apis.data.go.kr/B551015"


def old(op: str, name: str) -> str:
    return f"{BASE}/{op}/{name}"


def new(name: str) -> str:
    return f"{BASE}/{name}/get{name}"


# probe 가 시도할 파라미터 조합. 앞에서부터 순서대로.
PARAM_VARIANTS = [
    {"meet": 1, "rc_date": "{date}"},
    {"meet": "1", "rc_date": "{date}"},
    {"meet": 1, "rcDate": "{date}"},
    {"meet": 1, "rc_month": "{month}"},
    {"meet": 1, "rc_year": "{year}"},
    {"meet": 1},
    {},
]


DATASETS = {
    # ══════════════════ 필수 (없으면 프로젝트 정지) ══════════════════
    "race_result": {
        "label": "경주성적정보 — ★ 정답 라벨(착순)",
        "portal_id": "15063979",
        "priority": 1,
        "paths": [
            old("API214_1", "RaceDetailResult_1"),
            old("API299", "Race_Result"),
            new("raceresult"),
            old("API4_2", "raceResult_2"),
            old("API4_1", "raceResult_1"),
        ],
    },
    "entries": {
        "label": "출전 등록말 정보(출마표) — ★ 발주 전 예측 입력",
        "portal_id": "15056699",
        "priority": 1,
        "paths": [
            old("API23_1", "entryRaceHorse_1"),
            old("API210", "EntryRaceInfo"),
            new("entryracehorse"),
        ],
    },
    "race_plan_ai": {
        "label": "AI학습용 경주계획 — ★ 신청 완료됨. 출마표 대체 가능성",
        "portal_id": "?",
        "priority": 1,
        "paths": [
            new("raceplan"),
            new("airaceplan"),
            new("racescheduleai"),
            old("API299_1", "RacePlan_1"),
        ],
    },
    "odds_daily": {
        "label": "경마시행당일 확정배당율종합 — ★ 인기도 축",
        "portal_id": "15119558",
        "priority": 1,
        "paths": [
            new("racedividend"),
            new("dividendtotal"),
            new("racedividendtotal"),
            old("API186", "Payoff_totalInfo"),
        ],
    },
    "odds_total": {
        "label": "확정배당율 통합정보 — 위의 대체/보완",
        "portal_id": "15058559",
        "priority": 1,
        "paths": [
            old("API186", "Payoff_totalInfo"),
            old("API186_1", "Payoff_total_1"),
            new("payofftotal"),
        ],
    },

    # ══════════════════ 피처용 (축을 만들려면 필요) ══════════════════
    "race_detail": {
        "label": "경주별상세성적표 — 구간기록·코너통과순위 → 주행 축 + 3D 재생",
        "portal_id": "15089492",
        "priority": 2,
        "paths": [new("racedetailresult")],
    },
    "section": {
        "label": "경주 구간별 성적 — 주행 축 정밀도",
        "portal_id": "15057847",
        "priority": 2,
        "paths": [new("racesectionresult"), new("sectionrecord")],
    },
    "horse_info": {
        "label": "경주마 상세정보 — 혈통·통산전적 (⚠️ as-of 재계산 필수)",
        "portal_id": "15058115",
        "priority": 2,
        "paths": [old("API8_2", "raceHorseInfo_2"), new("racehorseinfo")],
    },
    "career_all": {
        "label": "마필·기수·조교사 통산경주기록 — 신청 완료됨",
        "portal_id": "?",
        "priority": 2,
        "paths": [
            new("totalrecord"),
            new("careerrecord"),
            new("horsejockeytrainerrecord"),
        ],
    },
    "horse_result": {
        "label": "경주마 성적정보 — 말별 과거 시퀀스 → 시퀀스 인코더 입력",
        "portal_id": "15058779",
        "priority": 2,
        "paths": [new("racehorseresult")],
    },
    "jockey_result": {
        "label": "기수 성적정보 — 기수 축",
        "portal_id": "15056591",
        "priority": 2,
        "paths": [new("jockeyresult")],
    },
    "rating": {
        "label": "경주마 레이팅 정보",
        "portal_id": "15057323",
        "priority": 2,
        "paths": [new("racehorserating")],
    },

    # ══════════════════ 보조 ══════════════════
    "race_info": {
        "label": "경마경주정보 — 거리·등급·주로상태 보완",
        "portal_id": "15063951",
        "priority": 3,
        "paths": [new("raceinfo")],
    },
    "horse_list": {
        "label": "경주마명단 — 마번↔마명 매핑",
        "portal_id": "15089503",
        "priority": 3,
        "paths": [new("racehorselist")],
    },
    "jockey": {
        "label": "현직기수정보",
        "portal_id": "15086329",
        "priority": 3,
        "paths": [old("API12_1", "jockeyInfo_1"), new("jockeyinfo")],
    },
}

for _k, _v in DATASETS.items():
    pid = _v["portal_id"]
    _v["portal_url"] = (
        f"https://www.data.go.kr/data/{pid}/openapi.do" if pid != "?"
        else "(포털에서 '요청주소' 확인 필요)"
    )

MEET = {1: "서울", 2: "제주", 3: "부산경남"}
