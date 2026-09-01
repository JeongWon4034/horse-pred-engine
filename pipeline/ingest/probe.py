"""엔드포인트 브루트포스 탐침.

경로 후보 × 파라미터 후보를 전부 대입해서 살아있는 조합을 찾아낸다.
성공한 조합은 found.json 에 저장되고, 이후 수집 스크립트가 그걸 읽는다.

    uv run python -m ingest.probe                 # 최근 일요일, 우선순위 1~2
    uv run python -m ingest.probe 20260830        # 날짜 지정
    uv run python -m ingest.probe 20260830 3      # 우선순위 3까지
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .client import ApiError, fetch, items_of
from .config import DATASETS, MEET, PARAM_VARIANTS

FOUND_PATH = Path(__file__).resolve().parents[1] / "found.json"

NO_KEY = """
  KRA_API_KEY 가 없습니다.

  pipeline/.env 파일을 만들고 아래 한 줄을 넣어주세요:

      KRA_API_KEY=발급받은_일반인증키_Decoding

  ※ 인증키는 계정당 1개이며, 활용신청한 데이터셋 전부에 그대로 쓰입니다.
"""


def last_sunday() -> str:
    d = date.today()
    back = (d.weekday() + 1) % 7 or 7
    return (d - timedelta(days=back)).strftime("%Y%m%d")


def probe_one(name: str, spec: dict, rc_date: str) -> dict | None:
    print()
    print("─" * 76)
    print(f"[{name}]  {spec['label']}")

    combos = [(p, v) for p in spec["paths"] for v in PARAM_VARIANTS]
    tried = 0

    for path, variant in combos:
        params = {
            k: (v.format(date=rc_date, month=rc_date[:6], year=rc_date[:4])
                if isinstance(v, str) else v)
            for k, v in variant.items()
        }
        tried += 1
        try:
            body = fetch(path, use_cache=False, num_of_rows=5, **params)
        except ApiError:
            continue
        except Exception:                                        # noqa: BLE001
            continue

        rows = items_of(body)
        total = int(body.get("totalCount") or 0)

        # 200 이지만 데이터가 0건이면 파라미터가 안 맞는 것 — 계속 시도
        if not rows and total == 0:
            continue

        short = path.split("/B551015/")[-1]
        print(f"  ✅ {short}")
        print(f"     params  {params}")
        print(f"     total={total}  수신={len(rows)}건  (시도 {tried}회)")

        if rows:
            sample = rows[0]
            print(f"     필드 {len(sample)}개:")
            for k, v in list(sample.items())[:45]:
                print(f"       {k:<22} = {str(v)[:38]}")

        return {"path": path, "params": variant, "n_fields": len(rows[0]) if rows else 0,
                "fields": list(rows[0].keys()) if rows else []}

    print(f"  ❌ 실패 — 경로 {len(spec['paths'])}개 × 파라미터 {len(PARAM_VARIANTS)}개 = {tried}조합 전부 실패")
    print(f"     포털: {spec['portal_url']}")
    print("     → 활용신청 상세의 '요청주소'와 '요청변수'를 config.py 에 넣으세요.")
    return None


def main() -> int:
    if not os.environ.get("KRA_API_KEY"):
        print(NO_KEY)
        return 1

    rc_date = sys.argv[1] if len(sys.argv) > 1 else last_sunday()
    max_prio = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"조회일 {rc_date} · 경마장 {MEET[1]}(meet=1) · 우선순위 ≤ {max_prio}")

    targets = {k: v for k, v in sorted(DATASETS.items(), key=lambda kv: kv[1]["priority"])
               if v["priority"] <= max_prio}

    found: dict[str, dict] = {}
    for name, spec in targets.items():
        hit = probe_one(name, spec, rc_date)
        if hit:
            found[name] = hit

    print()
    print("═" * 76)
    print(f"성공 {len(found)}/{len(targets)}")

    missing = [n for n in targets if n not in found]
    if missing:
        print("\n실패 — 포털에서 '요청주소'를 확인해 config.py 에 추가:")
        for n in missing:
            print(f"  · {targets[n]['label']}")
            print(f"    {targets[n]['portal_url']}")

    if found:
        FOUND_PATH.write_text(
            json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n→ {FOUND_PATH.name} 저장됨. 수집 스크립트가 이걸 읽습니다.")

    # 필수 데이터셋 경고
    critical = {"race_result": "정답 라벨(착순)", "entries": "발주 전 예측 입력"}
    blocked = [f"{v}" for k, v in critical.items() if k not in found]
    if blocked:
        print("\n⚠️  아래가 없으면 모델 학습 자체가 불가능합니다:")
        for b in blocked:
            print(f"    · {b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
