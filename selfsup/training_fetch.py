"""일별훈련(조교) 원장 수집 — 공공데이터 15058782, 경로 API18_1/dailyTraining_1 (포털은 앞 세그먼트를 안 보여준다. 브루트포스로 찾음).

  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.training_fetch            # 2010-01 ~ 오늘, 3개장
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.training_fetch 2016 2026

월 × 경마장 한 콜(numOfRows=20000, 월 최대 ~17k 행). 진행상황 JSON 으로 이어받기. 출력 pipeline/data/raw/training_api.csv (gitignore).
필드: meet · trDate · hrNo · hrName · prGubun(기승자 구분) · prNo · run1Cnt(구보) · run2Cnt(습보) · trTerm(훈련시간 초) · stTime · spTime · chulGubun · part · partNo · trName
"""
from __future__ import annotations

import csv
import json
import ssl
import sys
import time
import urllib.request
from datetime import date

from . import data as D

OUT = D.PIPE / "data" / "raw" / "training_api.csv"
PROG = D.PIPE / "data" / "raw" / "training_progress.json"
BASE = "https://apis.data.go.kr/B551015/API18_1/dailyTraining_1"
COLS = ["meet", "trDate", "hrNo", "hrName", "prGubun", "prNo", "run1Cnt", "run2Cnt", "trTerm",
        "stTime", "spTime", "chulGubun", "part", "partNo", "trName"]
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE


def key() -> str:
    for l in (D.PIPE / ".env").open(encoding="utf-8"):
        if l.startswith("KRA_API_KEY_ENCODED"):
            return l.split("=", 1)[1].strip()
    raise SystemExit("pipeline/.env 에 KRA_API_KEY_ENCODED 가 없다")


def call(k: str, meet: int, month: str, page: int = 1, rows: int = 20000):
    url = f"{BASE}?serviceKey={k}&_type=json&numOfRows={rows}&pageNo={page}&meet={meet}&tr_month={month}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), context=CTX, timeout=180) as r:
                b = json.loads(r.read().decode("utf-8", "replace"))["response"]["body"]
            items = b.get("items") or {}
            rows_ = items.get("item", []) if isinstance(items, dict) else []
            if isinstance(rows_, dict):
                rows_ = [rows_]
            return rows_, int(b.get("totalCount", 0))
        except Exception as e:
            last = e; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"4회 실패 {meet} {month}: {last}")


def main(y0=2010, y1=None):
    y1 = y1 or date.today().year
    k = key()
    done = set(json.load(PROG.open(encoding="utf-8"))) if PROG.exists() else set()
    new_file = not OUT.exists() or not done
    f = OUT.open("w" if new_file else "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", restval="")
    if new_file:
        w.writeheader()
    total, t0 = 0, time.time()
    for y in range(y0, y1 + 1):
        ycount = 0
        for m in range(1, 13):
            if (y, m) > (date.today().year, date.today().month):
                break
            month = f"{y}{m:02d}"
            for meet in (1, 2, 3):
                tag = f"{month}_{meet}"
                if tag in done:
                    continue
                rows, tc = call(k, meet, month)
                page = 2
                while len(rows) < tc:                      # 안전장치
                    more, _ = call(k, meet, month, page)
                    if not more:
                        break
                    rows += more; page += 1
                w.writerows(rows); f.flush()
                total += len(rows); ycount += len(rows); done.add(tag)
                PROG.write_text(json.dumps(sorted(done)), encoding="utf-8")
        print(f"  {y}: {ycount:>8,}행  (누적 {total:,} / {time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"\n끝. {total:,}행 → {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2010, int(a[1]) if len(a) > 1 else None)
