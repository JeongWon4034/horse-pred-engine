"""경주 원장 백필 — KRA 경주상세성적 API(API214_1) 로 2010~2026 전량 수집.

    PYTHONUTF8=1 uv run python -m ingest.ledger              # 2010~2026 전량
    PYTHONUTF8=1 uv run python -m ingest.ledger 2026 2026    # 범위 지정
    PYTHONUTF8=1 uv run python -m ingest.ledger --date 20260906   # 하루만 추가

왜 이 엔드포인트인가
  팀 백필(`docs/dataset/tools/backfill_ledger.py`)은 `API4_3/raceResult_3` 를 쓰는데
  우리 키로는 403 이다(그 API 는 신청 안 됨). `API214_1/RaceDetailResult_1` 은 통하고,
  팀 빌더 `build_v2.load_ledger()` 가 요구하는 컬럼 22개를 전부 준다.
  2026-08-30 기준 우리 test.parquet 176행과 착순 100% 일치 확인.

산출물
  data/raw/ledger_api.csv        말-경주 단위 원장 (컬럼 90개)
  data/raw/ledger_progress.json  완료한 (meet, year) — 재실행 시 건너뛴다

주의
  · 응답 컬럼 순서가 호출마다 다를 수 있어 헤더는 첫 배치 기준으로 고정하고
    이후 행은 그 헤더에 맞춰 정렬한다. 새 컬럼이 나오면 경고만 하고 버린다.
  · `ord == 0` 은 아직 안 열린 예정 경주다. 원장에는 그대로 담고 빌더가 거른다.
  · 개발계정 한도(1만회/일)에 견줘 호출 수는 60회 안팎이라 여유 있다.
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
OUT = ROOT / "data" / "raw" / "ledger_api.csv"
PROG = ROOT / "data" / "raw" / "ledger_progress.json"

BASE = "https://apis.data.go.kr/B551015/API214_1/RaceDetailResult_1"
MEETS = {1: "서울", 2: "제주", 3: "부경"}
ROWS = 10000              # 한 페이지 상한 (실측: 1만이 최대, 초과분은 pageNo 로)
MIN_INTERVAL = 0.2        # 호출 간 최소 간격(초)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
_last = 0.0


def key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("KRA_API_KEY_ENCODED"):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{ENV} 에 KRA_API_KEY_ENCODED 가 없다")


def call(k: str, query: str, page: int = 1) -> tuple[list[dict], int]:
    """(행 목록, 전체 건수). data.go.kr 은 인증 실패에도 200 을 주므로 resultCode 를 본다."""
    global _last
    gap = time.time() - _last
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    url = f"{BASE}?serviceKey={k}&numOfRows={ROWS}&pageNo={page}&_type=json&{query}"
    with urllib.request.urlopen(url, context=CTX, timeout=180) as r:
        body = r.read().decode("utf-8", "replace")
    _last = time.time()
    j = json.loads(body)
    resp = j.get("response", {})
    code = str(resp.get("header", {}).get("resultCode", "?"))
    if code not in ("0", "00"):
        raise RuntimeError(f"API 오류 {code}: {resp.get('header', {}).get('resultMsg')}  ({query} p{page})")
    b = resp.get("body", {})
    items = b.get("items")
    rows = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(rows, dict):          # 1건이면 dict 로 온다
        rows = [rows]
    return rows, int(b.get("totalCount", 0))


def fetch(k: str, query: str) -> list[dict]:
    """페이지를 끝까지 넘겨 전량."""
    rows, total = call(k, query, 1)
    page = 2
    while len(rows) < total:
        more, _ = call(k, query, page)
        if not more:
            break
        rows += more
        page += 1
    if len(rows) != total:
        print(f"    ! {query}: 받은 {len(rows):,} ≠ 전체 {total:,}")
    return rows


def load_progress() -> set[str]:
    return set(json.loads(PROG.read_text(encoding="utf-8"))) if PROG.exists() else set()


def save_progress(done: set[str]) -> None:
    PROG.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")


def header_of(path: Path) -> list[str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return next(csv.reader(f), None)


def append(rows: list[dict]) -> int:
    """CSV 에 덧붙인다. 헤더는 최초 1회만 쓰고 이후 행은 그 순서에 맞춘다."""
    if not rows:
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = header_of(OUT)
    new_file = head is None
    if new_file:
        head = sorted({c for r in rows for c in r})
    extra = {c for r in rows for c in r} - set(head)
    if extra:
        print(f"    ! 헤더에 없는 컬럼 무시: {sorted(extra)[:5]}")
    with OUT.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in head})
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("y0", nargs="?", type=int, default=2010)
    ap.add_argument("y1", nargs="?", type=int, default=2026)
    ap.add_argument("--date", help="YYYYMMDD 하루만 추가 (증분 갱신용)")
    ap.add_argument("--reset", action="store_true", help="기존 CSV·진행상황을 지우고 처음부터")
    a = ap.parse_args()

    k = key()
    if a.reset:
        OUT.unlink(missing_ok=True)
        PROG.unlink(missing_ok=True)
        print("기존 원장 삭제")

    if a.date:
        n = append(fetch(k, f"rc_date={a.date}"))
        print(f"{a.date} → {n:,}행 추가  ({OUT})")
        return

    done = load_progress()
    t0 = time.time()
    total = 0
    for meet, name in MEETS.items():
        for year in range(a.y0, a.y1 + 1):
            tag = f"{meet}-{year}"
            if tag in done:
                continue
            try:
                rows = fetch(k, f"meet={meet}&rc_year={year}")
            except Exception as e:
                print(f"  {name} {year}: 실패 {e}")
                continue
            n = append(rows)
            total += n
            done.add(tag)
            save_progress(done)
            print(f"  {name} {year}  {n:>6,}행  (누적 {total:,}, {time.time()-t0:.0f}초)", flush=True)

    print(f"\n끝. {total:,}행 → {OUT}  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
