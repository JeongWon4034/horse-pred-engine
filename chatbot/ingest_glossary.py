"""마사회 경마용어 사전 수집 — 1,528개 용어.

    PYTHONUTF8=1 uv run --project ../pipeline python ingest_glossary.py

출처: https://race.kra.co.kr/raceguide/RaceWordSearchService.do (로그인·API키 불필요)
산출: data/glossary.jsonl  — {term_ko, term_en, body, source_url, page}

주의
  · 페이지 인코딩이 EUC-KR 이다. UTF-8 로 읽으면 깨진다.
  · 목록 페이지에 정의문이 이미 들어 있다(숨은 div). 상세 페이지를 따로 안 열어도 된다.
  · 본문에 <img> 가 섞여 있어 태그를 지우고 공백을 정리한다.
"""
from __future__ import annotations

import html as html_mod
import json
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://race.kra.co.kr/raceguide/RaceWordSearchService.do"
OUT = Path(__file__).parent / "data" / "glossary.jsonl"
PER_PAGE = 10
MIN_INTERVAL = 0.3

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

ITEM = re.compile(r'<li>\s*<a href="#">(.*?)</a>\s*<div>(.*?)</div>\s*</li>', re.S)
TAG = re.compile(r"<[^>]+>")
# 국문 뒤에 영문이 붙어 있다: "복승식DOUBLE WIN" / "동요병(動搖病)WOBBLER SYSDROM"
SPLIT_EN = re.compile(r"^(.*?)([A-Z][A-Z0-9 ,.\-()=/&']*)$")


def fetch(page: int) -> str:
    req = urllib.request.Request(f"{BASE}?pageIndex={page}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, context=CTX, timeout=40).read().decode("euc-kr", "replace")


def clean(s: str) -> str:
    s = TAG.sub(" ", s)
    s = html_mod.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def split_term(raw: str) -> tuple[str, str]:
    """'복승식DOUBLE WIN' → ('복승식', 'DOUBLE WIN'). 영문이 없으면 빈 문자열."""
    t = clean(raw)
    m = SPLIT_EN.match(t)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    return t, ""


def total_count() -> int:
    """'총 1528건' 을 찾는다. 숫자가 <strong> 등으로 감싸여 있어 태그를 먼저 지운다."""
    text = clean(fetch(1))
    m = re.search(r"총\s*([\d,]+)\s*건", text)
    return int(m.group(1).replace(",", "")) if m else 0


def main() -> None:
    n = total_count()
    pages = (n + PER_PAGE - 1) // PER_PAGE
    if pages == 0:                                  # 건수 파싱 실패 시 빈 페이지 나올 때까지
        pages = 200
        print("건수 파싱 실패 — 빈 페이지가 나올 때까지 진행", flush=True)
    print(f"용어 {n:,}개 / {pages}페이지", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    with OUT.open("w", encoding="utf-8") as f:
        for p in range(1, pages + 1):
            try:
                items = ITEM.findall(fetch(p))
            except Exception as e:
                print(f"  [{p}] 실패 {str(e)[:60]}", file=sys.stderr, flush=True)
                continue
            if not items:                            # 마지막 페이지를 지나면 빈 목록
                print(f"  {p}페이지에서 항목 없음 — 종료", flush=True)
                break
            for raw_term, raw_body in items:
                ko, en = split_term(raw_term)
                body = clean(raw_body)
                key = f"{ko}|{en}"
                if not ko or not body or key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps({
                    "term_ko": ko, "term_en": en, "body": body,
                    "source_url": f"{BASE}?pageIndex={p}", "page": p,
                    "doc_type": "GLOSSARY", "title": "마사회 경마용어 사전",
                }, ensure_ascii=False) + "\n")
                kept += 1
            if p % 20 == 0:
                print(f"  {p}/{pages}페이지  누적 {kept:,}개", flush=True)
            time.sleep(MIN_INTERVAL)
    print(f"완료: {kept:,}개 → {OUT}")


if __name__ == "__main__":
    main()
