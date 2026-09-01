"""공공데이터포털 KRA API 클라이언트.

설계 원칙
  1. 응답 원본을 그대로 캐시한다 — 파싱 로직이 바뀌어도 재수집하지 않는다.
  2. 개발계정 한도(보통 10,000회/일)를 아끼기 위해 캐시 히트 시 네트워크를 타지 않는다.
  3. data.go.kr 은 인증 실패에도 HTTP 200 을 준다 → 본문의 resultCode 를 반드시 본다.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_INTERVAL = 0.15          # 호출 간 최소 간격(초)
_last_call = 0.0


class ApiError(RuntimeError):
    """data.go.kr 이 HTTP 200 과 함께 돌려준 업무 오류."""


class TransientError(RuntimeError):
    """재시도로 풀릴 수 있는 오류."""


def _cache_key(path: str, params: dict[str, Any]) -> Path:
    payload = json.dumps(
        {"path": path, "params": {k: v for k, v in sorted(params.items()) if k != "serviceKey"}},
        sort_keys=True,
        ensure_ascii=False,
    )
    return CACHE_DIR / f"{hashlib.sha1(payload.encode()).hexdigest()}.json"


def _unwrap(body: dict) -> dict:
    """공통 응답 봉투를 벗기고 오류를 예외로 올린다."""
    resp = body.get("response") or body
    header = resp.get("header") or {}
    code = str(header.get("resultCode", "")).strip()
    msg = header.get("resultMsg", "")

    # 00/0 = 정상. 22=한도초과, 30/31=키문제 → 재시도 무의미
    if code and code not in ("00", "0"):
        raise ApiError(f"resultCode={code} {msg}")

    return resp.get("body") or {}


@retry(
    retry=retry_if_exception_type(TransientError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _request(client: httpx.Client, path: str, params: dict[str, Any]) -> dict:
    global _last_call
    gap = time.monotonic() - _last_call
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)

    try:
        resp = client.get(path, params=params)
        _last_call = time.monotonic()
    except httpx.TransportError as e:
        raise TransientError(str(e)) from e

    if resp.status_code >= 500:
        raise TransientError(f"HTTP {resp.status_code}")
    if resp.status_code == 404:
        raise ApiError("HTTP 404 (엔드포인트 없음)")
    resp.raise_for_status()

    text = resp.text.strip()
    if text.startswith("<"):
        # 인증키 오류 등은 JSON 요청이어도 XML 로 온다
        snippet = text[:300].replace("\n", " ")
        raise ApiError(f"XML 응답: {snippet}")

    return json.loads(text)


def fetch(
    path: str,
    *,
    use_cache: bool = True,
    num_of_rows: int = 500,
    page_no: int = 1,
    **params: Any,
) -> dict:
    """단일 페이지 조회. 반환값은 response.body."""
    key = os.environ.get("KRA_API_KEY")
    if not key:
        raise RuntimeError(
            "KRA_API_KEY 가 없습니다. pipeline/.env 를 만들고 "
            "공공데이터포털의 '일반 인증키(Decoding)' 를 넣어주세요."
        )

    q = {
        "serviceKey": key,          # Decoding 키. 인코딩은 httpx 에 맡긴다
        "_type": "json",
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        **{k: v for k, v in params.items() if v is not None},
    }

    cache_path = _cache_key(path, q)
    if use_cache and cache_path.exists():
        return _unwrap(json.loads(cache_path.read_text(encoding="utf-8")))

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        body = _request(client, path, q)

    cache_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return _unwrap(body)


def items_of(body: dict) -> list[dict]:
    """body 에서 item 리스트를 꺼낸다. 1건이면 dict 로 오는 것을 정규화."""
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    return items


def fetch_all(path: str, *, num_of_rows: int = 500, max_pages: int = 200, **params: Any) -> list[dict]:
    """totalCount 를 보고 전 페이지를 순회해 item 리스트를 모은다."""
    rows: list[dict] = []
    page = 1

    while page <= max_pages:
        body = fetch(path, num_of_rows=num_of_rows, page_no=page, **params)
        batch = items_of(body)
        rows.extend(batch)

        total = int(body.get("totalCount") or 0)
        if page * num_of_rows >= total or not batch:
            break
        page += 1

    return rows
