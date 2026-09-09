"""경마 AI 사전 챗봇 서버 — 팀 API 명세서 §7 · §11-4 대응.

    PYTHONUTF8=1 uv run --project ../pipeline python server.py
    → http://127.0.0.1:8100  (문서: /docs)

제공하는 것 (명세서와 같은 요청·응답 형식)

    POST /dictionary/chat        RAG 챗봇 질의        §7
    GET  /dictionary/terms       용어 검색            §7
    GET  /dictionary/terms/{id}  용어 상세            §7
    POST /ai/rag/query           내부용 질의          §11-4
    GET  /ai/rag/documents       색인 문서 목록       §11-4
    GET  /ai/health              상태·데이터 버전     §11-5

아웃바운드
    LLM 호출 한 곳만 외부를 탄다. 키가 없으면 사전 원문을 그대로 돌려주므로
    아웃바운드가 막힌 환경에서도 서버는 정상 동작한다.

세션
    메모리에만 보관한다. 영속화는 백엔드의 chat_session · chat_message 테이블이 맡는다
    (명세서 스키마). AI 서버는 순수 계산 서버라 DB 를 갖지 않는다.
"""
from __future__ import annotations

import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rag import ask, load_env
from retriever import DATA, Retriever

app = FastAPI(title="경마 AI 사전", version="0.1.0")

R: Retriever | None = None
ENV: dict[str, str] = {}
SESS: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))


@app.on_event("startup")
def startup() -> None:
    global R, ENV
    R = Retriever()
    ENV = load_env()
    print(f"색인 {len(R.docs):,}개 · LLM {'있음' if ENV.get('LLM_API_KEY') else '없음(원문 반환 모드)'}")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ── §7 사전 ──────────────────────────────────────────────────────────
class ChatReq(BaseModel):
    sessionId: str | None = None
    question: str = Field(min_length=1, max_length=500)


@app.post("/dictionary/chat")
def chat(req: ChatReq) -> dict:
    sid = req.sessionId or str(uuid.uuid4())
    a = ask(req.question, R, ENV)
    SESS[sid].append({"q": req.question, "a": a.answer, "at": now()})
    return {
        "sessionId": sid,
        "answer": a.answer,
        "sources": [{"title": s["title"], "term": s["term"], "sourceUrl": s["sourceUrl"],
                     "termId": next(d.idx for d in R.docs if d.term_ko == s["term"])}
                    for s in a.sources],
        "beginnerMode": True,
        "usedLlm": a.used_llm,
    }


@app.get("/dictionary/terms")
def terms(q: str = Query("", max_length=100), page: int = 0, size: int = 20) -> dict:
    if q:
        hits = [d for d, _ in R.search(q, k=size * (page + 1))]
    else:
        hits = R.docs
    total = len(hits)
    page_items = hits[page * size: (page + 1) * size]
    return {
        "total": total, "page": page, "size": size,
        "items": [{"termId": d.idx, "termKo": d.term_ko, "termEn": d.term_en,
                   "summary": d.body[:120]} for d in page_items],
    }


@app.get("/dictionary/terms/{term_id}")
def term_detail(term_id: int) -> dict:
    if not 0 <= term_id < len(R.docs):
        raise HTTPException(404, "term not found")
    d = R.docs[term_id]
    return {"termId": d.idx, "termKo": d.term_ko, "termEn": d.term_en,
            "body": d.body, "title": d.title, "sourceUrl": d.source_url}


# ── §11-4 내부 RAG ───────────────────────────────────────────────────
class RagReq(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    topK: int = 3


@app.post("/ai/rag/query")
def rag_query(req: RagReq) -> dict:
    a = ask(req.query, R, ENV)
    return {"answer": a.answer, "sources": a.sources, "usedLlm": a.used_llm,
            "note": a.note, "computedAt": now()}


@app.get("/ai/rag/documents")
def documents() -> dict:
    return {"documents": [{
        "title": "마사회 경마용어 사전",
        "docType": "GLOSSARY",
        "sourceUrl": "https://race.kra.co.kr/raceguide/RaceWordSearchService.do",
        "chunkCount": len(R.docs),
        "indexedAt": datetime.fromtimestamp(DATA.stat().st_mtime).isoformat(timespec="seconds"),
    }]}


# ── §11-5 공통 ───────────────────────────────────────────────────────
@app.get("/ai/health")
def health() -> dict:
    return {"status": "UP", "terms": len(R.docs),
            "llm": bool(ENV.get("LLM_API_KEY")),
            "dataFile": DATA.name, "checkedAt": now()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
