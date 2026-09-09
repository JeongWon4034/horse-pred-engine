"""RAG — 검색한 사전 항목을 근거로 답변을 만든다.

    PYTHONUTF8=1 uv run --project ../pipeline python rag.py "복승식이 뭐예요?"

설계
  · **근거 없으면 답하지 않는다.** 검색 점수가 임계값 미만이면 모른다고 답하고
    sources 를 빈 배열로 준다 (API 명세 §7 요구사항).
  · LLM 은 **문장을 다듬는 역할만** 한다. 사실은 검색된 사전 원문에서만 온다.
    프롬프트에서 "주어진 근거 밖의 내용을 말하지 말 것"을 강제한다.
  · LLM 키가 없거나 호출이 실패하면 **사전 원문을 그대로 돌려준다.**
    챗봇이 죽지 않고, 답변 품질만 떨어진다.

LLM 설정 (.env)
    LLM_API_KEY=...                 필수
    LLM_BASE_URL=https://...        OpenAI 호환 엔드포인트 (GMS 등)
    LLM_MODEL=gpt-4o-mini           모델명
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from retriever import Doc, Retriever

ENV = Path(__file__).parent / ".env"
MIN_SCORE = 12.0          # 이 아래면 "근거 없음" 으로 본다
MAX_CTX = 3               # 프롬프트에 넣을 근거 문서 수

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

SYSTEM = """너는 한국 경마 입문자를 돕는 사전 도우미다.

규칙
1. 아래 [근거]에 있는 내용만 사용한다. 근거에 없는 사실은 절대 지어내지 않는다.
2. 근거로 답할 수 없으면 "제가 가진 자료에는 없는 내용이에요"라고만 말한다.
3. 경마를 처음 접하는 사람에게 말하듯 쉽게, 2~4문장으로 답한다.
4. 전문용어를 쓰면 괄호로 짧게 풀어 준다.
5. 돈을 걸라고 권하거나 수익을 약속하지 않는다.
6. 영어 원문이 근거에 섞여 있으면 한국어로 옮겨 설명한다."""


def load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    used_llm: bool = False
    note: str = ""


def call_llm(question: str, ctx: str, env: dict[str, str]) -> str | None:
    key = env.get("LLM_API_KEY")
    base = env.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = env.get("LLM_MODEL", "gpt-4o-mini")
    if not key:
        return None
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"[근거]\n{ctx}\n\n[질문]\n{question}"},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=40) as r:
            j = json.loads(r.read().decode("utf-8"))
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [LLM 실패] {str(e)[:100]}")
        return None


def fallback(hits: list[tuple[Doc, float]]) -> str:
    """LLM 없이 — 사전 원문을 그대로 정리해 돌려준다."""
    d = hits[0][0]
    body = re.sub(r"\s+", " ", d.body).strip()
    if len(body) > 400:
        body = body[:400].rsplit(" ", 1)[0] + " …"
    en = f" ({d.term_en})" if d.term_en else ""
    return f"{d.term_ko}{en}\n\n{body}"


def ask(question: str, r: Retriever, env: dict[str, str] | None = None) -> Answer:
    env = env if env is not None else load_env()
    # 같은 용어가 영문 표기만 달리해 여러 항목으로 실려 있다("복승식 QUINELLA" / "복승식 EITHER-ORDER").
    # 근거로는 하나만 남기되 설명은 합쳐서 넘긴다.
    raw = r.search(question, MAX_CTX * 3)
    merged: dict[str, tuple[Doc, float]] = {}
    for d, s in raw:
        prev = merged.get(d.term_ko)
        if prev is None:
            merged[d.term_ko] = (d, s)
        elif d.body not in prev[0].body:
            prev[0].body = f"{prev[0].body} / {d.body}"
    hits = sorted(merged.values(), key=lambda x: -x[1])[:MAX_CTX]
    if not hits or hits[0][1] < MIN_SCORE:
        return Answer("제가 가진 자료에는 없는 내용이에요. 마사회 경마용어 사전에 실린 용어만 답할 수 있어요.",
                      [], False, "근거 없음")

    ctx = "\n\n".join(
        f"- 용어: {d.term_ko}{f' ({d.term_en})' if d.term_en else ''}\n  설명: {d.body[:800]}"
        for d, _ in hits)
    text = call_llm(question, ctx, env)
    used = text is not None
    if not used:
        text = fallback(hits)

    sources = [{"title": d.title, "term": d.term_ko, "termEn": d.term_en,
                "sourceUrl": d.source_url, "score": round(s, 1)} for d, s in hits]
    return Answer(text, sources, used, "" if used else "LLM 없이 사전 원문 반환")


if __name__ == "__main__":
    import sys
    r = Retriever()
    env = load_env()
    print(f"색인 {len(r.docs):,}개 · LLM {'있음' if env.get('LLM_API_KEY') else '없음(원문 반환 모드)'}\n")
    for q in (sys.argv[1:] or ["복승식이 뭐예요?", "단승식이랑 연승식 뭐가 달라요?",
                               "삼쌍승식 어려운가요?", "오늘 저녁 뭐 먹지?"]):
        a = ask(q, r, env)
        print(f"Q. {q}\nA. {a.answer}")
        print(f"   근거 {len(a.sources)}개" + (f" — {', '.join(s['term'] for s in a.sources)}" if a.sources else "")
              + (f"  [{a.note}]" if a.note else "") + "\n")
