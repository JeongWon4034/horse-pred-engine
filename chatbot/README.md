# 경마 AI 사전 챗봇 (RAG)

마사회 경마용어 사전 1,528개를 근거로 답하는 챗봇. 팀 API 명세서 §7 · §11-4 형식을 따른다.

**근거가 없으면 답하지 않는다.** 검색 점수가 임계값 미만이면 "자료에 없다"고 답하고
`sources` 를 빈 배열로 준다 (명세서 요구사항).

---

## 빠른 시작

```bash
cd chatbot
uv sync
PYTHONUTF8=1 uv run python ingest_glossary.py    # 용어 수집 (약 2분, 1회만)
PYTHONUTF8=1 uv run python server.py             # http://127.0.0.1:8100/docs
```

LLM 키 없이도 동작한다. 그 경우 **사전 원문을 그대로 반환**하고 답변 문장만 덜 다듬어진다.

### LLM 붙이기

`chatbot/.env` 에 넣는다. **커밋 금지** (gitignore 됨).

```
LLM_API_KEY=키
LLM_BASE_URL=https://.../v1     # OpenAI 호환 엔드포인트 (GMS 등)
LLM_MODEL=gpt-4o-mini
```

LLM 은 **문장을 다듬는 역할만** 한다. 사실은 검색된 사전 원문에서만 나온다.
프롬프트에서 "근거 밖 내용 금지"를 강제하고, 호출이 실패하면 원문 반환으로 되돌아간다.

---

## 구조

| 파일 | 역할 |
|---|---|
| `ingest_glossary.py` | 마사회 사전 수집 → `data/glossary.jsonl` |
| `retriever.py` | BM25 + 용어명 정확매칭 검색 |
| `rag.py` | 검색 → 근거 정리 → LLM 답변 (실패 시 원문) |
| `server.py` | FastAPI. 명세서 형식 엔드포인트 |

### 왜 임베딩이 아니라 BM25 인가

사전 질문은 "복승식이 뭐예요?" 처럼 **용어가 질문에 그대로 들어 있다.**
이런 경우 키워드 검색이 임베딩보다 정확하고 모델 다운로드도 필요 없다.
"돈 어떻게 걸어요?" 같은 우회 표현에는 임베딩이 유리하므로, 나중에 붙일 수 있게
`Retriever.search()` 인터페이스를 분리해 두었다.

한국어는 형태소 분석기 없이 처리한다 — 조사 제거 + 2·3글자 n-gram 색인으로
"복승식이" → "복승식" 이 걸린다. konlpy 는 자바 의존이라 넣지 않았다.

---

## API

| Method | Path | 명세서 |
|---|---|---|
| POST | `/dictionary/chat` | §7 챗봇 질의 |
| GET | `/dictionary/terms?q=&page=&size=` | §7 용어 검색 |
| GET | `/dictionary/terms/{termId}` | §7 용어 상세 |
| POST | `/ai/rag/query` | §11-4 내부 질의 |
| GET | `/ai/rag/documents` | §11-4 색인 문서 |
| GET | `/ai/health` | §11-5 상태 |

**POST `/dictionary/chat`**

```json
// 요청
{ "sessionId": null, "question": "복승식이 뭐예요?" }
// 응답
{ "sessionId": "uuid", "answer": "…", "beginnerMode": true, "usedLlm": true,
  "sources": [ { "title": "마사회 경마용어 사전", "term": "복승식",
                 "termId": 173, "sourceUrl": "https://race.kra.co.kr/…" } ] }
```

> 윈도우 `curl` 로 한글을 POST 하면 인코딩 때문에 실패한다. `/docs` 나 파이썬으로 시험할 것.

---

## 아웃바운드

LLM 호출 한 곳만 외부를 탄다. 막혀 있으면 원문 반환 모드로 동작한다.
용어 수집(`ingest_glossary.py`)도 외부를 타지만 **1회성**이므로 서버 밖에서 돌려
`data/glossary.jsonl` 만 넣어 주어도 된다.

세션은 메모리에만 둔다. 영속화는 백엔드의 `chat_session` · `chat_message` 테이블이 맡는다
(명세서 스키마). AI 서버는 순수 계산 서버라 DB 를 갖지 않는다.

---

## 알려진 한계

| # | 내용 |
|---|---|
| 1 | **사전 원문에 영어가 많다.** LLM 이 한국어로 옮기게 프롬프트에 넣었으나, LLM 없이 쓰면 영어가 그대로 나온다 |
| 2 | **마사회 사전에 없는 용어는 답 못 한다.** 예: `함수율`(우리 데이터 용어). 우리 프로젝트 문서를 두 번째 코퍼스로 추가하면 해결 |
| 3 | **경마 시행규정을 아직 안 넣었다.** 스키마의 `doc_type=RULE` 자리가 비어 있다 |
| 4 | **답변 품질을 정량 평가하지 않았다.** 질문 30개쯤으로 검색 정확도를 재야 한다 |
| 5 | 세션 대화 맥락을 답변에 쓰지 않는다. 매 질문이 독립적이다 |

---

## 다음

1. LLM 키 연결 후 답변 품질 확인
2. 우리 프로젝트 용어(피처 설명·지표 정의)를 두 번째 코퍼스로 추가
3. 경마 시행규정 수집 (`doc_type=RULE`)
4. 질문 30개로 검색 정확도 측정
