"""검색기 — 질문에 맞는 사전 항목을 찾는다.

무거운 의존성 없이 BM25(키워드) + 용어 직접매칭을 섞는다.

왜 임베딩이 아니라 BM25 인가
  · 사전 질문은 "복승식이 뭐예요?" 처럼 **용어가 질문에 그대로 들어 있다.**
    이런 경우 키워드 검색이 임베딩보다 정확하고, 모델 다운로드도 필요 없다.
  · 임베딩은 "돈 어떻게 걸어요?" 같은 우회 표현에 강하다. 나중에 붙일 수 있게
    search() 인터페이스를 분리해 두었다.

한국어 처리
  · 형태소 분석기를 쓰지 않는다(konlpy 는 자바 의존이라 설치가 무겁다).
  · 대신 ① 공백 토큰 ② 조사 제거 ③ 2·3글자 부분문자열(n-gram)을 함께 색인한다.
    "복승식이" → "복승식", "복승", "승식", "복승식" … 로 걸린다.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data" / "glossary.jsonl"

# 조사·어미 — 끝에서 떼어낸다. 긴 것부터 시도한다.
JOSA = ("이라고", "이라는", "에서는", "에게서", "라고는", "이란", "라는", "에서", "에게", "부터",
        "까지", "으로", "이나", "이든", "이면", "인가", "인지", "은요", "는요", "이요",
        "과", "와", "은", "는", "이", "가", "을", "를", "의", "에", "로", "도", "만", "요", "야", "냐")

STOP = {"뭐", "무엇", "뭔가", "어떤", "어떻게", "왜", "언제", "누가", "알려줘", "설명", "해줘",
        "인가요", "예요", "이에요", "인가", "일까", "하나요", "있나요", "뜻", "의미", "차이", "그리고"}


def strip_josa(w: str) -> str:
    for j in JOSA:
        if len(w) > len(j) + 1 and w.endswith(j):
            return w[: -len(j)]
    return w


def tokens(text: str) -> list[str]:
    """공백 토큰 + 조사 제거형 + 2·3글자 n-gram."""
    out: list[str] = []
    for w in re.findall(r"[가-힣]+|[a-zA-Z]+|\d+", text.lower()):
        if w in STOP:
            continue
        out.append(w)
        s = strip_josa(w)
        if s != w and len(s) >= 2:
            out.append(s)
        base = s if len(s) >= 2 else w
        if len(base) >= 3:                       # 한글 부분문자열
            for n in (2, 3):
                out += [base[i:i + n] for i in range(len(base) - n + 1)]
    return out


@dataclass
class Doc:
    idx: int
    term_ko: str
    term_en: str
    body: str
    title: str
    source_url: str

    @property
    def text(self) -> str:
        return f"{self.term_ko} {self.term_en} {self.body}"


class Retriever:
    """BM25 + 용어명 정확매칭 가산점."""

    K1, B = 1.4, 0.75
    TERM_BONUS = 6.0          # 질문에 용어명이 그대로 들어 있으면 크게 가산

    def __init__(self, path: Path = DATA):
        self.docs: list[Doc] = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            self.docs.append(Doc(i, d["term_ko"], d.get("term_en", ""), d["body"],
                                 d.get("title", ""), d.get("source_url", "")))
        self._build()

    def _build(self) -> None:
        self.tf: list[Counter] = []
        self.len: list[int] = []
        df: Counter = Counter()
        for d in self.docs:
            t = Counter(tokens(d.text))
            self.tf.append(t)
            self.len.append(sum(t.values()))
            df.update(t.keys())
        n = len(self.docs)
        self.avg = sum(self.len) / max(n, 1)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.post: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.post[w].append(i)
        # 용어명 → 문서 (정확매칭용). 괄호 안 한자는 떼고 색인한다.
        self.by_term: dict[str, list[int]] = defaultdict(list)
        for d in self.docs:
            for name in {d.term_ko, re.sub(r"\(.*?\)", "", d.term_ko).strip()}:
                if len(name) >= 2:
                    self.by_term[name.lower()].append(d.idx)

    def search(self, query: str, k: int = 4) -> list[tuple[Doc, float]]:
        q = tokens(query)
        if not q:
            return []
        scores: dict[int, float] = defaultdict(float)
        for w in set(q):
            if w not in self.post:
                continue
            idf = self.idf[w]
            for i in self.post[w]:
                f = self.tf[i][w]
                dl = self.len[i]
                scores[i] += idf * f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * dl / self.avg))
        # 질문에 용어명이 통째로 들어 있으면 가산
        low = query.lower()
        for name, idxs in self.by_term.items():
            if name in low:
                for i in idxs:
                    scores[i] += self.TERM_BONUS * len(name)
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.docs[i], s) for i, s in top]


if __name__ == "__main__":
    import sys
    r = Retriever()
    print(f"색인 {len(r.docs):,}개 용어\n")
    for q in (sys.argv[1:] or ["복승식이 뭐예요?", "단승식과 연승식 차이", "함수율이 높으면 어떻게 되나요"]):
        print(f"Q. {q}")
        for d, s in r.search(q, 3):
            print(f"   [{s:6.1f}] {d.term_ko} {d.term_en}  — {d.body[:70]}...")
        print()
