"""As-of retrieval over static documents and live facts (ADR-010).

Scoring, per candidate:

- lexical: Okapi BM25 over one corpus made of the documents plus the facts valid at `as_of`.
  The corpus is rebuilt per call because the set of live facts depends on `as_of`; at support
  knowledge-base sizes that is cheaper than keeping an index in sync with a moving clock.
- recency prior, facts only: `bm25 * (1 + recency_weight * 0.5 ** (age / half_life_s))`, where
  age is time since the fact became valid. A fresh fact gets up to `1 + recency_weight` times
  the weight of a static document with the same text match; an old one decays toward parity.
- incident boost: an active incident (predicate "incident", valid at `as_of`) whose subject
  shares a term with the query gets `incident_boost` added. This is what puts "Order delivery
  delays" above the generic "order is late" article while the incident is open. Resolved
  incidents are not valid at a later `as_of`, so they drop out with no extra rule.

Only candidates with a lexical match or a boost are returned. Ties break on kind then id, so
results are deterministic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from switchboard.knowledge.bitemporal import Fact, FactStore
from switchboard.knowledge.statuspage import INCIDENT_PREDICATE

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "why",
        "will",
        "with",
        "you",
        "your",
    )
)
# Suffixes that end in "s" without being plurals (status, analysis, address).
_KEEP_S = ("ss", "us", "is")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords dropped, simple plural folding."""
    out: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOPWORDS:
            continue
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith(_KEEP_S):
            raw = raw[:-1]
        out.append(raw)
    return out


class BM25:
    """Okapi BM25 with the non-negative idf variant `ln(1 + (N - n + 0.5) / (n + 0.5))`."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs = [Counter(doc) for doc in corpus]
        self._lengths = [len(doc) for doc in corpus]
        self._n = len(corpus)
        self._avgdl = (sum(self._lengths) / self._n) if self._n else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        self._df = df

    def idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        return math.log(1.0 + (self._n - n + 0.5) / (n + 0.5))

    def scores(self, query: Iterable[str]) -> list[float]:
        terms = list(dict.fromkeys(query))
        out: list[float] = []
        for counts, length in zip(self._docs, self._lengths, strict=True):
            norm = self.k1 * (1.0 - self.b + self.b * length / self._avgdl) if self._avgdl else 0.0
            total = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    total += self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm)
            out.append(total)
        return out


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Retrieved:
    """One result with enough provenance for the grounding gate to cite it as evidence."""

    kind: Literal["document", "fact"]
    source_id: str  # doc_id or fact_id
    score: float
    text: str
    valid_from: float | None = None
    valid_to: float | None = None
    recorded_at: float | None = None
    boosted: bool = False
    fact: Fact | None = None
    document: Document | None = None


def _flatten(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _flatten(v)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def fact_text(fact: Fact) -> str:
    return " ".join([fact.subject, fact.predicate, *_flatten(fact.value)])


class KnowledgeIndex:
    def __init__(
        self,
        store: FactStore,
        *,
        half_life_s: float = 6 * 3600.0,
        recency_weight: float = 0.5,
        incident_boost: float = 5.0,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        self._store = store
        self._half_life_s = half_life_s
        self._recency_weight = recency_weight
        self._incident_boost = incident_boost
        self._k1 = k1
        self._b = b
        self._documents: dict[str, Document] = {}

    def add_document(self, document: Document) -> None:
        if document.doc_id in self._documents:
            raise ValueError(f"duplicate document id {document.doc_id!r}")
        self._documents[document.doc_id] = document

    def retrieve(
        self, query: str, as_of: float, k: int = 5, *, known_at: float | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []
        terms = tokenize(query)
        docs = [self._documents[i] for i in sorted(self._documents)]
        facts = self._store.query(as_of=as_of, known_at=known_at)
        doc_texts = [" ".join([d.title, d.body, *d.tags]) for d in docs]
        fact_texts = [fact_text(f) for f in facts]
        bm25 = BM25([tokenize(t) for t in doc_texts + fact_texts], k1=self._k1, b=self._b)
        lexical = bm25.scores(terms)
        query_terms = set(terms)

        results: list[Retrieved] = []
        for doc, text, score in zip(docs, doc_texts, lexical, strict=False):
            if score > 0:
                results.append(
                    Retrieved(
                        kind="document", source_id=doc.doc_id, score=score, text=text, document=doc
                    )
                )
        for fact, text, score in zip(facts, fact_texts, lexical[len(docs) :], strict=True):
            age = max(0.0, as_of - fact.valid_from)
            decay = 0.5 ** (age / self._half_life_s)
            total = score * (1.0 + self._recency_weight * decay)
            boosted = fact.predicate == INCIDENT_PREDICATE and bool(
                query_terms & set(tokenize(fact.subject))
            )
            if boosted:
                total += self._incident_boost
            if total > 0:
                results.append(
                    Retrieved(
                        kind="fact",
                        source_id=fact.fact_id,
                        score=total,
                        text=text,
                        valid_from=fact.valid_from,
                        valid_to=fact.valid_to,
                        recorded_at=fact.recorded_at,
                        boosted=boosted,
                        fact=fact,
                    )
                )
        results.sort(key=lambda r: (-r.score, r.kind, r.source_id))
        return results[:k]
