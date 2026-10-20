import json
from pathlib import Path

import pytest

from switchboard.knowledge.bitemporal import BitemporalStore
from switchboard.knowledge.retrieval import BM25, Document, KnowledgeIndex, tokenize
from switchboard.knowledge.statuspage import StatusPageIngestor, parse_timestamp

FIXTURE = Path(__file__).parent / "fixtures" / "incidents_open.json"
HOUR = 3600.0


def test_tokenize_lowercases_drops_stopwords_and_folds_plurals() -> None:
    assert tokenize("My Orders are LATE, and delays!") == ["order", "late", "delay"]
    assert tokenize("status 48213") == ["status", "48213"]
    assert tokenize("") == []


def test_bm25_ranks_matching_documents_and_weights_rare_terms() -> None:
    bm25 = BM25(
        [
            tokenize("refund policy for damaged items"),
            tokenize("delivery tracking and courier eta"),
            tokenize("refund delivery fee when courier is late"),
        ]
    )
    scores = bm25.scores(tokenize("courier refund"))
    assert scores[0] > 0 and scores[1] > 0
    assert scores[2] == max(scores)
    assert bm25.scores(tokenize("passport")) == [0.0, 0.0, 0.0]


def test_bm25_prefers_rarer_terms() -> None:
    bm25 = BM25([["common", "rare"], ["common"], ["common"], ["common"]])
    assert bm25.idf("rare") > bm25.idf("common") > 0


def test_bm25_normalizes_for_length() -> None:
    bm25 = BM25([["refund"], ["refund", "x", "y", "z", "w", "v"], ["other"]])
    short, long, _ = bm25.scores(["refund"])
    assert short > long > 0


def test_bm25_term_frequency_saturates() -> None:
    bm25 = BM25([["a"], ["a"] * 2, ["a"] * 50, ["b"]], b=0.0)
    one, two, fifty, _ = bm25.scores(["a"])
    assert one < two < fifty
    assert fifty < (bm25.k1 + 1) * bm25.idf("a")


def test_bm25_handles_empty_corpus() -> None:
    assert BM25([]).scores(["a"]) == []


def docs_index(store: BitemporalStore, **kwargs: float) -> KnowledgeIndex:
    index = KnowledgeIndex(store, **kwargs)
    index.add_document(
        Document(
            doc_id="kb_refunds",
            title="Refund policy",
            body="Refunds up to 25 dollars are issued for damaged items.",
            tags=("billing",),
        )
    )
    index.add_document(
        Document(
            doc_id="kb_late",
            title="When an order is late",
            body="If an order is late, check the courier ETA and offer a delivery credit. "
            "Late orders over two hours qualify for a refund of the delivery fee.",
            tags=("delivery", "late"),
        )
    )
    index.add_document(
        Document(
            doc_id="kb_address",
            title="Changing a delivery address",
            body="Confirm the new address with the customer before updating the order.",
            tags=("delivery",),
        )
    )
    return index


def test_retrieve_returns_documents_with_provenance() -> None:
    index = docs_index(BitemporalStore())
    results = index.retrieve("refund for damaged items", as_of=0.0, k=2)
    assert results[0].kind == "document"
    assert results[0].source_id == "kb_refunds"
    assert results[0].valid_from is None and results[0].valid_to is None
    assert len(results) <= 2
    assert all(r.score > 0 for r in results)


def test_retrieve_matches_tags_and_respects_k() -> None:
    index = docs_index(BitemporalStore())
    assert [r.source_id for r in index.retrieve("billing", as_of=0.0, k=5)] == ["kb_refunds"]
    assert len(index.retrieve("delivery order", as_of=0.0, k=1)) == 1
    assert index.retrieve("passport renewal", as_of=0.0, k=5) == []
    assert index.retrieve("delivery", as_of=0.0, k=0) == []


def test_duplicate_document_ids_are_rejected() -> None:
    index = docs_index(BitemporalStore())
    with pytest.raises(ValueError):
        index.add_document(Document(doc_id="kb_late", title="x", body="y"))


def test_recency_prior_prefers_newer_facts() -> None:
    store = BitemporalStore()
    store.assert_fact(
        "Warehouse", "notice", "restock schedule", 0.0, recorded_at=0.0, fact_id="old"
    )
    store.assert_fact(
        "Warehouse", "notice", "restock schedule", 9 * HOUR, recorded_at=9 * HOUR, fact_id="new"
    )
    index = KnowledgeIndex(store, half_life_s=HOUR)
    ranked = index.retrieve("restock schedule", as_of=10 * HOUR, k=2)
    assert [r.source_id for r in ranked] == ["new", "old"]
    assert ranked[0].score > ranked[1].score

    flat = KnowledgeIndex(store, half_life_s=1e12).retrieve("restock schedule", as_of=10 * HOUR)
    assert flat[0].score == pytest.approx(flat[1].score, rel=1e-6)


def test_fact_results_carry_their_valid_interval() -> None:
    store = BitemporalStore()
    store.assert_fact("Store hours", "notice", "closed for holiday", 0.0, 100.0, recorded_at=0.0)
    (hit,) = KnowledgeIndex(store).retrieve("holiday hours", as_of=50.0)
    assert hit.kind == "fact"
    assert hit.fact is not None
    assert hit.source_id == hit.fact.fact_id
    assert (hit.valid_from, hit.valid_to) == (0.0, 100.0)
    assert KnowledgeIndex(store).retrieve("holiday hours", as_of=150.0) == []


def test_known_at_is_passed_through_to_the_store() -> None:
    store = BitemporalStore()
    store.assert_fact("Pickup", "notice", "pickup counter moved", 0.0, recorded_at=10.0)
    index = KnowledgeIndex(store)
    assert index.retrieve("pickup counter", as_of=20.0, known_at=5.0) == []
    assert len(index.retrieve("pickup counter", as_of=20.0, known_at=10.0)) == 1


def test_active_incident_ranks_first_then_drops_out_after_resolution() -> None:
    store = BitemporalStore()
    ingestor = StatusPageIngestor(store, "statuspage:demo")
    payload = json.loads(FIXTURE.read_text())
    ingestor.ingest(payload)
    index = docs_index(store)
    during = parse_timestamp("2026-10-19T10:00:00-07:00")

    results = index.retrieve("my order is late", as_of=during, k=3)
    top = results[0]
    assert top.kind == "fact"
    assert top.fact is not None
    assert top.fact.value["name"] == "Order delivery delays"
    assert top.boosted is True
    # The knowledge base article is still there as supporting evidence.
    assert "kb_late" in [r.source_id for r in results]

    resolved_at = "2026-10-19T11:30:00-07:00"
    payload["incidents"][0]["resolved_at"] = resolved_at
    payload["incidents"][0]["updated_at"] = resolved_at
    ingestor.ingest(payload)

    after = parse_timestamp(resolved_at) + 60
    later = index.retrieve("my order is late", as_of=after, k=5)
    assert all(r.kind == "document" for r in later)
    assert later[0].source_id == "kb_late"
    # Asking about the past still finds the incident.
    past = index.retrieve("my order is late", as_of=during, k=1)
    assert past[0].fact is not None and past[0].valid_to == parse_timestamp(resolved_at)


def test_incident_boost_needs_a_subject_match() -> None:
    store = BitemporalStore()
    store.assert_fact(
        "Payments",
        "incident",
        {"name": "Card errors", "update": "late refunds"},
        0.0,
        recorded_at=0,
    )
    index = docs_index(store)
    (top, *_) = index.retrieve("my order is late", as_of=10.0, k=3)
    assert top.source_id == "kb_late"
    hits = {r.source_id: r for r in index.retrieve("my order is late", as_of=10.0, k=5)}
    incident = [r for r in hits.values() if r.kind == "fact"]
    assert incident and incident[0].boosted is False


def test_retrieval_is_deterministic_on_ties() -> None:
    store = BitemporalStore()
    index = KnowledgeIndex(store)
    for doc_id in ("b", "a", "c"):
        index.add_document(Document(doc_id=doc_id, title="same", body="same words"))
    assert [r.source_id for r in index.retrieve("same", as_of=0.0)] == ["a", "b", "c"]
