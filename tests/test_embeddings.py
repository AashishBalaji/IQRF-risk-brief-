"""Unit tests for the embedding layer: the deterministic offline
TF-IDF floor (fast, no network, still the fallback path) and the
real local semantic embedder that is now the default."""

from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings import (
    LocalSentenceTransformerEmbeddings,
    OfflineTfidfEmbeddings,
    get_embeddings,
)


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_embed_documents_fits_and_returns_one_vector_per_text():
    emb = OfflineTfidfEmbeddings()
    vectors = emb.embed_documents([
        "RELIANCE lost money on multiple trades this week.",
        "A market-wide stress day hit most tracked instruments.",
    ])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0


def test_embed_query_before_fit_raises():
    emb = OfflineTfidfEmbeddings()
    with pytest.raises(RuntimeError):
        emb.embed_query("anything")


def test_embed_documents_called_twice_raises_instead_of_reusing_stale_vocab():
    """Regression test for a round-table review finding: a second
    embed_documents call on an already-fitted instance used to call
    .transform() with the OLD vocabulary, silently dropping any new
    term from a changed corpus with no error. It must now raise
    rather than guess."""
    emb = OfflineTfidfEmbeddings()
    emb.embed_documents(["drawdown breach today", "overtrading flagged"])
    with pytest.raises(RuntimeError):
        emb.embed_documents(["a completely different second batch of documents"])


def test_embed_query_after_fit_uses_same_vocabulary_dimension():
    emb = OfflineTfidfEmbeddings()
    doc_vectors = emb.embed_documents(["drawdown breach today", "overtrading flagged"])
    query_vector = emb.embed_query("drawdown breach")
    assert len(query_vector) == len(doc_vectors[0])


def test_similar_text_scores_closer_than_unrelated_text():
    emb = OfflineTfidfEmbeddings()
    corpus = [
        "RELIANCE symbol_repeat_loser flag fired for five trading days running.",
        "overtrading_frequency flagged an outlier trade count today.",
        "payments reconciliation judge decided LIKELY_SAME_TRANSACTION.",
    ]
    doc_vectors = np.array(emb.embed_documents(corpus))
    query_vector = np.array(emb.embed_query("RELIANCE keeps losing money, symbol repeat loser"))
    scores = [_cosine(query_vector, v) for v in doc_vectors]
    assert scores[0] == max(scores)


# --- real semantic embedder -------------------------------------------------


def test_local_sentence_transformer_returns_one_vector_per_text():
    emb = LocalSentenceTransformerEmbeddings()
    vectors = emb.embed_documents([
        "RELIANCE lost money on multiple trades this week.",
        "A market-wide stress day hit most tracked instruments.",
    ])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # all-MiniLM-L6-v2's known output dimension
    assert len(vectors[0]) == len(vectors[1])


def test_local_sentence_transformer_can_embed_documents_more_than_once():
    """Unlike OfflineTfidfEmbeddings, there is no fitted vocabulary to
    go stale here -- a second embed_documents call must work exactly
    like the first, with no RuntimeError."""
    emb = LocalSentenceTransformerEmbeddings()
    first = emb.embed_documents(["first batch"])
    second = emb.embed_documents(["second batch", "still fine"])
    assert len(first) == 1
    assert len(second) == 2


def test_semantic_retrieval_finds_a_synonymous_match_tfidf_cannot():
    """The actual proof that LocalSentenceTransformerEmbeddings
    implements real semantic search rather than keyword overlap: a
    query sharing almost no vocabulary with the right document (a
    car/traffic-signal paraphrase of a vehicle/red-light sentence)
    must still score that document above an unrelated one. TF-IDF
    provably cannot do this -- shown first as the contrast this test
    exists to demonstrate, not merely asserted in a docstring."""
    doc_vehicle = "The vehicle came to a complete stop at the red light."
    doc_unrelated = "Quarterly earnings reports are published every three months."
    query = "The car halted at the traffic signal."

    tfidf = OfflineTfidfEmbeddings()
    tfidf_doc_vectors = tfidf.embed_documents([doc_vehicle, doc_unrelated])
    tfidf_query_vector = tfidf.embed_query(query)
    tfidf_scores = [_cosine(tfidf_query_vector, v) for v in tfidf_doc_vectors]
    # The whole point of this test: confirm TF-IDF really does fail
    # here (near-zero overlap after stop-word removal) before showing
    # the semantic embedder succeeding on the exact same inputs.
    assert tfidf_scores[0] == pytest.approx(0.0, abs=1e-6)

    semantic = LocalSentenceTransformerEmbeddings()
    semantic_doc_vectors = semantic.embed_documents([doc_vehicle, doc_unrelated])
    semantic_query_vector = semantic.embed_query(query)
    semantic_scores = [_cosine(semantic_query_vector, v) for v in semantic_doc_vectors]
    assert semantic_scores[0] > 0.5
    assert semantic_scores[0] > semantic_scores[1]


def test_get_embeddings_defaults_to_local_semantic_embedder_without_openai_key(monkeypatch):
    """Since 2026-11-04 the real semantic embedder is the default
    (an earlier version of this test asserted the TF-IDF fallback
    was the default -- that changed when the local model was
    confirmed to actually load in this sandbox; see embeddings.py's
    module docstring)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    emb = get_embeddings()
    assert isinstance(emb, LocalSentenceTransformerEmbeddings)


def test_get_embeddings_falls_back_to_offline_tfidf_if_local_model_fails_to_load(monkeypatch):
    """If the local model can't load for any reason (dependency
    missing, no network for the one-time download, corrupted cache),
    get_embeddings() must still return a working embedder rather than
    raising and failing the whole vector-store build."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated: local model failed to load")

    monkeypatch.setattr("rag.embeddings.LocalSentenceTransformerEmbeddings", _boom)
    emb = get_embeddings()
    assert isinstance(emb, OfflineTfidfEmbeddings)
