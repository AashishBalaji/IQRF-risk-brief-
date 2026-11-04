"""Unit tests for the offline TF-IDF embedder -- fast, no network."""

from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings import OfflineTfidfEmbeddings, get_embeddings


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

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    scores = [cosine(query_vector, v) for v in doc_vectors]
    assert scores[0] == max(scores)


def test_get_embeddings_falls_back_to_offline_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    emb = get_embeddings()
    assert isinstance(emb, OfflineTfidfEmbeddings)
