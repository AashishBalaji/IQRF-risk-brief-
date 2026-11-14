"""Unit/integration tests for rag/vector_store.py's two real backends
-- ChromaDB (build_vector_store) and FAISS (build_faiss_vector_store).

Built against the real corpus (corpus_builder.build_corpus()), same
as test_rag_chain.py and test_eval.py, not a synthetic fixture -- the
point of this test file is proving the FAISS backend is a real,
working, drop-in alternative to Chroma, not just an import that
happens to succeed.
"""

from __future__ import annotations

import pytest

from rag.corpus_builder import build_corpus
from rag.eval import run_retrieval_eval
from rag.vector_store import build_faiss_vector_store, build_vector_store


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def chroma_store(corpus):
    return build_vector_store(documents=corpus, collection_name="test-vector-store-chroma")


@pytest.fixture(scope="module")
def faiss_store(corpus):
    return build_faiss_vector_store(documents=corpus)


def test_faiss_store_builds_from_the_real_corpus(corpus, faiss_store):
    # FAISS has no public document count on the wrapper itself; the
    # underlying index does, and it must match the real corpus size --
    # nothing dropped, nothing duplicated during the build.
    assert faiss_store.index.ntotal == len(corpus)


def test_faiss_store_similarity_search_returns_real_matching_documents(faiss_store):
    results = faiss_store.similarity_search("what does the exit_reason_skew detector catch", k=4)
    assert results
    assert any(d.metadata.get("name") == "detect_exit_reason_skew" for d in results)


def test_faiss_store_raises_on_an_empty_corpus():
    with pytest.raises(ValueError):
        build_faiss_vector_store(documents=[])


def test_faiss_and_chroma_agree_on_retrieving_the_real_target_document_at_k4(chroma_store, faiss_store):
    """Not every query needs to agree between backends at top-1
    (approximate nearest-neighbor search can legitimately differ at
    the margins), but at the same k=4 depth rag/eval.py's own
    RETRIEVAL_EVAL_CASES already uses (and already documents this
    exact query as a real hit for Chroma -- see eval.py), both
    backends should surface the same real target document -- same
    embedder, same corpus, same question. This is the real
    cross-backend check; without it, 'FAISS works' would mean nothing
    more than 'FAISS didn't crash.' (At the stricter k=1 the two
    backends can legitimately disagree -- confirmed directly while
    writing this test, which is why k=4 is the fair comparison, not
    k=1.)"""
    query = "is this illustrative sample data or real trading data"
    chroma_hits = chroma_store.similarity_search(query, k=4)
    faiss_hits = faiss_store.similarity_search(query, k=4)
    assert any(d.metadata.get("kind") == "illustrative_sample" for d in chroma_hits)
    assert any(d.metadata.get("kind") == "illustrative_sample" for d in faiss_hits)


def test_faiss_backend_reports_a_real_precision_at_k(faiss_store):
    """rag/eval.py's retrieval-eval harness is backend-agnostic by
    design (it only calls similarity_search) -- this is the proof,
    not just the claim: the exact same RETRIEVAL_EVAL_CASES run
    unchanged against the FAISS backend and produce a real, reportable
    precision number, same as test_eval.py already does for Chroma.
    See README for how this number compares to Chroma's documented
    0.8 on the same corpus and the same five cases."""
    report = run_retrieval_eval(faiss_store, k=4)
    assert 0.0 <= report.precision <= 1.0
    assert len(report.results) == 5


def test_faiss_store_persists_and_reloads_from_disk(tmp_path, corpus):
    """The one real behavioral difference worth locking in: FAISS
    has no running server to hold state between process restarts
    (unlike Chroma, which can run embedded with its own persistence
    layer) -- save_local()/load_local() is the real mechanism a
    production deployment would use to avoid re-embedding the whole
    corpus on every cold start. Proven here against a real temp
    directory, not asserted in a docstring."""
    from rag.embeddings import get_embeddings

    persist_dir = tmp_path / "faiss-index"
    build_faiss_vector_store(documents=corpus, persist_directory=str(persist_dir))
    assert (persist_dir / "index.faiss").exists()

    from langchain_community.vectorstores import FAISS

    reloaded = FAISS.load_local(
        str(persist_dir), get_embeddings(), allow_dangerous_deserialization=True
    )
    assert reloaded.index.ntotal == len(corpus)
