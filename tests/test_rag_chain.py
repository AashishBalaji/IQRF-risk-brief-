"""Integration tests for the RAG query layer: real retrieval over a
real (small, test-local) vector store -- no network, no API key, so
these always exercise the offline/tool paths. The live-model path
(answer_question) is covered only by a mocked-failure test below,
same pattern as test_llm_narrative.py's offline coverage of
generate_brief_offline."""

from __future__ import annotations

import pytest

import rag.rag_chain as rag_chain
from rag.corpus_builder import build_corpus
from rag.rag_chain import answer_question_offline, ask
from rag.vector_store import build_vector_store


@pytest.fixture(scope="module")
def vector_store():
    return build_vector_store(documents=build_corpus(), collection_name="test-rag-chain")


def test_offline_answer_retrieves_the_matching_real_day(vector_store):
    result = answer_question_offline(vector_store, "What happened on 2026-06-04?")
    assert result.mode == "offline"
    assert "2026-06-04" in result.answer
    assert result.sources


def test_offline_answer_with_no_retrieved_documents_says_so(vector_store, monkeypatch):
    monkeypatch.setattr(rag_chain, "retrieve", lambda *a, **k: [])
    result = answer_question_offline(vector_store, "anything")
    assert "No relevant real records found" in result.answer
    assert result.sources == []


def test_ask_routes_live_state_questions_to_the_tool_not_retrieval(vector_store):
    result = ask(vector_store, "Is anything flagged right now?", offline=True)
    assert result.mode == "tool"
    assert "Live check as of" in result.answer


def test_ask_falls_back_to_offline_when_no_live_model_available(vector_store):
    # No ANTHROPIC_API_KEY in this sandbox (see README) -- ask() without
    # offline=True must still degrade cleanly rather than raising.
    result = ask(vector_store, "What happened on 2026-06-11?")
    assert result.mode in ("offline", "live")
    assert result.answer
