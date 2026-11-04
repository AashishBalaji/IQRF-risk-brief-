"""Unit tests for rag/eval.py -- the retrieval precision@k and
groundedness evaluation harness."""

from __future__ import annotations

import pytest

from rag.eval import RETRIEVAL_EVAL_CASES, groundedness_score, run_retrieval_eval
from rag.vector_store import build_vector_store


@pytest.fixture(scope="module")
def vector_store():
    return build_vector_store()


def test_run_retrieval_eval_covers_every_declared_case(vector_store):
    report = run_retrieval_eval(vector_store)
    assert len(report.results) == len(RETRIEVAL_EVAL_CASES)


def test_run_retrieval_eval_reports_the_documented_real_precision(vector_store):
    """Pins the actual, reported-in-README precision@4 number against
    the real corpus and the real semantic embedder (4/5 = 0.8 as of
    2026-11-04) -- if this ever drifts (corpus changes, embedder
    changes), this test should be the thing that notices, not a
    stale README claim nobody re-checks."""
    report = run_retrieval_eval(vector_store, k=4)
    assert report.precision == pytest.approx(0.8)


def test_every_eval_case_carries_a_non_empty_explanatory_note():
    for case in RETRIEVAL_EVAL_CASES:
        assert case["note"]


def test_groundedness_score_is_1_when_answer_has_no_numbers():
    assert groundedness_score("No relevant numeric claim here.", ["some source text"]) == 1.0


def test_groundedness_score_is_1_for_a_real_extractive_offline_answer():
    """An offline extractive answer is the source text verbatim, so
    every number in it must, by construction, also be in the source
    -- this is the actual verification that answer_question_offline's
    'never say more than the data supports' discipline holds, run as
    a number instead of only read by eye."""
    source = "day_realized_pnl=-5860.78, trade_count=13, stop_outs=4"
    answer = f"(offline retrieval -- no model call) Most relevant real records:\n\n{source}"
    assert groundedness_score(answer, [source]) == 1.0


def test_groundedness_score_catches_an_invented_number():
    """The real point of this metric: a hypothetical live-model
    answer that states a figure none of its real source documents
    contain must score below 1.0, not pass silently."""
    source_docs = ["day_realized_pnl=-5860.78, trade_count=13"]
    hallucinated_answer = "The day's realized P&L was -5860.78 across 13 trades, down 42.7% YoY."
    score = groundedness_score(hallucinated_answer, source_docs)
    assert 0.0 < score < 1.0


def test_groundedness_score_is_0_when_every_number_is_invented():
    score = groundedness_score("Loss was 999999.99 today.", ["nothing matches that at all"])
    assert score == 0.0
