"""Unit tests for the RAG corpus builder -- verifies the corpus is
built from the project's own real data, correctly labelled, and
non-empty for each source."""

from __future__ import annotations

from rag.corpus_builder import (
    _daily_history_documents,
    _detector_reference_documents,
    _real_usage_log_documents,
    _sample_data_documents,
    build_corpus,
)


def test_daily_history_documents_cover_all_13_real_days():
    docs = _daily_history_documents()
    assert len(docs) == 13
    dates = {d.metadata["date"] for d in docs}
    assert "2026-06-01" in dates
    assert "2026-06-17" in dates


def test_daily_history_document_for_known_real_finding_day():
    docs = _daily_history_documents()
    june_4 = next(d for d in docs if d.metadata["date"] == "2026-06-04")
    assert "symbol_repeat_loser" in june_4.text
    assert "overtrading_frequency" in june_4.text
    assert june_4.metadata["source"] == "real_history"


def test_real_usage_log_documents_nonempty_and_real():
    docs = _real_usage_log_documents()
    assert len(docs) >= 1
    assert all(d.metadata["source"] == "real_usage_log.md" for d in docs)
    assert "SUPPRESS_AS_NOISE" in docs[0].text or "NOTIFY_NOW" in docs[0].text


def test_sample_data_documents_are_labelled_illustrative_not_real():
    docs = _sample_data_documents()
    assert len(docs) >= 1
    for d in docs:
        assert "ILLUSTRATIVE SAMPLE" in d.text
        assert d.metadata["kind"] == "illustrative_sample"


def test_detector_reference_documents_cover_every_detector():
    docs = _detector_reference_documents()
    names = {d.metadata["name"] for d in docs}
    assert "detect_exit_reason_skew" in names
    assert "detect_cross_instrument_stress" in names
    assert "judge_escalation_offline" in names


def test_build_corpus_combines_all_sources_with_no_empty_documents():
    docs = build_corpus()
    assert len(docs) > 20
    assert all(d.text.strip() for d in docs)
    kinds = {d.metadata.get("kind") for d in docs}
    assert "daily_summary" in kinds
    assert "reference" in kinds
