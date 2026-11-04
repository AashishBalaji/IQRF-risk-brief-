"""Unit tests for query_log.py -- the real SQLite relational store
for /ask's query history. Uses sqlite3's special ':memory:' path so
every test gets an isolated, on-disk-free database."""

from __future__ import annotations

from rag import query_log


def _fresh_conn():
    return query_log.connect(":memory:")


def test_connect_creates_the_queries_table_empty():
    conn = _fresh_conn()
    assert query_log.total_count(conn) == 0


def test_log_query_is_a_real_insert_readable_back():
    conn = _fresh_conn()
    query_log.log_query(
        conn, question="what happened on 2026-06-04?", mode="offline",
        source_count=4, answer_chars=120,
    )
    assert query_log.total_count(conn) == 1
    recent = query_log.recent_queries(conn, limit=5)
    assert len(recent) == 1
    assert recent[0].question == "what happened on 2026-06-04?"
    assert recent[0].mode == "offline"
    assert recent[0].source_count == 4
    assert recent[0].answer_chars == 120


def test_recent_queries_orders_newest_first_and_respects_limit():
    conn = _fresh_conn()
    for i in range(5):
        query_log.log_query(
            conn, question=f"question {i}", mode="offline",
            source_count=1, answer_chars=10,
        )
    recent = query_log.recent_queries(conn, limit=3)
    assert len(recent) == 3
    assert [r.question for r in recent] == ["question 4", "question 3", "question 2"]


def test_mode_counts_is_a_real_group_by_aggregate():
    conn = _fresh_conn()
    for mode in ("offline", "offline", "live", "tool"):
        query_log.log_query(
            conn, question="q", mode=mode, source_count=1, answer_chars=10,
        )
    counts = query_log.mode_counts(conn)
    assert counts == {"offline": 2, "live": 1, "tool": 1}


def test_reset_for_tests_clears_the_table():
    conn = _fresh_conn()
    query_log.log_query(conn, question="q", mode="offline", source_count=1, answer_chars=1)
    assert query_log.total_count(conn) == 1
    query_log.reset_for_tests(conn)
    assert query_log.total_count(conn) == 0
