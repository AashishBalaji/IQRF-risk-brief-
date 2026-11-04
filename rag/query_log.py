"""
query_log.py -- a real relational store (SQLite, stdlib sqlite3, no
extra dependency) for the RAG layer's own query history.

This exists to cover a JD line ("basic SQL and database concepts")
that nothing else in this project touches -- everything else here is
either in-memory (the vector store) or flat files (real_usage_log.md,
sample_data/*.json). It is deliberately not a vector DB substitute
and not a cache: it is a plain relational log of what was asked, what
mode answered it (offline/live/tool), and how many sources came back,
queried back with real SQL (a GROUP BY aggregate, a bounded ORDER BY),
not just inserted and never read.

Schema is one table, kept intentionally small:

    queries(
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc        TEXT NOT NULL,   -- ISO-8601, e.g. 2026-11-04T12:00:00+00:00
        question      TEXT NOT NULL,
        mode          TEXT NOT NULL,   -- 'offline' | 'live' | 'tool'
        source_count  INTEGER NOT NULL,
        answer_chars  INTEGER NOT NULL
    )

The full answer text is not stored (it's already in the RagAnswer
returned to the caller, and this log's job is query *history*, not a
second copy of every answer) -- only its length, so growth stays
bounded regardless of how long individual answers get.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "rag_query_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT NOT NULL,
    question      TEXT NOT NULL,
    mode          TEXT NOT NULL,
    source_count  INTEGER NOT NULL,
    answer_chars  INTEGER NOT NULL
)
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Opens (and, on first use, creates) the query-log database.
    Safe to call once per process and hold onto the connection, or
    once per request -- sqlite3 handles either fine for this volume.

    Resolves the path from IQRF_QUERY_LOG_PATH at call time (not at
    import time) when no explicit path is given, so tests can point
    this at a throwaway file via conftest.py without the real
    api.py:app ever writing rag_query_log.db into the repo during a
    test run. Pass db_path=':memory:' directly for a pure in-memory
    database (used by tests/test_query_log.py)."""
    if db_path is None:
        db_path = os.environ.get("IQRF_QUERY_LOG_PATH", str(DEFAULT_DB_PATH))
    # check_same_thread=False: FastAPI runs each sync endpoint in a
    # worker thread from a pool, but this one connection is opened
    # once at lifespan startup and shared across every /ask and
    # /query-history request -- sqlite3's default same-thread check
    # would otherwise raise on the second request handled by a
    # different worker thread. Safe here: every real caller in this
    # project (api.py) only ever does one INSERT or one SELECT at a
    # time, never a concurrent write race this project needs to
    # guard against -- see README's "Noted, correctly not fixed"
    # concurrency caveat about _state for the same class of tradeoff.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def log_query(conn: sqlite3.Connection, *, question: str, mode: str,
              source_count: int, answer_chars: int) -> None:
    """Real INSERT -- called once per /ask request in api.py, after
    rag_chain.ask() already returned its answer. Logging failures
    must never break the request, same degrade-don't-fail discipline
    as rag_chain.ask()'s own live-model fallback -- callers should
    wrap this in try/except if they can't tolerate a disk error."""
    conn.execute(
        "INSERT INTO queries (ts_utc, question, mode, source_count, answer_chars) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            mode,
            source_count,
            answer_chars,
        ),
    )
    conn.commit()


@dataclass(frozen=True)
class QueryRecord:
    id: int
    ts_utc: str
    question: str
    mode: str
    source_count: int
    answer_chars: int


def recent_queries(conn: sqlite3.Connection, limit: int = 10) -> List[QueryRecord]:
    """Real SELECT with ORDER BY + LIMIT -- the most recently logged
    questions, newest first."""
    rows = conn.execute(
        "SELECT id, ts_utc, question, mode, source_count, answer_chars "
        "FROM queries ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [QueryRecord(*row) for row in rows]


def mode_counts(conn: sqlite3.Connection) -> dict:
    """Real SELECT with GROUP BY -- how many logged queries were
    answered by each mode (offline/live/tool). Used by
    /query-history to show the live-vs-offline split without the
    caller re-deriving it from raw rows."""
    rows = conn.execute(
        "SELECT mode, COUNT(*) FROM queries GROUP BY mode"
    ).fetchall()
    return {mode: count for mode, count in rows}


def total_count(conn: sqlite3.Connection) -> int:
    """Real SELECT COUNT(*) over the whole table."""
    (count,) = conn.execute("SELECT COUNT(*) FROM queries").fetchone()
    return count


def reset_for_tests(conn: sqlite3.Connection) -> None:
    """Test-only helper: clears the table so each test starts from a
    known-empty log instead of accumulating rows across test runs
    sharing a connection."""
    conn.execute("DELETE FROM queries")
    conn.commit()
