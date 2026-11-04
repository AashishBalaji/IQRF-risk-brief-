"""
api.py -- FastAPI service wrapping this project's existing CLI tools
(generate_daily_brief.py, multi_day_runner.py) plus the new RAG
layer (rag/) in a Python web API, as the Junior Generative AI
Engineer JD this project was upgraded for explicitly asks for
("Python-based APIs and backend services using FastAPI or Flask").

The vector store is built once at startup (from the real corpus --
see rag/corpus_builder.py) and reused for every /ask request, so a
request never pays the cost of re-embedding the whole corpus.

Every /ask call is also logged to a real SQLite database
(rag/query_log.py) -- not a cache, a plain relational query-history
table, read back with real SQL (GROUP BY, ORDER BY ... LIMIT) by
GET /query-history.

Run it:
    uvicorn api:app --reload
    curl -X POST localhost:8000/ask \
        -H 'content-type: application/json' \
        -d '{"question": "what happened on 2026-06-04?", "offline": true}'
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_narrative import generate_brief_offline
from rag import query_log
from rag.eval import run_retrieval_eval
from rag.rag_chain import ask as rag_ask
from rag.tools import check_live_pattern_state
from rag.vector_store import build_vector_store
from real_history import load_real_history
from risk_brief import compute_risk_summary

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["vector_store"] = build_vector_store()
    _state["real_days_by_date"] = {d.date: d for d in load_real_history()}
    # Real SQLite connection, opened once at startup and reused for
    # every /ask request -- a JD line ("basic SQL and database
    # concepts") that nothing else in this service touches otherwise.
    _state["query_log_conn"] = query_log.connect()
    yield
    _state["query_log_conn"].close()
    _state.clear()


app = FastAPI(title="iqrf-risk-brief API", lifespan=lifespan)


class AskRequest(BaseModel):
    # max_length is a cheap, real safeguard even for a local demo --
    # a round-table review pointed out an unbounded string here could
    # be pushed straight into the TF-IDF vectorizer or an LLM call
    # with no limit. 2000 chars is generous for a question.
    question: str = Field(max_length=2000)
    offline: bool = False
    k: int = Field(default=4, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    mode: str


class BriefRequest(BaseModel):
    date: str


class BriefResponse(BaseModel):
    date: str
    brief: str
    day_realized_pnl: float
    trade_count: int
    pnl_reconciliation_drift: float


class PatternCheckResponse(BaseModel):
    result: str


class QueryHistoryResponse(BaseModel):
    total_logged: int
    mode_counts: dict
    recent: List[dict]


class RetrievalEvalResponse(BaseModel):
    precision_at_k: float
    k: int
    cases: List[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = rag_ask(_state["vector_store"], req.question, offline=req.offline, k=req.k)
    try:
        query_log.log_query(
            _state["query_log_conn"],
            question=req.question,
            mode=result.mode,
            source_count=len(result.sources),
            answer_chars=len(result.answer),
        )
    except Exception:  # noqa: BLE001 -- logging must never break a real answer
        pass
    return AskResponse(answer=result.answer, sources=result.sources, mode=result.mode)


@app.post("/brief", response_model=BriefResponse)
def brief_endpoint(req: BriefRequest) -> BriefResponse:
    """Offline brief for one real historical day (from the real
    13-day RELIANCE history, same data multi_day_runner.py runs
    against) -- deterministic, no model call, so this endpoint has
    no live/offline toggle the way /ask does."""
    day = _state["real_days_by_date"].get(req.date)
    if day is None:
        available = sorted(_state["real_days_by_date"].keys())
        raise HTTPException(
            status_code=404,
            detail=f"no real data for date {req.date!r}; available dates: {available}",
        )
    summary = compute_risk_summary(day.portfolio, day.trades)
    return BriefResponse(
        date=day.date,
        brief=generate_brief_offline(summary),
        day_realized_pnl=summary.day_realized_pnl,
        trade_count=summary.trade_count,
        pnl_reconciliation_drift=summary.pnl_reconciliation_drift,
    )


@app.post("/pattern-check", response_model=PatternCheckResponse)
def pattern_check_endpoint() -> PatternCheckResponse:
    """Runs the real pattern-detector + escalation-judge tool live
    against the real history -- the same function the RAG agent's
    tool router calls for a live-state question (rag/tools.py)."""
    return PatternCheckResponse(result=check_live_pattern_state())


@app.get("/query-history", response_model=QueryHistoryResponse)
def query_history_endpoint() -> QueryHistoryResponse:
    """Reads back the real SQLite query log every /ask call writes
    to -- a real SELECT ... GROUP BY for the mode split and a real
    SELECT ... ORDER BY ... LIMIT for the most recent entries, not
    just an INSERT nobody ever reads again."""
    conn = _state["query_log_conn"]
    recent = [
        {
            "id": r.id,
            "ts_utc": r.ts_utc,
            "question": r.question,
            "mode": r.mode,
            "source_count": r.source_count,
            "answer_chars": r.answer_chars,
        }
        for r in query_log.recent_queries(conn, limit=10)
    ]
    return QueryHistoryResponse(
        total_logged=query_log.total_count(conn),
        mode_counts=query_log.mode_counts(conn),
        recent=recent,
    )


@app.get("/eval/retrieval", response_model=RetrievalEvalResponse)
def retrieval_eval_endpoint(k: int = 4) -> RetrievalEvalResponse:
    """Runs rag/eval.py's real retrieval precision@k harness against
    the live vector store this process actually serves /ask from --
    the same honestly-reported number (currently 0.8 at k=4) that
    README documents, computed fresh on every call rather than
    hardcoded."""
    report = run_retrieval_eval(_state["vector_store"], k=k)
    return RetrievalEvalResponse(
        precision_at_k=report.precision,
        k=k,
        cases=[{"query": r.query, "hit": r.hit, "note": r.note} for r in report.results],
    )
