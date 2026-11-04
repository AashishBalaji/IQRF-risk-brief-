"""
api.py -- FastAPI service wrapping this project's existing CLI tools
(generate_daily_brief.py, multi_day_runner.py) plus the new RAG
layer (rag/) in a Python web API, as the Junior Generative AI
Engineer JD this project was upgraded for explicitly asks for
("Python-based APIs and backend services using FastAPI or Flask").

The vector store is built once at startup (from the real corpus --
see rag/corpus_builder.py) and reused for every /ask request, so a
request never pays the cost of re-embedding the whole corpus.

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
    yield
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = rag_ask(_state["vector_store"], req.question, offline=req.offline, k=req.k)
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
