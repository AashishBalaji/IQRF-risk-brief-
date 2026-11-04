"""
tools.py -- the agentic-tool-calling layer: exposes the project's
existing real detectors (pattern_detector.py, escalation_judge.py) as
LangChain Tools, so a question like "is anything worth escalating
right now?" gets answered by actually *running* the real detectors
against the real history, not by retrieving a stale pre-computed
document from the vector store.

Same discipline as the rest of this project: a real tool-calling
agent (an LLM that decides whether to call the tool) is the intended
production path, but it needs a live model call -- which this
sandbox can't independently verify (no bare ANTHROPIC_API_KEY, same
constraint as llm_narrative.py and escalation_judge.py). A
deterministic keyword router is the tested, no-network floor the
agent is meant to improve on, not a hidden second decision-maker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.tools import Tool

from escalation_judge import judge_escalation_offline
from pattern_detector import run_pattern_detectors
from real_history import cross_instrument_loss_fraction_by_day, load_real_history
from risk_brief import compute_risk_summary

_LIVE_STATE_KEYWORDS = (
    "pattern", "flag", "escalat", "notify", "suppress", "right now",
    "currently", "today", "live",
)


def check_live_pattern_state(_input: str = "") -> str:
    """Runs the real pattern-detector suite and the real offline
    escalation rule over the FULL real 13-day RELIANCE history (the
    same detectors and the same data multi_day_runner.py runs), and
    reports the final real day's state -- not a cached answer, a live
    recomputation every time this tool is called."""
    real_days = load_real_history()
    cross_fraction_by_day = cross_instrument_loss_fraction_by_day(real_days)
    daily_trades_seen = [day.trades for day in real_days]

    latest = real_days[-1]
    summary = compute_risk_summary(latest.portfolio, latest.trades)
    flags = run_pattern_detectors(
        daily_trades_seen,
        cross_instrument_loss_fraction_by_day=cross_fraction_by_day,
        latest_date=latest.date,
    )
    recent_fire_counts = {f.name: 1 for f in flags}  # single-call view, no multi-day fire history
    escalation = judge_escalation_offline(flags, recent_fire_counts, recent_window_days=1)

    if not flags:
        flag_text = "No pattern flags fired."
    else:
        flag_text = "; ".join(f"[{f.severity}] {f.name}: {f.message}" for f in flags)

    return (
        f"Live check as of {latest.date} (real history, real detectors): "
        f"{flag_text} Escalation: {escalation.decision} -- {escalation.rationale} "
        f"(day_realized_pnl={summary.day_realized_pnl})."
    )


def build_tools() -> List[Tool]:
    """The tool list a real LangChain tool-calling agent would bind
    to. One tool today -- check_live_pattern_state -- kept to one so
    the model's choice of whether to call it stays legible; more
    detectors could be split into their own tools later without
    changing this shape."""
    return [
        Tool(
            name="check_live_pattern_state",
            description=(
                "Runs the real pattern-detector and escalation-judge code "
                "live against the real trading history and returns the "
                "current flags and escalation decision. Use this for any "
                "question about CURRENT or LIVE pattern/escalation state "
                "('is anything flagged right now', 'should I be notified "
                "today') -- not for historical questions about a specific "
                "past date, which the RAG retrieval layer already answers "
                "from stored real records."
            ),
            func=check_live_pattern_state,
        ),
    ]


@dataclass(frozen=True)
class ToolAnswer:
    answer: str
    tool_used: Optional[str]


def route_offline(question: str) -> Optional[ToolAnswer]:
    """Deterministic keyword router -- the tested, no-network floor.
    Returns a ToolAnswer when the question looks like it's asking
    about LIVE/CURRENT state (so the real tool should run), or None
    when it looks like a historical/reference question that the RAG
    retrieval layer should answer instead. This is a routing decision,
    not a content judgment -- it never answers the question itself
    beyond what the tool already returns."""
    lowered = question.lower()
    if any(kw in lowered for kw in _LIVE_STATE_KEYWORDS):
        return ToolAnswer(answer=check_live_pattern_state(), tool_used="check_live_pattern_state")
    return None
