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

from collections import deque
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

# Same window multi_day_runner.py uses for its rolling fire-history --
# kept as the same named constant here rather than a bare literal so a
# change to one is a visible decision to make in the other too.
ESCALATION_WINDOW_DAYS = 5


def check_live_pattern_state(_input: str = "") -> str:
    """Runs the real pattern-detector suite and the real offline
    escalation rule over the FULL real 13-day RELIANCE history (the
    same detectors and the same data multi_day_runner.py runs), and
    reports the final real day's state -- not a cached answer, a live
    recomputation every time this tool is called.

    The escalation decision replays the SAME rolling fire-history
    multi_day_runner.py builds (a deque of the last ESCALATION_WINDOW_DAYS
    days' fired-flag-name sets), not a single-day approximation -- a
    round-table review found an earlier version of this function
    hardcoded recent_fire_counts to 1 for every flag, which is below
    judge_escalation_offline's own >=3 recurrence threshold and so
    could never detect recurrence at all: it silently reported
    HOLD_FOR_DIGEST on days the real multi-day history says NOTIFY_NOW
    (confirmed on 2026-06-16, where symbol_repeat_loser actually
    recurs 5/5 days). See BUGS-style note in README.md's "GenAI
    upgrade" section."""
    real_days = load_real_history()
    cross_fraction_by_day = cross_instrument_loss_fraction_by_day(real_days)

    daily_trades_seen: List[list] = []
    fire_history: deque = deque()
    flags = []
    for day in real_days:
        daily_trades_seen.append(day.trades)
        flags = run_pattern_detectors(
            daily_trades_seen,
            cross_instrument_loss_fraction_by_day=cross_fraction_by_day,
            latest_date=day.date,
        )
        fire_history.append({f.name for f in flags})
        while len(fire_history) > ESCALATION_WINDOW_DAYS:
            fire_history.popleft()

    latest = real_days[-1]
    summary = compute_risk_summary(latest.portfolio, latest.trades)
    recent_fire_counts = {
        f.name: sum(1 for day_names in fire_history if f.name in day_names) for f in flags
    }
    escalation = judge_escalation_offline(flags, recent_fire_counts, ESCALATION_WINDOW_DAYS)

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
