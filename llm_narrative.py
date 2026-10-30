"""
llm_narrative.py -- turns an already-computed RiskSummary into a
short, plain-English brief.

The model is given ONLY the numbers risk_brief.py already computed
-- never the raw trades or portfolio state -- and is explicitly
told not to introduce a number that isn't in that input. This keeps
the LLM's job strictly "narrate", never "compute": a wrong risk
figure can only come from risk_brief.py's arithmetic (which is
plain Python and testable), not from the model.
"""

from __future__ import annotations

import os

import anthropic

from risk_brief import RiskSummary

_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = (
    "You write a short daily risk brief for an intraday equity trader, "
    "from numbers you are given. Rules: use ONLY the numbers in the "
    "input -- never invent, estimate, or round a figure that isn't "
    "there. 3 to 5 sentences, plain English, no bullet points, no "
    "headers. If distance_to_drawdown_limit_pct is negative or under "
    "1 percentage point, say so plainly and put it in the first "
    "sentence -- that is the one thing that must not get buried. "
    "Mention day P&L, open exposure, and the stop/target split. If "
    "pnl_reconciliation_drift is not zero, flag it as a data check, "
    "not a trading fact."
)


def _summary_to_prompt(summary: RiskSummary) -> str:
    winner = (
        f"{summary.biggest_winner.symbol} +{summary.biggest_winner.net_pnl}"
        if summary.biggest_winner
        else "none"
    )
    loser = (
        f"{summary.biggest_loser.symbol} {summary.biggest_loser.net_pnl}"
        if summary.biggest_loser
        else "none"
    )
    return (
        f"day_realized_pnl: {summary.day_realized_pnl}\n"
        f"open_exposure: {summary.open_exposure}\n"
        f"open_position_count: {summary.open_position_count}\n"
        f"current_drawdown_pct: {summary.current_drawdown_pct * 100:.2f}%\n"
        f"drawdown_limit_pct: {summary.drawdown_limit_pct * 100:.2f}%\n"
        f"distance_to_drawdown_limit_pct: "
        f"{summary.distance_to_drawdown_limit_pct * 100:.2f} percentage points\n"
        f"trade_count: {summary.trade_count}\n"
        f"stop_outs: {summary.stop_outs}\n"
        f"target_hits: {summary.target_hits}\n"
        f"signal_exits: {summary.signal_exits}\n"
        f"biggest_winner: {winner}\n"
        f"biggest_loser: {loser}\n"
        f"pnl_reconciliation_drift: {summary.pnl_reconciliation_drift}\n"
    )


def generate_brief(summary: RiskSummary, api_key: str | None = None) -> str:
    """Real call to the Anthropic API. Raises if no key is available --
    callers that want a no-network path should use
    generate_brief_offline() instead."""
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=_MODEL,
        max_tokens=300,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _summary_to_prompt(summary)}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def generate_brief_offline(summary: RiskSummary) -> str:
    """A deterministic, template-based fallback -- no network, no API
    key required. Used by tests and as a graceful degrade if the API
    call fails, so a broken network never means no brief gets sent."""
    lines = []
    if summary.distance_to_drawdown_limit_pct < 0.01:
        lines.append(
            f"Drawdown is at {summary.current_drawdown_pct * 100:.1f}%, "
            f"{'past' if summary.distance_to_drawdown_limit_pct < 0 else 'within 1 point of'} "
            f"your {summary.drawdown_limit_pct * 100:.0f}% limit."
        )
    sign = "+" if summary.day_realized_pnl >= 0 else ""
    lines.append(
        f"Day P&L is {sign}{summary.day_realized_pnl} across {summary.trade_count} trades "
        f"({summary.target_hits} targets, {summary.stop_outs} stops, "
        f"{summary.signal_exits} signal exits)."
    )
    lines.append(
        f"Open exposure is {summary.open_exposure} across "
        f"{summary.open_position_count} position(s)."
    )
    if summary.pnl_reconciliation_drift != 0:
        lines.append(
            f"Data check: today's trades sum to a P&L that differs from the "
            f"portfolio's own session figure by {summary.pnl_reconciliation_drift}."
        )
    return " ".join(lines)
