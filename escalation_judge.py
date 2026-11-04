"""
escalation_judge.py -- the one place in this project a MODEL decides
something, rather than narrating a decision a threshold already made.

Every number in this project is decided by plain Python
(risk_brief.py) and every pattern flag is decided by a stated,
testable threshold (pattern_detector.py) -- on purpose, see those
modules' own docstrings. That leaves a real gap: nothing in the
pipeline ever weighs flags AGAINST EACH OTHER or against their own
recent history. A single INFO-level exit_reason_skew on day one
means something different from the same flag firing for the fifth
day running; two WARNING flags together are not just "twice as bad"
as one. A fixed threshold can't easily hold that kind of context --
it's a judgment call, which is exactly what a model is for.

So this module asks a model exactly one question: given the flags
that ALREADY fired today (nothing recomputed, nothing re-derived)
plus how often each has fired recently, should this interrupt the
trader right now, wait for the ordinary end-of-day digest, or is it
noise not worth surfacing at all? The model picks one of three fixed
labels and gives a one-line reason. It is never shown raw trades or
asked to produce a number -- the same "math decides, model only
judges or narrates" discipline as llm_narrative.py, applied to a
genuinely judgment-shaped question instead of a wording one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import anthropic

from pattern_detector import PatternFlag
from risk_brief import RiskSummary

# See llm_narrative.py's _MODEL comment -- same deprecation fix.
_MODEL = "claude-sonnet-5"

NOTIFY_NOW = "NOTIFY_NOW"
HOLD_FOR_DIGEST = "HOLD_FOR_DIGEST"
SUPPRESS_AS_NOISE = "SUPPRESS_AS_NOISE"
_VALID_DECISIONS = {NOTIFY_NOW, HOLD_FOR_DIGEST, SUPPRESS_AS_NOISE}

_SYSTEM_PROMPT = (
    "You triage risk alerts for an intraday equity trader. You are given "
    "today's already-decided pattern flags (each one already passed a "
    "fixed, tested threshold -- you never re-evaluate whether a flag is "
    "real) and how many of the last 5 trading days each flag name fired "
    "on. Decide ONLY whether to interrupt the trader now, hold it for "
    "the ordinary end-of-day digest, or suppress it as noise they've "
    "already seen too much of. Rules: base the decision only on the "
    "flags, their severities, and their recent firing counts given to "
    "you -- never invent a number, a flag, or a fact not present in the "
    "input. Multiple WARNING flags together, or a flag recurring for "
    "several days running, usually means notify. A single INFO flag "
    "firing for the first time in 5 days usually does not. Respond with "
    "ONLY a JSON object: {\"decision\": one of NOTIFY_NOW, "
    "HOLD_FOR_DIGEST, SUPPRESS_AS_NOISE, \"rationale\": one short "
    "sentence}."
)


@dataclass(frozen=True)
class EscalationDecision:
    decision: str
    rationale: str


def _flags_to_prompt(
    flags: List[PatternFlag],
    recent_fire_counts: Dict[str, int],
    recent_window_days: int,
    summary: RiskSummary,
) -> str:
    if not flags:
        flag_lines = "none"
    else:
        flag_lines = "\n".join(
            f"- {f.name} [{f.severity}]: {f.message} "
            f"(fired on {recent_fire_counts.get(f.name, 0)} of the last "
            f"{recent_window_days} trading days, including today)"
            for f in flags
        )
    return (
        f"Today's flags:\n{flag_lines}\n\n"
        f"Context: day_realized_pnl={summary.day_realized_pnl}, "
        f"distance_to_drawdown_limit_pct="
        f"{summary.distance_to_drawdown_limit_pct * 100:.2f} points, "
        f"pnl_reconciliation_drift={summary.pnl_reconciliation_drift}.\n"
    )


def judge_escalation(
    flags: List[PatternFlag],
    recent_fire_counts: Dict[str, int],
    recent_window_days: int,
    summary: RiskSummary,
    api_key: str | None = None,
) -> EscalationDecision:
    """Real call to the Anthropic API. Raises if no key is available or
    the response isn't valid JSON with a recognized decision -- callers
    that want a no-network path should use judge_escalation_offline()
    instead, same pattern as llm_narrative.py."""
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=_MODEL,
        max_tokens=200,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": _flags_to_prompt(flags, recent_fire_counts, recent_window_days, summary),
        }],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    parsed = json.loads(text)
    decision = parsed["decision"]
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"model returned an unrecognized decision: {decision!r}")
    return EscalationDecision(decision=decision, rationale=parsed.get("rationale", ""))


def judge_escalation_offline(
    flags: List[PatternFlag],
    recent_fire_counts: Dict[str, int],
    recent_window_days: int,
) -> EscalationDecision:
    """Deterministic fallback -- no network, no API key required. Used
    by tests and as the graceful degrade when the live call fails, so a
    broken network never means a real warning gets silently dropped.

    Intentionally cruder than the model: this is the floor the model is
    expected to improve on, not a hidden second judge. Rule: 2+ WARNING
    flags, or any flag recurring on 3+ of the last recent_window_days
    (including today), notifies now. Exactly one WARNING with no
    recurrence holds for the digest. INFO-only, or nothing at all,
    is suppressed.
    """
    warnings = [f for f in flags if f.severity == "WARNING"]
    recurring = any(
        recent_fire_counts.get(f.name, 0) >= 3 for f in flags
    )

    if len(warnings) >= 2 or recurring:
        return EscalationDecision(
            decision=NOTIFY_NOW,
            rationale=(
                f"{len(warnings)} WARNING flag(s) and/or a flag recurring "
                f"{max(recent_fire_counts.values(), default=0)} of the last "
                f"{recent_window_days} days (offline rule, not the model)."
            ),
        )
    if len(warnings) == 1:
        return EscalationDecision(
            decision=HOLD_FOR_DIGEST,
            rationale="One new WARNING flag, not yet recurring (offline rule, not the model).",
        )
    if flags:
        return EscalationDecision(
            decision=SUPPRESS_AS_NOISE,
            rationale="INFO-level flags only, no WARNING (offline rule, not the model).",
        )
    return EscalationDecision(
        decision=SUPPRESS_AS_NOISE,
        rationale="No flags fired today (offline rule, not the model).",
    )
