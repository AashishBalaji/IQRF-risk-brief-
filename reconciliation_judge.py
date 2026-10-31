"""
reconciliation_judge.py -- the payments-domain twin of
escalation_judge.py: the one place in this module a model decides
something, applied to the ambiguous leftover reconcile.py produces.

reconcile.py already decided, deterministically, what cleanly
matches and what has no candidate at all. What's left --
AmbiguousPair objects -- is the genuinely judgment-shaped set: a
reference ID that matches but an amount that doesn't, or an amount
that matches but no reference ID to confirm it. A model is asked
exactly one question per pair: LIKELY_SAME_TRANSACTION,
LIKELY_DIFFERENT, or NEEDS_HUMAN_REVIEW, with a one-line rationale --
never asked to re-derive the amounts or timestamps it's given, same
discipline as escalation_judge.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import anthropic

from reconcile import ROUNDING_TOLERANCE, AmbiguousPair, _hours_between

_MODEL = "claude-sonnet-4-5-20250929"

LIKELY_SAME_TRANSACTION = "LIKELY_SAME_TRANSACTION"
LIKELY_DIFFERENT = "LIKELY_DIFFERENT"
NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
_VALID_DECISIONS = {LIKELY_SAME_TRANSACTION, LIKELY_DIFFERENT, NEEDS_HUMAN_REVIEW}

# How close in time counts as "close enough that a missing reference
# ID is probably just a logging gap, not a real question mark."
CLOSE_TIME_HOURS = 2.0

_SYSTEM_PROMPT = (
    "You review one ambiguous pair at a time from a payments "
    "reconciliation system: a gateway settlement record and an "
    "internal ledger entry that a deterministic matcher could not "
    "cleanly match (reference ID missing on one side, an amount "
    "that's close but not exact, or a reference ID that matches but "
    "an amount that doesn't). Decide only whether this looks like "
    "the same transaction, a different transaction, or a case that "
    "genuinely needs a human to look at it -- never recompute the "
    "amounts or invent a fact not given to you. A matching reference "
    "ID with only a rounding-level amount difference is normally the "
    "same transaction. A matching reference ID with a real amount "
    "difference needs a human, since that reference match could mean "
    "a partial refund or fee rather than a clean duplicate. With no "
    "reference ID at all, lean on amount and timing: an exact amount "
    "match close in time is probably the same transaction; an exact "
    "amount match far apart in time still needs a human, since there "
    "is nothing else tying the two together; a real amount "
    "difference with no reference ID is probably two different "
    "transactions. Respond with ONLY a JSON object: "
    "{\"decision\": one of LIKELY_SAME_TRANSACTION, LIKELY_DIFFERENT, "
    "NEEDS_HUMAN_REVIEW, \"rationale\": one short sentence}."
)


@dataclass(frozen=True)
class MatchDecision:
    decision: str
    rationale: str


def _pair_to_prompt(pair: AmbiguousPair) -> str:
    g, led = pair.gateway, pair.ledger
    return (
        f"Gateway settlement: amount={g.amount}, timestamp={g.timestamp}, "
        f"reference_id={g.reference_id!r}.\n"
        f"Ledger entry: amount={led.amount}, timestamp={led.timestamp}, "
        f"reference_id={led.reference_id!r}.\n"
        f"Why the matcher flagged this pair rather than matching or "
        f"dropping it: {pair.reason}.\n"
    )


def judge_match(pair: AmbiguousPair, api_key: Optional[str] = None) -> MatchDecision:
    """Real call to the Anthropic API. Raises if no key is available
    or the response isn't valid JSON with a recognized decision --
    callers that want a no-network path should use
    judge_match_offline() instead, same pattern as
    escalation_judge.judge_escalation."""
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=_MODEL,
        max_tokens=200,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _pair_to_prompt(pair)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    parsed = json.loads(text)
    decision = parsed["decision"]
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"model returned an unrecognized decision: {decision!r}")
    return MatchDecision(decision=decision, rationale=parsed.get("rationale", ""))


def judge_match_offline(pair: AmbiguousPair) -> MatchDecision:
    """Deterministic fallback -- no network, no API key required.
    The floor the model is meant to improve on, used by tests and as
    the graceful degrade when the live call fails. Three-way split,
    stated plainly rather than buried in branching:

    - Reference IDs match: a rounding-level amount difference
      (<= ROUNDING_TOLERANCE) is the same transaction; anything
      bigger needs a human, since a matching reference with a real
      amount gap could be a partial refund, not a clean duplicate.
    - No reference ID on one side: an exact-enough amount match close
      in time is the same transaction; the same amount match far
      apart in time still needs a human (nothing else ties them
      together); a real amount difference with no reference ID is
      probably two different transactions.
    """
    g, led = pair.gateway, pair.ledger
    amount_diff = abs(g.amount - led.amount)
    ref_match = bool(g.reference_id and led.reference_id and g.reference_id == led.reference_id)

    if ref_match:
        if amount_diff <= ROUNDING_TOLERANCE:
            return MatchDecision(
                decision=LIKELY_SAME_TRANSACTION,
                rationale=(
                    f"Reference ID matches and the amount differs by only "
                    f"{amount_diff:.2f} -- rounding, not a different transaction "
                    f"(offline rule, not the model)."
                ),
            )
        return MatchDecision(
            decision=NEEDS_HUMAN_REVIEW,
            rationale=(
                f"Reference ID matches but the amount differs by {amount_diff:.2f}, "
                f"more than a rounding margin -- could be a partial refund or fee, "
                f"not assumed automatically (offline rule, not the model)."
            ),
        )

    # No shared reference ID to lean on.
    hours_apart = _hours_between(g.timestamp, led.timestamp)
    if amount_diff <= ROUNDING_TOLERANCE:
        if hours_apart <= CLOSE_TIME_HOURS:
            return MatchDecision(
                decision=LIKELY_SAME_TRANSACTION,
                rationale=(
                    f"No reference ID on one side, but the amount matches and "
                    f"they're only {hours_apart:.1f}h apart -- looks like a "
                    f"logging gap, not a different transaction (offline rule, "
                    f"not the model)."
                ),
            )
        return MatchDecision(
            decision=NEEDS_HUMAN_REVIEW,
            rationale=(
                f"No reference ID on one side, and {hours_apart:.1f}h apart is "
                f"too long a gap to call this automatically even with a "
                f"matching amount (offline rule, not the model)."
            ),
        )
    return MatchDecision(
        decision=LIKELY_DIFFERENT,
        rationale=(
            f"No reference ID on one side, and the amounts differ by "
            f"{amount_diff:.2f} -- not close enough to assume these are "
            f"the same transaction (offline rule, not the model)."
        ),
    )
