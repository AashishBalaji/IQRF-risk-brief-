"""
reconcile.py -- deterministic transaction matching, the payments-domain
generalization of this project's "math decides" discipline.

Given two real transaction sources (here: a payment-gateway settlement
feed and an internal ledger), matches records that plainly agree and
leaves everything else in one of two honest buckets:

  - ambiguous_candidates: a plausible counterpart exists on the other
    side, but something about it doesn't cleanly satisfy the matching
    rule (reference ID missing on one side, a rounding-level amount
    difference, a timestamp outside the clean-match window). This is
    the ONLY bucket reconciliation_judge.py ever sees -- it is never
    shown a clean match (nothing to judge) or a genuinely unmatched
    entry (no candidate exists to compare against, so there is
    nothing for a model to weigh either).
  - unmatched: no candidate found on the other side at all. A real,
    deterministic finding -- flagged for direct human investigation,
    not sent to a model, because there's no comparison to make.

No model is involved anywhere in this file. Matching, not-matching,
and "there is a candidate but it's ambiguous" are all decided by
plain arithmetic and timestamp comparisons, stated here as named
constants, not buried in logic.

Scope limitation, stated plainly rather than discovered later: this
matcher assumes 1:1 cardinality -- one gateway settlement maps to at
most one ledger entry, and vice versa. It does not model partial
captures, split refunds, or a single payout batching several ledger
entries into one settlement (N:1 or 1:N). A real reconciliation
system eventually needs that; this one does not attempt it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Clean match: same reference ID, amount equal (to the cent), and
# settled within this many hours of each other.
MATCH_WINDOW_HOURS = 4.0
AMOUNT_EXACT_TOLERANCE = 0.001

# A rounding-level amount difference a judge should treat as "the
# same transaction, fees/rounding aside" -- used by
# reconciliation_judge.py, defined here since it's part of the same
# real-world tolerance model as the matcher's own thresholds.
ROUNDING_TOLERANCE = 1.00

# How far apart two entries' amounts may be and still be considered
# a *candidate* pair at all when there's no reference ID to go on.
AMOUNT_AMBIGUOUS_TOLERANCE = 5.00

# How far apart two entries' timestamps may be and still be
# considered a candidate pair (wide on purpose -- a real T+1
# settlement delay is normal and shouldn't be thrown away before a
# human or the judge ever sees it; this is a search net, not a
# verdict).
AMBIGUOUS_WINDOW_HOURS = 24.0


@dataclass(frozen=True)
class GatewayTxn:
    gateway_txn_id: str
    amount: float
    timestamp: str
    reference_id: Optional[str]


@dataclass(frozen=True)
class LedgerEntry:
    ledger_entry_id: str
    amount: float
    timestamp: str
    reference_id: Optional[str]


@dataclass(frozen=True)
class AmbiguousPair:
    gateway: GatewayTxn
    ledger: LedgerEntry
    reason: str  # "ref_match_amount_or_time_off" | "missing_ref_amount_time_close"


@dataclass(frozen=True)
class ReconciliationResult:
    matched: List[Tuple[GatewayTxn, LedgerEntry]]
    ambiguous: List[AmbiguousPair]
    unmatched_gateway: List[GatewayTxn]
    unmatched_ledger: List[LedgerEntry]


def _hours_between(ts_a: str, ts_b: str) -> float:
    a = datetime.fromisoformat(ts_a)
    b = datetime.fromisoformat(ts_b)
    return abs((a - b).total_seconds()) / 3600.0


def load_gateway_settlements(records: List[Dict]) -> List[GatewayTxn]:
    return [
        GatewayTxn(
            gateway_txn_id=r["gateway_txn_id"],
            amount=float(r["amount"]),
            timestamp=r["timestamp"],
            reference_id=r.get("reference_id"),
        )
        for r in records
    ]


def load_ledger_entries(records: List[Dict]) -> List[LedgerEntry]:
    return [
        LedgerEntry(
            ledger_entry_id=r["ledger_entry_id"],
            amount=float(r["amount"]),
            timestamp=r["timestamp"],
            reference_id=r.get("reference_id"),
        )
        for r in records
    ]


def _is_clean_match(g: GatewayTxn, led: LedgerEntry) -> bool:
    if not (g.reference_id and led.reference_id and g.reference_id == led.reference_id):
        return False
    if abs(g.amount - led.amount) > AMOUNT_EXACT_TOLERANCE:
        return False
    return _hours_between(g.timestamp, led.timestamp) <= MATCH_WINDOW_HOURS


def _find_ambiguous_candidate(
    g: GatewayTxn, remaining_ledger: List[LedgerEntry], used: set
) -> Optional[Tuple[LedgerEntry, str]]:
    for led in remaining_ledger:
        if led.ledger_entry_id in used:
            continue
        ref_match = bool(g.reference_id and led.reference_id and g.reference_id == led.reference_id)
        time_close = _hours_between(g.timestamp, led.timestamp) <= AMBIGUOUS_WINDOW_HOURS
        if ref_match and time_close:
            return led, "ref_match_amount_or_time_off"
        amount_close = abs(g.amount - led.amount) <= AMOUNT_AMBIGUOUS_TOLERANCE
        ref_missing_one_side = not (g.reference_id and led.reference_id)
        if amount_close and time_close and ref_missing_one_side:
            return led, "missing_ref_amount_time_close"
    return None


def reconcile(
    gateway_settlements: List[GatewayTxn], internal_ledger: List[LedgerEntry]
) -> ReconciliationResult:
    """Two passes, both deterministic. Pass 1: exact matches only.
    Pass 2: among what's left, find candidates that are close but
    not clean -- the genuinely ambiguous set. Anything still left
    after both passes has no candidate at all and is a real,
    standalone finding, not sent anywhere for judgment."""
    matched: List[Tuple[GatewayTxn, LedgerEntry]] = []
    used_ledger: set = set()
    remaining_gateway: List[GatewayTxn] = []

    for g in gateway_settlements:
        match = None
        for led in internal_ledger:
            if led.ledger_entry_id in used_ledger:
                continue
            if _is_clean_match(g, led):
                match = led
                break
        if match:
            matched.append((g, match))
            used_ledger.add(match.ledger_entry_id)
        else:
            remaining_gateway.append(g)

    remaining_ledger = [led for led in internal_ledger if led.ledger_entry_id not in used_ledger]

    ambiguous: List[AmbiguousPair] = []
    used_ledger_2: set = set()
    unmatched_gateway: List[GatewayTxn] = []

    for g in remaining_gateway:
        found = _find_ambiguous_candidate(g, remaining_ledger, used_ledger_2)
        if found:
            led, reason = found
            ambiguous.append(AmbiguousPair(gateway=g, ledger=led, reason=reason))
            used_ledger_2.add(led.ledger_entry_id)
        else:
            unmatched_gateway.append(g)

    unmatched_ledger = [led for led in remaining_ledger if led.ledger_entry_id not in used_ledger_2]

    return ReconciliationResult(
        matched=matched,
        ambiguous=ambiguous,
        unmatched_gateway=unmatched_gateway,
        unmatched_ledger=unmatched_ledger,
    )
