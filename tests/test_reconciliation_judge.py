"""
Tests for reconciliation_judge.py's offline rule -- the deterministic
floor the model is meant to improve on. The live judge_match() call
is not independently tested here, same discipline as
llm_narrative.generate_brief and escalation_judge.judge_escalation:
no bare ANTHROPIC_API_KEY in this sandbox, see README.md's "What's
actually verified" section.
"""

from reconcile import AmbiguousPair, GatewayTxn, LedgerEntry
from reconciliation_judge import (
    LIKELY_DIFFERENT,
    LIKELY_SAME_TRANSACTION,
    NEEDS_HUMAN_REVIEW,
    judge_match_offline,
)


def _pair(g_amount, g_ts, g_ref, l_amount, l_ts, l_ref, reason):
    g = GatewayTxn("GW-T", g_amount, g_ts, g_ref)
    led = LedgerEntry("LG-T", l_amount, l_ts, l_ref)
    return AmbiguousPair(gateway=g, ledger=led, reason=reason)


def test_matching_reference_rounding_diff_is_likely_same():
    pair = _pair(1999.99, "2026-10-30T10:00:00+05:30", "REF-A",
                 2000.00, "2026-10-30T10:01:00+05:30", "REF-A",
                 "ref_match_amount_or_time_off")
    decision = judge_match_offline(pair)
    assert decision.decision == LIKELY_SAME_TRANSACTION


def test_matching_reference_real_amount_gap_needs_human():
    pair = _pair(749.00, "2026-10-30T11:00:00+05:30", "REF-B",
                 736.00, "2026-10-30T11:01:30+05:30", "REF-B",
                 "ref_match_amount_or_time_off")
    decision = judge_match_offline(pair)
    assert decision.decision == NEEDS_HUMAN_REVIEW


def test_missing_reference_exact_amount_close_in_time_is_likely_same():
    pair = _pair(3200.00, "2026-10-30T09:30:00+05:30", "REF-C",
                 3200.00, "2026-10-30T09:31:00+05:30", None,
                 "missing_ref_amount_time_close")
    decision = judge_match_offline(pair)
    assert decision.decision == LIKELY_SAME_TRANSACTION


def test_missing_reference_exact_amount_far_apart_needs_human():
    pair = _pair(450.00, "2026-10-30T16:00:00+05:30", "REF-D",
                 450.00, "2026-10-30T22:30:00+05:30", None,
                 "missing_ref_amount_time_close")
    decision = judge_match_offline(pair)
    assert decision.decision == NEEDS_HUMAN_REVIEW


def test_missing_reference_real_amount_gap_is_likely_different():
    pair = _pair(610.00, "2026-10-30T14:50:00+05:30", "REF-E",
                 605.50, "2026-10-30T14:52:00+05:30", None,
                 "missing_ref_amount_time_close")
    decision = judge_match_offline(pair)
    assert decision.decision == LIKELY_DIFFERENT


def test_matching_reference_ignores_a_large_time_gap():
    # Reference IDs matching is the strong signal here -- a T+1
    # settlement delay shouldn't flip a rounding-level amount match
    # to NEEDS_HUMAN_REVIEW just because of the clock.
    pair = _pair(1200.00, "2026-10-30T10:00:00+05:30", "REF-F",
                 1200.00, "2026-10-31T09:00:00+05:30", "REF-F",
                 "ref_match_amount_or_time_off")
    decision = judge_match_offline(pair)
    assert decision.decision == LIKELY_SAME_TRANSACTION


def test_rationale_is_never_empty():
    pairs = [
        _pair(100.0, "2026-10-30T10:00:00+05:30", "REF-G", 100.5, "2026-10-30T10:01:00+05:30",
              "REF-G", "ref_match_amount_or_time_off"),
        _pair(100.0, "2026-10-30T10:00:00+05:30", "REF-H", 150.0, "2026-10-30T10:01:00+05:30",
              "REF-H", "ref_match_amount_or_time_off"),
        _pair(100.0, "2026-10-30T10:00:00+05:30", "REF-I", 100.0, "2026-10-30T10:30:00+05:30",
              None, "missing_ref_amount_time_close"),
    ]
    for pair in pairs:
        decision = judge_match_offline(pair)
        assert decision.rationale
