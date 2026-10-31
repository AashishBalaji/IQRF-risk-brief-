"""
Tests for escalation_judge.py's offline rule -- the deterministic
floor the model is meant to improve on. The live judge_escalation()
call is not independently tested here, same discipline as
llm_narrative.generate_brief(): no bare ANTHROPIC_API_KEY in this
sandbox, see README.md's "What's actually verified" section.
"""

from pattern_detector import PatternFlag
from escalation_judge import (
    HOLD_FOR_DIGEST,
    NOTIFY_NOW,
    SUPPRESS_AS_NOISE,
    judge_escalation_offline,
)


def _flag(name, severity):
    return PatternFlag(name=name, severity=severity, message="msg", evidence="ev")


def test_no_flags_suppresses():
    decision = judge_escalation_offline([], {}, 5)
    assert decision.decision == SUPPRESS_AS_NOISE


def test_info_only_suppresses():
    flags = [_flag("exit_reason_skew", "INFO")]
    decision = judge_escalation_offline(flags, {"exit_reason_skew": 1}, 5)
    assert decision.decision == SUPPRESS_AS_NOISE


def test_single_new_warning_holds_for_digest():
    flags = [_flag("overtrading_frequency", "WARNING")]
    decision = judge_escalation_offline(flags, {"overtrading_frequency": 1}, 5)
    assert decision.decision == HOLD_FOR_DIGEST


def test_two_warnings_notify_now():
    flags = [_flag("stop_out_clustering", "WARNING"), _flag("overtrading_frequency", "WARNING")]
    counts = {"stop_out_clustering": 1, "overtrading_frequency": 1}
    decision = judge_escalation_offline(flags, counts, 5)
    assert decision.decision == NOTIFY_NOW


def test_recurring_flag_notifies_even_if_alone():
    flags = [_flag("symbol_repeat_loser", "WARNING")]
    counts = {"symbol_repeat_loser": 4}  # fired 4 of the last 5 days
    decision = judge_escalation_offline(flags, counts, 5)
    assert decision.decision == NOTIFY_NOW


def test_recurring_info_flag_notifies_too():
    # recurrence matters even for an INFO flag -- 5 days running of the
    # same signal is no longer "just noise", regardless of severity
    flags = [_flag("exit_reason_skew", "INFO")]
    counts = {"exit_reason_skew": 5}
    decision = judge_escalation_offline(flags, counts, 5)
    assert decision.decision == NOTIFY_NOW


def test_rationale_is_never_empty():
    for flags, counts in [
        ([], {}),
        ([_flag("a", "INFO")], {"a": 1}),
        ([_flag("a", "WARNING")], {"a": 1}),
        ([_flag("a", "WARNING"), _flag("b", "WARNING")], {"a": 1, "b": 1}),
    ]:
        decision = judge_escalation_offline(flags, counts, 5)
        assert decision.rationale
