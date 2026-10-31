"""
Tests for multi_day_runner.py's rolling fire-history bookkeeping --
the piece that lets escalation_judge reason about recurrence rather
than just today's flags in isolation.
"""

from collections import deque

from multi_day_runner import ESCALATION_WINDOW_DAYS, _update_fire_history


def test_single_day_counts_once():
    history = deque()
    counts = _update_fire_history(history, {"a", "b"})
    assert counts == {"a": 1, "b": 1}


def test_recurrence_accumulates_across_days():
    history = deque()
    _update_fire_history(history, {"a"})
    _update_fire_history(history, {"a"})
    counts = _update_fire_history(history, {"a"})
    assert counts == {"a": 3}


def test_window_caps_at_escalation_window_days():
    history = deque()
    for _ in range(ESCALATION_WINDOW_DAYS + 3):
        counts = _update_fire_history(history, {"a"})
    assert counts == {"a": ESCALATION_WINDOW_DAYS}
    assert len(history) == ESCALATION_WINDOW_DAYS


def test_a_flag_not_fired_today_is_absent_from_counts():
    history = deque()
    _update_fire_history(history, {"a"})
    counts = _update_fire_history(history, {"b"})
    assert "a" not in counts
    assert counts == {"b": 1}  # "b" only fired today, nothing to carry over


def test_gap_then_refire_does_not_inherit_old_streak():
    history = deque()
    _update_fire_history(history, {"a"})  # day 1: a fires
    _update_fire_history(history, {})      # day 2: nothing fires
    counts = _update_fire_history(history, {"a"})  # day 3: a fires again
    assert counts == {"a": 2}  # fired on 2 of the last 3 days, not back-to-back
