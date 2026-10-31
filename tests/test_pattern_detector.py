"""
Tests for pattern_detector.py. Each detector is tested in isolation
with small, hand-built sequences first (to pin down exactly when it
should and shouldn't fire), then the whole suite is run once against
the real 13-day history (real_data/) as an integration check that it
runs end to end on real data without crashing and produces at least
the findings this project's own README documents having observed.
"""

from datetime import datetime, timedelta

from models import Trade
from pattern_detector import (
    detect_cross_instrument_stress,
    detect_exit_reason_skew,
    detect_overtrading_frequency,
    detect_position_size_drift_after_loss,
    detect_stop_out_clustering,
    detect_symbol_repeat_loser,
    run_pattern_detectors,
)
from real_history import cross_instrument_loss_fraction_by_day, load_real_history


def _trade(symbol="RELIANCE", net_pnl=-100.0, exit_reason="STOP", quantity=10, day_offset=0, seq=0):
    ts = datetime(2026, 6, 1) + timedelta(days=day_offset, minutes=seq)
    return Trade(
        symbol=symbol,
        direction="LONG",
        entry_timestamp=ts,
        entry_price=100.0,
        exit_timestamp=ts + timedelta(minutes=5),
        exit_price=100.0 + (net_pnl / max(quantity, 1)),
        quantity=quantity,
        exit_reason=exit_reason,
        gross_pnl=net_pnl,
        commission=0.0,
        slippage_cost=0.0,
        net_pnl=net_pnl,
    )


def test_stop_out_clustering_fires_on_trend_not_single_day():
    # 2 stop-outs each day for 3 days -> 6 total, over threshold
    days = [[_trade(exit_reason="STOP", day_offset=d, seq=i) for i in range(2)] for d in range(3)]
    flag = detect_stop_out_clustering(days, window=3, threshold=5)
    assert flag is not None
    assert flag.name == "stop_out_clustering"


def test_stop_out_clustering_silent_below_threshold():
    days = [[_trade(exit_reason="STOP", day_offset=d, seq=0)] for d in range(3)]
    flag = detect_stop_out_clustering(days, window=3, threshold=5)
    assert flag is None


def test_stop_out_clustering_silent_with_too_little_history():
    days = [[_trade(exit_reason="STOP", day_offset=0, seq=i) for i in range(10)]]
    flag = detect_stop_out_clustering(days, window=3, threshold=5)
    assert flag is None  # only 1 day of history, window asks for 3


def test_exit_reason_skew_fires_on_dominant_signal_exit():
    days = [[_trade(exit_reason="SIGNAL_EXIT", day_offset=0, seq=i) for i in range(10)]]
    flag = detect_exit_reason_skew(days, window=5, signal_exit_fraction_threshold=0.6, min_trades=8)
    assert flag is not None
    assert "SIGNAL_EXIT" in flag.message


def test_exit_reason_skew_silent_when_balanced():
    days = [[
        _trade(exit_reason="STOP", seq=0), _trade(exit_reason="TARGET", seq=1),
        _trade(exit_reason="STOP", seq=2), _trade(exit_reason="TARGET", seq=3),
        _trade(exit_reason="STOP", seq=4), _trade(exit_reason="TARGET", seq=5),
        _trade(exit_reason="STOP", seq=6), _trade(exit_reason="TARGET", seq=7),
    ]]
    flag = detect_exit_reason_skew(days, min_trades=8)
    assert flag is None


def test_symbol_repeat_loser_fires_for_worst_offender():
    days = [[
        _trade(symbol="TCS", net_pnl=-10, seq=0),
        _trade(symbol="TCS", net_pnl=-10, seq=1),
        _trade(symbol="TCS", net_pnl=-10, seq=2),
        _trade(symbol="INFY", net_pnl=-10, seq=3),
    ]]
    flag = detect_symbol_repeat_loser(days, min_losses=3)
    assert flag is not None
    assert "TCS" in flag.message


def test_symbol_repeat_loser_silent_below_threshold():
    days = [[_trade(symbol="TCS", net_pnl=-10, seq=0), _trade(symbol="INFY", net_pnl=-10, seq=1)]]
    flag = detect_symbol_repeat_loser(days, min_losses=3)
    assert flag is None


def test_overtrading_frequency_flags_real_outlier():
    counts = [3, 4, 5, 4]  # a little real day-to-day variance, not a flat baseline
    baseline = [[_trade(seq=i) for i in range(n)] for n in counts]
    today = [_trade(seq=i) for i in range(20)]  # way more than baseline
    flag = detect_overtrading_frequency(baseline + [today], z_threshold=1.5)
    assert flag is not None
    assert flag.name == "overtrading_frequency"


def test_overtrading_frequency_silent_with_insufficient_baseline():
    days = [[_trade(seq=0)], [_trade(seq=0), _trade(seq=1)]]
    flag = detect_overtrading_frequency(days)
    assert flag is None


def test_position_size_drift_fires_when_sizing_up_after_losses():
    trades = []
    for i in range(6):
        if i % 2 == 0:
            trades.append(_trade(net_pnl=-50.0, quantity=10, seq=i))  # a loss
        else:
            trades.append(_trade(net_pnl=100.0, quantity=50, seq=i))  # sized up right after
    flag = detect_position_size_drift_after_loss([trades], lookback_trades=10, drift_threshold=0.25)
    assert flag is not None
    assert flag.name == "position_size_drift_after_loss"


def test_position_size_drift_silent_with_flat_quantity():
    # exactly the shape of this project's real_data/ extraction: every
    # trade has the same quantity, so there's no real signal to find
    trades = [_trade(net_pnl=(-50.0 if i % 2 == 0 else 50.0), quantity=1, seq=i) for i in range(10)]
    flag = detect_position_size_drift_after_loss([trades])
    assert flag is None


def test_cross_instrument_stress_fires_above_threshold():
    flag = detect_cross_instrument_stress({"2026-06-03": 0.75}, "2026-06-03", threshold=0.6)
    assert flag is not None


def test_cross_instrument_stress_silent_when_missing():
    flag = detect_cross_instrument_stress({"2026-06-03": None}, "2026-06-03")
    assert flag is None


def test_run_pattern_detectors_against_real_history_runs_end_to_end():
    real_days = load_real_history()
    cross_fraction = cross_instrument_loss_fraction_by_day(real_days)

    seen = []
    all_flag_names = set()
    for day in real_days:
        seen.append(day.trades)
        flags = run_pattern_detectors(
            seen, cross_instrument_loss_fraction_by_day=cross_fraction, latest_date=day.date
        )
        all_flag_names.update(f.name for f in flags)

    # the real 13-day history genuinely exhibits these, per manual
    # verification in README.md -- a regression here means either the
    # fixture or the detector thresholds changed
    assert "exit_reason_skew" in all_flag_names
    assert "cross_instrument_stress" in all_flag_names
    assert "overtrading_frequency" in all_flag_names
    # flat quantity=1 in the real fixture (see real_history.py) means
    # this one genuinely never fires on this dataset -- not a bug
    assert "position_size_drift_after_loss" not in all_flag_names
