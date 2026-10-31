"""
Tests for real_history.py -- the loader for the real 13-day RELIANCE
backtest history extracted from IQRF_V2's own report files (see
real_data/reliance_june_2026.json's _provenance block).
"""

import pytest

from models import PortfolioState, Trade
from real_history import cross_instrument_loss_fraction_by_day, load_real_history


def test_loads_all_thirteen_real_trading_days():
    days = load_real_history()
    assert len(days) == 13
    assert days[0].date == "2026-06-01"
    assert days[-1].date == "2026-06-17"


def test_days_are_in_chronological_order():
    days = load_real_history()
    dates = [d.date for d in days]
    assert dates == sorted(dates)


def test_each_day_has_real_trade_objects():
    days = load_real_history()
    total_trades = sum(len(d.trades) for d in days)
    assert total_trades == 92  # the real count from signal_exit_root_cause_report.json
    for d in days:
        for t in d.trades:
            assert isinstance(t, Trade)
            assert t.symbol == "RELIANCE"


def test_each_day_has_a_portfolio_state():
    days = load_real_history()
    for d in days:
        assert isinstance(d.portfolio, PortfolioState)
        assert d.portfolio.open_positions == []


def test_cross_instrument_loss_fraction_by_day_covers_all_dates():
    days = load_real_history()
    mapping = cross_instrument_loss_fraction_by_day(days)
    assert len(mapping) == 13
    # the real worst day in this window, per IQRF_V2's own
    # loss_clustering diagnosis
    assert mapping["2026-06-11"] == pytest.approx(0.8333333333333334)
