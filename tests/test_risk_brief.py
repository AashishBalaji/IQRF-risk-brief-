"""
Real tests for risk_brief.py, run against the actual sample_data
files -- the exact numbers asserted here are the same ones printed
by `python3 generate_daily_brief.py sample_data/... --offline`
during manual verification, not independently re-derived.
"""

import json
from pathlib import Path

import pytest

from models import portfolio_state_from_dict, trade_from_dict
from risk_brief import compute_risk_summary

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _load(filename):
    data = json.loads((SAMPLE_DIR / filename).read_text())
    portfolio = portfolio_state_from_dict(data["portfolio"])
    trades = [trade_from_dict(t) for t in data["trades_today"]]
    return portfolio, trades


def test_steady_day_summary():
    portfolio, trades = _load("day_2026-10-28.json")
    summary = compute_risk_summary(portfolio, trades)

    assert summary.day_realized_pnl == 659.0
    assert summary.trade_count == 3
    assert summary.target_hits == 2
    assert summary.stop_outs == 1
    assert summary.signal_exits == 0
    assert summary.open_position_count == 1
    assert summary.open_exposure == pytest.approx(2940.5 * 20)
    assert summary.pnl_reconciliation_drift == 0.0
    assert summary.biggest_winner.symbol == "HDFCBANK"
    assert summary.biggest_loser.symbol == "TCS"
    # 215000 peak, 204300 current -> well inside the default 6% limit
    assert summary.distance_to_drawdown_limit_pct > 0


def test_rough_day_breaches_drawdown_limit():
    portfolio, trades = _load("day_2026-10-29.json")
    summary = compute_risk_summary(portfolio, trades)

    assert summary.day_realized_pnl == -784.0
    assert summary.trade_count == 4
    assert summary.stop_outs == 3
    assert summary.target_hits == 1
    assert summary.pnl_reconciliation_drift == 0.0
    # 215000 peak, 201850 current = 6.12% drawdown, past the default 6% limit
    assert summary.current_drawdown_pct == pytest.approx(0.0612, abs=1e-4)
    assert summary.distance_to_drawdown_limit_pct < 0


def test_custom_drawdown_limit_changes_the_verdict():
    portfolio, trades = _load("day_2026-10-29.json")
    # the same rough day, but against a looser 10% limit, should no
    # longer be breached -- proves the limit is a real parameter,
    # not a hardcoded threshold
    summary = compute_risk_summary(portfolio, trades, drawdown_limit_pct=0.10)
    assert summary.distance_to_drawdown_limit_pct > 0


def test_pnl_reconciliation_drift_is_detected():
    portfolio, trades = _load("day_2026-10-28.json")
    # deliberately feed a portfolio whose session_realized_pnl disagrees
    # with the real sum of today's trades, and confirm the drift shows up
    from dataclasses import replace

    stale_portfolio = replace(portfolio, session_realized_pnl=portfolio.session_realized_pnl - 50)
    summary = compute_risk_summary(stale_portfolio, trades)
    assert summary.pnl_reconciliation_drift == pytest.approx(50.0)


def test_no_trades_today_handles_empty_list_cleanly():
    portfolio, _ = _load("day_2026-10-28.json")
    summary = compute_risk_summary(portfolio, [])
    assert summary.trade_count == 0
    assert summary.day_realized_pnl == 0.0
    assert summary.biggest_winner is None
    assert summary.biggest_loser is None
