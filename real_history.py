"""
real_history.py -- loads the real multi-day RELIANCE trade history in
real_data/reliance_june_2026.json (see that file's own "_provenance"
block for exactly where it came from: IQRF_V2's own already-computed
backtest reports, read read-only, 2026-10-31).

This is the real-usage data source for multi_day_runner.py. It is
NOT live data and NOT hand-authored like sample_data/ -- it's a real
backtest IQRF_V2 itself already ran and recorded. See README.md's
"What's real here, and what isn't" for the honest framing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from models import PortfolioState, Trade, trade_from_dict

DEFAULT_PATH = Path(__file__).resolve().parent / "real_data" / "reliance_june_2026.json"


@dataclass(frozen=True)
class RealDay:
    date: str
    trades: List[Trade]
    portfolio: PortfolioState
    day_realized_pnl_reference_report: float
    cross_instrument_loss_fraction: Optional[float]


def load_real_history(path: Path = DEFAULT_PATH) -> List[RealDay]:
    data = json.loads(path.read_text())
    days_out: List[RealDay] = []

    for day in data["days"]:
        trades = [trade_from_dict(t) for t in day["trades"]]
        portfolio = PortfolioState(
            current_equity=day["end_of_day_equity"],
            peak_equity=day["peak_equity_to_date"],
            open_positions=[],
            # the real, externally-reported number for the day (IQRF_V2's
            # own daily_pnl), deliberately kept distinct from the sum of
            # this day's own trades -- see risk_brief.pnl_reconciliation_drift
            session_realized_pnl=day["day_realized_pnl_reference_report"],
        )
        days_out.append(RealDay(
            date=day["date"],
            trades=trades,
            portfolio=portfolio,
            day_realized_pnl_reference_report=day["day_realized_pnl_reference_report"],
            cross_instrument_loss_fraction=day["cross_instrument_loss_fraction"],
        ))

    return days_out


def cross_instrument_loss_fraction_by_day(days: List[RealDay]) -> Dict[str, Optional[float]]:
    return {d.date: d.cross_instrument_loss_fraction for d in days}
