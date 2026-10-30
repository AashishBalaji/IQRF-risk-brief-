"""
models.py -- standalone data shapes for the risk-brief agent.

These are DELIBERATELY independent of IQRF_V2's own
src/models/data_models.py: this project never imports from, or
writes to, the IQRF_V2 repository, by design (see README.md). But
to be genuinely useful -- i.e. to be able to read IQRF_V2's real
trade/portfolio logs directly, if the user ever points this at them
-- these dataclasses are structurally identical (same field names,
same types) to IQRF_V2's real Trade/Position/PortfolioState, as
read from a reference copy of src/models/data_models.py on
2026-10-31. A real IQRF_V2 log, serialized to JSON, should load
into these with no translation layer.

If IQRF_V2's own schema changes later, these will drift from it --
that's an accepted, explicit tradeoff for keeping the two projects
fully decoupled, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Position:
    """An open position, as IQRF_V2's Risk Management layer understands it."""

    symbol: str
    direction: str
    entry_price: float
    stop_price: float
    quantity: float
    entry_timestamp: datetime
    target_price: Optional[float] = None


@dataclass(frozen=True)
class PortfolioState:
    """Explicit portfolio state, as IQRF_V2's Risk Management layer takes it."""

    current_equity: float
    peak_equity: float
    open_positions: List[Position] = field(default_factory=list)
    session_realized_pnl: float = 0.0


@dataclass(frozen=True)
class Trade:
    """A single completed round-trip trade, as IQRF_V2's BacktestEngine produces it."""

    symbol: str
    direction: str
    entry_timestamp: datetime
    entry_price: float
    exit_timestamp: datetime
    exit_price: float
    quantity: float
    exit_reason: str  # STOP | TARGET | SIGNAL_EXIT | END_OF_DATA
    gross_pnl: float
    commission: float
    slippage_cost: float
    net_pnl: float
    r_multiple: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def trade_from_dict(d: Dict[str, Any]) -> Trade:
    return Trade(
        symbol=d["symbol"],
        direction=d["direction"],
        entry_timestamp=datetime.fromisoformat(d["entry_timestamp"]),
        entry_price=float(d["entry_price"]),
        exit_timestamp=datetime.fromisoformat(d["exit_timestamp"]),
        exit_price=float(d["exit_price"]),
        quantity=float(d["quantity"]),
        exit_reason=d["exit_reason"],
        gross_pnl=float(d["gross_pnl"]),
        commission=float(d["commission"]),
        slippage_cost=float(d["slippage_cost"]),
        net_pnl=float(d["net_pnl"]),
        r_multiple=d.get("r_multiple"),
        metadata=d.get("metadata", {}),
    )


def position_from_dict(d: Dict[str, Any]) -> Position:
    return Position(
        symbol=d["symbol"],
        direction=d["direction"],
        entry_price=float(d["entry_price"]),
        stop_price=float(d["stop_price"]),
        quantity=float(d["quantity"]),
        entry_timestamp=datetime.fromisoformat(d["entry_timestamp"]),
        target_price=d.get("target_price"),
    )


def portfolio_state_from_dict(d: Dict[str, Any]) -> PortfolioState:
    return PortfolioState(
        current_equity=float(d["current_equity"]),
        peak_equity=float(d["peak_equity"]),
        open_positions=[position_from_dict(p) for p in d.get("open_positions", [])],
        session_realized_pnl=float(d.get("session_realized_pnl", 0.0)),
    )
