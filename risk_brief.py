"""
risk_brief.py -- real risk-metric computation over a day's trades
and current portfolio state.

No LLM involvement here: every number this module produces is a
plain, checkable arithmetic result over the Trade/PortfolioState
records it's given (models.py). The LLM layer (llm_narrative.py)
only ever narrates these already-computed numbers -- it never
computes a risk figure itself, so a wrong brief can't come from the
model inventing a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from models import PortfolioState, Trade


@dataclass(frozen=True)
class RiskSummary:
    day_realized_pnl: float
    open_exposure: float
    open_position_count: int
    current_drawdown_pct: float
    drawdown_limit_pct: float
    distance_to_drawdown_limit_pct: float
    stop_outs: int
    target_hits: int
    signal_exits: int
    trade_count: int
    biggest_winner: Optional[Trade]
    biggest_loser: Optional[Trade]
    # A real consistency check: today's trades should sum to the
    # portfolio's own session_realized_pnl. A nonzero drift here is
    # itself a genuine finding -- it means either a trade is missing
    # from `trades_today`, or the portfolio snapshot is stale.
    pnl_reconciliation_drift: float


def compute_risk_summary(
    portfolio: PortfolioState,
    trades_today: List[Trade],
    drawdown_limit_pct: float = 0.20,
) -> RiskSummary:
    """
    drawdown_limit_pct is the maximum current_equity drop from
    peak_equity the operator has decided to tolerate before their
    own kill-switch/risk layer would act. This project does not
    import IQRF_V2's config (see README's "Why a separate project"),
    so this default (0.20) is a value copied by hand from a one-time
    read-only read of IQRF_V2's real src/config/settings.py
    (MAX_DRAWDOWN_PERCENT = 20.0) on 2026-10-31, not a live link to
    it -- if that setting changes later, this default will silently
    go stale until someone updates it here too, or passes the real
    current value explicitly via --drawdown-limit-pct.
    """
    day_realized_pnl = sum(t.net_pnl for t in trades_today)

    open_exposure = sum(
        abs(p.entry_price * p.quantity) for p in portfolio.open_positions
    )

    current_drawdown_pct = (
        (portfolio.peak_equity - portfolio.current_equity) / portfolio.peak_equity
        if portfolio.peak_equity > 0
        else 0.0
    )
    distance_to_drawdown_limit_pct = drawdown_limit_pct - current_drawdown_pct

    stop_outs = sum(1 for t in trades_today if t.exit_reason == "STOP")
    target_hits = sum(1 for t in trades_today if t.exit_reason == "TARGET")
    signal_exits = sum(1 for t in trades_today if t.exit_reason == "SIGNAL_EXIT")

    biggest_winner = max(trades_today, key=lambda t: t.net_pnl, default=None)
    biggest_loser = min(trades_today, key=lambda t: t.net_pnl, default=None)
    if biggest_winner is not None and biggest_winner.net_pnl <= 0:
        biggest_winner = None
    if biggest_loser is not None and biggest_loser.net_pnl >= 0:
        biggest_loser = None

    pnl_reconciliation_drift = round(
        day_realized_pnl - portfolio.session_realized_pnl, 2
    )

    return RiskSummary(
        day_realized_pnl=round(day_realized_pnl, 2),
        open_exposure=round(open_exposure, 2),
        open_position_count=len(portfolio.open_positions),
        current_drawdown_pct=round(current_drawdown_pct, 4),
        drawdown_limit_pct=drawdown_limit_pct,
        distance_to_drawdown_limit_pct=round(distance_to_drawdown_limit_pct, 4),
        stop_outs=stop_outs,
        target_hits=target_hits,
        signal_exits=signal_exits,
        trade_count=len(trades_today),
        biggest_winner=biggest_winner,
        biggest_loser=biggest_loser,
        pnl_reconciliation_drift=pnl_reconciliation_drift,
    )
