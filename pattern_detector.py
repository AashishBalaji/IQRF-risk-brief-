"""
pattern_detector.py -- the agentic layer.

risk_brief.py answers "what happened today, in numbers." This module
answers a different question: "looking at the last several real
trading days together, is a pattern forming that a single day's
numbers would hide, and is it worth escalating?" That's the gap
between a dashboard and an agent -- deciding whether to flag
something, not just reporting a number that was already computed.

Same discipline as risk_brief.py: every detector here is a plain,
deterministic, testable function over real Trade/RiskSummary data.
No LLM involvement, no invented thresholds dressed up as "AI
judgment" -- each detector states its own threshold explicitly, and
the LLM layer (llm_narrative.py) only ever narrates a PatternFlag
after a detector has already decided to raise it, same as it only
narrates RiskSummary's numbers.

Detectors:

- stop_out_clustering      -- stop-outs trending UP across several
                               days, not just high on one day.
- exit_reason_skew         -- exits dominated by SIGNAL_EXIT rather
                               than STOP/TARGET across a window --
                               the exact question IQRF_V2's own
                               signal_exit_root_cause_report.json was
                               written to investigate.
- symbol_repeat_loser      -- one symbol losing repeatedly across a
                               window, not just once.
- overtrading_frequency    -- a day's trade count is a real outlier
                               against the recent rolling average,
                               not just "more than usual."
- position_size_drift_after_loss -- position size creeping up right
                               after a loss (chasing), rather than
                               down or flat -- a classic
                               risk-of-ruin precursor.
- cross_instrument_stress  -- reuses a real, already-computed,
                               external signal (IQRF_V2's own
                               cross-instrument loss_clustering
                               diagnosis) rather than re-deriving it
                               here from a single symbol's data,
                               which this agent cannot do on its own.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from models import Trade


@dataclass(frozen=True)
class PatternFlag:
    name: str
    severity: str  # "INFO" | "WARNING"
    message: str
    evidence: str


def detect_stop_out_clustering(
    daily_trades: Sequence[List[Trade]],
    window: int = 3,
    threshold: int = 5,
) -> Optional[PatternFlag]:
    """
    Flags when stop-outs across the last `window` real trading days
    (not any single day) reach or exceed `threshold`. A single bad
    day's stop-out count is already in that day's RiskSummary --
    this exists specifically to catch the trend a day-by-day read
    would miss.
    """
    if len(daily_trades) < window:
        return None

    recent = daily_trades[-window:]
    stop_outs = sum(1 for day in recent for t in day if t.exit_reason == "STOP")

    if stop_outs >= threshold:
        return PatternFlag(
            name="stop_out_clustering",
            severity="WARNING",
            message=(
                f"{stop_outs} stop-outs across the last {len(recent)} "
                "trading days -- an escalating pattern, not a one-off "
                "bad day."
            ),
            evidence=f"window={len(recent)}d, stop_outs={stop_outs}, threshold={threshold}",
        )
    return None


def detect_exit_reason_skew(
    daily_trades: Sequence[List[Trade]],
    window: int = 5,
    signal_exit_fraction_threshold: float = 0.60,
    min_trades: int = 8,
) -> Optional[PatternFlag]:
    """
    Flags when SIGNAL_EXIT accounts for an outsized share of all
    exits across the window -- the exit logic may be cutting trades
    before they reach their stop or target, rather than the strategy
    simply having a run of bad signals. This is the real question
    IQRF_V2's own signal_exit_root_cause_report.json exists to
    investigate; this detector raises it automatically from ordinary
    trade logs instead of requiring a hand-run diagnostic script.
    """
    recent = daily_trades[-window:]
    trades = [t for day in recent for t in day]
    if len(trades) < min_trades:
        return None

    signal_exits = sum(1 for t in trades if t.exit_reason == "SIGNAL_EXIT")
    fraction = signal_exits / len(trades)

    if fraction >= signal_exit_fraction_threshold:
        return PatternFlag(
            name="exit_reason_skew",
            severity="INFO",
            message=(
                f"{signal_exits}/{len(trades)} exits over the last "
                f"{len(recent)} days ({fraction:.0%}) were SIGNAL_EXIT, "
                "not STOP or TARGET -- worth checking whether the exit "
                "logic is cutting trades short rather than the strategy "
                "having a genuinely bad run."
            ),
            evidence=(
                f"window={len(recent)}d, signal_exits={signal_exits}, "
                f"total_exits={len(trades)}, fraction={fraction:.4f}, "
                f"threshold={signal_exit_fraction_threshold}"
            ),
        )
    return None


def detect_symbol_repeat_loser(
    daily_trades: Sequence[List[Trade]],
    window: int = 5,
    min_losses: int = 3,
) -> Optional[PatternFlag]:
    """
    Flags a symbol with min_losses+ losing trades across the window
    -- a repeat offender, not a single bad trade on that symbol.
    """
    recent = daily_trades[-window:]
    losses_by_symbol: Dict[str, int] = {}
    for day in recent:
        for t in day:
            if t.net_pnl < 0:
                losses_by_symbol[t.symbol] = losses_by_symbol.get(t.symbol, 0) + 1

    worst_symbol, worst_count = None, 0
    for symbol, count in losses_by_symbol.items():
        if count > worst_count:
            worst_symbol, worst_count = symbol, count

    if worst_symbol is not None and worst_count >= min_losses:
        return PatternFlag(
            name="symbol_repeat_loser",
            severity="WARNING",
            message=(
                f"{worst_symbol} has lost money on {worst_count} separate "
                f"trades over the last {len(recent)} trading days -- worth "
                "checking whether this symbol fits the strategy at all "
                "right now, rather than re-trading it unchanged."
            ),
            evidence=(
                f"window={len(recent)}d, symbol={worst_symbol}, "
                f"losing_trades={worst_count}, threshold={min_losses}"
            ),
        )
    return None


def detect_overtrading_frequency(
    daily_trades: Sequence[List[Trade]],
    baseline_window: int = 5,
    z_threshold: float = 1.5,
) -> Optional[PatternFlag]:
    """
    Flags when today's trade count is a real statistical outlier
    against the recent rolling baseline (z-score), not just "a
    bigger number than usual." Needs at least 3 prior days to form a
    meaningful baseline stdev; otherwise this is silent rather than
    guessing.
    """
    if len(daily_trades) < 4:
        return None

    today = daily_trades[-1]
    baseline = daily_trades[max(0, len(daily_trades) - 1 - baseline_window):-1]
    if len(baseline) < 3:
        return None

    counts = [len(day) for day in baseline]
    mean = statistics.mean(counts)
    stdev = statistics.pstdev(counts)
    today_count = len(today)

    if stdev == 0:
        return None

    z = (today_count - mean) / stdev
    if z >= z_threshold:
        return PatternFlag(
            name="overtrading_frequency",
            severity="WARNING",
            message=(
                f"Today's {today_count} trades is a real outlier against "
                f"the last {len(baseline)} days' average of {mean:.1f} "
                f"(z={z:.2f}) -- worth checking for overtrading rather "
                "than assuming it's just an active day."
            ),
            evidence=(
                f"today={today_count}, baseline_mean={mean:.2f}, "
                f"baseline_stdev={stdev:.2f}, z={z:.2f}, threshold={z_threshold}"
            ),
        )
    return None


def detect_position_size_drift_after_loss(
    daily_trades: Sequence[List[Trade]],
    lookback_trades: int = 10,
    drift_threshold: float = 0.25,
) -> Optional[PatternFlag]:
    """
    Flags when average position size immediately AFTER a losing
    trade runs meaningfully larger than average size immediately
    AFTER a winning trade, within the last `lookback_trades` real
    trades -- a classic "chasing losses by sizing up" precursor to
    risk-of-ruin, not visible from any single day's numbers.

    Needs real per-trade `quantity` variation to say anything -- if
    every trade in the window has the same size (as in a trade log
    that doesn't record quantity separately, e.g. this project's own
    real_data/ extraction from IQRF_V2's signal_exit_root_cause_report,
    which stores net_pnl/r_multiple but not quantity), this correctly
    returns None rather than fabricating a verdict from flat data.
    """
    trades = [t for day in daily_trades for t in day]
    trades = sorted(trades, key=lambda t: t.entry_timestamp)
    window = trades[-lookback_trades - 1:]
    if len(window) < 4:
        return None

    sizes_after_loss = []
    sizes_after_win = []
    for prev, cur in zip(window[:-1], window[1:]):
        if prev.net_pnl < 0:
            sizes_after_loss.append(cur.quantity)
        elif prev.net_pnl > 0:
            sizes_after_win.append(cur.quantity)

    if len(sizes_after_loss) < 2 or len(sizes_after_win) < 2:
        return None

    avg_after_loss = statistics.mean(sizes_after_loss)
    avg_after_win = statistics.mean(sizes_after_win)
    if avg_after_win == 0:
        return None

    drift = (avg_after_loss - avg_after_win) / avg_after_win
    if drift >= drift_threshold:
        return PatternFlag(
            name="position_size_drift_after_loss",
            severity="WARNING",
            message=(
                f"Average position size right after a loss ({avg_after_loss:.2f}) "
                f"runs {drift:.0%} larger than right after a win "
                f"({avg_after_win:.2f}) over the last {len(window) - 1} trades -- "
                "a classic sizing-up-to-chase-losses pattern worth a manual check."
            ),
            evidence=(
                f"n_trades={len(window) - 1}, avg_after_loss={avg_after_loss:.4f}, "
                f"avg_after_win={avg_after_win:.4f}, drift={drift:.4f}, "
                f"threshold={drift_threshold}"
            ),
        )
    return None


def detect_cross_instrument_stress(
    cross_instrument_loss_fraction_by_day: Dict[str, Optional[float]],
    latest_date: str,
    threshold: float = 0.60,
) -> Optional[PatternFlag]:
    """
    Reuses a real signal this agent cannot compute on its own: the
    fraction of instruments IQRF_V2 tracks (12, in the real backtest
    this project's real_data/ was extracted from) that lost money on
    a given day, as already computed by IQRF_V2's own cross-sectional
    loss_clustering diagnosis. Flags when that fraction, for the
    current day, crosses threshold -- a market-wide stress day, not a
    symbol-specific problem, which changes what action makes sense
    (sit out broadly vs. review one symbol).
    """
    fraction = cross_instrument_loss_fraction_by_day.get(latest_date)
    if fraction is None:
        return None

    if fraction >= threshold:
        return PatternFlag(
            name="cross_instrument_stress",
            severity="WARNING",
            message=(
                f"{fraction:.0%} of IQRF_V2's tracked instruments lost money "
                f"on {latest_date} -- a market-wide stress day, not a "
                "problem specific to this symbol's trades."
            ),
            evidence=(
                f"date={latest_date}, fraction_instruments_negative={fraction:.4f}, "
                f"threshold={threshold}"
            ),
        )
    return None


def run_pattern_detectors(
    daily_trades: Sequence[List[Trade]],
    cross_instrument_loss_fraction_by_day: Optional[Dict[str, Optional[float]]] = None,
    latest_date: Optional[str] = None,
) -> List[PatternFlag]:
    """
    Runs every detector that has enough data to speak, in a fixed
    order (escalating severity roughly first), and returns only the
    flags that actually fired -- the decision of whether to escalate
    at all lives here, one call site, rather than being left to
    whatever calls this.
    """
    flags: List[PatternFlag] = []

    for flag in (
        detect_stop_out_clustering(daily_trades),
        detect_symbol_repeat_loser(daily_trades),
        detect_overtrading_frequency(daily_trades),
        detect_position_size_drift_after_loss(daily_trades),
        detect_exit_reason_skew(daily_trades),
    ):
        if flag is not None:
            flags.append(flag)

    if cross_instrument_loss_fraction_by_day is not None and latest_date is not None:
        cross_flag = detect_cross_instrument_stress(
            cross_instrument_loss_fraction_by_day, latest_date
        )
        if cross_flag is not None:
            flags.append(cross_flag)

    return flags
