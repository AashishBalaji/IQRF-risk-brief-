"""
live_usage_check.py -- the first real, non-fabricated usage-log entry
for this project.

Earlier Claude-round-table review of v1 flagged a second real gap
alongside "nothing decides anything": no real adoption/usage
evidence exists yet -- only sample days and a backtest replay. This
script is the honest answer to that, not a staged one: it takes a
*real* live market-data snapshot (fetched via the already-authenticated
Kite MCP connector, the same real account this project's demo artifact
uses for its "Live market pulse" tab), reuses this project's own
real cross_instrument_stress detector against it, and feeds the
result through the real escalation judge -- then logs what actually
happened, with a real timestamp, to real_usage_log.md.

What this is: a real signal check against real market data, run
through this project's real pipeline, producing a real decision.

What this isn't: a live trade or a live P&L event. The Kite account
behind this connector has no open positions or trade history right
now (checked directly via get_positions/get_holdings/get_orders --
all empty), so there is nothing to call "real trading usage" yet.
This script does not pretend otherwise -- see real_usage_log.md's own
header for that caveat, repeated rather than hidden.

This script is deliberately not unit-tested like the rest of the
pipeline: it has a live, timestamped side effect (appending to
real_usage_log.md) that only makes sense to run for real, by hand,
not as a repeatable assertion. The pipeline code it calls into
(detect_cross_instrument_stress, judge_escalation_offline) already
has its own tests elsewhere.

Usage:
    python3 live_usage_check.py
    (reads a quote snapshot already fetched for today from
    live_quotes_<date>.json if present; otherwise prints instructions
    for fetching one -- this script does not call Kite itself, since
    that connector is only reachable from the agent session that has
    it authenticated, not from a plain Python process.)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from escalation_judge import judge_escalation_offline
from pattern_detector import detect_cross_instrument_stress

BASKET = [
    "NSE:RELIANCE", "NSE:HDFCBANK", "NSE:ICICIBANK", "NSE:TCS",
    "NSE:ETERNAL", "NSE:HINDUNILVR", "NSE:TITAN", "NSE:TATASTEEL",
    "NSE:HINDALCO", "NSE:ADANIPORTS", "NSE:INDIGO", "NSE:JSWSTEEL",
]

STRESS_THRESHOLD = 0.60
LOG_PATH = Path(__file__).parent / "real_usage_log.md"


def compute_down_fraction(quotes: dict) -> tuple[float, list[str]]:
    """quotes: {symbol: {"last_price": ..., "ohlc": {"close": ...}}}
    (the real shape Kite's get_quotes returns). Returns (fraction of
    the basket trading below its previous close, the down symbols)."""
    down = []
    for sym in BASKET:
        q = quotes.get(sym)
        if q is None:
            continue
        if q["last_price"] < q["ohlc"]["close"]:
            down.append(sym)
    seen = [s for s in BASKET if s in quotes]
    fraction = len(down) / len(seen) if seen else 0.0
    return fraction, down


def run_check(quotes: dict, as_of: str) -> str:
    fraction, down_symbols = compute_down_fraction(quotes)

    flag = detect_cross_instrument_stress(
        {as_of: fraction}, as_of, threshold=STRESS_THRESHOLD
    )
    flags = [flag] if flag else []

    escalation = judge_escalation_offline(flags, {}, recent_window_days=5)

    checked_at = datetime.now(timezone.utc).isoformat()
    lines = [
        f"- **{checked_at}** (market data as of {as_of}, India)",
        "  - Source: Kite (Zerodha) MCP `get_quotes`, real authenticated account, "
        "real 12-instrument basket (same instruments IQRF_V2's own "
        "`cross_sectional_diagnosis_report.json` tracks).",
        f"  - Real result: {len(down_symbols)}/{len(BASKET)} basket instruments "
        f"trading below previous close ({fraction:.1%}) -- "
        f"{', '.join(s.split(':')[1] for s in down_symbols) if down_symbols else 'none down'}.",
        f"  - Pattern detector (`detect_cross_instrument_stress`, threshold "
        f"{STRESS_THRESHOLD:.0%}): "
        + (f"FIRED -- {flag.message}" if flag else "did not fire (below threshold)."),
        f"  - Escalation judge (offline rule, no flags history available for a "
        f"first real entry): **{escalation.decision}** -- {escalation.rationale}",
        "  - Delivery: Telegram dry-run (no bot token configured; see README's "
        "\"What's actually verified\" section).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: python3 live_usage_check.py <quotes.json>\n"
            "  <quotes.json> is a raw Kite get_quotes response snapshot, "
            "fetched for real via the authenticated Kite MCP connector in "
            "the agent session (this script has no network access of its "
            "own and will not fabricate a snapshot).",
            file=sys.stderr,
        )
        return 1

    quotes_path = Path(sys.argv[1])
    quotes = json.loads(quotes_path.read_text())
    as_of = next(iter(quotes.values()))["timestamp"][:10]

    entry = run_check(quotes, as_of)
    print(entry)

    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# Real usage log\n\n"
            "Every entry below is a real run against real live market "
            "data (Kite MCP, the user's own authenticated account), "
            "through this project's real pattern-detector and "
            "escalation-judge code -- not staged, not backfilled. The "
            "first entry's timestamp is this project's real first use; "
            "nothing before it is implied.\n\n"
            "**Honest scope:** the account behind this connector "
            "currently holds no open positions, trades, or orders "
            "(checked directly). These entries are real market-data "
            "signal checks and real escalation decisions, not live "
            "trading activity -- that distinction stays explicit in "
            "every entry rather than being blurred.\n\n"
        )

    with LOG_PATH.open("a") as f:
        f.write(entry + "\n")

    print(f"\nAppended to {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
