"""
multi_day_runner.py -- CLI: runs the risk-brief agent across a real,
multi-day sequence instead of one hand-picked sample day.

    python3 multi_day_runner.py
    python3 multi_day_runner.py --offline
    python3 multi_day_runner.py --drawdown-limit-pct 0.20

For each real day (oldest to newest), in order:
  1. computes RiskSummary (risk_brief.py) exactly as the single-day
     CLI does
  2. runs the full pattern-detector suite (pattern_detector.py) over
     every day SEEN SO FAR (not the whole dataset -- a day's flags
     only ever use information that would have been available by
     that day, same look-ahead discipline as IQRF_V2's own
     walk-forward split)
  3. narrates the day's numbers (llm_narrative.py) and appends any
     pattern flags as plain, deterministic text (flags are already
     decided by step 2 -- the LLM never re-derives or embellishes
     them)
  4. delivers the combined brief over Telegram (dry-run by default)

This is the real multi-day run referenced in README.md's "real usage"
section.
"""

from __future__ import annotations

import argparse
import sys

from llm_narrative import generate_brief, generate_brief_offline
from pattern_detector import run_pattern_detectors
from real_history import cross_instrument_loss_fraction_by_day, load_real_history
from risk_brief import compute_risk_summary
from telegram_notifier import TelegramNotifier


def format_flags(flags) -> str:
    if not flags:
        return ""
    lines = ["", "Pattern alerts (multi-day):"]
    for f in flags:
        marker = "⚠" if f.severity == "WARNING" else "ℹ"
        lines.append(f"  {marker} [{f.name}] {f.message}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="multi_day_runner")
    parser.add_argument(
        "--drawdown-limit-pct", type=float, default=0.20,
        help="real drawdown-limit fraction (default: 0.20, see risk_brief.py)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="use the deterministic template narrator instead of calling the Anthropic API",
    )
    args = parser.parse_args(argv)

    real_days = load_real_history()
    cross_fraction_by_day = cross_instrument_loss_fraction_by_day(real_days)
    notifier = TelegramNotifier()

    daily_trades_seen = []
    exit_code = 0

    for day in real_days:
        daily_trades_seen.append(day.trades)

        summary = compute_risk_summary(
            day.portfolio, day.trades, drawdown_limit_pct=args.drawdown_limit_pct
        )

        flags = run_pattern_detectors(
            daily_trades_seen,
            cross_instrument_loss_fraction_by_day=cross_fraction_by_day,
            latest_date=day.date,
        )

        if args.offline:
            brief = generate_brief_offline(summary)
        else:
            try:
                brief = generate_brief(summary)
            except Exception as e:  # noqa: BLE001 -- degrade, never drop the brief
                print(
                    f"note: live Anthropic API call failed ({e}); "
                    "falling back to offline narrator",
                    file=sys.stderr,
                )
                brief = generate_brief_offline(summary)

        full_message = f"[{day.date}] {brief}{format_flags(flags)}"

        level = "WARNING" if (summary.distance_to_drawdown_limit_pct < 0 or flags) else "INFO"
        notifier.send(full_message, level=level)
        print(full_message)
        print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
