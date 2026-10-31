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
  3. narrates the day's numbers (llm_narrative.py) -- numbers only,
     never the flags (flags are already decided by step 2)
  4. hands the fired flags, plus how often each has fired over the
     last ESCALATION_WINDOW_DAYS, to escalation_judge.py, which
     decides whether to interrupt now, hold for the digest, or
     suppress as noise -- the one decision in this pipeline a model
     makes rather than a threshold
  5. delivers the combined message over Telegram at the level the
     judge picked (dry-run by default)

This is the real multi-day run referenced in README.md's "real usage"
section.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

from escalation_judge import (
    HOLD_FOR_DIGEST,
    NOTIFY_NOW,
    judge_escalation,
    judge_escalation_offline,
)
from llm_narrative import generate_brief, generate_brief_offline
from pattern_detector import run_pattern_detectors
from real_history import cross_instrument_loss_fraction_by_day, load_real_history
from risk_brief import compute_risk_summary
from telegram_notifier import TelegramNotifier

ESCALATION_WINDOW_DAYS = 5


def format_flags(flags) -> str:
    if not flags:
        return ""
    lines = ["", "Pattern alerts (multi-day):"]
    for f in flags:
        marker = "⚠" if f.severity == "WARNING" else "ℹ"
        lines.append(f"  {marker} [{f.name}] {f.message}")
    return "\n".join(lines)


def _update_fire_history(fire_history: deque, today_flag_names: set) -> dict:
    """fire_history holds one set of flag names per recent day (oldest
    first, capped at ESCALATION_WINDOW_DAYS). Returns, for each flag
    name that fired today, how many of the days in that window
    (today included) it fired on."""
    fire_history.append(today_flag_names)
    while len(fire_history) > ESCALATION_WINDOW_DAYS:
        fire_history.popleft()
    counts = {}
    for name in today_flag_names:
        counts[name] = sum(1 for day_names in fire_history if name in day_names)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="multi_day_runner")
    parser.add_argument(
        "--drawdown-limit-pct", type=float, default=0.20,
        help="real drawdown-limit fraction (default: 0.20, see risk_brief.py)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="use the deterministic template narrator AND the offline escalation rule "
             "instead of calling the Anthropic API for either",
    )
    args = parser.parse_args(argv)

    real_days = load_real_history()
    cross_fraction_by_day = cross_instrument_loss_fraction_by_day(real_days)
    notifier = TelegramNotifier()

    daily_trades_seen = []
    fire_history: deque = deque()
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

        recent_fire_counts = _update_fire_history(fire_history, {f.name for f in flags})

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

        if args.offline:
            escalation = judge_escalation_offline(flags, recent_fire_counts, ESCALATION_WINDOW_DAYS)
        else:
            try:
                escalation = judge_escalation(flags, recent_fire_counts, ESCALATION_WINDOW_DAYS, summary)
            except Exception as e:  # noqa: BLE001 -- degrade, never drop a real WARNING
                print(
                    f"note: live escalation-judge call failed ({e}); "
                    "falling back to the offline escalation rule",
                    file=sys.stderr,
                )
                escalation = judge_escalation_offline(flags, recent_fire_counts, ESCALATION_WINDOW_DAYS)

        full_message = (
            f"[{day.date}] {brief}{format_flags(flags)}\n"
            f"Escalation: {escalation.decision} -- {escalation.rationale}"
        )

        # the drawdown breach itself is never suppressible -- that one
        # stays a hard rule, same as llm_narrative.py leading with it
        breach = summary.distance_to_drawdown_limit_pct < 0
        if breach or escalation.decision == NOTIFY_NOW:
            level = "WARNING"
        elif escalation.decision == HOLD_FOR_DIGEST:
            level = "INFO"
        else:
            level = "INFO"  # SUPPRESS_AS_NOISE: still logged, just never a WARNING

        notifier.send(full_message, level=level)
        print(full_message)
        print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
