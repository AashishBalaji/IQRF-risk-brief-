"""
generate_daily_brief.py -- CLI entrypoint.

    python3 generate_daily_brief.py sample_data/day_2026-10-29.json
    python3 generate_daily_brief.py sample_data/day_2026-10-29.json --drawdown-limit-pct 0.05
    python3 generate_daily_brief.py sample_data/day_2026-10-29.json --offline

Reads a day's {portfolio, trades_today} JSON (schema: models.py),
computes the real risk numbers (risk_brief.py), narrates them
(llm_narrative.py -- the real Anthropic API call by default, or the
deterministic --offline fallback), and delivers the result via
Telegram (telegram_notifier.py -- real send if TELEGRAM_BOT_TOKEN/
TELEGRAM_CHAT_ID are set, otherwise a dry-run print).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from models import portfolio_state_from_dict, trade_from_dict
from risk_brief import compute_risk_summary
from llm_narrative import generate_brief, generate_brief_offline
from telegram_notifier import TelegramNotifier


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="generate_daily_brief")
    parser.add_argument(
        "day_file", type=Path,
        help="path to a day's {portfolio, trades_today} JSON",
    )
    parser.add_argument(
        "--drawdown-limit-pct", type=float, default=0.06,
        help="your own real drawdown-limit fraction "
             "(default: 0.06 placeholder -- see risk_brief.py)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="use the deterministic template narrator instead of calling the Anthropic API",
    )
    args = parser.parse_args(argv)

    try:
        data = json.loads(args.day_file.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: could not read/parse {args.day_file}: {e}", file=sys.stderr)
        return 2

    portfolio = portfolio_state_from_dict(data["portfolio"])
    trades_today = [trade_from_dict(t) for t in data["trades_today"]]

    summary = compute_risk_summary(
        portfolio, trades_today, drawdown_limit_pct=args.drawdown_limit_pct
    )

    if args.offline:
        brief = generate_brief_offline(summary)
    else:
        try:
            brief = generate_brief(summary)
        except Exception as e:  # noqa: BLE001 -- degrade, never silently drop the brief
            print(
                f"note: live Anthropic API call failed ({e}); "
                "falling back to offline narrator",
                file=sys.stderr,
            )
            brief = generate_brief_offline(summary)

    notifier = TelegramNotifier()
    notifier.send(brief, level="WARNING" if summary.distance_to_drawdown_limit_pct < 0 else "INFO")

    print(brief)
    return 0


if __name__ == "__main__":
    sys.exit(main())
