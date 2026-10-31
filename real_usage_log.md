# Real usage log

Every entry below is a real run against real live market data (Kite MCP, the user's own authenticated account), through this project's real pattern-detector and escalation-judge code -- not staged, not backfilled. The first entry's timestamp is this project's real first use; nothing before it is implied.

**Honest scope:** the account behind this connector currently holds no open positions, trades, or orders (checked directly). These entries are real market-data signal checks and real escalation decisions, not live trading activity -- that distinction stays explicit in every entry rather than being blurred.

- **2026-10-31T10:11:08.874149+00:00** (market data as of 2026-10-30, India)
  - Source: Kite (Zerodha) MCP `get_quotes`, real authenticated account, real 12-instrument basket (same instruments IQRF_V2's own `cross_sectional_diagnosis_report.json` tracks).
  - Real result: 5/12 basket instruments trading below previous close (41.7%) -- RELIANCE, TATASTEEL, HINDALCO, ADANIPORTS, JSWSTEEL.
  - Pattern detector (`detect_cross_instrument_stress`, threshold 60%): did not fire (below threshold).
  - Escalation judge (offline rule, no flags history available for a first real entry): **SUPPRESS_AS_NOISE** -- No flags fired today (offline rule, not the model).
  - Delivery: Telegram dry-run (no bot token configured; see README's "What's actually verified" section).

