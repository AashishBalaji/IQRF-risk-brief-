# iqrf-risk-brief

A small daily risk-brief agent for an intraday equity trader. Built
for a Razorpay "AI Builder" portfolio piece — the AI-Builder loop
performed on myself: a real problem I actually have (reading my own
trading system's risk state every day takes manual digging), a v1
shipped in a day with AI tools, used on a real cadence, kept or
killed based on whether it's actually useful.

It is **deliberately a separate, independent project from
[`IQRF_V2`](../IQRF-)**, my real intraday trading system — see
"Why a separate project" below.

## What it does

Given a day's trades and current portfolio state, it:

1. Computes real risk numbers — day realized P&L, open exposure,
   current drawdown vs a configurable limit, stop-outs vs
   target-hits, biggest winner/loser, and a P&L reconciliation
   check (today's trades should sum to the portfolio's own
   `session_realized_pnl`; a nonzero drift is itself a real finding
   — a missing trade or a stale snapshot) — `risk_brief.py`.
2. Turns those numbers into a 3–5 sentence plain-English brief via
   a real Claude API call, told to use *only* the numbers it's
   given, never invent one — `llm_narrative.py`. A deterministic,
   no-network offline narrator (`generate_brief_offline`) exists as
   both a tested fallback and the honest path for this README (see
   "What's actually verified" below).
3. Delivers it over Telegram — `telegram_notifier.py`.

Run it:

```
pip install -r requirements.txt
python3 generate_daily_brief.py sample_data/day_2026-10-29.json --offline
```

## Why a separate project

I was explicit with myself about this while building it: this
project **never imports from, executes, or writes to** the
`IQRF_V2` repository. I cloned it once, read-only, purely to copy
the real field names/types of its `Trade`/`Position`/
`PortfolioState` dataclasses into `models.py` here — structurally
identical, zero code dependency — so this agent could, if pointed
at a real exported log, read it with no translation layer. Nothing
in this repo touches IQRF_V2's code, its data, or its live trading
in any way.

## What's actually verified, and what isn't

Following the same discipline as my other portfolio projects (see
[`kiln`](../kiln)) — build for real, verify end to end, say
honestly what wasn't checked:

- **Risk math (`risk_brief.py`)** — fully verified. 15/15 tests
  pass, run against two real sample days (`sample_data/`): a steady
  day and a day deliberately constructed so `current_equity` vs
  `peak_equity` crosses a 6% drawdown. The drawdown-breach
  detection, the P&L reconciliation drift check, and the
  custom-limit override are all exercised against real numbers, not
  mocked.
- **Offline narrator (`generate_brief_offline`)** — fully verified,
  run end to end against both sample days; this is the path
  actually demonstrated in this README's example output.
- **Live Claude API call (`generate_brief`)** — implemented against
  the real `anthropic` SDK (`client.messages.create(...)`,
  `claude-sonnet-4-5-20250929`), but **not independently run** in
  the sandbox this was built in, which has no bare
  `ANTHROPIC_API_KEY` available to it. `generate_daily_brief.py`
  calls it by default and falls back to the offline narrator on any
  failure (tested: a missing-key run genuinely triggers this
  fallback, printing the real authentication error to stderr
  first). To verify the live path yourself: `export
  ANTHROPIC_API_KEY=...` and run without `--offline`.
- **Telegram delivery (`telegram_notifier.py`)** — the dry-run path
  (no credentials configured) is fully verified: it prints exactly
  what would be sent, and this is what every example in this README
  actually ran. The real-send path is unit-tested against a mocked
  `requests.post` (correct URL, correct payload, correct error
  handling on `ok: false`), but **not yet proven against the real
  Telegram Bot API** — that needs a real bot token and chat ID,
  which I don't have in hand while writing this. To verify for
  real: message `@BotFather` on Telegram, `/newbot`, then `export
  TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...` and run without
  `dry_run=True`.

## Example output (offline narrator, real run)

```
$ python3 generate_daily_brief.py sample_data/day_2026-10-29.json --offline
[telegram dry-run, level=WARNING] Drawdown is at 6.1%, past your 6% limit. Day P&L is -784.0 across 4 trades (1 targets, 3 stops, 0 signal exits). Open exposure is 32500.0 across 1 position(s).
Drawdown is at 6.1%, past your 6% limit. Day P&L is -784.0 across 4 trades (1 targets, 3 stops, 0 signal exits). Open exposure is 32500.0 across 1 position(s).
```

## Wiring into IQRF_V2 (documented, not done here)

IQRF_V2 already has a clean extension point for exactly this: its
`src/live_trading/alerts.py` defines `BaseAlertChannel` (one method,
`send(message, level="INFO") -> None`) and an `AlertManager` that
takes a list of channels — its own docstring says "a real Slack/
email channel implements the same contract and joins AlertManager's
channel list with no other changes needed downstream."

`TelegramNotifier.send()` here matches that exact signature on
purpose. If I ever want this wired into IQRF_V2 for real, the
integration is one line in that repo:

```python
from telegram_notifier import TelegramNotifier  # this repo, not copied in
alert_manager = AlertManager(channels=[LogAlertChannel(), TelegramNotifier()])
```

I have not made that change — it would mean editing IQRF_V2, which
this project exists specifically to avoid.

## Files

```
models.py                 Trade/Position/PortfolioState (schema-matched to IQRF_V2, independent)
risk_brief.py              real risk-metric computation
llm_narrative.py           Claude API narration + offline fallback
telegram_notifier.py       Telegram Bot API delivery, dry-run by default
generate_daily_brief.py    CLI entrypoint
sample_data/                two illustrative sample days (not live-exported)
tests/                     15 tests, all passing
```
