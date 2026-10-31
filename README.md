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
2. Looks across the sequence of real days seen so far — not just
   today — and decides whether a pattern is forming that a single
   day's numbers would hide: stop-outs clustering across days, one
   symbol losing repeatedly, a day's trade count that's a real
   statistical outlier against its own recent baseline, position
   size creeping up right after a loss, the exit logic being
   dominated by `SIGNAL_EXIT` rather than `STOP`/`TARGET`, and a
   market-wide stress day (reusing IQRF_V2's own real cross-
   instrument diagnosis rather than re-deriving it from one
   symbol's data, which isn't possible here) — `pattern_detector.py`.
   This is the actual decision-making layer: each detector decides
   for itself whether to escalate, against a stated threshold, and
   only a fired flag ever reaches the brief. See "Agentic layer" below.
3. Turns the day's numbers into a 3–5 sentence plain-English brief
   via a real Claude API call, told to use *only* the numbers it's
   given, never invent one — `llm_narrative.py`. A deterministic,
   no-network offline narrator (`generate_brief_offline`) exists as
   both a tested fallback and the honest path for this README (see
   "What's actually verified" below). Any pattern flags from step 2
   are appended as plain, deterministic text — the LLM never
   re-derives or embellishes a flag that's already been decided.
4. Decides whether today's fired flags are worth interrupting the
   trader for right now, worth holding for the end-of-day digest, or
   worth suppressing as noise — `escalation_judge.py`. This is the
   one place in the whole pipeline a model actually judges something
   rather than narrating a decision a threshold already made. See
   "Escalation layer" below.
5. Delivers it over Telegram — `telegram_notifier.py`.

Run it against one hand-built sample day:

```
pip install -r requirements.txt
python3 generate_daily_brief.py sample_data/day_2026-10-29.json --offline
```

Run it across the real 13-day history (see "Real multi-day history, and real usage" below):

```
python3 multi_day_runner.py --offline
```

## Agentic layer (`pattern_detector.py`)

v1 of this project only narrated numbers that were already computed
— useful, but not really *agentic*: nothing in it decided anything.
`pattern_detector.py` is the response to that gap. Each detector is
a plain, deterministic, testable function (same "math decides,
narration only describes" discipline as `risk_brief.py` — no LLM
judgment calls dressed up as pattern detection) that looks across
**several real days at once** and decides, against a threshold it
states explicitly, whether to raise a flag:

| Detector | What it catches | Real finding on this project's own 13-day history |
|---|---|---|
| `stop_out_clustering` | Stop-outs trending up across days, not one bad day | did not fire in this window |
| `exit_reason_skew` | Exits dominated by `SIGNAL_EXIT` over `STOP`/`TARGET` | **fired from day 2 onward** — 60–92% of exits across the window were `SIGNAL_EXIT`, every day. This is the exact question IQRF_V2's own `signal_exit_root_cause_report.json` exists to investigate; this detector raises it automatically from ordinary trade logs. |
| `symbol_repeat_loser` | One symbol losing repeatedly | fired every day — honestly, close to a given with only one traded symbol in this fixture (see caveat below) |
| `overtrading_frequency` | Today's trade count is a real statistical outlier (z-score) vs its own recent baseline | fired on 2026-06-04 (z=7.35, 13 trades vs a 7.0 baseline) and twice more later in the window |
| `position_size_drift_after_loss` | Position size creeping up right after a loss (chasing) | did not fire — the real fixture has no per-trade quantity data (see "Known gaps" below), so this correctly stays silent rather than fabricating a verdict. Covered by its own unit tests with real quantity variation. |
| `cross_instrument_stress` | A market-wide stress day, reusing IQRF_V2's own already-computed cross-sectional diagnosis | fired on 5 of the 13 real days (2026-06-03/04/05/09/11), matching IQRF_V2's own `loss_clustering.worst_days` exactly |

**Honest caveat on `symbol_repeat_loser`:** it's correct and
generically useful, but in this single-symbol (RELIANCE-only) real
fixture it's close to trivially true — there's only one symbol that
*could* be the repeat offender. It would differentiate properly on
a multi-symbol trade log; it's included here, not hidden, because
pretending every detector is equally interesting on this particular
dataset would be the same kind of overselling this whole project is
trying to avoid.

## Escalation layer (`escalation_judge.py`)

A round-table review of v1 surfaced a real gap underneath the
"agentic" pattern layer above: every one of those six detectors is
still a fixed threshold. Nothing in the pipeline ever *weighs* flags
against each other, or against their own recent history — a single
INFO-level `exit_reason_skew` on day one means something different
from the same flag firing for the fifth day running, and two WARNING
flags together aren't just "twice as bad" as one. That's a judgment
call, not a threshold, which is exactly what a model is for — so
`escalation_judge.py` asks a model exactly one question: given
today's already-fired flags (nothing recomputed, nothing
re-derived) plus how often each has fired over the last 5 trading
days, should this `NOTIFY_NOW`, `HOLD_FOR_DIGEST`, or
`SUPPRESS_AS_NOISE`? Same "math decides, model only judges or
narrates" discipline as `llm_narrative.py` — the model is never
shown raw trades and never asked to produce a number, only a label
and a one-line rationale.

A deterministic `judge_escalation_offline` exists alongside the real
API call (`judge_escalation`), same pattern as
`generate_brief`/`generate_brief_offline`: it's the floor the model
is meant to improve on, not a hidden second judge, used by tests and
as the graceful degrade if the live call fails (a broken network
should never mean a real WARNING gets silently dropped).

**Real finding, running the offline rule against the real 13-day
history (`python3 multi_day_runner.py --offline`):** 2 of the 13
real days hold at `HOLD_FOR_DIGEST` (2026-06-01, before
`symbol_repeat_loser` has had time to recur) and the other 11
escalate to `NOTIFY_NOW` — mostly because `symbol_repeat_loser`
recurs on 5 of the last 5 days for almost the entire window (see the
"Honest caveat" above: single-symbol fixture). Said plainly, the
offline rule saturates to "notify" almost immediately on this
particular dataset and stays there — which is itself an honest
finding about the *rule*, not a reason to hide it: a real triage
layer needs more than a 5-day recurrence count to stay useful on a
single-symbol history, and a multi-symbol trade log is the real test
this escalation layer hasn't had yet. The live model path
(`judge_escalation`) is implemented but not run against this real
data in this sandbox — see "What's actually verified" below — so
whether it behaves less bluntly than the offline floor on these same
13 days is still an open question, not a claim this README makes.

## Real multi-day history, and real usage

v1 shipped with two hand-written sample days (`sample_data/`) —
honest, but synthetic, and single-day, so it couldn't demonstrate
anything about patterns *across* days or give a real "I ran this and
it caught something" story.

`real_data/reliance_june_2026.json` fixes that with **real, not
synthetic, data**: 92 real trades across 13 real trading days
(2026-06-01 to 2026-06-17), extracted read-only from two backtest
reports IQRF_V2 had *already run and recorded itself* —
`data/signal_exit_root_cause_report.json` (`per_trade`) and
`data/cross_sectional_diagnosis_report.json`
(`per_instrument.RELIANCE.daily_pnl`, `loss_clustering.worst_days`).
Extracting it took nothing but reading two JSON files — no IQRF_V2
code was executed, no IQRF_V2 file was touched. Full provenance,
including exactly what was and wasn't available in the source
reports, is in the fixture's own `_provenance` block.

**What this is, and isn't:** it's a real backtest result from a real
historical run of IQRF_V2's own strategy over real RELIANCE market
data — not hand-authored, not invented. It is *not* live money and
not "trades I personally made today" — I don't have a live Upstox
feed or broker token in the environment this was built in. I'm
treating this as the honest, available version of "real usage": the
agent runs against IQRF_V2's own real historical performance, in
order, day by day, and both the risk numbers and the pattern flags
above are genuine outputs of that run, not illustrations.

`multi_day_runner.py` runs the full agent (risk numbers + pattern
detection + escalation judgment + narration + Telegram delivery)
across all 13 real days in order, each day only ever seeing the days
before it (same no-lookahead discipline as IQRF_V2's own
walk-forward split) — `python3 multi_day_runner.py --offline`
reproduces the findings table above, plus the escalation decisions
in "Escalation layer" above, end to end.

**Known gap, stated plainly:** `signal_exit_root_cause_report.json`
stores `net_pnl` and `r_multiple` per trade but not a
quantity/commission/slippage breakdown, so `real_history.py` sets
`quantity=1`, `gross_pnl=net_pnl`, `commission=0`,
`slippage_cost=0` for every real trade rather than inventing a split
IQRF_V2 itself didn't report. This also means
`position_size_drift_after_loss` has no real signal to find in this
fixture (expected and tested, not a bug). Separately, the two real
source reports' own daily P&L figures don't agree with each other
exactly (likely different experiment configs) — rather than forcing
them to reconcile, this project's own `pnl_reconciliation_drift`
check surfaces that disagreement as a real finding on nearly every
day, which is arguably the single most honest thing about this
fixture.

## Real usage log (`real_usage_log.md`)

The backtest replay above is real historical data, but it's still a
batch run, not evidence that anyone actually *uses* this day to day.
`live_usage_check.py` is the honest start of that evidence: it takes
a real live quote snapshot (fetched via the same authenticated Kite
MCP connector as the demo artifact's "Live market pulse" tab) for
the real 12-instrument basket, reuses this project's own
`detect_cross_instrument_stress` against it, and runs the result
through `judge_escalation_offline` — then appends what actually
happened, with a real UTC timestamp, to `real_usage_log.md`. Each
entry there is a real check, not a staged one; `real_usage_log.md`'s
own header states plainly that the account behind it has no open
positions, trades, or orders right now, so these are real
market-data signal checks and real escalation decisions, not live
trading activity — a distinction kept explicit rather than blurred.
This is one real entry so far (2026-10-30 market data: 5/12 basket
instruments down, 41.7%, below the 60% stress threshold, correctly
suppressed as noise), not a backfilled history — more entries land
here only as the project actually gets run again, by hand for now.

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

- **Risk math (`risk_brief.py`)** — fully verified. Tests pass
  against two hand-built sample days (`sample_data/`) and the real
  13-day history below. The drawdown-breach detection, the P&L
  reconciliation drift check, and the custom-limit override are all
  exercised against real numbers, not mocked.
- **Pattern detection (`pattern_detector.py`)** — fully verified.
  Each detector has isolated unit tests pinning down exactly when it
  should and shouldn't fire, plus one integration test running the
  whole suite against the real 13-day history end to end and
  asserting the specific findings documented in "Agentic layer"
  above — a regression there means the fixture or a threshold
  changed, and the test will catch it.
- **Real multi-day history (`real_history.py`, `real_data/`)** —
  fully verified: loads all 13 real days, all 92 real trades, and
  the real cross-instrument loss fractions, with the exact counts
  asserted in tests.
- **Escalation judgment (`escalation_judge.py`)** — the offline rule
  (`judge_escalation_offline`) is fully verified: isolated tests pin
  down exactly when it suppresses, holds, or notifies, including the
  recurrence-weighted cases (a lone recurring flag notifies even
  without a second WARNING; a gap then a re-fire doesn't inherit an
  old streak), plus the end-to-end real 13-day run documented in
  "Escalation layer" above. The rolling fire-history bookkeeping that
  feeds it (`multi_day_runner._update_fire_history`) has its own
  tests, including the window-cap and the no-false-carryover case.
  The **live model call (`judge_escalation`)** is implemented against
  the real `anthropic` SDK, same model and client pattern as
  `generate_brief`, but **not independently run** in this sandbox for
  the same reason: no bare `ANTHROPIC_API_KEY` available. Every path
  through `multi_day_runner.py` falls back to the offline rule on any
  failure, so a broken live call degrades the judgment, never drops a
  real WARNING. To verify the live path yourself: `export
  ANTHROPIC_API_KEY=...` and run `multi_day_runner.py` without
  `--offline`.
- **46/46 tests pass** across `risk_brief.py`, `llm_narrative.py`,
  `telegram_notifier.py`, `pattern_detector.py`, `real_history.py`,
  `escalation_judge.py`, and `multi_day_runner.py`. Flake8 clean
  (`--max-line-length=110 --exclude=real_data`).
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

Single hand-built sample day:

```
$ python3 generate_daily_brief.py sample_data/day_2026-10-29.json --offline
[telegram dry-run, level=WARNING] Drawdown is at 20.5%, past your 20% limit. Day P&L is -784.0 across 4 trades (1 targets, 3 stops, 0 signal exits). Open exposure is 32500.0 across 1 position(s).
Drawdown is at 20.5%, past your 20% limit. Day P&L is -784.0 across 4 trades (1 targets, 3 stops, 0 signal exits). Open exposure is 32500.0 across 1 position(s).
```

Real 13-day history, with the agentic pattern layer firing on real
findings (`multi_day_runner.py --offline`, two representative days
shown):

```
$ python3 multi_day_runner.py --offline
[2026-06-04] Day P&L is -5860.78 across 13 trades (0 targets, 0 stops, 13 signal exits). Open exposure is 0 across 0 position(s). Data check: today's trades sum to a P&L that differs from the portfolio's own session figure by -2239.46.
Pattern alerts (multi-day):
  ⚠ [symbol_repeat_loser] RELIANCE has lost money on 26 separate trades over the last 4 trading days -- worth checking whether this symbol fits the strategy at all right now, rather than re-trading it unchanged.
  ⚠ [overtrading_frequency] Today's 13 trades is a real outlier against the last 3 days' average of 7.0 (z=7.35) -- worth checking for overtrading rather than assuming it's just an active day.
  ℹ [exit_reason_skew] 31/34 exits over the last 4 days (91%) were SIGNAL_EXIT, not STOP or TARGET -- worth checking whether the exit logic is cutting trades short rather than the strategy having a genuinely bad run.
  ⚠ [cross_instrument_stress] 67% of IQRF_V2's tracked instruments lost money on 2026-06-04 -- a market-wide stress day, not a problem specific to this symbol's trades.
Escalation: NOTIFY_NOW -- 3 WARNING flag(s) and/or a flag recurring 4 of the last 5 days (offline rule, not the model).

[2026-06-11] Day P&L is -3701.67 across 9 trades (0 targets, 2 stops, 7 signal exits). Open exposure is 0 across 0 position(s). Data check: today's trades sum to a P&L that differs from the portfolio's own session figure by -1265.71.
Pattern alerts (multi-day):
  ⚠ [symbol_repeat_loser] RELIANCE has lost money on 20 separate trades over the last 5 trading days -- worth checking whether this symbol fits the strategy at all right now, rather than re-trading it unchanged.
  ℹ [exit_reason_skew] 20/32 exits over the last 5 days (62%) were SIGNAL_EXIT, not STOP or TARGET -- worth checking whether the exit logic is cutting trades short rather than the strategy having a genuinely bad run.
  ⚠ [cross_instrument_stress] 83% of IQRF_V2's tracked instruments lost money on 2026-06-11 -- a market-wide stress day, not a problem specific to this symbol's trades.
Escalation: NOTIFY_NOW -- 2 WARNING flag(s) and/or a flag recurring 5 of the last 5 days (offline rule, not the model).
```

Full 13-day escalation tally from this same run: 2 days hold for the
digest (`HOLD_FOR_DIGEST`), 11 notify now (`NOTIFY_NOW`), 0 suppressed
— see "Escalation layer" above for what that actually means (and
doesn't).

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
pattern_detector.py        agentic cross-day pattern detection (task A)
escalation_judge.py        the model's one real judgment call: notify / hold / suppress
real_history.py            loads the real 13-day RELIANCE history
multi_day_runner.py        CLI: runs the agent across the real history, incl. escalation (task B)
llm_narrative.py           Claude API narration + offline fallback
telegram_notifier.py       Telegram Bot API delivery, dry-run by default
generate_daily_brief.py    CLI entrypoint for a single day
live_usage_check.py        real usage-log entry generator (task B): real live Kite quotes -> real decision
real_usage_log.md          the append-only real usage log live_usage_check.py writes to
sample_data/                two hand-built illustrative sample days
real_data/                  the real 13-day RELIANCE history, extracted read-only from IQRF_V2's own reports
tests/                     46 tests, all passing
```
