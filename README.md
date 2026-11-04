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

**Upgraded for a "Junior Generative AI Engineer" JD** (RAG, LLM
frameworks, vector databases, FastAPI, agentic tool-calling) with a
real RAG layer (LangChain + ChromaDB) over this project's own real
data, an agentic tool that re-runs the real detectors live instead of
retrieving a stale answer, and a FastAPI service wrapping all of it —
see "GenAI upgrade" below. This is an addition, not a rewrite:
everything above this point in the README describes the original
Razorpay-targeted agent exactly as it was before the upgrade, and all
of it still works unchanged.

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

## GenAI upgrade: RAG, vector search, and a served API (`rag/`, `api.py`)

Everything above this section predates the upgrade and still runs
exactly as described. This section adds the three things a "Junior
Generative AI Engineer" JD asks for that the original agent didn't
have: a RAG pipeline over real data, a vector database, and a served
Python API — plus a step toward real agentic tool-calling, building
on the judgment layer (`escalation_judge.py`) this project already
had.

**What it does:** given a natural-language question, it retrieves
the most relevant real documents from a ChromaDB vector store and
either answers from them directly (offline, extractive) or asks
Claude to narrate an answer *from only those documents* (live, same
"never invent a fact" discipline as `llm_narrative.py`) — or, for a
question about the *current* live state ("is anything flagged right
now?"), it skips retrieval entirely and actually re-runs the real
pattern detectors and escalation rule, because a cached document
would go stale the moment a new real day lands.

```
rag/corpus_builder.py   builds the real document corpus: every real day's
                        walk-forward output (re-derived, not cached), the
                        real usage log, labelled illustrative samples,
                        reference docs pulled from the detectors' own
                        docstrings, and README.md split into real,
                        overlapping chunks (see "Real chunking" below)
rag/embeddings.py       LocalSentenceTransformerEmbeddings (a real,
                        pretrained local semantic-embedding model) is the
                        default; OfflineTfidfEmbeddings (scikit-learn
                        TF-IDF, no network at all) is the tested fallback
                        if the model can't load. Both implement
                        LangChain's Embeddings interface; get_embeddings()
                        is the one seam a hosted provider (OpenAI) plugs
                        into instead
rag/vector_store.py     builds the ChromaDB collection from the corpus
rag/query_log.py        a real SQLite table logging every /ask call (see
                        "Real SQL" below) -- not a cache, a query-history
                        log read back with real SQL
rag/tools.py            check_live_pattern_state -- a LangChain Tool that
                        re-runs the real detectors live; route_offline is
                        the deterministic keyword router (the tested floor
                        a real tool-calling agent is meant to improve on)
rag/agent.py            run_tool_calling_agent() -- the real two-turn
                        tool-calling loop: binds rag/tools.py's real Tool
                        to a model via bind_tools(), lets the model decide
                        whether to call it, and feeds the real tool output
                        back for a final answer (see "Real agentic
                        tool-calling" below)
rag/eval.py             a real evaluation harness: retrieval precision@k
                        against the real corpus, and a groundedness score
                        checking an answer's stated numbers against its
                        real sources (see "Real evaluation" below)
rag/rag_chain.py         ask() -- the single entry point: tool router, then
                        RAG retrieval, live-model-with-offline-fallback
api.py                  FastAPI service: /ask, /brief, /pattern-check,
                        /query-history, /eval/retrieval, /health
```

**Real semantic embeddings (updated 2026-11-04):** the vector
embeddings now come from a real, pretrained local model
(`sentence-transformers/all-MiniLM-L6-v2`, 384 dims) via
`LocalSentenceTransformerEmbeddings` — no API key, no per-call
network, only a one-time weights download. An earlier version of
this README said `huggingface.co` was blocked by this sandbox's
egress allowlist (same block `veridex`'s README documents for its
own FAISS layer); re-tested directly, the download now succeeds.
`tests/test_embeddings.py::test_semantic_retrieval_finds_a_synonymous_match_tfidf_cannot`
is the actual proof this is real semantic matching and not a relabel:
it shows TF-IDF scoring a synonym-only paraphrase at `0.0` similarity
against the document it should match, and the local model scoring
the same pair above `0.5` — a gap TF-IDF cannot close by construction
(it only ever matches shared vocabulary). `OfflineTfidfEmbeddings` is
kept as the tested, no-network fallback `get_embeddings()` reaches
for if the local model ever fails to load — logged when that happens,
never silent — not a placeholder, a real exercised path.

**Real chunking (`corpus_builder._readme_documents`):** README.md
(500+ real lines, nothing fabricated for this purpose) is the
corpus's only document long enough to need splitting, so it's the
one that gets a real chunk-size/overlap decision instead of an
invented one: `RecursiveCharacterTextSplitter`, 800-char chunks with
120-char overlap, splitting on markdown headers first, then
paragraphs, then sentences, only falling back to a mid-sentence
break as a last resort. 800/120 were chosen for this specific
document (most README sections are shorter than 800 chars and so
survive as one chunk; the overlap is sized to usually keep a short
code example attached to the sentence introducing it), not left at a
library default — see `corpus_builder.py`'s comment for the full
reasoning.

**Real SQL (`rag/query_log.py`, `GET /query-history`):** every
`/ask` call is logged to a real SQLite table (one `INSERT` per
call) and read back with real SQL — a `GROUP BY` for the
live/offline/tool mode split, an `ORDER BY ... LIMIT` for the most
recent queries — not just written and never queried again. This is
deliberately a plain relational log, not a second vector store or a
cache: "basic SQL and database concepts" is a JD line nothing else
in this project touched.

**Live-model path (`rag_chain.answer_question`, used by `/ask` when
`offline` isn't forced):** implemented against the real
`langchain-anthropic` `ChatAnthropic` client, same "narrate only
what's given" system prompt discipline as `llm_narrative.py`, now
pinned to `claude-sonnet-5` (the previous `claude-sonnet-4-5-*` pin
started emitting a deprecation warning, EOL 2026-11-30, confirmed
live against the real API). Narrowed what "not verified" actually
means here, rather than leaving it vague: a direct call to the real
`api.anthropic.com` with a dummy key returns a clean `401
authentication_error` for `claude-sonnet-5` — proof the model id and
endpoint are both valid, not a wrong-model or connectivity failure —
but no bare `ANTHROPIC_API_KEY` is available in this sandbox, so a
real authenticated call still can't be independently run here. Same
honest status as `generate_brief`'s and `judge_escalation`'s live
paths; `ask()` degrades to the offline extractive path on any
failure, so this never costs a dropped answer — export
`ANTHROPIC_API_KEY` yourself to verify the live path for real.

**Real agentic tool-calling (`rag/agent.py`, updated 2026-11-04):**
`run_tool_calling_agent()` is a real two-turn loop, not just a bound
tool sitting unused — it binds `rag/tools.py`'s real
`check_live_pattern_state` Tool to a model via `bind_tools()`, sends
the question, and if the model's response carries a `tool_call` (the
model *deciding* to call it, not a keyword match), the real tool
actually runs and its real output is fed back as a `ToolMessage` for
a final answer. An actual end-to-end attempt against the real API
was made (`ChatAnthropic(model="claude-sonnet-5").invoke(...)`
through this exact loop) and failed with a connection error — the
same "no bare `ANTHROPIC_API_KEY` available" constraint as every
other live path in this project, confirmed directly, not assumed.
What's independently verified instead:
`tests/test_agent.py::test_agent_calls_the_real_tool_when_the_model_requests_it_and_feeds_back_the_real_result`
drives this exact function with a scripted model double that returns
a real Anthropic-shaped `tool_call` message, and asserts the loop
dispatches to the real tool, the real tool's real output
(`"Live check as of ..."`, not a stub) is what gets fed back, and the
final answer is returned correctly — proving this project's own loop
logic is correct, which is the one part of this a sandbox without a
model to actually call still can test honestly. The deterministic
keyword router (`route_offline`) remains the tested, no-network floor
this agent is meant to improve on in production, not a hidden second
decision-maker.

**Real evaluation (`rag/eval.py`, `GET /eval/retrieval`):** two
automated metrics address the JD's "LLM response evaluation, and
model optimization" line, which nothing else in this project
measured. (1) Retrieval precision@k: five representative real
questions against the real corpus, each with a documented
real-answer check — run fresh against the live vector store at
`/eval/retrieval`, not hardcoded. Honest result as of 2026-11-04:
**0.8** (4/5) at k=4. The one miss is reported, not hidden: the 13
daily-summary documents are near-identically templated, so a
semantic embedder (unlike TF-IDF's literal token match) can't
reliably surface one specific date in the top k — a real, documented
tradeoff of the embeddings upgrade above, not a bug in the harness.
(2) Groundedness: checks what fraction of the numbers an answer
states actually appear in the real documents it was retrieved from —
this project's long-standing "never invent a number" rule
(`llm_narrative.py`, `escalation_judge.py`, `rag_chain.answer_question`'s
system prompt) turned into a metric instead of only ever read by
eye. It needs no live model to compute (no model-graded-by-model
step — no live model is reachable in this sandbox to do that
grading), and `tests/test_eval.py` proves it actually catches an
invented number rather than just restating the rule in a docstring.

**Real bug found and fixed (round-table review):** the first version
of `check_live_pattern_state` approximated the escalation judge's
recurrence input as `{flag_name: 1}` for every flag — below
`judge_escalation_offline`'s own `>=3` recurrence threshold, so
recurrence could never be detected at all. On 2026-06-17, the real
rolling fire-history (the same one `multi_day_runner.py` builds)
shows `symbol_repeat_loser` recurring 5 of the last 5 days and
correctly decides `NOTIFY_NOW`; the broken approximation silently
reported `HOLD_FOR_DIGEST` for the same day and flags instead — a
real wrong answer from the one tool meant to represent current state,
not just a cruder floor. Fixed by replaying the same rolling
fire-history window `multi_day_runner.py` uses, inside the tool
itself, rather than approximating it; see `tests/test_tools.py`'s
regression test pinning the real `NOTIFY_NOW` decision.

**What's actually verified:** `rag/corpus_builder.py` (all 13 real
days + the real usage log + labelled samples + detector reference
docs + the README chunking, asserted against known real findings
like the `2026-06-04` `symbol_repeat_loser`/`overtrading_frequency`
flags), both embedders (the real local semantic model's output shape
and its synonym-retrieval advantage over TF-IDF; TF-IDF's own
fit/embed/relative-similarity and "query before fit raises" guard as
the tested fallback), the ChromaDB-backed offline retrieval path end
to end, the tool router's live re-run of the real detectors, the real
agentic tool-calling loop's dispatch-and-feedback logic (via a
scripted model double), the retrieval precision@k and groundedness
eval metrics (against the real corpus and real hand-checked
examples), the real SQLite query log (insert, the `GROUP BY`
mode-count aggregate, the `ORDER BY ... LIMIT` recency query), and
every FastAPI endpoint (`/health`, `/ask` offline and tool-routed
plus its SQLite logging, `/brief` for a known and an unknown date,
`/pattern-check`, `/query-history`, `/eval/retrieval`) through
`TestClient` against the real startup lifespan. 116 tests total
across the project, all passing, flake8-clean.

Run the API:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
curl -X POST localhost:8000/ask \
    -H 'content-type: application/json' \
    -d '{"question": "what happened on 2026-06-04?", "offline": true}'
curl -X POST localhost:8000/pattern-check -H 'content-type: application/json' -d '{}'
curl localhost:8000/query-history
curl localhost:8000/eval/retrieval
```

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

## Payments-domain generalization (`reconcile.py` / `reconciliation_judge.py`)

Everything above is trading-domain, built on real IQRF_V2 data — but
Razorpay's own business is payments, not trading. This pair of
modules generalizes the exact pattern proved out in
`escalation_judge.py` — deterministic work decides the bulk of it,
a model only judges the genuinely ambiguous leftover — into a
payments reconciliation agent, to show the pattern transfers rather
than only working in the one domain it was first built for.
"Automated reconciliation" is named directly, in those words, in a
real, current Razorpay Central AI team posting (Product Manager II —
AI), alongside agentic payments, fraud detection, and smart payouts.

**What it does:**

1. `reconcile.py` deterministically matches a gateway-settlement
   feed against an internal ledger, by reference ID, amount, and a
   clean-match timing window. No model involved — plain matching
   logic, same "math decides" discipline as `risk_brief.py`.
2. Everything that matches cleanly is done. What's left — a
   reference ID that matches but an amount that doesn't, an amount
   that matches but no reference ID to confirm it, a real settlement
   delay — is the genuinely ambiguous set, and nothing else.
3. `reconciliation_judge.py` looks at *only* that ambiguous set and
   decides, per pair: `LIKELY_SAME_TRANSACTION`, `LIKELY_DIFFERENT`,
   or `NEEDS_HUMAN_REVIEW`, with a stated rationale — never asked to
   recompute an amount or invent a fact it wasn't given.
4. Anything with no candidate at all on the other side is flagged
   directly as a real, standalone discrepancy — never sent to a
   model, since there's nothing to compare it against.

**Scope limitation, stated plainly:** the matcher assumes 1:1
cardinality — one gateway settlement maps to at most one ledger
entry, and vice versa. It does not model partial captures, split
refunds, or a single payout batching several ledger entries into
one settlement (N:1 or 1:N). A production reconciliation system
eventually needs that; this one does not attempt it.

**Honest scope difference, stated plainly:** unlike IQRF_V2's real
trade history, there is no real payment-gateway settlement file or
real internal ledger available to this project. `synthetic_data/reconciliation_demo.json`
is exactly that — synthetic, hand-authored, its own `_provenance`
block says so — 18 scenarios built to hold real reconciliation
patterns (10 clean matches, 6 ambiguous pairs, 2 genuinely unmatched
singles), not drawn from any real system. It is never presented as
real transaction data, the same honesty line this project holds
everywhere else.

**Real finding, running the offline rule against the full demo set**
(`python3 run_reconciliation.py synthetic_data/reconciliation_demo.json --offline`):
of the 6 ambiguous pairs, 3 resolve to `LIKELY_SAME_TRANSACTION`
(rounding-level amount gaps, or a missing reference ID with a close
timestamp), 2 to `NEEDS_HUMAN_REVIEW` (a real amount gap behind a
matching reference ID; a matching amount but too long a gap with no
reference ID to confirm it), and 1 to `LIKELY_DIFFERENT` (no
reference ID and a real amount gap). One result worth calling out
specifically: a pair ~23 hours apart (`GW-0015`/`LG-0015`, a T+1
settlement delay) still resolves to `LIKELY_SAME_TRANSACTION`,
because the offline rule leans on the matching reference ID rather
than the clock — a deliberately realistic case, since next-day
settlement is normal in real payment systems and shouldn't be
treated as suspicious just because time passed.

**Verification status:** the offline rule (`judge_match_offline`) and
the deterministic matcher are fully verified — 15 tests, including
the T+1-settlement case and all three decision labels exercised
directly. The live model call (`judge_match`) is implemented against
the real `anthropic` SDK, same pattern as `escalation_judge.judge_escalation`,
but **not independently run** in this sandbox for the same reason —
no bare `ANTHROPIC_API_KEY` available.

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
  `claude-sonnet-5` — updated from `claude-sonnet-4-5-20250929` after
  that pin started emitting a deprecation warning, EOL 2026-11-30),
  but **not independently run** in the sandbox this was built in,
  which has no bare `ANTHROPIC_API_KEY` available to it. `generate_daily_brief.py`
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
reconcile.py                payments-domain generalization: deterministic transaction matching
reconciliation_judge.py     the model's one judgment call in the payments domain: same / different / needs review
run_reconciliation.py       CLI: runs the reconciliation agent end to end
rag/                       GenAI-JD upgrade: RAG corpus/embeddings/vector store/SQL
                           query log, real agentic tool-calling loop, eval harness
api.py                     FastAPI service wrapping the CLI tools and the RAG layer
sample_data/                two hand-built illustrative sample days
real_data/                  the real 13-day RELIANCE history, extracted read-only from IQRF_V2's own reports
synthetic_data/              hand-authored, clearly-labeled synthetic reconciliation demo data
tests/                     116 tests, all passing
.github/workflows/ci.yml   lint (flake8) + full test suite on every push
```
