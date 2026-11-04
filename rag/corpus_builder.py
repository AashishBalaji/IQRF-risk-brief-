"""
corpus_builder.py -- builds the RAG layer's document corpus out of this
project's own real, already-computed output. Nothing here is written
*for* the RAG layer: every document is either

  (a) a genuine re-run of the exact same real pipeline
      multi_day_runner.py already runs (same walk-forward,
      no-lookahead discipline -- a day's document only ever reflects
      what pattern_detector.py and escalation_judge.py would have
      known by that day), over the real 13-day RELIANCE history, or
  (b) a real, already-written project artifact (real_usage_log.md,
      sample_data/, the modules' own docstrings), or
  (c) explicitly labelled as illustrative/synthetic when the
      underlying data is (sample_data/, synthetic_data/) -- never
      presented as real without that label, same discipline as
      README.md.

Keeping this separate from multi_day_runner.py is deliberate: the
walk-forward loop needs to run once here to produce RAG documents,
and the CLI needs to run it once to produce a human-readable
transcript -- two different callers of the same real detectors, not
two different sources of truth.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from escalation_judge import judge_escalation_offline
from llm_narrative import generate_brief_offline
from pattern_detector import run_pattern_detectors
from real_history import cross_instrument_loss_fraction_by_day, load_real_history
from risk_brief import compute_risk_summary

REPO_ROOT = Path(__file__).resolve().parent.parent
ESCALATION_WINDOW_DAYS = 5


@dataclass(frozen=True)
class RagDocument:
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


def _daily_history_documents() -> List[RagDocument]:
    """Re-runs the exact walk-forward loop multi_day_runner.py runs
    (offline narrator + offline escalation rule -- no network, no API
    key, so this always produces the same real documents regardless
    of whether ANTHROPIC_API_KEY is set), and turns each real day's
    already-computed output into one retrievable document."""
    real_days = load_real_history()
    cross_fraction_by_day = cross_instrument_loss_fraction_by_day(real_days)

    daily_trades_seen: List[list] = []
    fire_history: deque = deque()
    docs: List[RagDocument] = []

    for day in real_days:
        daily_trades_seen.append(day.trades)

        summary = compute_risk_summary(day.portfolio, day.trades)
        flags = run_pattern_detectors(
            daily_trades_seen,
            cross_instrument_loss_fraction_by_day=cross_fraction_by_day,
            latest_date=day.date,
        )

        fire_history.append({f.name for f in flags})
        while len(fire_history) > ESCALATION_WINDOW_DAYS:
            fire_history.popleft()
        recent_fire_counts = {
            f.name: sum(1 for day_names in fire_history if f.name in day_names)
            for f in flags
        }

        escalation = judge_escalation_offline(flags, recent_fire_counts, ESCALATION_WINDOW_DAYS)
        brief = generate_brief_offline(summary)

        flag_lines = "\n".join(f"- [{f.severity}] {f.name}: {f.message}" for f in flags) or "none"
        text = (
            f"Real trading day {day.date} (IQRF_V2's own RELIANCE backtest history).\n"
            f"Risk brief: {brief}\n"
            f"Pattern flags fired this day:\n{flag_lines}\n"
            f"Escalation decision: {escalation.decision} -- {escalation.rationale}\n"
            f"Raw numbers: day_realized_pnl={summary.day_realized_pnl}, "
            f"trade_count={summary.trade_count}, stop_outs={summary.stop_outs}, "
            f"target_hits={summary.target_hits}, signal_exits={summary.signal_exits}, "
            f"current_drawdown_pct={summary.current_drawdown_pct * 100:.2f}%, "
            f"pnl_reconciliation_drift={summary.pnl_reconciliation_drift}."
        )
        docs.append(RagDocument(
            text=text,
            metadata={"source": "real_history", "date": day.date, "kind": "daily_summary"},
        ))

    return docs


def _real_usage_log_documents() -> List[RagDocument]:
    """Splits real_usage_log.md's own real entries (each one a genuine
    run against live market data) into one document per entry, by its
    '- **<timestamp>**' bullet marker."""
    path = REPO_ROOT / "real_usage_log.md"
    if not path.exists():
        return []
    text = path.read_text()
    entries = text.split("\n- **")[1:]
    docs = []
    for entry in entries:
        body = "- **" + entry.strip()
        docs.append(RagDocument(
            text=body,
            metadata={"source": "real_usage_log.md", "kind": "real_usage_entry"},
        ))
    return docs


def _sample_data_documents() -> List[RagDocument]:
    """sample_data/ is hand-built, illustrative, and says so in its own
    '_note' field -- every document built from it carries that label
    through to the metadata and the text itself, so a RAG answer can
    never cite it as real without saying it isn't."""
    docs = []
    sample_dir = REPO_ROOT / "sample_data"
    if not sample_dir.exists():
        return docs
    for path in sorted(sample_dir.glob("*.json")):
        data = json.loads(path.read_text())
        note = data.get("_note", "illustrative sample data, not a real trading day")
        docs.append(RagDocument(
            text=(
                f"ILLUSTRATIVE SAMPLE (not real trading data) from {path.name}: {note} "
                f"session_realized_pnl={data['portfolio']['session_realized_pnl']}, "
                f"open_positions={len(data['portfolio']['open_positions'])}, "
                f"trades_today={len(data['trades_today'])}."
            ),
            metadata={"source": path.name, "kind": "illustrative_sample"},
        ))
    return docs


def _detector_reference_documents() -> List[RagDocument]:
    """One reference document per pattern detector / escalation rule,
    built from that function's own real docstring -- this is the
    project's own documentation, not content invented for the RAG
    layer, so questions like 'what does exit_reason_skew catch?' are
    answered from the same text a developer reading the source would
    see."""
    import escalation_judge
    import pattern_detector

    docs = []
    for name in (
        "detect_stop_out_clustering",
        "detect_exit_reason_skew",
        "detect_symbol_repeat_loser",
        "detect_overtrading_frequency",
        "detect_position_size_drift_after_loss",
        "detect_cross_instrument_stress",
    ):
        fn = getattr(pattern_detector, name)
        doc = (fn.__doc__ or "").strip()
        docs.append(RagDocument(
            text=f"Reference -- pattern detector `{name}`:\n{doc}",
            metadata={"source": "pattern_detector.py", "kind": "reference", "name": name},
        ))

    for name in ("judge_escalation_offline",):
        fn = getattr(escalation_judge, name)
        doc = (fn.__doc__ or "").strip()
        docs.append(RagDocument(
            text=f"Reference -- escalation rule `{name}`:\n{doc}",
            metadata={"source": "escalation_judge.py", "kind": "reference", "name": name},
        ))

    return docs


# README.md is this project's own real, long-form documentation --
# 500+ lines mixing prose and code blocks, nothing fabricated for
# this purpose. 800 chars / 120 overlap was chosen, not defaulted,
# for two concrete reasons specific to this document: (a) README's
# sections run from a couple of sentences to several paragraphs, so
# 800 chars keeps most sections in one chunk while still splitting
# the handful of long ones instead of returning all-or-nothing; (b)
# the overlap is sized to usually keep a short code block's own
# surrounding sentence attached to it even when a chunk boundary
# lands close by, which a 0-overlap split would frequently sever.
# Splitting on markdown headers first, then paragraphs, then
# sentences (the separator list below, in that order) means a chunk
# only breaks mid-paragraph as a last resort.
_README_CHUNK_SIZE = 800
_README_CHUNK_OVERLAP = 120


def _readme_documents() -> List[RagDocument]:
    """Splits this project's own real README.md into retrievable
    chunks via a real chunk-size/overlap strategy (RecursiveCharacter
    TextSplitter), rather than one giant document or one document per
    arbitrary line count. This is the corpus's only long real
    document, which is exactly why it is the one that needed a real
    chunking decision -- the daily-history and usage-log documents
    above are already short enough that splitting them would only
    fragment an already-atomic fact."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    path = REPO_ROOT / "README.md"
    if not path.exists():
        return []
    text = path.read_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_README_CHUNK_SIZE,
        chunk_overlap=_README_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        RagDocument(
            text=chunk,
            metadata={"source": "README.md", "kind": "readme_chunk", "chunk_index": str(i)},
        )
        for i, chunk in enumerate(chunks)
    ]


def build_corpus() -> List[RagDocument]:
    """Assembles the full corpus: real multi-day history (walk-forward,
    re-derived, never fabricated), the real usage log, labelled
    illustrative samples, reference documentation, and chunked README
    content. Called once at vector-store build time."""
    docs: List[RagDocument] = []
    docs.extend(_daily_history_documents())
    docs.extend(_real_usage_log_documents())
    docs.extend(_sample_data_documents())
    docs.extend(_detector_reference_documents())
    docs.extend(_readme_documents())
    return docs
