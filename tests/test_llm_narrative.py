import json
from pathlib import Path

from models import portfolio_state_from_dict, trade_from_dict
from risk_brief import compute_risk_summary
from llm_narrative import generate_brief_offline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _summary_for(filename):
    data = json.loads((SAMPLE_DIR / filename).read_text())
    portfolio = portfolio_state_from_dict(data["portfolio"])
    trades = [trade_from_dict(t) for t in data["trades_today"]]
    return compute_risk_summary(portfolio, trades)


def test_offline_narrator_mentions_pnl_and_trade_count():
    summary = _summary_for("day_2026-10-28.json")
    brief = generate_brief_offline(summary)
    assert "+659.0" in brief
    assert "3 trades" in brief


def test_offline_narrator_leads_with_drawdown_when_breached():
    summary = _summary_for("day_2026-10-29.json")
    brief = generate_brief_offline(summary)
    assert brief.startswith("Drawdown is at")
    assert "past your 6% limit" in brief


def test_offline_narrator_omits_drawdown_sentence_when_comfortable():
    summary = _summary_for("day_2026-10-28.json")
    brief = generate_brief_offline(summary)
    assert "Drawdown is at" not in brief


def test_offline_narrator_flags_reconciliation_drift():
    from dataclasses import replace

    summary = _summary_for("day_2026-10-28.json")
    drifted = replace(summary, pnl_reconciliation_drift=12.5)
    brief = generate_brief_offline(drifted)
    assert "Data check" in brief
    assert "12.5" in brief
