"""Unit tests for the agentic tool-calling layer (rag/tools.py) --
the live pattern-check tool and the deterministic keyword router."""

from __future__ import annotations

from rag.tools import build_tools, check_live_pattern_state, route_offline


def test_check_live_pattern_state_runs_real_detectors_over_real_history():
    result = check_live_pattern_state()
    assert "Live check as of 2026-06-17" in result
    assert "Escalation:" in result


def test_build_tools_returns_one_named_tool():
    tools = build_tools()
    assert len(tools) == 1
    assert tools[0].name == "check_live_pattern_state"
    assert tools[0].func is check_live_pattern_state


def test_route_offline_fires_on_live_state_keywords():
    hit = route_offline("Is anything flagged right now?")
    assert hit is not None
    assert hit.tool_used == "check_live_pattern_state"
    assert "Live check as of" in hit.answer


def test_route_offline_returns_none_for_historical_questions():
    assert route_offline("What happened on 2026-06-04?") is None
    assert route_offline("What is RELIANCE's biggest loser trade in the sample data?") is None
