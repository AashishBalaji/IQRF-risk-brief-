"""Integration tests for the FastAPI service (api.py) -- black-box,
through FastAPI's TestClient, exercising the real startup lifespan
(real vector-store build) and the real endpoints end to end. No
network, no API key needed: every endpoint here either has no
live/offline toggle (/brief, /pattern-check) or is explicitly called
with offline=True (/ask)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_offline_returns_a_real_retrieved_record(client):
    response = client.post(
        "/ask", json={"question": "What happened on 2026-06-04?", "offline": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "offline"
    assert "2026-06-04" in body["answer"]
    assert body["sources"]


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_rejects_oversized_question_at_the_pydantic_layer(client):
    """Regression test for a round-table review finding: question had
    no max_length, so an arbitrarily large string could be pushed
    straight into the TF-IDF vectorizer / LLM call. FastAPI/Pydantic
    should reject it before it ever reaches rag_chain.ask()."""
    response = client.post("/ask", json={"question": "x" * 2001})
    assert response.status_code == 422


def test_ask_rejects_k_outside_allowed_range(client):
    response = client.post("/ask", json={"question": "anything", "k": 0, "offline": True})
    assert response.status_code == 422
    response = client.post("/ask", json={"question": "anything", "k": 21, "offline": True})
    assert response.status_code == 422


def test_ask_routes_live_state_question_to_the_tool(client):
    response = client.post("/ask", json={"question": "Is anything flagged right now?"})
    assert response.status_code == 200
    assert response.json()["mode"] == "tool"


def test_brief_for_a_known_real_date(client):
    response = client.post("/brief", json={"date": "2026-06-04"})
    assert response.status_code == 200
    body = response.json()
    assert body["trade_count"] == 13
    assert body["day_realized_pnl"] == -5860.78


def test_brief_for_unknown_date_returns_404_with_available_dates(client):
    response = client.post("/brief", json={"date": "2099-01-01"})
    assert response.status_code == 404
    assert "2026-06-04" in response.json()["detail"]


def test_pattern_check_runs_the_real_tool(client):
    response = client.post("/pattern-check", json={})
    assert response.status_code == 200
    assert "Live check as of" in response.json()["result"]
