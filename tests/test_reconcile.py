"""
Tests for reconcile.py's deterministic matcher, against the real
synthetic_data/reconciliation_demo.json fixture (18 hand-authored
scenarios: 10 clean matches, 6 ambiguous pairs, 2 genuinely unmatched
singles -- see that file's own _provenance block for what each
scenario is testing).
"""

import json
from pathlib import Path

from reconcile import load_gateway_settlements, load_ledger_entries, reconcile

FIXTURE = Path(__file__).parent.parent / "synthetic_data" / "reconciliation_demo.json"


def _load():
    data = json.loads(FIXTURE.read_text())
    return (
        load_gateway_settlements(data["gateway_settlements"]),
        load_ledger_entries(data["internal_ledger"]),
    )


def test_fixture_counts():
    gateway, ledger = _load()
    assert len(gateway) == 17
    assert len(ledger) == 17


def test_ten_clean_matches():
    gateway, ledger = _load()
    result = reconcile(gateway, ledger)
    assert len(result.matched) == 10
    matched_refs = {g.reference_id for g, _ in result.matched}
    assert matched_refs == {f"REF100{i}" for i in range(1, 10)} | {"REF1010"}


def test_six_ambiguous_pairs():
    gateway, ledger = _load()
    result = reconcile(gateway, ledger)
    assert len(result.ambiguous) == 6
    ambiguous_gateway_ids = {p.gateway.gateway_txn_id for p in result.ambiguous}
    assert ambiguous_gateway_ids == {
        "GW-0011", "GW-0012", "GW-0013", "GW-0014", "GW-0015", "GW-0016",
    }


def test_two_genuinely_unmatched_singles():
    gateway, ledger = _load()
    result = reconcile(gateway, ledger)
    assert [g.gateway_txn_id for g in result.unmatched_gateway] == ["GW-0017"]
    assert [led.ledger_entry_id for led in result.unmatched_ledger] == ["LG-0018"]


def test_rounding_pair_flagged_ref_match_reason():
    gateway, ledger = _load()
    result = reconcile(gateway, ledger)
    pair = next(p for p in result.ambiguous if p.gateway.gateway_txn_id == "GW-0011")
    assert pair.reason == "ref_match_amount_or_time_off"
    assert abs(pair.gateway.amount - pair.ledger.amount) < 1.0


def test_missing_reference_pair_flagged_correctly():
    gateway, ledger = _load()
    result = reconcile(gateway, ledger)
    pair = next(p for p in result.ambiguous if p.gateway.gateway_txn_id == "GW-0013")
    assert pair.reason == "missing_ref_amount_time_close"
    assert pair.ledger.reference_id is None


def test_t_plus_one_settlement_still_a_candidate_via_matching_reference():
    # ~23h apart, but the reference IDs match -- should still surface
    # as an ambiguous candidate rather than being thrown away as
    # "too far apart to even consider."
    gateway, ledger = _load()
    result = reconcile(gateway, ledger)
    pair = next(p for p in result.ambiguous if p.gateway.gateway_txn_id == "GW-0015")
    assert pair.ledger.ledger_entry_id == "LG-0015"


def test_clean_match_requires_both_amount_and_reference():
    from reconcile import GatewayTxn, LedgerEntry, reconcile as _reconcile

    g = GatewayTxn("GW-X", 100.0, "2026-10-30T10:00:00+05:30", "REF-X")
    l_wrong_amount = LedgerEntry("LG-X", 150.0, "2026-10-30T10:01:00+05:30", "REF-X")
    result = _reconcile([g], [l_wrong_amount])
    assert result.matched == []
    assert len(result.ambiguous) == 1
