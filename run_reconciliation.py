"""
run_reconciliation.py -- CLI: runs the full reconciliation agent
(deterministic matcher + judgment on the ambiguous leftover) against
a real reconciliation dataset.

    python3 run_reconciliation.py synthetic_data/reconciliation_demo.json
    python3 run_reconciliation.py synthetic_data/reconciliation_demo.json --offline

Prints three sections: clean matches (count only -- nothing to say
about these, which is the point), the judged ambiguous pairs (each
with the matcher's reason for flagging it and the judge's decision),
and the genuinely unmatched singles (flagged directly for human
investigation, never sent to a model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reconcile import load_gateway_settlements, load_ledger_entries, reconcile
from reconciliation_judge import judge_match, judge_match_offline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="run_reconciliation")
    parser.add_argument("data_file", help="JSON file with gateway_settlements + internal_ledger")
    parser.add_argument(
        "--offline", action="store_true",
        help="use the deterministic offline rule instead of calling the Anthropic API",
    )
    args = parser.parse_args(argv)

    data = json.loads(Path(args.data_file).read_text())
    gateway = load_gateway_settlements(data["gateway_settlements"])
    ledger = load_ledger_entries(data["internal_ledger"])

    result = reconcile(gateway, ledger)

    print(f"Clean matches: {len(result.matched)} (matched by amount + reference ID + timing, "
          "no model involved)\n")

    print(f"Ambiguous pairs needing judgment: {len(result.ambiguous)}")
    for pair in result.ambiguous:
        if args.offline:
            decision = judge_match_offline(pair)
        else:
            try:
                decision = judge_match(pair)
            except Exception as e:  # noqa: BLE001 -- degrade, never drop a real pair unjudged
                print(f"note: live Anthropic API call failed ({e}); falling back to offline rule",
                      file=sys.stderr)
                decision = judge_match_offline(pair)
        print(
            f"  {pair.gateway.gateway_txn_id} <-> {pair.ledger.ledger_entry_id} "
            f"(amounts {pair.gateway.amount} vs {pair.ledger.amount}, reason={pair.reason}): "
            f"{decision.decision} -- {decision.rationale}"
        )

    print("\nGenuinely unmatched, no candidate found (flagged directly, never sent to a model):")
    for g in result.unmatched_gateway:
        print(f"  gateway-only: {g.gateway_txn_id} amount={g.amount} ref={g.reference_id}")
    for led in result.unmatched_ledger:
        print(f"  ledger-only: {led.ledger_entry_id} amount={led.amount} ref={led.reference_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
