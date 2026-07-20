"""Summarize forward-tested official paper wagers and closing-line value.

Usage: python validate_paper.py

Unlike ``validate_production.py``, this file never reconstructs historical
bets.  It evaluates only timestamped official wagers locked before the event.
"""

import argparse
import json
from pathlib import Path

from paper_ledger import summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="paper_trades.csv")
    ap.add_argument("--settlements", default="paper_settlements.csv")
    ap.add_argument("--report", default="paper_validation.json")
    args = ap.parse_args()
    report = summary(args.trades, args.settlements)
    report["status"] = "paper_only"
    report["interpretation"] = (
        "Forward-test evidence only. Positive closing-line value should be "
        "established before ROI is treated as meaningful."
    )
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
