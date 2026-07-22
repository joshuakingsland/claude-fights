"""Fetch upcoming UFC fights and auditable market consensus prices.

Automatic mode requires ``ODDS_API_KEY``. The GitHub workflow also passes
``--require-key`` so a missing secret cannot silently reuse a stale card.
Manual CSVs remain supported; the two consensus columns are optional there.
"""

import argparse
import csv
import json
import os
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import ODDS_CONSENSUS_VERSION


API = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
UPCOMING_FIELDS = [
    "date", "commence_time", "fighter_a", "fighter_b", "odds_a", "odds_b",
    "market_prob_a", "market_books", "weightclass", "five_rounds",
    "odds_source", "fetched_at",
]
LOG_FIELDS = [
    "fetched_at", "commence_time", "date", "fighter_a", "fighter_b",
    "odds_a", "odds_b", "market_prob_a", "market_books", "odds_source",
]


def _american_to_prob(odds):
    odds = float(odds)
    return -odds / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _upper_median(values):
    values = sorted(values)
    return values[len(values) // 2]


def consensus_quote(event):
    """Return paired median prices and median per-book de-vig probability."""
    fighter_a = event.get("home_team", "")
    fighter_b = event.get("away_team", "")
    paired = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            prices = {
                outcome.get("name"): outcome.get("price")
                for outcome in market.get("outcomes", [])
            }
            odds_a = prices.get(fighter_a)
            odds_b = prices.get(fighter_b)
            try:
                odds_a = float(odds_a)
                odds_b = float(odds_b)
            except (TypeError, ValueError):
                continue
            if abs(odds_a) < 100 or abs(odds_b) < 100:
                continue
            pa = _american_to_prob(odds_a)
            pb = _american_to_prob(odds_b)
            paired.append((odds_a, odds_b, pa / (pa + pb)))
            break
    if not paired:
        return None
    return {
        "odds_a": _upper_median([row[0] for row in paired]),
        "odds_b": _upper_median([row[1] for row in paired]),
        "market_prob_a": round(float(statistics.median(row[2] for row in paired)), 8),
        "market_books": len(paired),
    }


def _write_atomic(path, fields, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_log(path, rows):
    """Append via an atomic rewrite, migrating older log schemas in place."""
    path = Path(path)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as source:
            existing.extend(csv.DictReader(source))
    normalized = [{field: row.get(field, "") for field in LOG_FIELDS}
                  for row in existing]
    normalized.extend({field: row.get(field, "") for field in LOG_FIELDS}
                      for row in rows)
    _write_atomic(path, LOG_FIELDS, normalized)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-key", action="store_true",
        help="fail instead of entering manual mode when ODDS_API_KEY is absent",
    )
    args = parser.parse_args(argv)

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        if args.require_key:
            raise SystemExit(
                "ODDS_API_KEY is required. Add it as a GitHub Actions secret."
            )
        print("No ODDS_API_KEY set; using manual mode.")
        print("Edit odds_upcoming.csv by hand with current book prices.")
        if not Path("odds_upcoming.csv").exists():
            _write_atomic("odds_upcoming.csv", UPCOMING_FIELDS, [])
            print("Empty odds_upcoming.csv template written.")
        return

    query = urllib.parse.urlencode({
        "apiKey": key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    })
    request = urllib.request.Request(
        f"{API}?{query}", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        events = json.load(response)

    stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    rows = []
    for event in events:
        quote = consensus_quote(event)
        if quote is None:
            continue
        commence = event.get("commence_time", "")
        rows.append({
            "date": commence[:10],
            "commence_time": commence,
            "fighter_a": event.get("home_team", ""),
            "fighter_b": event.get("away_team", ""),
            **quote,
            "weightclass": "",
            "five_rounds": "0",
            "odds_source": f"the-odds-api-{ODDS_CONSENSUS_VERSION}",
            "fetched_at": stamp,
        })

    _write_atomic("odds_upcoming.csv", UPCOMING_FIELDS, rows)
    log_rows = [{field: row.get(field, "") for field in LOG_FIELDS} for row in rows]
    append_log("odds_log.csv", log_rows)
    print(
        f"wrote odds_upcoming.csv ({len(rows)} fights) and appended "
        f"a paired-book consensus snapshot at {stamp}"
    )


if __name__ == "__main__":
    main()
