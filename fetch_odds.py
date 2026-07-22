"""Fetch upcoming UFC fights and auditable market consensus prices.

Automatic mode requires ``ODDS_API_KEY``. The GitHub workflow also passes
``--require-key`` so a missing secret cannot silently reuse a stale card.
Manual CSVs remain supported; consensus and execution columns are optional.
"""

import argparse
import csv
import hashlib
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
    "market_prob_a", "market_books", "market_spread", "best_odds_a",
    "best_book_a", "best_odds_b", "best_book_b", "weightclass",
    "five_rounds", "odds_source", "fetched_at",
]
LOG_FIELDS = [
    "fetched_at", "commence_time", "date", "fighter_a", "fighter_b",
    "odds_a", "odds_b", "market_prob_a", "market_books", "market_spread",
    "best_odds_a", "best_book_a", "best_odds_b", "best_book_b",
    "odds_source",
]
MARKET_QUOTE_FIELDS = [
    "snapshot_id", "fetched_at", "event_id", "commence_time", "date",
    "fighter_a", "fighter_b", "book_key", "book_title", "book_updated_at",
    "odds_a", "odds_b", "devig_prob_a",
]


def _american_to_prob(odds):
    odds = float(odds)
    return -odds / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _upper_median(values):
    values = sorted(values)
    return values[len(values) // 2]


def paired_book_quotes(event):
    """Return valid, paired H2H quotes with sportsbook provenance."""
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
            paired.append({
                "book_key": book.get("key", ""),
                "book_title": book.get("title", book.get("key", "")),
                "book_updated_at": book.get("last_update", ""),
                "odds_a": odds_a,
                "odds_b": odds_b,
                "devig_prob_a": pa / (pa + pb),
            })
            break
    return paired


def consensus_quote(event, paired=None):
    """Return consensus inputs plus the best executable price on each side."""
    paired = paired_book_quotes(event) if paired is None else paired
    if not paired:
        return None
    best_a = max(paired, key=lambda row: row["odds_a"])
    best_b = max(paired, key=lambda row: row["odds_b"])
    probabilities = [row["devig_prob_a"] for row in paired]
    return {
        "odds_a": _upper_median([row["odds_a"] for row in paired]),
        "odds_b": _upper_median([row["odds_b"] for row in paired]),
        "market_prob_a": round(float(statistics.median(probabilities)), 8),
        "market_books": len(paired),
        "market_spread": round(float(max(probabilities) - min(probabilities)), 8),
        "best_odds_a": best_a["odds_a"],
        "best_book_a": best_a["book_title"],
        "best_odds_b": best_b["odds_b"],
        "best_book_b": best_b["book_title"],
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


def append_quote_log(path, rows):
    """Append full book quotes and de-duplicate deterministic snapshots."""
    path = Path(path)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as source:
            existing.extend(csv.DictReader(source))
    normalized = [
        {field: row.get(field, "") for field in MARKET_QUOTE_FIELDS}
        for row in existing
    ]
    known = {row["snapshot_id"] for row in normalized}
    for row in rows:
        item = {field: row.get(field, "") for field in MARKET_QUOTE_FIELDS}
        if item["snapshot_id"] not in known:
            normalized.append(item)
            known.add(item["snapshot_id"])
    _write_atomic(path, MARKET_QUOTE_FIELDS, normalized)


def _quote_rows(event, paired, stamp):
    rows = []
    commence = event.get("commence_time", "")
    for quote in paired:
        raw = "|".join(str(value) for value in (
            stamp, event.get("id", ""), quote["book_key"], quote["odds_a"],
            quote["odds_b"], quote["book_updated_at"],
        ))
        rows.append({
            "snapshot_id": hashlib.sha256(raw.encode()).hexdigest()[:20],
            "fetched_at": stamp,
            "event_id": event.get("id", ""),
            "commence_time": commence,
            "date": commence[:10],
            "fighter_a": event.get("home_team", ""),
            "fighter_b": event.get("away_team", ""),
            **quote,
        })
    return rows


def _write_snapshot_manifest(path, stamp, rows, quote_rows, quote_path):
    payload = {
        "fetched_at": stamp,
        "events": len(rows),
        "paired_book_quotes": len(quote_rows),
        "quote_file": str(quote_path).replace("\\", "/"),
        "first_event_date": min((row["date"] for row in rows), default=None),
        "last_event_date": max((row["date"] for row in rows), default=None),
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
    os.replace(temporary, path)


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
    all_quotes = []
    for event in events:
        paired = paired_book_quotes(event)
        quote = consensus_quote(event, paired)
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
        all_quotes.extend(_quote_rows(event, paired, stamp))

    _write_atomic("odds_upcoming.csv", UPCOMING_FIELDS, rows)
    log_rows = [{field: row.get(field, "") for field in LOG_FIELDS} for row in rows]
    append_log("odds_log.csv", log_rows)
    quote_path = Path("data/market_quotes") / f"quotes_{stamp[:7]}.csv"
    quote_path.parent.mkdir(parents=True, exist_ok=True)
    append_quote_log(quote_path, all_quotes)
    _write_snapshot_manifest(
        "market_snapshot_manifest.json", stamp, rows, all_quotes, quote_path
    )
    print(
        f"wrote odds_upcoming.csv ({len(rows)} fights) and appended "
        f"{len(all_quotes)} paired book quotes at {stamp}"
    )


if __name__ == "__main__":
    main()
