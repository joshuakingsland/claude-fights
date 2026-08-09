"""Capped discovery of event-specific MMA market keys.

This used to record only keys matching a list of prop-sounding terms, and
report an empty list as "no props offered". The filter could not see the two
markets most worth knowing about. `totals` - the over/under on rounds, and a
standard market rather than a prop - contains none of the terms. Neither does
a distance market such as `fight_to_go_distance`; the previous test fixture
included that exact key and asserted it was discarded.

So an empty catalogue meant "no key matched seven guessed substrings", not
"no market exists", and the difference matters because the whole point of the
catalogue is to answer whether a market can be priced at all.

Every key is recorded now, with the number of books offering it. The book
count is the part that decides anything: the moneyline already loses most
fights to the three-paired-book minimum, and a market quoted by one book is
not tradable no matter how attractive the model finds it.

The first run of the fixed version, over 12 events in the `us` region on
2026-08-09, found exactly one key: `h2h`, on up to 7 books. So the blind spot
was real but was not concealing anything - no US book exposes a method,
distance, or totals market for MMA through this API. Region is a parameter
now rather than a hardcoded "us", because that is the remaining thing worth
testing before concluding the markets are unreachable rather than just
unreachable from one region.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SPORT = "mma_mixed_martial_arts"
BASE = "https://api.the-odds-api.com/v4"

# Kept only to label the catalogue for a human reader. Nothing is dropped on
# the strength of it.
PROP_TERMS = ("method", "victory", "finish", "round", "decision", "submission",
              "ko", "distance", "total")


def _fetch(path, key, **params):
    query = urllib.parse.urlencode({"apiKey": key, **params})
    request = urllib.request.Request(
        f"{BASE}{path}?{query}", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def market_book_counts(payload):
    """Every market key offered for an event, and how many books offer it."""
    counts = Counter()
    for bookmaker in payload.get("bookmakers", []):
        for key in {str(m.get("key", "")) for m in bookmaker.get("markets", [])}:
            if key:
                counts[key] += 1
    return dict(sorted(counts.items()))


def prop_keys(payload):
    """Non-moneyline keys. A label over market_book_counts, not a filter."""
    return sorted(k for k in market_book_counts(payload) if k != "h2h")


def run(key, max_requests=0, output="prop_market_catalog.json", fetcher=_fetch,
        regions="us"):
    if not key:
        raise SystemExit("ODDS_API_KEY is required for prop discovery.")
    events = fetcher(f"/sports/{SPORT}/events", key)
    cap = max(0, min(int(max_requests), len(events)))
    discoveries = []
    offered = Counter()
    depth = Counter()
    for event in events[:cap]:
        payload = fetcher(
            f"/sports/{SPORT}/events/{event['id']}/markets", key, regions=regions
        )
        counts = market_book_counts(payload)
        for market, books in counts.items():
            offered[market] += 1
            depth[market] = max(depth[market], books)
        discoveries.append({
            "event_id": event["id"],
            "commence_time": event.get("commence_time"),
            "fighter_a": event.get("home_team"),
            "fighter_b": event.get("away_team"),
            "market_book_counts": counts,
            "prop_market_keys": prop_keys(payload),
        })
    summary = [
        {
            "market": market,
            "events_offering": offered[market],
            "events_share": round(offered[market] / cap, 3) if cap else 0.0,
            "max_books_on_one_event": depth[market],
        }
        for market in sorted(offered, key=lambda m: (-offered[m], m))
    ]
    report = {
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "available_events": len(events),
        "discovery_requests": cap,
        "request_cap": int(max_requests),
        "regions": regions,
        "market_summary": summary,
        "events": discoveries,
        "note": ("Market-key discovery only; no prices are fetched. Every key "
                 "offered is recorded, not a term-matched subset."),
    }
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--output", default="prop_market_catalog.json")
    parser.add_argument("--regions", default="us",
                        help="comma-separated Odds API regions; each region "
                             "billed separately per request")
    args = parser.parse_args()
    run(os.environ.get("ODDS_API_KEY"), args.max_requests, args.output,
        regions=args.regions)


if __name__ == "__main__":
    main()
