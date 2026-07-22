"""Capped discovery of event-specific MMA prop market keys."""

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SPORT = "mma_mixed_martial_arts"
BASE = "https://api.the-odds-api.com/v4"
PROP_TERMS = ("method", "victory", "finish", "round", "decision", "submission", "ko")


def _fetch(path, key, **params):
    query = urllib.parse.urlencode({"apiKey": key, **params})
    request = urllib.request.Request(
        f"{BASE}{path}?{query}", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def prop_keys(payload):
    found = set()
    for bookmaker in payload.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            key = str(market.get("key", ""))
            if any(term in key.lower() for term in PROP_TERMS):
                found.add(key)
    return sorted(found)


def run(key, max_requests=0, output="prop_market_catalog.json", fetcher=_fetch):
    if not key:
        raise SystemExit("ODDS_API_KEY is required for prop discovery.")
    events = fetcher(f"/sports/{SPORT}/events", key)
    cap = max(0, min(int(max_requests), len(events)))
    discoveries = []
    for event in events[:cap]:
        payload = fetcher(
            f"/sports/{SPORT}/events/{event['id']}/markets", key, regions="us"
        )
        discoveries.append({
            "event_id": event["id"],
            "commence_time": event.get("commence_time"),
            "fighter_a": event.get("home_team"),
            "fighter_b": event.get("away_team"),
            "prop_market_keys": prop_keys(payload),
        })
    report = {
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "available_events": len(events),
        "discovery_requests": cap,
        "request_cap": int(max_requests),
        "events": discoveries,
        "note": "Market-key discovery only; no prop prices are fetched.",
    }
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--output", default="prop_market_catalog.json")
    args = parser.parse_args()
    run(os.environ.get("ODDS_API_KEY"), args.max_requests, args.output)


if __name__ == "__main__":
    main()
