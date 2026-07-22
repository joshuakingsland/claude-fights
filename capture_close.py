"""Capture one standardized pre-fight H2H snapshot near T-30 minutes."""

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import ODDS_CONSENSUS_VERSION
from fetch_odds import _write_atomic, consensus_quote


SPORT = "mma_mixed_martial_arts"
BASE = "https://api.the-odds-api.com/v4"
FIELDS = [
    "captured_at", "event_id", "commence_time", "lead_minutes",
    "fighter_a", "fighter_b", "odds_a", "odds_b", "market_prob_a",
    "market_books", "snapshot_kind", "odds_source",
]


def _utc(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return stamp.astimezone(timezone.utc)


def _fetch(path, key, **params):
    query = urllib.parse.urlencode({"apiKey": key, **params})
    request = urllib.request.Request(
        f"{BASE}{path}?{query}", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def due_events(events, captured_ids, now, minimum_minutes=10, maximum_minutes=50):
    now = _utc(now)
    due = []
    for event in events:
        event_id = str(event.get("id", ""))
        try:
            commence = _utc(event.get("commence_time"))
        except (TypeError, ValueError):
            continue
        lead = (commence - now).total_seconds() / 60.0
        if (event_id and event_id not in captured_ids
                and minimum_minutes <= lead <= maximum_minutes):
            due.append((event_id, lead))
    return due


def _existing(path):
    path = Path(path)
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return rows, {row.get("event_id", "") for row in rows}


def run(key, output="close_snapshots.csv", now=None, minimum_minutes=10,
        maximum_minutes=50, dry_run=False, fetcher=_fetch):
    if not key:
        raise SystemExit("ODDS_API_KEY is required for close capture.")
    now = _utc(now or datetime.now(timezone.utc))
    rows, captured = _existing(output)
    events = fetcher(f"/sports/{SPORT}/events", key)
    due = due_events(events, captured, now, minimum_minutes, maximum_minutes)
    print(f"close capture: {len(events)} upcoming events, {len(due)} due")
    if dry_run or not due:
        return {"events": len(events), "due": len(due), "paid_requests": 0}

    odds = fetcher(
        f"/sports/{SPORT}/odds", key, regions="us", markets="h2h",
        oddsFormat="american",
    )
    by_id = {str(event.get("id", "")): event for event in odds}
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    added = 0
    for event_id, lead in due:
        event = by_id.get(event_id)
        quote = consensus_quote(event) if event else None
        if quote is None:
            continue
        rows.append({
            "captured_at": stamp,
            "event_id": event_id,
            "commence_time": event.get("commence_time", ""),
            "lead_minutes": round(lead, 2),
            "fighter_a": event.get("home_team", ""),
            "fighter_b": event.get("away_team", ""),
            **quote,
            "snapshot_kind": "standardized_t30_window",
            "odds_source": f"the-odds-api-{ODDS_CONSENSUS_VERSION}",
        })
        added += 1
    if added:
        _write_atomic(output, FIELDS, rows)
    print(f"close capture: added {added} snapshot(s) with one paid H2H request")
    return {"events": len(events), "due": len(due), "added": added,
            "paid_requests": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="close_snapshots.csv")
    parser.add_argument("--min-minutes", type=float, default=10.0)
    parser.add_argument("--max-minutes", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(
        os.environ.get("ODDS_API_KEY"), args.output,
        minimum_minutes=args.min_minutes, maximum_minutes=args.max_minutes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
