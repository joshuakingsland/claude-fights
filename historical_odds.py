"""Backfill auditable pre-event MMA moneyline snapshots from The Odds API.

The importer deliberately does *not* alter the production training table.
It writes immutable, per-book input quotes first. ``prepare_api_odds_history.py``
then creates a point-in-time entry/close-proxy dataset for a separate audit.

The API returns the closest snapshot at or before the requested time.  We
therefore discover each UFC card's API commence time from a pre-card snapshot,
then collect an entry snapshot (default: 24 hours before the first bout) and a
close proxy (default: 15 minutes before the first bout). The actual returned
snapshot can be earlier and is recorded exactly. It is never a model input.

Usage (the key belongs in the environment, never a command line or CSV):
    python historical_odds.py --dry-run
    python historical_odds.py --max-requests 30
    python historical_odds.py --max-requests 100
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import timezone
from pathlib import Path

import pandas as pd

from identity import norm_name


SPORT = "mma_mixed_martial_arts"
HISTORICAL_START = pd.Timestamp("2020-06-06", tz="UTC")
QUOTE_FIELDS = [
    "event_uid", "event_name", "event_date", "snapshot_kind",
    "requested_snapshot", "actual_snapshot", "event_start_utc",
    "api_event_id", "commence_time", "fighter_a", "fighter_b",
    "book_key", "book_title", "book_last_update", "market_last_update",
    "odds_a", "odds_b", "response_sha256",
]
MANIFEST_FIELDS = [
    "event_uid", "event_name", "event_date", "snapshot_kind",
    "requested_snapshot", "actual_snapshot", "event_start_utc", "status",
    "matched_fights", "quote_rows", "response_sha256", "response_file",
    "requests_remaining", "requests_used", "requests_last",
]


def event_uid(date, event):
    """Stable local ID; the historical API has no UFC promotion/card ID."""
    key = f"{pd.Timestamp(date).date()}|{str(event).strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def pair_key(a, b):
    return "|".join(sorted((norm_name(a), norm_name(b))))


def _append_csv(path, fields, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path):
    """Return attempted snapshots and reusable discovered card starts.

    Historical responses are immutable, so deterministic no-match/no-quote
    results are terminal unless ``--force`` is requested. This prevents a
    resume from repeatedly spending credits on the same failed snapshot.
    """
    attempted = set()
    card_starts = {}
    if not Path(path).exists():
        return attempted, card_starts
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            uid = row.get("event_uid", "")
            kind = row.get("snapshot_kind", "")
            status = row.get("status", "")
            if uid and kind and status in {
                    "ok", "no_match", "no_quotes", "invalid_snapshot_time"}:
                attempted.add((uid, kind))
            if uid and kind.startswith("discovery_") and status == "ok":
                start = pd.to_datetime(row.get("event_start_utc"), utc=True,
                                       errors="coerce")
                if not pd.isna(start):
                    card_starts[uid] = start
    return attempted, card_starts


def _schedule(fights_path, start, end):
    fights = pd.read_csv(fights_path, parse_dates=["date"])
    fights = fights.dropna(subset=["date", "event", "fighter_a", "fighter_b"]).copy()
    fights["date_utc"] = pd.to_datetime(fights["date"], utc=True)
    fights = fights[(fights["date_utc"] >= start) & (fights["date_utc"] < end)]
    events = []
    for (date, name), group in fights.groupby(["date", "event"], sort=True):
        events.append({
            "event_uid": event_uid(date, name),
            "event_name": str(name),
            "event_date": str(pd.Timestamp(date).date()),
            "pairs": {pair_key(a, b) for a, b in zip(group["fighter_a"], group["fighter_b"])},
        })
    return events


def _request(key, query_time, regions):
    params = {
        "apiKey": key, "regions": regions, "markets": "h2h",
        "oddsFormat": "american", "date": query_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    url = "https://api.the-odds-api.com/v4/historical/sports/" + SPORT + "/odds?"
    request = urllib.request.Request(url + urllib.parse.urlencode(params),
                                     headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            headers = {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Odds API HTTP {exc.code}: {detail[:500]}") from exc
    return json.loads(body), headers


def _matched_events(payload, wanted_pairs):
    return [ev for ev in payload.get("data", [])
            if pair_key(ev.get("home_team", ""), ev.get("away_team", "")) in wanted_pairs]


def _event_start(events):
    values = [pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")
              for ev in events]
    values = [v for v in values if not pd.isna(v)]
    return min(values) if values else None


def _quotes(events, event, kind, requested, actual, card_start, digest):
    rows = []
    for ev in events:
        a, b = ev.get("home_team", ""), ev.get("away_team", "")
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
                if prices.get(a) is None or prices.get(b) is None:
                    continue
                rows.append({
                    "event_uid": event["event_uid"], "event_name": event["event_name"],
                    "event_date": event["event_date"], "snapshot_kind": kind,
                    "requested_snapshot": requested.isoformat(), "actual_snapshot": actual,
                    "event_start_utc": card_start.isoformat(),
                    "api_event_id": ev.get("id", ""), "commence_time": ev.get("commence_time", ""),
                    "fighter_a": a, "fighter_b": b,
                    "book_key": book.get("key", ""), "book_title": book.get("title", ""),
                    "book_last_update": book.get("last_update", ""),
                    "market_last_update": market.get("last_update", ""),
                    "odds_a": prices[a], "odds_b": prices[b], "response_sha256": digest,
                })
    return rows


def _save_used_response(directory, event, kind, requested, payload, events):
    """Store exactly the API response fragment used by this project, compressed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    actual = str(payload.get("timestamp", "unknown")).replace(":", "-")
    path = directory / f"{event['event_uid']}__{kind}__{actual}.json.gz"
    public = {
        "request": {"sport": SPORT, "regions": "us", "markets": "h2h",
                    "requested_snapshot": requested.isoformat()},
        "response": {k: payload.get(k) for k in ("timestamp", "previous_timestamp", "next_timestamp")},
        "data": events,
    }
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    with gzip.open(path, "wb") as f:
        f.write(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def _record(manifest_path, quote_path, response_dir, event, kind, requested,
            payload, headers, matched, card_start):
    actual = pd.to_datetime(payload.get("timestamp"), utc=True, errors="coerce")
    valid_time = card_start is not None and not pd.isna(actual) and actual < card_start
    if not valid_time:
        status = "invalid_snapshot_time"
        quotes, response_file, digest = [], "", ""
    elif not matched:
        status = "no_match"
        quotes, response_file, digest = [], "", ""
    else:
        response_file, digest = _save_used_response(response_dir, event, kind, requested,
                                                    payload, matched)
        quotes = _quotes(matched, event, kind, requested, str(actual), card_start, digest)
        status = "ok" if quotes else "no_quotes"
    _append_csv(quote_path, QUOTE_FIELDS, quotes)
    _append_csv(manifest_path, MANIFEST_FIELDS, [{
        "event_uid": event["event_uid"], "event_name": event["event_name"],
        "event_date": event["event_date"], "snapshot_kind": kind,
        "requested_snapshot": requested.isoformat(), "actual_snapshot": str(actual),
        "event_start_utc": card_start.isoformat() if card_start is not None else "",
        "status": status, "matched_fights": len(matched), "quote_rows": len(quotes),
        "response_sha256": digest, "response_file": str(response_file),
        "requests_remaining": headers.get("x-requests-remaining", ""),
        "requests_used": headers.get("x-requests-used", ""),
        "requests_last": headers.get("x-requests-last", ""),
    }])
    return status, len(quotes)


def run(args):
    start = max(pd.Timestamp(args.start, tz="UTC"), HISTORICAL_START)
    end = pd.Timestamp(args.end, tz="UTC")
    events = _schedule(args.fights, start, end)
    root = Path(args.output_dir)
    manifest_path = root / "snapshot_manifest.csv"
    quote_path = root / "historical_h2h_quotes.csv"
    if args.force:
        attempted, cached_starts = set(), {}
    else:
        attempted, cached_starts = _read_manifest(manifest_path)
    worst_requests = 0
    for event in events:
        uid = event["event_uid"]
        pending_snapshots = sum(
            (uid, kind) not in attempted for kind in ("entry", "close")
        )
        if not pending_snapshots:
            continue
        if uid not in cached_starts:
            worst_requests += sum(
                (uid, f"discovery_{hours}h") not in attempted
                for hours in (72, 48, 24)
            )
        worst_requests += pending_snapshots
    print(f"eligible UFC events: {len(events)} ({start.date()} to {end.date()})")
    print(f"remaining worst-case cost: {worst_requests * args.credits_per_request} credits "
          f"({worst_requests} requests; actual is normally lower)")
    if args.dry_run:
        return {"events": len(events), "worst_requests": worst_requests}

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("Set ODDS_API_KEY in the environment; do not put it in a file or command line.")
    requests, quote_count, completed = 0, 0, 0

    def fetch(query_time):
        nonlocal requests
        if requests >= args.max_requests:
            return None, None
        payload, headers = _request(key, query_time, args.regions)
        requests += 1
        return payload, headers

    for event in events:
        uid = event["event_uid"]
        if (uid, "entry") in attempted and (uid, "close") in attempted:
            continue
        day = pd.Timestamp(event["event_date"], tz="UTC")
        card_start = cached_starts.get(uid)
        discovered = card_start is not None
        # A 72h query is inexpensive relative to per-fight polling and usually
        # finds an already-open UFC card.  Later probes handle late-openers.
        if not discovered:
            for hours in (72, 48, 24):
                kind = f"discovery_{hours}h"
                if (uid, kind) in attempted:
                    continue
                requested = day - pd.Timedelta(hours=hours)
                payload, headers = fetch(requested)
                if payload is None:
                    break
                matched = _matched_events(payload, event["pairs"])
                candidate = _event_start(matched)
                if candidate is not None:
                    card_start = candidate
                    discovered = True
                    _record(manifest_path, quote_path, root / "responses", event,
                            kind, requested, payload, headers, matched, card_start)
                    break
                _record(manifest_path, quote_path, root / "responses", event,
                        kind, requested, payload, headers, matched,
                        day + pd.Timedelta(days=1))
        if not discovered:
            if requests >= args.max_requests:
                break
            continue

        for kind, requested in (
            ("entry", card_start - pd.Timedelta(hours=args.entry_hours)),
            ("close", card_start - pd.Timedelta(minutes=args.close_minutes)),
        ):
            if (uid, kind) in attempted:
                continue
            payload, headers = fetch(requested)
            if payload is None:
                break
            matched = _matched_events(payload, event["pairs"])
            status, n = _record(manifest_path, quote_path, root / "responses", event,
                                kind, requested, payload, headers, matched, card_start)
            quote_count += n
            if status == "ok":
                completed += 1
        if requests >= args.max_requests:
            break
    print(f"completed snapshot kinds: {completed}; quote rows: {quote_count}; requests: {requests}")
    print(f"raw inputs and manifest: {root}")
    return {"requests": requests, "quote_rows": quote_count, "completed": completed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--start", default="2020-06-06")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--entry-hours", type=float, default=24.0)
    parser.add_argument("--close-minutes", type=float, default=15.0)
    parser.add_argument("--max-requests", type=int, default=30,
                        help="safety cap; run repeated batches after checking coverage")
    parser.add_argument("--credits-per-request", type=int, default=10)
    parser.add_argument("--output-dir", default="raw/odds_api_historical")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-fetch completed snapshot kinds")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
