"""Harvest historical MMA odds snapshots from The Odds API.

Run LOCALLY (do not commit output to a public repo — bulk redistribution
of their data likely violates their terms):

    set ODDS_API_KEY=yourkey        (Windows)
    python harvest_history.py --start 2020-06-01 --max-requests 700

Resumable: already-harvested snapshot timestamps are skipped. Two
snapshots per week: Wednesday 14:00 UTC (early lines) and Saturday
20:00 UTC (near-close for most cards). Each request costs ~10 credits;
remaining quota is printed from response headers as you go.

Output: odds_history.csv with one row per fight per snapshot:
    snapshot_ts, commence_time, fighter_a, fighter_b,
    odds_a, odds_b (median across US books), n_books
"""

import argparse
import csv
import os
import sys
import time
import urllib.parse
import urllib.request
import json
from datetime import datetime, timedelta, timezone

BASE = ("https://api.the-odds-api.com/v4/historical/sports/"
        "mma_mixed_martial_arts/odds")
OUT = "odds_history.csv"
FIELDS = ["snapshot_ts", "commence_time", "fighter_a", "fighter_b",
          "odds_a", "odds_b", "n_books"]


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def weekly_stamps(start, end):
    """Every Wednesday 14:00 and Saturday 20:00 UTC in [start, end]."""
    d = start
    while d <= end:
        if d.weekday() == 2:      # Wednesday
            yield d.replace(hour=14, minute=0)
        if d.weekday() == 5:      # Saturday
            yield d.replace(hour=20, minute=0)
        d += timedelta(days=1)


def fetch(key, stamp):
    qs = urllib.parse.urlencode({
        "apiKey": key, "regions": "us", "markets": "h2h",
        "oddsFormat": "american",
        "date": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    req = urllib.request.Request(f"{BASE}?{qs}")
    with urllib.request.urlopen(req, timeout=30) as r:
        remaining = r.headers.get("x-requests-remaining", "?")
        return json.loads(r.read().decode()), remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-requests", type=int, default=700)
    args = ap.parse_args()

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("Set the ODDS_API_KEY environment variable first.")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))

    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            done = {row["snapshot_ts"] for row in csv.DictReader(f)}
        print(f"resuming: {len(done)} snapshot timestamps already harvested")

    new_file = not os.path.exists(OUT)
    made = 0
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for stamp in weekly_stamps(start, end):
            ts = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            if ts in done:
                continue
            if made >= args.max_requests:
                print(f"hit --max-requests={args.max_requests}; "
                      f"re-run later to continue (resumable)")
                break
            try:
                payload, remaining = fetch(key, stamp)
            except Exception as exc:
                print(f"  {ts}: ERROR {exc} — stopping; re-run to resume")
                break
            made += 1
            events = payload.get("data", [])
            rows = 0
            for ev in events:
                pa, pb = [], []
                home, away = ev.get("home_team"), ev.get("away_team")
                for bk in ev.get("bookmakers", []):
                    for mk in bk.get("markets", []):
                        if mk.get("key") != "h2h":
                            continue
                        for o in mk.get("outcomes", []):
                            if o["name"] == home:
                                pa.append(o["price"])
                            elif o["name"] == away:
                                pb.append(o["price"])
                if pa and pb:
                    w.writerow({"snapshot_ts": ts,
                                "commence_time": ev.get("commence_time"),
                                "fighter_a": home, "fighter_b": away,
                                "odds_a": median(pa), "odds_b": median(pb),
                                "n_books": min(len(pa), len(pb))})
                    rows += 1
            print(f"  {ts}: {rows} fights  (credits left: {remaining})")
            time.sleep(1.0)
    print(f"done: {made} requests this run -> {OUT}")


if __name__ == "__main__":
    main()
