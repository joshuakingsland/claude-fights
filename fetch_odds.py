"""Fetch upcoming UFC fights + moneylines into odds_upcoming.csv.

Two modes:
  1. Automatic — set env var ODDS_API_KEY (free tier at the-odds-api.com,
     500 requests/month; this uses 1 per run).
  2. Manual — no key: edit odds_upcoming.csv yourself with lines from
     your book. Columns: date,commence_time,fighter_a,fighter_b,
     odds_a,odds_b,weightclass,five_rounds,odds_source,fetched_at

Usage: python fetch_odds.py
"""

import csv
import json
import os
import urllib.request
from datetime import datetime, timezone

API = ("https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
       "?apiKey={key}&regions=us&markets=h2h&oddsFormat=american")


def main():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("No ODDS_API_KEY set — using manual mode.")
        print("Edit odds_upcoming.csv by hand with your book's lines.")
        if not os.path.exists("odds_upcoming.csv"):
            with open("odds_upcoming.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "commence_time", "fighter_a", "fighter_b",
                            "odds_a", "odds_b", "weightclass",
                            "five_rounds", "odds_source", "fetched_at"])
                w.writerow(["2099-01-02", "2099-01-02T03:00:00Z",
                            "Example Fighter", "Other Fighter", "+150",
                            "-180", "Lightweight Bout", "0", "manual", ""])
            print("Template written to odds_upcoming.csv")
        return

    with urllib.request.urlopen(API.format(key=key), timeout=30) as r:
        events = json.load(r)

    stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    rows = []
    for ev in events:
        home, away = ev["home_team"], ev["away_team"]
        commence = ev["commence_time"]
        t = commence[:10]
        # median price across books for robustness
        pa, pb = [], []
        for book in ev.get("bookmakers", []):
            for mk in book.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                for o in mk["outcomes"]:
                    if o["name"] == home:
                        pa.append(o["price"])
                    elif o["name"] == away:
                        pb.append(o["price"])
                    # anything else (Draw etc.) is ignored
        if not pa or not pb:
            continue
        med = lambda xs: sorted(xs)[len(xs) // 2]
        rows.append([t, commence, home, away, med(pa), med(pb), "", "0",
                     "the-odds-api-median", stamp])

    with open("odds_upcoming.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "commence_time", "fighter_a", "fighter_b",
                    "odds_a", "odds_b", "weightclass", "five_rounds",
                    "odds_source", "fetched_at"])
        w.writerows(rows)

    # append timestamped snapshot -> over months this becomes an
    # open/close line-movement dataset AND a prop-validation base
    new_file = not os.path.exists("odds_log.csv")
    with open("odds_log.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["fetched_at", "commence_time", "date",
                        "fighter_a", "fighter_b", "odds_a", "odds_b",
                        "odds_source"])
        for r in rows:
            w.writerow([stamp, r[1], r[0], r[2], r[3], r[4], r[5], r[8]])
    print(f"wrote odds_upcoming.csv ({len(rows)} fights) and appended "
          f"snapshot to odds_log.csv at {stamp}")


if __name__ == "__main__":
    main()
