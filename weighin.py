"""Line movement across the weigh-in window.

A weigh-in is one of the few scheduled information events in the sport: it
happens the morning before the card, it is public, and it is televised. If a
fighter misses weight, the market learns at a known moment. That makes it the
cleanest natural experiment available for the timing question - unlike a
gradual drift, there is a before and an after with a known boundary.

No scraping is required to measure the movement. The six-hour capture already
brackets the window: quotes exist for 51 fights both more than 40 hours and
less than 30 hours before the first bell, which straddles a Friday-morning
weigh-in for a Saturday card.

What is NOT available is the weigh-in result itself. UFCStats publishes no
scale readings, so `fights_v2.csv` and the bundled datasets carry only a
fighter's profile weight - static across a career apart from division moves -
and no missed-weight flag exists anywhere in this repository. The
`missed_weight` column here is therefore always empty. It is plumbed through
so that adding a source later is a join rather than a rewrite, and so that
its absence is visible in every report instead of being silently assumed
away.

Until that source exists, a large move across the window is a *proxy* for a
weigh-in event, not evidence of one. Fights are also pulled, fighters fall
ill, and news breaks on Friday for reasons that have nothing to do with the
scale. The report says `inferred` for exactly that reason.
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

# UFC weigh-ins run the morning before the card. A Saturday event beginning
# 22:00-04:00 UTC puts the weigh-in roughly 30-40 hours before the first
# bout, so quotes outside that band sit cleanly either side of it. The gap is
# deliberately left empty rather than split at a point estimate, because a
# quote taken mid-weigh-in belongs to neither side.
PRE_HOURS_MIN = 40.0
POST_HOURS_MAX = 30.0

# A move this large across the window is unlikely to be ordinary drift.
NOTABLE_MOVE_POINTS = 3.0


def load_quotes(pattern="data/market_quotes/quotes_*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no quote files matched {pattern}")
    quotes = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    quotes["fetched"] = pd.to_datetime(quotes["fetched_at"], utc=True,
                                       errors="coerce")
    quotes["commence"] = pd.to_datetime(quotes["commence_time"], utc=True,
                                        errors="coerce")
    quotes = quotes.dropna(subset=["fetched", "commence", "devig_prob_a"])
    # A quote taken after the bout began is an in-play price, not a pre-fight
    # one, and must never enter a before/after comparison.
    quotes = quotes[quotes["fetched"] < quotes["commence"]].copy()
    quotes["hours_out"] = ((quotes["commence"] - quotes["fetched"])
                           .dt.total_seconds() / 3600.0)
    return quotes


def _side(quotes, low, high, label, min_books):
    window = quotes[(quotes["hours_out"] >= low) & (quotes["hours_out"] < high)]
    if not len(window):
        return pd.DataFrame()
    grouped = window.groupby(["date", "fighter_a", "fighter_b"])
    frame = grouped.agg(**{
        f"{label}_prob": ("devig_prob_a", "median"),
        f"{label}_books": ("book_key", "nunique"),
        f"{label}_hours": ("hours_out", "median"),
    }).reset_index()
    return frame[frame[f"{label}_books"] >= min_books]


def build(quotes, min_books=2):
    """One row per fight with a consensus either side of the weigh-in."""
    pre = _side(quotes, PRE_HOURS_MIN, 1e9, "pre", min_books)
    post = _side(quotes, 0.0, POST_HOURS_MAX, "post", min_books)
    if not len(pre) or not len(post):
        return pd.DataFrame()
    frame = pre.merge(post, on=["date", "fighter_a", "fighter_b"], how="inner")
    frame["move_points"] = (frame["post_prob"] - frame["pre_prob"]) * 100.0
    frame["abs_move"] = frame["move_points"].abs()
    # No weigh-in result source exists yet. Kept explicit and empty rather
    # than defaulted, so a missing value can never be read as "made weight".
    frame["missed_weight"] = pd.NA
    return frame.sort_values("abs_move", ascending=False).reset_index(drop=True)


def report(frame):
    if not len(frame):
        return {"status": "no fights bracket the weigh-in window yet",
                "fights": 0}
    moves = frame["abs_move"]
    notable = frame[moves >= NOTABLE_MOVE_POINTS]
    out = {
        "fights": int(len(frame)),
        "window": {"pre_hours_min": PRE_HOURS_MIN,
                   "post_hours_max": POST_HOURS_MAX},
        "absolute_move_points": {
            "median": round(float(moves.median()), 3),
            "mean": round(float(moves.mean()), 3),
            "p90": round(float(moves.quantile(0.90)), 3),
            "max": round(float(moves.max()), 3),
        },
        "notable_moves": {
            "threshold_points": NOTABLE_MOVE_POINTS,
            "count": int(len(notable)),
            "share": round(float(len(notable) / len(frame)), 4),
            "basis": "inferred",
            "caveat": ("A large move is a proxy for a weigh-in event, not "
                       "evidence of one. Cancellations, injury news and "
                       "ordinary Friday drift move lines too."),
            "fights": [
                {"date": r.date, "fight": f"{r.fighter_a} vs {r.fighter_b}",
                 "move_points": round(float(r.move_points), 2),
                 "pre_books": int(r.pre_books), "post_books": int(r.post_books)}
                for r in notable.head(10).itertuples()
            ],
        },
        "missed_weight": {
            "status": "unavailable",
            "reason": ("UFCStats publishes no weigh-in results, so no scale "
                       "reading or missed-weight flag exists in this "
                       "repository. The column is plumbed and empty; adding a "
                       "commission or weigh-in feed makes it a join."),
        },
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/market_quotes/quotes_*.csv")
    parser.add_argument("--min-books", type=int, default=2)
    parser.add_argument("--out", default="weighin_moves.csv")
    parser.add_argument("--report", default="weighin_report.json")
    args = parser.parse_args()
    frame = build(load_quotes(args.quotes), min_books=args.min_books)
    if len(frame):
        frame.to_csv(args.out, index=False)
    payload = report(frame)
    Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
