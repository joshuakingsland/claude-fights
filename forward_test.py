"""Forward paper log for H22, the only route left to a clean answer.

The archive cannot settle this. H22 replicated out of sample at the same size
it showed in development - 82.53% against 81.88%, +4.26% against +3.59% - and
failed only because 166 bets is too few to resolve a four-point edge. The
holdout is now spent, so nothing further from the archive is out-of-sample.
The fights that would settle it have not happened yet.

So this logs what the rule would bet, at prices captured before the fight, and
settles them as cards land. Three properties make it evidence rather than
bookkeeping:

- **Point-in-time.** A bet is written once, from a snapshot taken before the
  event, and never rewritten. The logged price is the price that existed at
  the logged moment, not one looked up afterwards.
- **One bet per fight, ever.** The archive test took one snapshot per fight;
  logging a fight twice as its line moved would double-count a single opinion
  and shrink the intervals to nothing.
- **No discretion.** The rule is the specification from PREREGISTRATION.md
  Addendum 8, unchanged. If it is edited, the log is no longer a forward test
  of the thing that was tested, and `RULE_VERSION` exists to make that break
  visible rather than silent.

What this cannot fix: the archive test could not see limits, and neither can
this. A logged price is one the API published, not one a book would have laid
in size to a winning account.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

import fair_line as fl
import sharp_fade as sf
from backtest import norm_name

# Bump only when the betting rule itself changes. A forward log that silently
# changes rules mid-flight measures nothing.
RULE_VERSION = "h22.1"

TARGET_LEAD_HOURS = 12.0
LEAD_WINDOW_HOURS = (6.0, 24.0)
FAIR_THRESHOLD = fl.PRIMARY_FAVOURITE_BUCKET
MIN_POOL_BOOKS = fl.MIN_POOL_BOOKS

# The archive holdout covered cards up to 2026-08-08. A fight on or before that
# date has already been counted once in the out-of-sample test, and logging it
# again would double-count the same evidence while looking like new data.
FORWARD_START = pd.Timestamp("2026-08-09", tz="UTC")

LOG_FIELDS = [
    "fight_key", "rule_version", "logged_at", "snapshot_at", "commence_time",
    "date", "fighter_a", "fighter_b", "side", "pick", "fair_prob",
    "bet_odds", "bet_book", "pool_books", "lead_hours", "logged_before_event",
]


def load_quotes(pattern="data/market_quotes/quotes_*.csv"):
    """Live per-book quotes, as captured by the six-hourly snapshot job."""
    files = sorted(Path().glob(pattern))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for column in ("odds_a", "odds_b"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["odds_a", "odds_b", "commence_time", "fetched_at"])
    frame["commence_time"] = pd.to_datetime(frame["commence_time"], utc=True,
                                            errors="coerce")
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True,
                                         errors="coerce")
    frame = frame.dropna(subset=["commence_time", "fetched_at"])
    frame["lead_hours"] = ((frame["commence_time"] - frame["fetched_at"])
                           .dt.total_seconds() / 3600.0)
    return frame


# A rematch inside this window does not happen; a rescheduled card inside it
# happens constantly. So the pair identifies the fight and the window separates
# genuine rematches.
REMATCH_WINDOW_DAYS = 60


def _fight_key(row):
    """Stable identity for a fight: the two fighters, corner order removed.

    The date is deliberately *not* part of this. It used to be, and that was a
    bug: the API refines a card's start time as it firms up, and one fight here
    moved from 2026-08-16 02:00 UTC to 2026-08-15 21:45 UTC between snapshots.
    That produced two keys for one fight, which would have been logged twice and
    counted as two independent bets - the precise double-count the log exists to
    prevent. event_id is no better; it is the API's and also changes.

    Rematches are handled by `append_log`, which treats a repeat of the same
    pair as new only when it is more than REMATCH_WINDOW_DAYS later.
    """
    pair = sorted((norm_name(row["fighter_a"]), norm_name(row["fighter_b"])))
    return f"{pair[0]}|{pair[1]}"


def candidates(quotes, target=TARGET_LEAD_HOURS, window=LEAD_WINDOW_HOURS,
               threshold=FAIR_THRESHOLD, min_pool=MIN_POOL_BOOKS,
               start=FORWARD_START):
    """H22 selections from live quotes, one row per qualifying fight.

    The snapshot used is the one closest to `target` hours out and inside
    `window`. The archive tested every horizon from 1h to 48h and all six were
    positive, so the exact lead is not delicate - but it is recorded per bet so
    drift between what was tested and what is being logged stays visible.

    Fights on or before the holdout's last card are refused outright. They are
    already counted in the out-of-sample result, and re-logging them here would
    present the same evidence twice while wearing the label of new data.
    """
    if quotes.empty:
        return pd.DataFrame(columns=LOG_FIELDS)
    live = sf.drop_exchanges(quotes)
    live = live[live["commence_time"] >= start]
    low, high = window
    live = live[(live["lead_hours"] >= low) & (live["lead_hours"] <= high)]
    if live.empty:
        return pd.DataFrame(columns=LOG_FIELDS)
    live = live.assign(_key=[_fight_key(r) for _, r in live.iterrows()])

    rows = []
    for key, fight in live.groupby("_key"):
        # One snapshot per fight: the capture nearest the tested horizon.
        distance = (fight["lead_hours"] - target).abs()
        chosen_at = fight.loc[distance.idxmin(), "fetched_at"]
        group = fight[fight["fetched_at"] == chosen_at]
        group = group.drop_duplicates(subset=["book_key"], keep="last")

        reference = group[group["book_key"] == fl.FAIR_BOOK]
        pool = group[group["book_key"] != fl.FAIR_BOOK]
        if len(reference) != 1 or len(pool) < min_pool:
            continue
        reference = reference.iloc[0]
        fair_a = float(fl._devig(np.array([reference["odds_a"]]),
                                 np.array([reference["odds_b"]]))[0])
        if not np.isfinite(fair_a):
            continue
        take_a = fair_a >= 0.5
        fair = fair_a if take_a else 1.0 - fair_a
        if fair < threshold:
            continue
        column = "odds_a" if take_a else "odds_b"
        payouts = sf._payout(pool[column].to_numpy(float))
        best = int(np.argmax(payouts))
        first = group.iloc[0]
        now = pd.Timestamp.now(tz="UTC")
        rows.append({
            "fight_key": key,
            "rule_version": RULE_VERSION,
            # A bet written after the bell used a price from before it, so the
            # price is honest, but the *decision* was not made blind. Those are
            # backfill and are reported apart from the forward number, because
            # only bets written before the outcome existed are evidence.
            "logged_before_event": bool(now < pd.Timestamp(first["commence_time"])),
            "logged_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "snapshot_at": pd.Timestamp(chosen_at).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commence_time": pd.Timestamp(first["commence_time"]).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "date": str(first["date"])[:10],
            "fighter_a": first["fighter_a"],
            "fighter_b": first["fighter_b"],
            "side": "a" if take_a else "b",
            "pick": first["fighter_a"] if take_a else first["fighter_b"],
            "fair_prob": round(fair, 6),
            "bet_odds": float(pool[column].iloc[best]),
            "bet_book": pool["book_key"].iloc[best],
            "pool_books": len(pool),
            "lead_hours": round(float(
                fight.loc[distance.idxmin(), "lead_hours"]), 2),
        })
    return pd.DataFrame(rows, columns=LOG_FIELDS)


def append_log(rows, path="h22_forward_log.csv"):
    """Append new bets, never rewriting one already logged.

    A fight already in the log is skipped rather than updated. That is the
    whole point: the record has to be what was knowable then, not what looks
    right now.
    """
    path = Path(path)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_FIELDS)
    if not len(rows):
        return 0
    if len(existing):
        seen = {}
        for _, row in existing.iterrows():
            seen.setdefault(str(row["fight_key"]), []).append(
                pd.Timestamp(row["commence_time"]))
        # Same pair within the rematch window is the same fight, however its
        # start time was revised between snapshots.
        def already_logged(row):
            when = pd.Timestamp(row["commence_time"])
            return any(abs((when - prior).days) <= REMATCH_WINDOW_DAYS
                       for prior in seen.get(str(row["fight_key"]), []))
        fresh = rows[~rows.apply(already_logged, axis=1)]
    else:
        fresh = rows
    if not len(fresh):
        return 0
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        for row in fresh.to_dict("records"):
            writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})
    return len(fresh)


def settle(path="h22_forward_log.csv", fights_path="fights_v2.csv"):
    """Attach results to logged bets. Unfought or unresolved bets stay open."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    log = pd.read_csv(path)
    if not len(log):
        return log
    results = sf.outcomes(fights_path)
    winners = []
    for _, row in log.iterrows():
        pair = frozenset((norm_name(row["fighter_a"]), norm_name(row["fighter_b"])))
        stamp = pd.Timestamp(str(row["date"])[:10])
        # A UFC card starting 22:00-02:00 UTC lands on either side of midnight,
        # so the date the quote carried and the date the result carries differ
        # by a day about as often as not. Matching only on the exact string
        # left those bets permanently unsettled - open forever, invisible in
        # every summary, and silently absent from the ROI they belonged in.
        found = None
        for offset in (0, -1, 1):
            found = results.get(
                (str((stamp + pd.Timedelta(days=offset)).date()), pair))
            if found is not None:
                break
        winners.append(found)
    log["winner_name"] = winners
    settled = log.dropna(subset=["winner_name"]).copy()
    if not len(settled):
        return settled
    settled["won"] = [norm_name(p) == w for p, w
                      in zip(settled["pick"], settled["winner_name"])]
    settled["profit"] = np.where(settled["won"],
                                 sf._payout(settled["bet_odds"].to_numpy()), -1.0)
    settled["event_date"] = settled["date"]
    return settled


def bets_needed(roi, standard_error, observed_n, confidence=1.645):
    """How many bets until a 90% lower bound clears zero, at this effect size.

    Turns "we need more data" into a number. The standard error is assumed to
    fall as 1/sqrt(n), which held between the two archive samples, and the
    estimate is only as good as the ROI it is handed - at a smaller true effect
    it grows quadratically.
    """
    if roi <= 0 or standard_error <= 0 or observed_n <= 0:
        return None
    target = roi / confidence
    return int(np.ceil(observed_n * (standard_error / target) ** 2))


def summary(path="h22_forward_log.csv", fights_path="fights_v2.csv", draws=2000):
    """Where the forward test stands, and how far it has left to run."""
    log = pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    settled = settle(path, fights_path)
    report = {
        "rule_version": RULE_VERSION,
        "logged": int(len(log)),
        "settled": int(len(settled)),
        "open": int(len(log) - len(settled)),
    }
    if not len(settled):
        return report
    # Only bets written before the bell count toward the headline. Backfill is
    # kept and reported, never pooled - mixing them would let a decision made
    # with the result available borrow the credibility of one made blind.
    blind = settled[settled.get("logged_before_event", True).astype(bool)]
    report["settled_backfilled"] = int(len(settled) - len(blind))
    if not len(blind):
        report["note"] = "no bets yet logged before their event; nothing to score"
        return report
    scored = sf.score(blind, "H22 forward", draws=draws)
    report.update({k: scored[k] for k in
                   ("n", "win_rate", "roi", "ci90_low", "ci90_high", "passes")})
    half_width = (scored["ci90_high"] - scored["ci90_low"]) / 2.0
    error = half_width / 1.645 if half_width > 0 else 0.0
    report["bets_needed_at_this_effect"] = bets_needed(
        scored["roi"], error, scored["n"])
    return report


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["log", "summary"])
    parser.add_argument("--quotes", default="data/market_quotes/quotes_*.csv")
    parser.add_argument("--log", default="h22_forward_log.csv")
    parser.add_argument("--fights", default="fights_v2.csv")
    args = parser.parse_args(argv)

    if args.command == "log":
        found = candidates(load_quotes(args.quotes))
        added = append_log(found, args.log)
        print(f"H22 forward log: {len(found)} candidate(s) in window, "
              f"{added} newly written")
        if added:
            fresh = pd.read_csv(args.log).tail(added)
            for row in fresh.to_dict("records"):
                mark = "" if row.get("logged_before_event") else "  [BACKFILL]"
                print(f"  {row['date']}  {row['pick']:<24} "
                      f"{int(row['bet_odds']):+d} @ {row['bet_book']}"
                      f"  fair {float(row['fair_prob']):.3f}{mark}")
    else:
        print(json.dumps(summary(args.log, args.fights), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
