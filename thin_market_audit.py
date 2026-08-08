"""Does the model do better where fewer books quote?

The production screen rejects any fight with fewer than three paired books,
on the reasoning that a one-book price is not a market. That filter is
defensible for staking and indefensible as an assumption, because thin books
are also where soft prices are most likely to survive. The filter has never
been examined as a research question.

It cannot be examined in the historical entry archive: that file was built
with the same three-book minimum, so `entry_n_books` bottoms out at 3 and the
interesting region is simply absent. Within the range it does cover, the
model's advantage over the market is flat in book count — beta -0.000018 per
book, r = -0.000, 90% interval [-0.0022, +0.0022] — and the thinnest bucket
available (3-5 books, 92 fights) is where the model does *worst*, losing by
0.019 of log loss. That is the opposite of the hypothesis, on a sample far
too small to settle it.

Forward capture does retain thin fights: `prediction_snapshots.csv` records
every scored fight with its `market_books` count and an eligibility reason,
including the 1,258 rows across 94 fights that quoted one or two books. What
was missing is the join to results, which is what this script does.

The first run reframed the question. Zero thin fights had resolved, despite
94 of them being captured, because book count is largely a proxy for time to
event: books post an early line on a marquee bout months out, so only one or
two quote it, and the same fight is thick a week before it happens. The thin
fights span 2026-07-31 to 2027-08-01 while the thick ones span 2026-08-01 to
2026-09-20 - they are mostly future fights, not obscure ones.

That means "thin market" and "far from the event" are close to the same
variable here, and a thin-book audit is a slower restatement of the timing
question rather than an independent one. It is still worth running as fights
resolve, because the two are not perfectly confounded: a genuinely obscure
bout is thin all the way to its start. The `days_to_event` column exists so
that confound can be separated rather than assumed away.

It reports rather than decides. Nothing here promotes a fight into the
tradable set, and a thin bucket that looks profitable on 20 fights is noise;
the sample column is printed next to every number so that stays visible.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import norm_name

BUCKETS = [(1, 1, "1 book"), (2, 2, "2 books"), (3, 5, "3-5"),
           (6, 8, "6-8"), (9, 99, "9+")]


def _pair_key(a, b):
    return "|".join(sorted((norm_name(a), norm_name(b))))


def load_outcomes(path="fights_v2.csv"):
    """Map (date, fighter pair) to whether the first-named fighter won."""
    fights = pd.read_csv(path, parse_dates=["date"])
    fights = fights.dropna(subset=["fighter_a", "fighter_b", "winner"])
    outcomes = {}
    for row in fights.itertuples():
        key = (row.date.strftime("%Y-%m-%d"),
               _pair_key(row.fighter_a, row.fighter_b))
        winner = str(row.winner).strip().upper()
        if winner not in ("A", "B"):
            continue
        outcomes[key] = norm_name(
            row.fighter_a if winner == "A" else row.fighter_b)
    return outcomes


def last_snapshot_per_fight(snapshots):
    """The final pre-event scoring of each fight is the one to evaluate."""
    snapshots = snapshots[snapshots["market_books"].notna()].copy()
    snapshots = snapshots.sort_values("recorded_at")
    return snapshots.groupby("fight_key", as_index=False).last()


def build(snapshots_path="prediction_snapshots.csv", fights_path="fights_v2.csv"):
    snapshots = pd.read_csv(snapshots_path)
    outcomes = load_outcomes(fights_path)
    rows = []
    for snap in last_snapshot_per_fight(snapshots).to_dict("records"):
        key = (str(snap["date"]), _pair_key(snap["pick"], snap["opp"]))
        winner = outcomes.get(key)
        if winner is None:
            continue  # not yet fought, or not a tracked UFC bout
        try:
            model = float(snap["model"]) / 100.0
            market = float(snap["market"]) / 100.0
            books = int(float(snap["market_books"]))
        except (TypeError, ValueError):
            continue
        if not (0 < model < 1 and 0 < market < 1):
            continue
        days_out = np.nan
        try:
            days_out = (pd.Timestamp(snap["date"])
                        - pd.Timestamp(str(snap["recorded_at"])[:10])).days
        except (TypeError, ValueError):
            pass
        rows.append({
            "fight_key": snap["fight_key"],
            "date": snap["date"],
            "books": books,
            "days_to_event": days_out,
            "model": model,
            "market": market,
            "won": int(winner == norm_name(snap["pick"])),
            "execution_price": snap.get("execution_price", ""),
            "net_edge": snap.get("net_edge", np.nan),
        })
    return pd.DataFrame(rows)


def _log_loss(probability, outcome):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    outcome = np.asarray(outcome, dtype=float)
    return float(-np.mean(outcome * np.log(probability)
                          + (1 - outcome) * np.log(1 - probability)))


def audit(frame, min_fights=20):
    report = {"fights_resolved": int(len(frame)), "buckets": []}
    if not len(frame):
        report["status"] = "no resolved fights yet"
        return report
    for low, high, label in BUCKETS:
        subset = frame[(frame["books"] >= low) & (frame["books"] <= high)]
        entry = {"books": label, "fights": int(len(subset))}
        if len(subset) < min_fights:
            # Printed anyway, so a thin bucket is visibly thin rather than
            # quietly excluded.
            entry["status"] = f"below {min_fights}-fight reporting minimum"
            report["buckets"].append(entry)
            continue
        model = _log_loss(subset["model"], subset["won"])
        market = _log_loss(subset["market"], subset["won"])
        entry.update({
            "log_loss_model": round(model, 5),
            "log_loss_market": round(market, 5),
            "delta": round(model - market, 5),
            "model_beats_market": bool(model < market),
            "pick_win_rate": round(float(subset["won"].mean()), 4),
            "median_days_to_event": (
                None if subset["days_to_event"].isna().all()
                else float(subset["days_to_event"].median())),
        })
        report["buckets"].append(entry)
    thin = frame[frame["books"] < 3]
    thick = frame[frame["books"] >= 3]
    if len(thin) >= min_fights and len(thick) >= min_fights:
        report["thin_versus_thick"] = {
            "thin_fights": int(len(thin)), "thick_fights": int(len(thick)),
            "thin_delta": round(_log_loss(thin["model"], thin["won"])
                                - _log_loss(thin["market"], thin["won"]), 5),
            "thick_delta": round(_log_loss(thick["model"], thick["won"])
                                 - _log_loss(thick["market"], thick["won"]), 5),
        }
    else:
        report["thin_versus_thick"] = {
            "status": "insufficient sample",
            "thin_fights": int(len(thin)), "thick_fights": int(len(thick)),
            "note": ("Thin fights accumulate at roughly the rate cards are "
                     "scored. This is a standing question, not a finished "
                     "one."),
        }
    report["interpretation"] = (
        "Research only. A thin bucket that looks profitable on a small sample "
        "is noise, and nothing here promotes a fight into the tradable set. "
        "The production screen still requires three paired books."
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default="prediction_snapshots.csv")
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--report", default="thin_market_audit.json")
    args = parser.parse_args()
    frame = build(args.snapshots, args.fights)
    report = audit(frame)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
