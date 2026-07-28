"""Assess and publish the freshness of completed fight results."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from identity import canonical_name


def _pairs(frame):
    return [
        "|".join(sorted((canonical_name(a), canonical_name(b))))
        for a, b in zip(frame["fighter_a"], frame["fighter_b"])
    ]


def _cancelled_keys(path):
    path = Path(path)
    if not path.exists():
        return set()
    cancellations = pd.read_csv(path)
    required = {"date", "fighter_a", "fighter_b", "status"}
    missing = required - set(cancellations.columns)
    if missing:
        raise ValueError(
            f"cancelled fight registry is missing columns: {sorted(missing)}"
        )
    statuses = cancellations["status"].astype(str).str.strip().str.lower()
    cancellations = cancellations[statuses.isin({"cancelled", "canceled"})].copy()
    dates = pd.to_datetime(cancellations["date"], errors="coerce", utc=True)
    return set(zip(dates.dt.date, _pairs(cancellations)))


def assess_freshness(
    fights,
    odds_log="odds_log.csv",
    now=None,
    cancelled_fights="cancelled_fights.csv",
):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    dates = pd.to_datetime(fights["date"], errors="coerce", utc=True)
    latest = dates.max()
    if pd.isna(latest):
        raise ValueError("fight results contain no valid dates")

    known_missing = []
    known_cancelled = []
    cancelled_keys = _cancelled_keys(cancelled_fights)
    path = Path(odds_log)
    if path.exists():
        odds = pd.read_csv(path)
        required = {"fighter_a", "fighter_b", "commence_time"}
        if required.issubset(odds.columns):
            odds["commence_time"] = pd.to_datetime(
                odds["commence_time"], errors="coerce", utc=True
            )
            odds["pair"] = _pairs(odds)
            result_pairs = set(zip(dates.dt.date, _pairs(fights)))
            completed = odds[odds["commence_time"] < now - pd.Timedelta(hours=12)]
            completed = completed.dropna(subset=["commence_time"]).copy()
            completed["fight_date"] = completed["commence_time"].dt.date
            completed = completed.drop_duplicates(["fight_date", "pair"])
            for row in completed.itertuples():
                key = (row.commence_time.date(), row.pair)
                item = {
                    "date": str(row.commence_time.date()),
                    "fighter_a": row.fighter_a,
                    "fighter_b": row.fighter_b,
                }
                if key in result_pairs:
                    continue
                if key in cancelled_keys:
                    known_cancelled.append(item)
                else:
                    known_missing.append(item)

    age_days = max(0, int((now.normalize() - latest.normalize()).days))
    if known_missing:
        status = "lagging"
        message = f"{len(known_missing)} completed tracked fight(s) await results"
    elif age_days > 21:
        status = "check"
        message = f"latest bundled result is {age_days} days old"
    else:
        status = "current"
        message = "no completed tracked fights are missing"
    return {
        "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results_through": str(latest.date()),
        "days_since_latest_result": age_days,
        "status": status,
        "message": message,
        "known_completed_missing": known_missing,
        "known_cancelled": known_cancelled,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--odds-log", default="odds_log.csv")
    parser.add_argument("--cancelled-fights", default="cancelled_fights.csv")
    parser.add_argument("--output", default="data_freshness.json")
    parser.add_argument("--require-current", action="store_true")
    args = parser.parse_args()
    fights = pd.read_csv(args.fights)
    report = assess_freshness(
        fights, args.odds_log, cancelled_fights=args.cancelled_fights
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_current and report["status"] == "lagging":
        raise SystemExit("Completed tracked fights are missing from the result source.")


if __name__ == "__main__":
    main()
