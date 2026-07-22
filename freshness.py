"""Assess and publish the freshness of completed fight results."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from identity import norm_name


def _pairs(frame):
    return [
        "|".join(sorted((norm_name(a), norm_name(b))))
        for a, b in zip(frame["fighter_a"], frame["fighter_b"])
    ]


def assess_freshness(fights, odds_log="odds_log.csv", now=None):
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
            for row in completed.dropna(subset=["commence_time"]).itertuples():
                key = (row.commence_time.date(), row.pair)
                if key not in result_pairs:
                    known_missing.append({
                        "date": str(row.commence_time.date()),
                        "fighter_a": row.fighter_a,
                        "fighter_b": row.fighter_b,
                    })

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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--odds-log", default="odds_log.csv")
    parser.add_argument("--output", default="data_freshness.json")
    parser.add_argument("--require-current", action="store_true")
    args = parser.parse_args()
    fights = pd.read_csv(args.fights)
    report = assess_freshness(fights, args.odds_log)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_current and report["status"] == "lagging":
        raise SystemExit("Completed tracked fights are missing from the result source.")


if __name__ == "__main__":
    main()
