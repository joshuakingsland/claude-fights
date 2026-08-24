"""Assess and publish the freshness of completed fight results."""

import argparse
import csv
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


def _roster_names(path):
    """Return canonical names for every fighter UFCStats has ever listed."""
    path = Path(path)
    if not path.exists():
        return set()
    roster = pd.read_csv(path)
    if not {"FIRST", "LAST"}.issubset(roster.columns):
        raise ValueError(
            "fighter roster is missing columns: ['FIRST', 'LAST']"
        )
    full = (
        roster["FIRST"].fillna("").astype(str)
        + " "
        + roster["LAST"].fillna("").astype(str)
    )
    return {name for name in full.map(canonical_name) if name}


def _debut_dates(fights):
    """Earliest recorded UFC bout per fighter, by canonical name."""
    dates = pd.to_datetime(fights["date"], errors="coerce", utc=True)
    debuts = {}
    for date, pair in zip(dates.dt.date, _names_by_row(fights)):
        if pd.isna(date):
            continue
        for name in pair:
            if name and (name not in debuts or date < debuts[name]):
                debuts[name] = date
    return debuts


def _tracked_universe(fights, fighter_roster):
    """Return ``(listed, debuts)`` for judging whether a bout is UFC scope.

    ``listed`` is everyone UFCStats knows of; ``debuts`` maps each fighter to
    their first recorded UFC bout. Returns ``None`` when the roster is missing,
    which keeps the check failing closed: scope cannot be judged, so nothing
    is excused.
    """
    roster = _roster_names(fighter_roster)
    if not roster:
        return None
    debuts = _debut_dates(fights)
    return roster | set(debuts), debuts


def _in_ufc_scope(fighter_a, fighter_b, tracked, fight_date=None):
    """Report whether UFCStats could ever carry a result for this bout.

    The odds feed covers every MMA promotion while results come from UFCStats
    alone, so a bout only counts as trackable when both fighters are known to
    UFCStats and at least one had already fought there **by the date of the
    bout**. That keeps genuine UFC debuts in scope, since a debutant is
    matched against somebody with prior UFC bouts.

    The date is what makes this correct, and leaving it out was a real bug.
    Scope used to be judged against today's veteran list, so a fighter's
    Contender Series bout became retroactively "trackable" the moment they
    later debuted in the UFC - for a result UFCStats was never going to carry.
    Matt Adams vs Anthony Wint on 2026-08-12 sat quietly out of scope until
    Wint's UFC debut on the 22nd flipped it in, whereupon it was already
    twelve days past the grace window and failed the run immediately.
    """
    listed, debuts = tracked
    names = [canonical_name(fighter_a), canonical_name(fighter_b)]
    if not all(name in listed for name in names):
        return False
    if fight_date is None:
        return any(name in debuts for name in names)
    return any(name in debuts and debuts[name] <= fight_date for name in names)


def _names(frame):
    for column in ("fighter_a", "fighter_b"):
        for value in frame[column]:
            name = canonical_name(value)
            if name:
                yield name


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


def _result_pair_dates(fights, dates):
    result_dates = {}
    for date, pair in zip(dates.dt.date, _pairs(fights)):
        result_dates.setdefault(pair, set()).add(date)
    return result_dates


def _result_matches(result_dates, fight_date, pair, tolerance_days=1):
    for result_date in result_dates.get(pair, set()):
        if abs((pd.Timestamp(fight_date) - pd.Timestamp(result_date)).days) <= tolerance_days:
            return True
    return False


def _result_name_dates(fights, dates):
    """When each fighter last has a recorded result, by canonical name."""
    seen = {}
    for date, frame_name in zip(dates.dt.date, _names_by_row(fights)):
        for name in frame_name:
            if name:
                seen.setdefault(name, set()).add(date)
    return seen


def _names_by_row(fights):
    for a, b in zip(fights["fighter_a"], fights["fighter_b"]):
        yield canonical_name(a), canonical_name(b)


def _fought_someone_else(name_dates, fight_date, fighter_a, fighter_b,
                         tolerance_days=1):
    """Did either fighter compete that night, against somebody else?

    The odds feed carries bookings that never happen. A fighter is announced
    against one opponent, the opponent changes, and the dead pairing keeps
    being quoted; books also spell the same man differently, so one bout can
    appear under several names. On 2026-08-16 the feed held Charles Johnson
    against Jose Ochoa, Eduardo Henrique and Eduardo Chapolin - one fight, and
    only the last name matched the result.

    Those unmatched pairings are not missing results, and waiting for them is
    waiting forever. But if the fighter has a result that night against
    anybody, the bout was resolved and the other pairings were superseded.
    A fighter competes once per card, so this cannot hide a genuine gap.
    """
    for name in (canonical_name(fighter_a), canonical_name(fighter_b)):
        for result_date in name_dates.get(name, set()):
            if abs((pd.Timestamp(fight_date)
                    - pd.Timestamp(result_date)).days) <= tolerance_days:
                return True
    return False


# How long a completed fight may sit without a result before it is a fault
# rather than a wait. The results feed is a third-party scrape that publishes
# some days after a card; on 2026-08-16 its newest event was 2026-08-08, which
# is normal for it and not a problem with anything here. Failing closed the
# moment a card ends therefore blocked odds capture and the card refresh too,
# freezing the published page until a stranger's repository caught up. Beyond
# this window the silence is no longer routine and should still stop the run.
RESULT_GRACE_DAYS = 7

# Bookings that stayed unmatchable past the grace window, recorded so they are
# excused on later runs. Three separate root causes have now produced the same
# symptom - an upstream publishing lag, a replaced opponent, and a bout that
# only became "trackable" once a fighter later debuted in the UFC - and each
# needed a code change and a human to notice. A fourth is a matter of time,
# because the guard cannot tell "the result is late" from "no result is ever
# coming", and only the second kind can be resolved by waiting.
QUARANTINE_LOG = "unmatchable_bookings.csv"
QUARANTINE_FIELDS = ["date", "fighter_a", "fighter_b", "quarantined_at",
                     "days_waited", "reason"]
# A handful of stragglers is the normal shape of this problem. A sudden pile is
# not: it means results stopped ingesting, names stopped matching, or the feed
# changed. That still deserves to stop the run, so the escape hatch has a lid.
MAX_AUTO_QUARANTINE = 5


def _quarantined_keys(path):
    """(date, pair) for every booking already written off as unmatchable."""
    path = Path(path)
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if not {"date", "fighter_a", "fighter_b"}.issubset(frame.columns):
        return set()
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    return set(zip(dates.dt.date, _pairs(frame)))


def quarantine(items, path=QUARANTINE_LOG, reason="unmatchable past grace"):
    """Write off bookings no result is coming for, leaving an audit trail.

    Excusing them is not the same as ignoring them: every row records what was
    excused, when, and after how long, so a wrong call is visible rather than
    silent.
    """
    if not items:
        return 0
    path = Path(path)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    known = _quarantined_keys(path)
    fresh = [item for item in items
             if (pd.Timestamp(item["date"]).date(),
                 "|".join(sorted((canonical_name(item["fighter_a"]),
                                  canonical_name(item["fighter_b"])))))
             not in known]
    if not fresh:
        return 0
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUARANTINE_FIELDS)
        if write_header:
            writer.writeheader()
        for item in fresh:
            writer.writerow({
                "date": item["date"], "fighter_a": item["fighter_a"],
                "fighter_b": item["fighter_b"], "quarantined_at": stamp,
                "days_waited": item.get("days_waiting", ""), "reason": reason,
            })
    return len(fresh)


def assess_freshness(
    fights,
    odds_log="odds_log.csv",
    now=None,
    cancelled_fights="cancelled_fights.csv",
    fighter_roster="raw/ufc_fighter_details.csv",
    grace_days=RESULT_GRACE_DAYS,
    quarantine_log=QUARANTINE_LOG,
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
    known_pending = []
    known_cancelled = []
    known_superseded = []
    known_quarantined = []
    known_out_of_scope = []
    cancelled_keys = _cancelled_keys(cancelled_fights)
    quarantined_keys = _quarantined_keys(quarantine_log)
    tracked = _tracked_universe(fights, fighter_roster)
    path = Path(odds_log)
    if path.exists():
        odds = pd.read_csv(path)
        required = {"fighter_a", "fighter_b", "commence_time"}
        if required.issubset(odds.columns):
            odds["commence_time"] = pd.to_datetime(
                odds["commence_time"], errors="coerce", utc=True
            )
            odds["pair"] = _pairs(odds)
            result_dates = _result_pair_dates(fights, dates)
            name_dates = _result_name_dates(fights, dates)
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
                if _result_matches(result_dates, row.commence_time.date(), row.pair):
                    continue
                if key in cancelled_keys:
                    known_cancelled.append(item)
                elif key in quarantined_keys:
                    known_quarantined.append(item)
                elif _fought_someone_else(name_dates, row.commence_time.date(),
                                          row.fighter_a, row.fighter_b):
                    known_superseded.append(item)
                elif tracked is not None and not _in_ufc_scope(
                    row.fighter_a, row.fighter_b, tracked,
                    row.commence_time.date()
                ):
                    known_out_of_scope.append(item)
                elif (now - row.commence_time).days < grace_days:
                    # Recent enough that the upstream scrape simply has not
                    # published it yet. Reported, so the wait is visible, but
                    # not treated as a fault.
                    item["days_waiting"] = int((now - row.commence_time).days)
                    known_pending.append(item)
                else:
                    item["days_waiting"] = int((now - row.commence_time).days)
                    known_missing.append(item)

    age_days = max(0, int((now.normalize() - latest.normalize()).days))
    if known_missing:
        status = "lagging"
        oldest = max(item["days_waiting"] for item in known_missing)
        message = (f"{len(known_missing)} completed tracked fight(s) have awaited "
                   f"results for over {grace_days} days (oldest {oldest})")
    elif age_days > 21:
        status = "check"
        message = f"latest bundled result is {age_days} days old"
    elif known_pending:
        status = "pending"
        message = (f"{len(known_pending)} recent fight(s) await results, within "
                   f"the {grace_days}-day upstream publishing window")
    else:
        status = "current"
        message = "no completed tracked fights are missing"
    return {
        "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results_through": str(latest.date()),
        "days_since_latest_result": age_days,
        "grace_days": grace_days,
        "status": status,
        "message": message,
        "known_completed_missing": known_missing,
        "known_awaiting_upstream": known_pending,
        "known_cancelled": known_cancelled,
        "known_superseded": known_superseded,
        "known_quarantined": known_quarantined,
        "known_out_of_scope": known_out_of_scope,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--odds-log", default="odds_log.csv")
    parser.add_argument("--cancelled-fights", default="cancelled_fights.csv")
    parser.add_argument("--fighter-roster", default="raw/ufc_fighter_details.csv")
    parser.add_argument("--output", default="data_freshness.json")
    parser.add_argument("--require-current", action="store_true")
    parser.add_argument("--grace-days", type=int, default=RESULT_GRACE_DAYS,
                        help="days a finished fight may await results before "
                             "the wait counts as a fault")
    parser.add_argument("--quarantine", action="store_true",
                        help="write off bookings no result is coming for, so "
                             "they stop failing every later run")
    parser.add_argument("--quarantine-log", default=QUARANTINE_LOG)
    parser.add_argument("--max-quarantine", type=int, default=MAX_AUTO_QUARANTINE,
                        help="refuse to auto-excuse more than this many at "
                             "once; a pile means something systemic broke")
    args = parser.parse_args()

    def assess():
        return assess_freshness(
            pd.read_csv(args.fights), args.odds_log,
            cancelled_fights=args.cancelled_fights,
            fighter_roster=args.fighter_roster,
            grace_days=args.grace_days,
            quarantine_log=args.quarantine_log,
        )

    report = assess()
    if args.quarantine and report["known_completed_missing"]:
        stuck = report["known_completed_missing"]
        if len(stuck) > args.max_quarantine:
            # Refusing here is the point of the lid. A handful of stragglers is
            # this problem's normal shape; a pile means results stopped
            # ingesting or names stopped matching, and quietly writing those
            # off would turn a broken pipeline into a clean-looking one.
            Path(args.output).write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
            raise SystemExit(
                f"{len(stuck)} bookings are unmatchable at once, over the "
                f"limit of {args.max_quarantine}. That is a broken feed rather "
                f"than a few stragglers; refusing to write them off.")
        written = quarantine(stuck, args.quarantine_log)
        print(f"quarantined {written} unmatchable booking(s); "
              f"they will not be awaited again")
        report = assess()

    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_current and report["status"] == "lagging":
        raise SystemExit("Completed tracked fights are missing from the result source.")


if __name__ == "__main__":
    main()
