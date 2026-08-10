"""Atomically refresh UFCStats source data and rebuild the fight table."""

import argparse
import glob
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import adapter
from data_quality import audit_fights


FILES = [
    "ufc_event_details.csv",
    "ufc_fight_results.csv",
    "ufc_fight_stats.csv",
    "ufc_fighter_tott.csv",
    "ufc_fighter_details.csv",
]
BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
MAX_RECOVERED_EVENT_DATES = 5


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "fight-ledger/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with Path(destination).open("wb") as output:
            shutil.copyfileobj(response, output)
        return {
            "url": url,
            "last_modified": response.headers.get("Last-Modified"),
            "etag": response.headers.get("ETag"),
        }


def _replace_from_staging(source, destination):
    """Copy into the target directory before replace so Windows inherits its ACL."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".refresh.tmp")
    with Path(source).open("rb") as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    os.replace(temporary, destination)


def _fighter_ids(frame):
    values = []
    for column in ("fighter_a_id", "fighter_b_id"):
        if column in frame:
            values.extend(frame[column].dropna().astype(str).str.strip())
    return {value for value in values if value}


def _fight_keys(frame):
    required = {"event", "fighter_a_id", "fighter_b_id"}
    if not required.issubset(frame.columns):
        return set()
    keys = set()
    for event, fighter_a_id, fighter_b_id in zip(
            frame["event"], frame["fighter_a_id"], frame["fighter_b_id"]):
        fighters = sorted((str(fighter_a_id).strip(), str(fighter_b_id).strip()))
        keys.add(f"{str(event).strip()}|{fighters[0]}|{fighters[1]}")
    return keys


def _dedupe_event_aliases(frame):
    """Collapse duplicate rows caused by upstream event-name aliases."""
    required = {"date", "fighter_a_id", "fighter_b_id", "winner"}
    if not required.issubset(frame.columns):
        return frame, 0
    out = frame.copy()
    out["_date_key"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["_pair_key"] = [
        "|".join(sorted((str(a).strip(), str(b).strip())))
        for a, b in zip(out["fighter_a_id"], out["fighter_b_id"])
    ]
    duplicate_groups = out.duplicated(["_date_key", "_pair_key"], keep=False)
    if duplicate_groups.any():
        conflicts = (
            out[duplicate_groups]
            .groupby(["_date_key", "_pair_key"])["winner"]
            .nunique(dropna=False)
        )
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            raise ValueError(
                "Data refresh rejected:\n- duplicate event aliases disagree on winners"
            )
    before = len(out)
    out = out.drop_duplicates(["_date_key", "_pair_key"], keep="first")
    return out.drop(columns=["_date_key", "_pair_key"]), before - len(out)


def _historical_event_dates(previous):
    if previous is None or not {"event", "date"}.issubset(previous.columns):
        return {}
    rows = previous[["event", "date"]].copy()
    rows["event"] = rows["event"].fillna("").astype(str).str.strip()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows[(rows["event"] != "") & rows["date"].notna()]
    counts = rows.groupby("event")["date"].nunique()
    unambiguous = set(counts[counts == 1].index)
    rows = rows[rows["event"].isin(unambiguous)].drop_duplicates("event")
    return dict(zip(rows["event"], rows["date"]))


def _event_date_recovery(previous, event_details, fight_results):
    """Recover a missing event date only from the prior validated fight table."""
    fallback = _historical_event_dates(previous)
    events = event_details.copy()
    events["EVENT"] = events["EVENT"].fillna("").astype(str).str.strip()
    events["date"] = pd.to_datetime(events["DATE"], format="mixed", errors="coerce")
    valid_events = set(events.loc[events["date"].notna(), "EVENT"])
    result_events = set(
        fight_results["EVENT"].fillna("").astype(str).str.strip()
    ) - {""}
    missing = result_events - valid_events
    recovered = {event: fallback[event] for event in sorted(missing) if event in fallback}
    unresolved = sorted(missing - set(recovered))
    return recovered, unresolved


def _regression_errors(new, old):
    errors = []
    if old is None or not len(old):
        return errors
    if len(new) < len(old):
        errors.append(f"fight rows shrank from {len(old)} to {len(new)}")
    new_max = pd.to_datetime(new["date"]).max()
    old_max = pd.to_datetime(old["date"]).max()
    if new_max < old_max:
        errors.append(f"latest result moved backward from {old_max.date()} to {new_max.date()}")
    old_ids = _fighter_ids(old)
    new_ids = _fighter_ids(new)
    missing_ids = sorted(old_ids - new_ids)
    if missing_ids:
        errors.append(
            f"refresh dropped {len(missing_ids)} historical fighter IDs: "
            + ", ".join(missing_ids[:5])
        )
    old_fights = _fight_keys(old)
    new_fights = _fight_keys(new)
    missing_fights = sorted(old_fights - new_fights)
    if missing_fights:
        errors.append(
            f"refresh dropped {len(missing_fights)} historical fights: "
            + "; ".join(missing_fights[:3])
        )
    return errors


def run(raw_dir="raw", output="fights_v2.csv", manifest="data_source_manifest.json"):
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(output, parse_dates=["date"]) if Path(output).exists() else None

    with tempfile.TemporaryDirectory(prefix="fight-ledger-refresh-") as directory:
        staging = Path(directory)
        sources = {}
        for filename in FILES:
            print(f"downloading {filename} ...")
            metadata = _download(BASE + filename, staging / filename)
            metadata["sha256"] = _sha256(staging / filename)
            metadata["bytes"] = int((staging / filename).stat().st_size)
            sources[filename] = metadata

        event_details = pd.read_csv(staging / "ufc_event_details.csv")
        fight_results = pd.read_csv(staging / "ufc_fight_results.csv")
        recovered_dates, unresolved_events = _event_date_recovery(
            previous, event_details, fight_results
        )
        if unresolved_events:
            raise ValueError(
                "Data refresh rejected:\n- results contain events without a valid date "
                "or prior validated fallback: " + ", ".join(unresolved_events[:5])
            )
        if len(recovered_dates) > MAX_RECOVERED_EVENT_DATES:
            raise ValueError(
                "Data refresh rejected:\n- refusing to recover "
                f"{len(recovered_dates)} missing event dates; limit is "
                f"{MAX_RECOVERED_EVENT_DATES}"
            )
        rebuilt = adapter.build(
            str(staging), fallback_event_dates=recovered_dates
        )
        rebuilt, deduped_alias_rows = _dedupe_event_aliases(rebuilt)
        errors = audit_fights(rebuilt) + _regression_errors(rebuilt, previous)
        if errors:
            raise ValueError("Data refresh rejected:\n- " + "\n- ".join(errors))

        staged_output = staging / "fights_v2.csv"
        rebuilt.to_csv(staged_output, index=False)
        report = {
            "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": BASE,
            "rows": int(len(rebuilt)),
            "result_date_min": str(pd.to_datetime(rebuilt["date"]).min().date()),
            "result_date_max": str(pd.to_datetime(rebuilt["date"]).max().date()),
            "fights_sha256": _sha256(staged_output),
            "recovered_event_dates": [
                {"event": event, "date": str(pd.Timestamp(date).date())}
                for event, date in recovered_dates.items()
            ],
            "deduped_event_alias_rows": int(deduped_alias_rows),
            "files": sources,
        }
        staged_manifest = staging / "data_source_manifest.json"
        staged_manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")

        for filename in FILES:
            _replace_from_staging(staging / filename, raw_path / filename)
        _replace_from_staging(staged_output, output)
        _replace_from_staging(staged_manifest, manifest)

    print(f"{output} rebuilt: {len(rebuilt)} fights through "
          f"{pd.to_datetime(rebuilt['date']).max().date()}")
    for event, date in recovered_dates.items():
        print(f"recovered missing upstream event date from prior validated data: "
              f"{event} ({pd.Timestamp(date).date()})")
    if report["deduped_event_alias_rows"]:
        print(f"deduped {report['deduped_event_alias_rows']} upstream event alias rows")
    for cache in glob.glob("cache_*.pkl"):
        os.remove(cache)
        print(f"cleared {cache}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--output", default="fights_v2.csv")
    parser.add_argument("--manifest", default="data_source_manifest.json")
    args = parser.parse_args()
    run(args.raw_dir, args.output, args.manifest)


if __name__ == "__main__":
    main()
