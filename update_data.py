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
    old_ids = set(old.get("fighter_a_id", [])) | set(old.get("fighter_b_id", []))
    new_ids = set(new["fighter_a_id"]) | set(new["fighter_b_id"])
    if old_ids and not old_ids.issubset(new_ids):
        errors.append(f"refresh dropped {len(old_ids - new_ids)} historical fighter IDs")
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

        rebuilt = adapter.build(str(staging))
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
