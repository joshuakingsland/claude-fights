"""Prepare a sparse historical MMA odds archive for point-in-time research.

The source archive contains consensus two-sided quotes, not individual-book
quotes.  It is therefore an input to market-model and CLV research, not a
best-price or bookmaker-disagreement study.  This command never changes the
production dataset; it creates a separately auditable entry/close dataset.

Usage:
    python prepare_odds_history.py C:\\path\\to\\odds-history.zip
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from backtest import american_to_prob
from identity import norm_name


REQUIRED = {"snapshot_ts", "commence_time", "fighter_a", "fighter_b",
            "odds_a", "odds_b", "n_books"}


def _pair(a, b):
    return "|".join(sorted((norm_name(a), norm_name(b))))


def _fight_id(date, event, a, b):
    raw = f"{pd.Timestamp(date).date()}|{event}|{_pair(a, b)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _devig_prob_a(odds_a, odds_b):
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    return pa / (pa + pb)


def _read_source(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("zip must contain exactly one CSV")
            return pd.read_csv(archive.open(names[0]))
    return pd.read_csv(path)


def run(args):
    input_sha256 = _file_sha256(args.input)
    source = _read_source(args.input)
    if not REQUIRED <= set(source.columns):
        raise ValueError(f"missing source columns: {sorted(REQUIRED - set(source.columns))}")
    raw_count = len(source)
    source = source.copy()
    source["snapshot_ts"] = pd.to_datetime(source["snapshot_ts"], utc=True, errors="coerce")
    source["commence_time"] = pd.to_datetime(source["commence_time"], utc=True, errors="coerce")
    source["odds_a"] = pd.to_numeric(source["odds_a"], errors="coerce")
    source["odds_b"] = pd.to_numeric(source["odds_b"], errors="coerce")
    source["n_books"] = pd.to_numeric(source["n_books"], errors="coerce")
    source = source.dropna(subset=["snapshot_ts", "commence_time", "fighter_a",
                                  "fighter_b", "odds_a", "odds_b", "n_books"])
    source = source[(source["odds_a"] != 0) & (source["odds_b"] != 0)].copy()
    source["pair"] = [_pair(a, b) for a, b in zip(source["fighter_a"], source["fighter_b"])]
    source["source_row"] = range(len(source))

    fights = pd.read_csv(args.fights, parse_dates=["date"])
    fights = fights.dropna(subset=["date", "event", "fighter_a", "fighter_b", "winner"]).copy()
    fights["date_utc"] = pd.to_datetime(fights["date"], utc=True)
    fights["pair"] = [_pair(a, b) for a, b in zip(fights["fighter_a"], fights["fighter_b"])]
    fights["fight_id"] = [_fight_id(d, e, a, b)
                          for d, e, a, b in zip(fights["date"], fights["event"],
                                                  fights["fighter_a"], fights["fighter_b"])]

    merged = source.merge(
        fights[["fight_id", "date_utc", "event", "fighter_a", "fighter_b", "winner", "pair"]],
        on="pair", how="inner", suffixes=("_source", "_actual"),
    )
    merged["date_error_days"] = (merged["commence_time"] - merged["date_utc"]).abs().dt.total_seconds() / 86400
    merged = merged[merged["date_error_days"] <= args.date_tolerance_days].copy()
    # A same-name pair might appear more than once in the source's anticipated
    # schedule.  Keep the result date closest to the listing, never an arbitrary
    # join result.
    merged = merged.sort_values(["source_row", "date_error_days", "date_utc"])
    merged = merged.drop_duplicates("source_row", keep="first").copy()
    merged["lead_hours"] = (merged["commence_time"] - merged["snapshot_ts"]).dt.total_seconds() / 3600
    post_start = int((merged["lead_hours"] <= 0).sum())
    clean = merged[merged["lead_hours"] > 0].copy()
    clean["source_orientation_matches_actual"] = [
        norm_name(a) == norm_name(b)
        for a, b in zip(clean["fighter_a_source"], clean["fighter_a_actual"])
    ]
    clean["actual_odds_a"] = clean["odds_a"]
    clean["actual_odds_b"] = clean["odds_b"]
    swapped = ~clean["source_orientation_matches_actual"]
    clean.loc[swapped, ["actual_odds_a", "actual_odds_b"]] = \
        clean.loc[swapped, ["odds_b", "odds_a"]].to_numpy()
    clean["actual_prob_a"] = _devig_prob_a(
        clean["actual_odds_a"], clean["actual_odds_b"]
    )

    keep = [
        "fight_id", "date_utc", "event", "fighter_a_actual", "fighter_b_actual", "winner",
        "snapshot_ts", "commence_time", "lead_hours", "n_books",
        "actual_odds_a", "actual_odds_b", "actual_prob_a",
        "fighter_a_source", "fighter_b_source",
        "date_error_days", "source_orientation_matches_actual",
    ]
    clean = clean[keep].rename(columns={
        "date_utc": "date", "fighter_a_actual": "fighter_a", "fighter_b_actual": "fighter_b",
    })
    clean = clean.sort_values(["fight_id", "snapshot_ts"]).reset_index(drop=True)

    eligible = clean[clean["n_books"] >= args.min_books].copy()
    def select_at_or_before(hours, prefix):
        selected = eligible[eligible["lead_hours"] >= hours].copy()
        selected = selected.sort_values(["fight_id", "snapshot_ts"])
        selected = selected.groupby("fight_id", as_index=False).tail(1).copy()
        return selected.rename(columns={
            "snapshot_ts": f"{prefix}_snapshot_ts",
            "lead_hours": f"{prefix}_lead_hours",
            "n_books": f"{prefix}_n_books",
            "actual_odds_a": f"{prefix}_odds_a",
            "actual_odds_b": f"{prefix}_odds_b",
            "actual_prob_a": f"{prefix}_prob_a",
        })

    entry = select_at_or_before(args.entry_hours, "entry")
    close = select_at_or_before(args.close_minutes / 60.0, "close")
    entry["entry_source"] = "archive_consensus"
    close["close_source"] = "archive_consensus"
    identity = ["fight_id", "date", "event", "fighter_a", "fighter_b", "winner", "commence_time"]
    entry = entry[identity + ["entry_snapshot_ts", "entry_lead_hours", "entry_n_books",
                              "entry_odds_a", "entry_odds_b", "entry_prob_a",
                              "entry_source"]]
    close = close[["fight_id", "close_snapshot_ts", "close_lead_hours", "close_n_books",
                   "close_odds_a", "close_odds_b", "close_prob_a", "close_source"]]
    with_close = entry.merge(close, on="fight_id", how="inner")
    # Hard invariant: a close is a benchmark, never a model input. Both lines
    # must precede the scheduled start, and a genuine close must be later than
    # the entry observation. Sparse archives commonly have only one snapshot.
    valid_entry_time = with_close["entry_snapshot_ts"] < with_close["commence_time"]
    valid_close_time = (
        (with_close["close_snapshot_ts"] < with_close["commence_time"])
        & (with_close["close_lead_hours"] <= args.max_close_hours)
    )
    valid_order = with_close["close_snapshot_ts"] > with_close["entry_snapshot_ts"]
    invalid_close_pairs = int((~(valid_entry_time & valid_close_time & valid_order)).sum())
    dataset = with_close[valid_entry_time & valid_close_time & valid_order].copy()
    close_columns = [
        "fight_id", "close_snapshot_ts", "close_lead_hours", "close_n_books",
        "close_odds_a", "close_odds_b", "close_prob_a",
        "close_source",
    ]
    combined = entry.merge(dataset[close_columns], on="fight_id", how="left")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source.to_csv(output / "odds_history_source.csv", index=False)
    clean.to_csv(output / "odds_history_clean.csv", index=False)
    entry.to_csv(output / "odds_history_entry.csv", index=False)
    dataset.to_csv(output / "odds_history_entry_close.csv", index=False)
    combined.to_csv(output / "odds_history_entry_with_close.csv", index=False)
    report = {
        "input_sha256": input_sha256,
        "input_rows": int(raw_count),
        "valid_source_rows": int(len(source)),
        "matched_pre_fight_rows": int(len(clean)),
        "unique_matched_fights": int(clean["fight_id"].nunique()),
        "source_post_start_rows": int((source["snapshot_ts"] >= source["commence_time"]).sum()),
        "matched_post_start_rows_dropped": post_start,
        "minimum_books": args.min_books,
        "entry_hours_before_start": args.entry_hours,
        "close_minutes_before_start": args.close_minutes,
        "maximum_close_proxy_lead_hours": args.max_close_hours,
        "entry_fights": int(entry["fight_id"].nunique()),
        "entry_and_later_close_fights": int(dataset["fight_id"].nunique()),
        "invalid_entry_close_pairs_dropped": invalid_close_pairs,
        "entry_median_lead_hours": float(entry["entry_lead_hours"].median()) if len(entry) else None,
        "close_median_lead_hours": float(dataset["close_lead_hours"].median()) if len(dataset) else None,
        "matched_date_min": str(entry["date"].min()) if len(entry) else None,
        "matched_date_max": str(entry["date"].max()) if len(entry) else None,
        "note": ("Consensus source only: individual bookmaker prices are not available. "
                 "Probabilities are de-vigged from each consensus quote. A close is emitted "
                 "only when it is later than the chosen entry snapshot."),
    }
    (output / "odds_history_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote cleaned history to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--output-dir", default="data/odds_history")
    parser.add_argument("--entry-hours", type=float, default=24.0)
    parser.add_argument("--close-minutes", type=float, default=15.0)
    parser.add_argument("--max-close-hours", type=float, default=12.0)
    parser.add_argument("--min-books", type=int, default=3)
    parser.add_argument("--date-tolerance-days", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
