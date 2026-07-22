"""Build fight-level entry/close odds history from The Odds API quotes.

``historical_odds.py`` stores immutable per-book quote rows.  This command
turns those rows into the same fight-level shape used by
``validate_entry_history.py``, with a genuine later pre-card close proxy when
both snapshots exist for a fight.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest import american_to_prob
from identity import norm_name


def _pair(a, b):
    return "|".join(sorted((norm_name(a), norm_name(b))))


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


def _read_quotes(path):
    quotes = pd.read_csv(path)
    needed = {
        "event_uid", "event_name", "event_date", "snapshot_kind",
        "actual_snapshot", "event_start_utc", "api_event_id",
        "commence_time", "fighter_a", "fighter_b", "book_key",
        "odds_a", "odds_b",
    }
    missing = needed - set(quotes.columns)
    if missing:
        raise ValueError(f"missing quote columns: {sorted(missing)}")
    quotes = quotes[quotes["snapshot_kind"].isin(["entry", "close"])].copy()
    quotes["actual_snapshot"] = pd.to_datetime(quotes["actual_snapshot"], utc=True, errors="coerce")
    quotes["event_start_utc"] = pd.to_datetime(quotes["event_start_utc"], utc=True, errors="coerce")
    quotes["commence_time"] = pd.to_datetime(quotes["commence_time"], utc=True, errors="coerce")
    quotes["event_date"] = pd.to_datetime(quotes["event_date"], utc=True, errors="coerce")
    quotes["odds_a"] = pd.to_numeric(quotes["odds_a"], errors="coerce")
    quotes["odds_b"] = pd.to_numeric(quotes["odds_b"], errors="coerce")
    quotes = quotes.dropna(subset=[
        "actual_snapshot", "event_start_utc", "commence_time",
        "event_date", "fighter_a", "fighter_b", "book_key", "odds_a", "odds_b",
    ])
    quotes = quotes[
        (quotes["odds_a"].abs() >= 100) & (quotes["odds_b"].abs() >= 100)
    ].copy()
    quotes = quotes[quotes["actual_snapshot"] < quotes["commence_time"]].copy()
    quotes["pair"] = [_pair(a, b) for a, b in zip(quotes["fighter_a"], quotes["fighter_b"])]
    quotes = quotes.sort_values(["event_uid", "snapshot_kind", "pair", "book_key", "actual_snapshot"])
    quotes = quotes.drop_duplicates(
        ["event_uid", "snapshot_kind", "pair", "book_key"], keep="last"
    )
    quotes["book_prob_a"] = _devig_prob_a(quotes["odds_a"], quotes["odds_b"])
    return quotes


def _consensus(quotes, min_books):
    grouped = quotes.groupby(["event_uid", "event_name", "event_date", "snapshot_kind", "pair"])
    rows = grouped.agg(
        snapshot_ts=("actual_snapshot", "max"),
        commence_time=("commence_time", "min"),
        quote_fighter_a=("fighter_a", "first"),
        quote_fighter_b=("fighter_b", "first"),
        odds_a=("odds_a", "median"),
        odds_b=("odds_b", "median"),
        consensus_prob_a=("book_prob_a", "median"),
        n_books=("book_key", "nunique"),
    ).reset_index()
    rows = rows[rows["n_books"] >= min_books].copy()
    rows["lead_hours"] = (rows["commence_time"] - rows["snapshot_ts"]).dt.total_seconds() / 3600.0
    return rows[rows["lead_hours"] > 0].copy()


def _fight_table(path):
    fights = pd.read_csv(path, parse_dates=["date"])
    fights = fights.dropna(subset=["date", "event", "fighter_a", "fighter_b", "winner"]).copy()
    fights["date"] = pd.to_datetime(fights["date"], utc=True)
    fights["pair"] = [_pair(a, b) for a, b in zip(fights["fighter_a"], fights["fighter_b"])]
    return fights[["date", "event", "fighter_a", "fighter_b", "winner", "pair"]]


def run(args):
    quote_sha256 = _file_sha256(args.quotes)
    quotes = _read_quotes(args.quotes)
    consensus = _consensus(quotes, args.min_books)
    fights = _fight_table(args.fights)
    matched = consensus.merge(
        fights,
        left_on=["event_date", "pair"],
        right_on=["date", "pair"],
        how="inner",
        suffixes=("_api", ""),
    )
    matched["quote_orientation_matches_fights"] = [
        norm_name(q) == norm_name(f)
        for q, f in zip(matched["quote_fighter_a"], matched["fighter_a"])
    ]
    matched["actual_odds_a"] = matched["odds_a"]
    matched["actual_odds_b"] = matched["odds_b"]
    matched["actual_prob_a"] = matched["consensus_prob_a"]
    swapped = ~matched["quote_orientation_matches_fights"]
    matched.loc[swapped, ["actual_odds_a", "actual_odds_b"]] = (
        matched.loc[swapped, ["odds_b", "odds_a"]].to_numpy()
    )
    matched.loc[swapped, "actual_prob_a"] = 1.0 - matched.loc[
        swapped, "consensus_prob_a"
    ]

    identity = ["date", "event", "fighter_a", "fighter_b", "winner", "pair"]
    entry = matched[matched["snapshot_kind"].eq("entry")].rename(columns={
        "snapshot_ts": "entry_snapshot_ts",
        "lead_hours": "entry_lead_hours",
        "n_books": "entry_n_books",
        "actual_odds_a": "entry_odds_a",
        "actual_odds_b": "entry_odds_b",
        "actual_prob_a": "entry_prob_a",
    })
    close = matched[matched["snapshot_kind"].eq("close")].rename(columns={
        "snapshot_ts": "close_snapshot_ts",
        "lead_hours": "close_lead_hours",
        "n_books": "close_n_books",
        "actual_odds_a": "close_odds_a",
        "actual_odds_b": "close_odds_b",
        "actual_prob_a": "close_prob_a",
    })
    entry["entry_source"] = "odds_api_book_consensus"
    close["close_source"] = "odds_api_book_consensus"
    entry = entry[identity + [
        "commence_time", "entry_snapshot_ts", "entry_lead_hours",
        "entry_n_books", "entry_odds_a", "entry_odds_b",
        "entry_prob_a", "entry_source",
    ]]
    close = close[identity + [
        "close_snapshot_ts", "close_lead_hours", "close_n_books",
        "close_odds_a", "close_odds_b",
        "close_prob_a", "close_source",
    ]]
    dataset = entry.merge(close, on=identity, how="inner")
    dataset = dataset[
        (dataset["entry_snapshot_ts"] < dataset["commence_time"])
        & (dataset["close_snapshot_ts"] < dataset["commence_time"])
        & (dataset["close_snapshot_ts"] > dataset["entry_snapshot_ts"])
        & (dataset["entry_lead_hours"] >= args.min_entry_hours)
        & (dataset["close_lead_hours"] <= args.max_close_hours)
    ].copy()
    dataset = dataset.drop_duplicates(["date", "pair"], keep="last")
    dataset = dataset.sort_values(["date", "commence_time", "fighter_a"]).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    combined_rows = None
    if args.base_history:
        base = pd.read_csv(args.base_history)
        base["date"] = pd.to_datetime(base["date"], utc=True)
        for col in ["commence_time", "entry_snapshot_ts"]:
            base[col] = pd.to_datetime(base[col], utc=True, errors="coerce")
        base["pair"] = [_pair(a, b) for a, b in zip(base["fighter_a"], base["fighter_b"])]
        if "entry_prob_a" not in base:
            base["entry_prob_a"] = _devig_prob_a(
                base["entry_odds_a"], base["entry_odds_b"]
            )
        if "entry_source" not in base:
            base["entry_source"] = "archive_consensus"
        close_columns = [
            "close_snapshot_ts", "close_lead_hours", "close_n_books",
            "close_odds_a", "close_odds_b", "close_prob_a",
            "close_source",
        ]
        for column in close_columns:
            if column not in base:
                base[column] = pd.NA
        base["close_snapshot_ts"] = pd.to_datetime(
            base["close_snapshot_ts"], utc=True, errors="coerce"
        )
        api_cols = [
            "date", "pair", "commence_time", "entry_snapshot_ts", "entry_lead_hours",
            "entry_n_books", "entry_odds_a", "entry_odds_b", "entry_prob_a",
            "entry_source",
            "close_snapshot_ts", "close_lead_hours", "close_n_books",
            "close_odds_a", "close_odds_b", "close_prob_a", "close_source",
        ]
        combined = base.merge(
            dataset[api_cols],
            on=["date", "pair"],
            how="left",
            suffixes=("", "_api"),
        )
        for col in ["commence_time", "entry_snapshot_ts", "entry_lead_hours",
                    "entry_n_books", "entry_odds_a", "entry_odds_b",
                    "entry_prob_a", "entry_source"]:
            api_col = f"{col}_api"
            combined[col] = combined[api_col].combine_first(combined[col])
            combined = combined.drop(columns=[api_col])
        combined["commence_time"] = pd.to_datetime(
            combined["commence_time"], utc=True, errors="coerce"
        )
        combined["entry_snapshot_ts"] = pd.to_datetime(
            combined["entry_snapshot_ts"], utc=True, errors="coerce"
        )
        api_close_time = pd.to_datetime(
            combined["close_snapshot_ts_api"], utc=True, errors="coerce"
        )
        base_close_time = pd.to_datetime(
            combined["close_snapshot_ts"], utc=True, errors="coerce"
        )
        base_close_lead = (
            combined["commence_time"] - base_close_time
        ).dt.total_seconds() / 3600.0
        api_close_lead = (
            combined["commence_time"] - api_close_time
        ).dt.total_seconds() / 3600.0
        base_close_valid = (
            (base_close_time > combined["entry_snapshot_ts"])
            & (base_close_time < combined["commence_time"])
            & (base_close_lead <= args.max_close_hours)
        )
        api_close_valid = (
            (api_close_time > combined["entry_snapshot_ts"])
            & (api_close_time < combined["commence_time"])
            & (api_close_lead <= args.max_close_hours)
        )
        use_api_close = api_close_valid & (
            ~base_close_valid | (api_close_time > base_close_time)
        )
        for column in close_columns:
            api_column = f"{column}_api"
            combined[column] = combined[column].where(base_close_valid)
            combined.loc[use_api_close, column] = combined.loc[
                use_api_close, api_column
            ].to_numpy()
            combined = combined.drop(columns=[api_column])
        combined["entry_lead_hours"] = (
            (combined["commence_time"] - combined["entry_snapshot_ts"])
            .dt.total_seconds() / 3600.0
        )
        combined["close_snapshot_ts"] = pd.to_datetime(
            combined["close_snapshot_ts"], utc=True, errors="coerce"
        )
        combined["close_lead_hours"] = (
            (combined["commence_time"] - combined["close_snapshot_ts"])
            .dt.total_seconds() / 3600.0
        )
        combined = combined.drop(columns=["pair"])
        combined_output = Path(args.combined_output)
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_output, index=False)
        combined_rows = int(len(combined))
        combined_close_rows = int(combined["close_snapshot_ts"].notna().sum())
    else:
        combined_close_rows = None
    report = {
        "quotes_sha256": quote_sha256,
        "quote_rows": int(len(quotes)),
        "consensus_snapshot_fights": int(len(consensus)),
        "entry_close_fights": int(len(dataset)),
        "events": int(dataset["date"].nunique()) if len(dataset) else 0,
        "minimum_books": args.min_books,
        "minimum_entry_lead_hours": args.min_entry_hours,
        "maximum_close_proxy_lead_hours": args.max_close_hours,
        "median_entry_lead_hours": float(dataset["entry_lead_hours"].median()) if len(dataset) else None,
        "median_close_lead_minutes": float(dataset["close_lead_hours"].median() * 60.0) if len(dataset) else None,
        "probability_consensus": "median of per-book de-vigged probabilities",
        "close_definition": "latest API snapshot at or before the requested pre-card close time",
        "output": str(output),
        "combined_output": str(args.combined_output) if args.base_history else None,
        "combined_rows": combined_rows,
        "combined_close_proxy_fights": combined_close_rows,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="raw/odds_api_historical/historical_h2h_quotes.csv")
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--output", default="data/odds_history/odds_history_api_entry_close.csv")
    parser.add_argument(
        "--base-history",
        default="data/odds_history/odds_history_entry_with_close.csv",
    )
    parser.add_argument("--combined-output", default="data/odds_history/odds_history_entry_with_api_close.csv")
    parser.add_argument("--report", default="data/odds_history/odds_history_api_entry_close_audit.json")
    parser.add_argument("--min-books", type=int, default=3)
    parser.add_argument("--min-entry-hours", type=float, default=24.0)
    parser.add_argument("--max-close-hours", type=float, default=12.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
