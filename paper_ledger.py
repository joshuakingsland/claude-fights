"""Immutable forward-test ledgers for prediction snapshots and paper wagers.

``prediction_snapshots.csv`` records every pre-event model run.  The separate
``paper_trades.csv`` records at most one official qualifying wager per fight,
only when the caller explicitly requests a lock.  Settlements are append-only
and reject any row whose prediction timing cannot be verified as pre-event.

This module never places or transmits a real wager.
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import american_payout, american_to_prob, norm_name
from config import MODEL_VERSION, STAKING_POLICY_VERSION

SNAPSHOT_FIELDS = [
    "snapshot_id", "recorded_at", "scheduled_start", "timing_precision",
    "date", "fight_key", "pick", "opp", "price", "market", "model",
    "edge", "se", "net_edge", "bet", "stake", "meta", "model_version",
    "manifest_hash", "odds_source", "odds_fetched_at", "consensus_price",
    "consensus_opp_price", "execution_price", "execution_book",
    "execution_implied", "market_books", "market_spread",
    "eligibility_reason", "staking_policy",
]
TRADE_FIELDS = [
    "trade_id", "snapshot_id", "locked_at", "scheduled_start",
    "timing_precision", "date", "fight_key", "pick", "opp", "price",
    "market", "model", "edge", "se", "net_edge", "stake", "meta",
    "model_version", "manifest_hash", "odds_source", "odds_fetched_at",
    "consensus_price", "consensus_opp_price", "execution_price",
    "execution_book", "execution_implied", "market_books", "market_spread",
    "eligibility_reason", "staking_policy",
]
SETTLEMENT_FIELDS = [
    "trade_id", "settled_at", "result", "pnl", "closing_price",
    "closing_market", "clv_prob", "closing_source",
]


def _utc(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _now_stamp(now=None):
    value = datetime.now(timezone.utc) if now is None else now
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path, fields):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=fields)


def scheduled_start(item):
    """Return (UTC timestamp, precision) from a prediction/input row.

    Exact ``commence_time``/``scheduled_start`` timestamps are preferred.  A
    date-only row is conservatively treated as beginning at 00:00 UTC, which
    means it cannot be recorded on the event date itself.
    """
    exact = None
    for key in ("scheduled_start", "commence_time"):
        value = item.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            exact = value
            break
    if exact is not None:
        return _utc(exact), "exact"
    date = str(item.get("date", ""))[:10]
    if not date:
        raise ValueError("Prediction row is missing scheduled_start/commence_time and date")
    return _utc(f"{date}T00:00:00Z"), "date_only"


def assert_pre_event(predictions, recorded_at=None):
    """Fail closed unless every row is demonstrably recorded pre-event."""
    stamp = _utc(recorded_at or datetime.now(timezone.utc))
    expired = []
    for item in predictions:
        start, precision = scheduled_start(item)
        if stamp >= start:
            expired.append(
                f"{item.get('pick', item.get('fighter_a', '?'))} vs "
                f"{item.get('opp', item.get('fighter_b', '?'))} "
                f"({start.isoformat()}, {precision})"
            )
    if expired:
        preview = "; ".join(expired[:5])
        more = f"; +{len(expired) - 5} more" if len(expired) > 5 else ""
        raise ValueError(
            "Refusing retroactive prediction rows. Supply an exact future "
            f"commence_time or a future event date: {preview}{more}"
        )


def _fight_key(item):
    start, _ = scheduled_start(item)
    pair = sorted((norm_name(item.get("pick", "")), norm_name(item.get("opp", ""))))
    raw = f"{start.date()}|{'|'.join(pair)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _provenance(provenance=None):
    provenance = provenance or {}
    return {
        "model_version": provenance.get("model_version", MODEL_VERSION),
        "manifest_hash": provenance.get("manifest_hash", ""),
    }


def _snapshot_row(item, stamp, provenance):
    start, precision = scheduled_start(item)
    fight_key = _fight_key(item)
    raw = "|".join(str(item.get(k, "")) for k in (
        "pick", "opp", "price", "execution_book", "model", "net", "stake",
    ))
    snapshot_id = hashlib.sha256(
        f"{stamp}|{fight_key}|{raw}".encode()
    ).hexdigest()[:20]
    return {
        "snapshot_id": snapshot_id,
        "recorded_at": stamp,
        "scheduled_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timing_precision": precision,
        "date": str(start.date()),
        "fight_key": fight_key,
        "pick": item.get("pick", ""),
        "opp": item.get("opp", ""),
        "price": item.get("price", ""),
        "market": item.get("market", ""),
        "model": item.get("model", ""),
        "edge": item.get("edge", ""),
        "se": item.get("se", ""),
        "net_edge": item.get("net", ""),
        "bet": bool(item.get("bet", False)),
        "stake": int(item.get("stake", 0)),
        "meta": item.get("meta", ""),
        "model_version": provenance["model_version"],
        "manifest_hash": provenance["manifest_hash"],
        "odds_source": item.get("odds_source", "manual_or_unknown"),
        "odds_fetched_at": item.get("odds_fetched_at", ""),
        "consensus_price": item.get("consensus_price", item.get("price", "")),
        "consensus_opp_price": item.get("consensus_opp_price", ""),
        "execution_price": item.get("execution_price", item.get("price", "")),
        "execution_book": item.get("execution_book", "consensus"),
        "execution_implied": item.get("execution_implied", ""),
        "market_books": item.get("market_books", ""),
        "market_spread": item.get("market_spread", ""),
        "eligibility_reason": item.get("eligibility_reason", ""),
        "staking_policy": STAKING_POLICY_VERSION,
    }


def _append_rows(path, fields, rows):
    if not rows:
        return 0
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            existing = list(reader)
            existing_fields = reader.fieldnames or []
        if existing_fields != fields:
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing)
            os.replace(temporary, path)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def record_prediction_snapshots(predictions, path="prediction_snapshots.csv",
                                recorded_at=None, provenance=None):
    """Append every verified pre-event model snapshot."""
    if not predictions:
        return 0
    stamp = _now_stamp(recorded_at)
    assert_pre_event(predictions, stamp)
    prov = _provenance(provenance)
    existing = _read(path, SNAPSHOT_FIELDS)
    known = set(existing.get("snapshot_id", pd.Series(dtype=str)).astype(str))
    rows = []
    for item in predictions:
        row = _snapshot_row(item, stamp, prov)
        if row["snapshot_id"] not in known:
            rows.append(row)
            known.add(row["snapshot_id"])
    return _append_rows(path, SNAPSHOT_FIELDS, rows)


def lock_paper_trades(predictions, snapshots_path="prediction_snapshots.csv",
                      trades_path="paper_trades.csv", locked_at=None,
                      provenance=None):
    """Lock at most one official qualifying paper wager per fight.

    The caller must opt into this action (the scheduled workflow does so only
    on Wednesday).  Repeated runs are idempotent by ``fight_key``.
    """
    qualifying = [p for p in predictions
                  if bool(p.get("bet", False)) and int(p.get("stake", 0)) > 0]
    if not qualifying:
        return 0
    stamp = _now_stamp(locked_at)
    assert_pre_event(qualifying, stamp)
    prov = _provenance(provenance)
    snapshots = _read(snapshots_path, SNAPSHOT_FIELDS)
    trades = _read(trades_path, TRADE_FIELDS)
    locked_fights = set(trades.get("fight_key", pd.Series(dtype=str)).astype(str))
    rows = []
    for item in qualifying:
        snap = _snapshot_row(item, stamp, prov)
        fight_key = snap["fight_key"]
        if fight_key in locked_fights:
            continue
        # The snapshot should normally have just been appended.  Fall back to
        # the deterministic current snapshot id if a caller invokes lock alone.
        snapshot_id = snap["snapshot_id"]
        if len(snapshots) and fight_key in set(snapshots["fight_key"].astype(str)):
            match = snapshots[snapshots["fight_key"].astype(str) == fight_key]
            snapshot_id = str(match.iloc[-1]["snapshot_id"])
        trade_id = hashlib.sha256(f"official|{fight_key}".encode()).hexdigest()[:20]
        rows.append({
            "trade_id": trade_id,
            "snapshot_id": snapshot_id,
            "locked_at": stamp,
            "scheduled_start": snap["scheduled_start"],
            "timing_precision": snap["timing_precision"],
            "date": snap["date"],
            "fight_key": fight_key,
            "pick": snap["pick"],
            "opp": snap["opp"],
            "price": snap["price"],
            "market": snap["market"],
            "model": snap["model"],
            "edge": snap["edge"],
            "se": snap["se"],
            "net_edge": snap["net_edge"],
            "stake": snap["stake"],
            "meta": snap["meta"],
            "model_version": snap["model_version"],
            "manifest_hash": snap["manifest_hash"],
            "odds_source": snap["odds_source"],
            "odds_fetched_at": snap["odds_fetched_at"],
            "consensus_price": snap["consensus_price"],
            "consensus_opp_price": snap["consensus_opp_price"],
            "execution_price": snap["execution_price"],
            "execution_book": snap["execution_book"],
            "execution_implied": snap["execution_implied"],
            "market_books": snap["market_books"],
            "market_spread": snap["market_spread"],
            "eligibility_reason": snap["eligibility_reason"],
            "staking_policy": snap["staking_policy"],
        })
        locked_fights.add(fight_key)
    return _append_rows(trades_path, TRADE_FIELDS, rows)


def _timing_is_valid(row):
    try:
        locked = _utc(row.locked_at)
        start = _utc(row.scheduled_start)
        return locked < start
    except Exception:
        return False


def _closing_lookup(master_path):
    if not Path(master_path).exists():
        return {}
    closing = pd.read_csv(master_path, low_memory=False)
    required = {"date", "R_fighter", "B_fighter", "R_odds", "B_odds"}
    if not required.issubset(closing.columns):
        return {}
    closing = closing.dropna(subset=["date", "R_odds", "B_odds"]).copy()
    closing["date"] = pd.to_datetime(closing["date"], errors="coerce")
    closing["pair"] = [frozenset((norm_name(a), norm_name(b)))
                       for a, b in zip(closing["R_fighter"], closing["B_fighter"])]
    return {(str(r.date.date()), r.pair): r for r in closing.itertuples()
            if pd.notna(r.date)}


def _captured_closing_lookup(path):
    if not Path(path).exists():
        return {}
    captured = pd.read_csv(path)
    required = {
        "captured_at", "commence_time", "fighter_a", "fighter_b",
        "odds_a", "odds_b", "market_prob_a",
    }
    if not required.issubset(captured.columns):
        return {}
    captured["captured_at"] = pd.to_datetime(
        captured["captured_at"], errors="coerce", utc=True
    )
    captured["commence_time"] = pd.to_datetime(
        captured["commence_time"], errors="coerce", utc=True
    )
    captured = captured[
        captured["captured_at"].notna()
        & captured["commence_time"].notna()
        & (captured["captured_at"] < captured["commence_time"])
    ].copy()
    captured["date"] = captured["commence_time"].dt.date.astype(str)
    captured["pair"] = [
        frozenset((norm_name(a), norm_name(b)))
        for a, b in zip(captured["fighter_a"], captured["fighter_b"])
    ]
    captured = captured.sort_values("captured_at").drop_duplicates(
        ["date", "pair"], keep="last"
    )
    return {(row.date, row.pair): row for row in captured.itertuples()}


def _near_date(lookup, date, pair):
    stamp = pd.Timestamp(date)
    for delta in (0, -1, 1):
        key = (str((stamp + pd.Timedelta(days=delta)).date()), pair)
        if key in lookup:
            return lookup[key]
    return None


def settle_completed(trades_path="paper_trades.csv",
                     settlements_path="paper_settlements.csv",
                     fights_path="fights_v2.csv",
                     closing_path="raw/ufc-master.csv",
                     captured_closing_path="close_snapshots.csv"):
    """Append settlements for valid, pre-event official paper trades."""
    trades = _read(trades_path, TRADE_FIELDS)
    if not len(trades):
        return 0
    settled = _read(settlements_path, SETTLEMENT_FIELDS)
    done = set(settled.get("trade_id", pd.Series(dtype=str)).astype(str))
    fights = pd.read_csv(fights_path, parse_dates=["date"])
    fights["pair"] = [frozenset((norm_name(a), norm_name(b)))
                      for a, b in zip(fights["fighter_a"], fights["fighter_b"])]
    fights = fights[fights["winner"].isin(["A", "B"])].copy()
    by_key = {(str(pd.Timestamp(r.date).date()), r.pair): r
              for r in fights.itertuples()}
    closing = _closing_lookup(closing_path)
    captured_closing = _captured_closing_lookup(captured_closing_path)
    rows = []
    invalid = 0
    for row in trades.itertuples():
        if str(row.trade_id) in done:
            continue
        if not _timing_is_valid(row):
            invalid += 1
            continue
        date = str(row.date)[:10]
        pair = frozenset((norm_name(row.pick), norm_name(row.opp)))
        fight = _near_date(by_key, date, pair)
        if fight is None:
            continue
        pick_is_a = norm_name(row.pick) == norm_name(fight.fighter_a)
        won = (fight.winner == "A") == pick_is_a
        stake = float(row.stake)
        try:
            price = float(str(row.price).replace("+", ""))
            pnl = stake * (float(american_payout(price)) if won else -1.0)
        except (TypeError, ValueError):
            pnl = 0.0
            won = False

        closing_price = closing_market = clv = ""
        close = _near_date(captured_closing, date, pair)
        close_source = "standardized-t30" if close is not None else ""
        if close is None:
            close = _near_date(closing, date, pair)
            close_source = "ufc-master" if close is not None else ""
        if close is not None:
            if close_source == "standardized-t30":
                pick_is_red = norm_name(row.pick) == norm_name(close.fighter_a)
                close_price = float(close.odds_a if pick_is_red else close.odds_b)
                probability_a = float(close.market_prob_a)
                close_market = (probability_a if pick_is_red
                                else 1.0 - probability_a) * 100.0
            else:
                pick_is_red = norm_name(row.pick) == norm_name(close.R_fighter)
                close_price = float(close.R_odds if pick_is_red else close.B_odds)
                pr = float(american_to_prob(float(close.R_odds)))
                pb = float(american_to_prob(float(close.B_odds)))
                close_market = (pr if pick_is_red else pb) / (pr + pb) * 100.0
            try:
                captured_market = float(row.market)
                clv = round(close_market - captured_market, 6)
            except (TypeError, ValueError):
                clv = ""
            closing_price = f"{int(close_price):+d}"
            closing_market = round(close_market, 6)

        rows.append({
            "trade_id": row.trade_id,
            "settled_at": _now_stamp(),
            "result": "WIN" if won else "LOSS",
            "pnl": round(pnl, 6),
            "closing_price": closing_price,
            "closing_market": closing_market,
            "clv_prob": clv,
            "closing_source": close_source,
        })
    if invalid:
        print(f"skipped {invalid} trade(s) with unverifiable pre-event timing")
    return _append_rows(settlements_path, SETTLEMENT_FIELDS, rows)


def summary(trades_path="paper_trades.csv", settlements_path="paper_settlements.csv"):
    trades = _read(trades_path, TRADE_FIELDS)
    settlements = _read(settlements_path, SETTLEMENT_FIELDS)
    settled = settlements.merge(trades[["trade_id", "stake"]], on="trade_id", how="left") \
        if len(settlements) and len(trades) else pd.DataFrame()
    staked = float(settled["stake"].sum()) if len(settled) else 0.0
    pnl = float(settled["pnl"].sum()) if len(settled) else 0.0
    clv = pd.to_numeric(settled.get("clv_prob", pd.Series(dtype=float)), errors="coerce")
    return {
        "official_trades": int(len(trades)),
        "settled": int(len(settlements)),
        "staked": round(staked, 4),
        "pnl": round(pnl, 4),
        "roi": round(pnl / staked, 6) if staked else None,
        "mean_clv_prob_points": round(float(clv.mean()), 4) if clv.notna().any() else None,
        "positive_clv_rate": round(float((clv > 0).mean()), 4) if clv.notna().any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["settle", "summary"])
    ap.add_argument("--trades", default="paper_trades.csv")
    ap.add_argument("--settlements", default="paper_settlements.csv")
    ap.add_argument("--fights", default="fights_v2.csv")
    ap.add_argument("--closing", default="raw/ufc-master.csv")
    ap.add_argument("--captured-closing", default="close_snapshots.csv")
    args = ap.parse_args()
    if args.command == "settle":
        count = settle_completed(args.trades, args.settlements, args.fights,
                                 args.closing, args.captured_closing)
        print(f"settled {count} official paper trades")
        return
    print(json.dumps(summary(args.trades, args.settlements), indent=2))


if __name__ == "__main__":
    main()
