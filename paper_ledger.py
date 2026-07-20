"""Append-only paper-trading ledger.

Predictions and settlements are separate append-only files.  A rerun never
rewrites history, which makes retrospective removal of losing bets visible.
This is deliberately paper trading: it does not place or transmit wagers.
"""

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import american_payout, norm_name

TRADE_FIELDS = ["trade_id", "recorded_at", "date", "pick", "opp", "price",
                "market", "model", "edge", "se", "net_edge", "bet",
                "stake", "meta"]
SETTLEMENT_FIELDS = ["trade_id", "settled_at", "result", "pnl"]


def _read(path, fields):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=fields)


def record_predictions(predictions, path="paper_trades.csv", recorded_at=None):
    """Append one row per displayed prediction and return rows added."""
    stamp = recorded_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = _read(path, TRADE_FIELDS)
    known = set(existing.get("trade_id", pd.Series(dtype=str)).astype(str))
    rows = []
    for item in predictions:
        date = str(item.get("date", ""))[:10]
        raw = "|".join(str(item.get(k, "")) for k in
                       ("pick", "opp", "price", "model", "net"))
        trade_id = hashlib.sha256(f"{stamp}|{date}|{raw}".encode()).hexdigest()[:20]
        if trade_id in known:
            continue
        rows.append({
            "trade_id": trade_id, "recorded_at": stamp, "date": date,
            "pick": item.get("pick", ""), "opp": item.get("opp", ""),
            "price": item.get("price", ""), "market": item.get("market", ""),
            "model": item.get("model", ""), "edge": item.get("edge", ""),
            "se": item.get("se", ""), "net_edge": item.get("net", ""),
            "bet": bool(item.get("bet", False)), "stake": int(item.get("stake", 0)),
            "meta": item.get("meta", ""),
        })
    if not rows:
        return 0
    write_header = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def settle_completed(trades_path="paper_trades.csv", settlements_path="paper_settlements.csv",
                     fights_path="fights_v2.csv"):
    """Append settlements for completed fights without modifying predictions."""
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
    rows = []
    for row in trades.itertuples():
        if str(row.trade_id) in done:
            continue
        date = str(row.date)[:10]
        pair = frozenset((norm_name(row.pick), norm_name(row.opp)))
        fight = by_key.get((date, pair))
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
        rows.append({"trade_id": row.trade_id,
                     "settled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "result": "WIN" if won else "LOSS", "pnl": round(pnl, 6)})
    if not rows:
        return 0
    write_header = not Path(settlements_path).exists()
    with open(settlements_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SETTLEMENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["settle", "summary"])
    ap.add_argument("--trades", default="paper_trades.csv")
    ap.add_argument("--settlements", default="paper_settlements.csv")
    ap.add_argument("--fights", default="fights_v2.csv")
    args = ap.parse_args()
    if args.command == "settle":
        print(f"settled {settle_completed(args.trades, args.settlements, args.fights)} trades")
        return
    trades = _read(args.trades, TRADE_FIELDS)
    settlements = _read(args.settlements, SETTLEMENT_FIELDS)
    print({"predictions": len(trades), "settled": len(settlements),
           "pnl": round(float(settlements["pnl"].sum()), 4)
           if len(settlements) else 0.0})


if __name__ == "__main__":
    main()
