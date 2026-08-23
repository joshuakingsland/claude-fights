"""Alert when the model locks a paper trade.

Until now the only thing that sent a notification was a broken workflow. A
fight clearing the edge rule happened silently, which is backwards for
something whose whole output is "this fight is worth a look".

Two things this deliberately does not do.

**It does not tell you to bet.** The rule that flags these fights was tested
and failed: H1 in PREREGISTRATION.md, and the ledger it has produced since is
negative on both money and closing-line value. Every notice carries that
record, because an alert that reads like a recommendation would be the single
most harmful thing this repository could send. What it is actually reporting is
that a forward test recorded a new row.

**It does not notify twice.** The snapshot job runs every six hours and the
card is regenerated each time, so a locked trade would otherwise be announced
until the fight happened. Trade ids already sent are kept in a log that is
committed alongside the ledger, so a re-run is silent and a restarted container
does not re-announce a week of trades.

The one number here worth acting on is the shopping line: best price, which
book, and what it saves. That is the only effect in twenty-four hypotheses that
survived testing, and it applies whether or not the model's opinion is any good.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

import notify_email
import shopping

NOTIFIED_LOG = "notified_trades.csv"
NOTIFIED_FIELDS = ["trade_id", "notified_at", "channel"]


def _already_notified(path):
    path = Path(path)
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["trade_id"] for row in csv.DictReader(handle)
                if row.get("trade_id")}


def _record(path, trade_ids, channel):
    path = Path(path)
    write_header = not path.exists()
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NOTIFIED_FIELDS)
        if write_header:
            writer.writeheader()
        for trade_id in trade_ids:
            writer.writerow({"trade_id": trade_id, "notified_at": stamp,
                             "channel": channel or "none"})


def _ledger_line(settlements_path="paper_settlements.csv",
                 trades_path="paper_trades.csv"):
    """The forward record so far, stated in every notice.

    Deliberately not optional. A notice about a new signal that omits how the
    previous signals did is an advertisement.
    """
    try:
        import paper_ledger
        report = paper_ledger.summary(trades_path, settlements_path)
    except Exception:
        return "ledger record unavailable"
    settled = report.get("settled") or 0
    if not settled:
        return "no settled paper trades yet"
    roi = report.get("roi")
    clv = report.get("mean_clv_prob_points")
    parts = [f"{settled} settled"]
    if roi is not None:
        parts.append(f"ROI {roi * 100:+.1f}%")
    if clv is not None:
        parts.append(f"CLV {clv:+.2f} pts")
    return ", ".join(parts)


def compose(trades, ledger_line):
    """The notice body. Prices, the shopping line, and the honest caveat."""
    lines = []
    for row in trades:
        price = str(row.get("execution_price") or row.get("price") or "?")
        book = str(row.get("execution_book") or "consensus")
        lines.append(
            f"{row.get('pick')} vs {row.get('opp')}  ({row.get('date')})\n"
            f"  {price} at {book}   net {row.get('net_edge')} pts, "
            f"{row.get('stake')}u paper")
        gain = shopping.gain_points(row.get("consensus_price"),
                                    row.get("execution_price"))
        if gain is not None and gain >= shopping.MIN_MEANINGFUL_POINTS:
            lines.append(f"  shopping: that price saves {gain:.1f} pts vs "
                         f"consensus {row.get('consensus_price')}")
    body = "\n".join(lines)
    return (
        f"{body}\n\n"
        f"This is a paper trade, not a recommendation.\n"
        f"The rule that flagged it failed its pre-registered test (H1), and "
        f"the forward ledger reads: {ledger_line}.\n"
        f"The shopping line above is the part with evidence behind it.\n"
    )


def run(trades_path="paper_trades.csv", notified_path=NOTIFIED_LOG,
        settlements_path="paper_settlements.csv", dry_run=False):
    path = Path(trades_path)
    if not path.exists():
        print("no paper trades file; nothing to notify")
        return 0
    trades = pd.read_csv(path)
    if not len(trades):
        print("no paper trades; nothing to notify")
        return 0
    seen = _already_notified(notified_path)
    fresh = trades[~trades["trade_id"].astype(str).isin(seen)].copy()

    # A trade for a fight that already happened is not actionable, and on the
    # first run - when nothing has been notified yet - every historical trade
    # in the ledger would otherwise go out at once. Past fights are recorded
    # as seen without being announced, so the backlog is absorbed silently and
    # a restarted container never re-announces a week of settled trades.
    if len(fresh):
        when = pd.to_datetime(fresh["date"], errors="coerce", utc=True)
        today = pd.Timestamp.now(tz="UTC").normalize()
        stale = fresh[when.notna() & (when < today)]
        fresh = fresh[when.isna() | (when >= today)]
        if len(stale):
            print(f"suppressing {len(stale)} trade(s) for fights already run")
            if not dry_run:
                _record(notified_path,
                        [str(t) for t in stale["trade_id"]], "suppressed")

    if not len(fresh):
        print(f"no new paper trades to announce ({len(trades)} in the ledger)")
        return 0

    rows = fresh.to_dict("records")
    subject = (f"claude-fights: {len(rows)} paper trade"
               f"{'s' if len(rows) > 1 else ''} locked")
    body = compose(rows, _ledger_line(settlements_path, trades_path))
    if dry_run:
        print(f"[dry run] would send: {subject}\n")
        print(body)
        return 0
    channel = notify_email.deliver(subject, body)
    print(f"notified {len(rows)} trade(s) via {channel or 'no channel'}")
    _record(notified_path, [str(r["trade_id"]) for r in rows], channel)
    return 0


def send_test():
    """Send the notice a real signal would produce, so the channel can be
    verified without waiting for a fight to qualify."""
    sample = [{
        "pick": "Denise Gomes", "opp": "Yan Xiaonan", "date": "2026-08-29",
        "execution_price": "+140", "execution_book": "DraftKings",
        "consensus_price": "+130", "net_edge": 4.8, "stake": 1,
    }]
    body = ("TEST NOTIFICATION - no trade was locked.\n\n"
            + compose(sample, _ledger_line()))
    channel = notify_email.deliver("claude-fights: test notification", body)
    if channel is None:
        print("no channel configured; set RESEND_API_KEY and BET_EMAIL_TO")
        return 1
    print(f"test notification sent via {channel}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trades", default="paper_trades.csv")
    parser.add_argument("--notified", default=NOTIFIED_LOG)
    parser.add_argument("--settlements", default="paper_settlements.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the notice instead of sending it")
    parser.add_argument("--test", action="store_true",
                        help="send a sample notice to verify the channel")
    args = parser.parse_args(argv)
    if args.test:
        return send_test()
    return run(args.trades, args.notified, args.settlements, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
