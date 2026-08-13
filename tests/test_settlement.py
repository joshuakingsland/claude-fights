import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import paper_ledger
from paper_ledger import SETTLEMENT_FIELDS, TRADE_FIELDS, settle_completed

FIGHT_DATE = "2025-03-01"


def _write(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _trade(trade_id, pick, opp, price="+150", stake="1"):
    return {
        "trade_id": trade_id,
        "locked_at": f"{FIGHT_DATE}T18:00:00+00:00",
        "scheduled_start": f"{FIGHT_DATE}T23:00:00+00:00",
        "date": FIGHT_DATE,
        "pick": pick,
        "opp": opp,
        "price": price,
        "market": "50",
        "stake": stake,
        "staking_policy": "v1",
    }


class SettlementTests(unittest.TestCase):
    """Two ways the ledger used to lie about a trade.

    Both were silent. A drawn fight left the trade permanently open, and an
    unparseable price wrote a 0.00 LOSS - so the win rate and the P&L
    disagreed about the same wager and neither flagged it.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.trades = self.root / "trades.csv"
        self.settlements = self.root / "settlements.csv"
        self.fights = self.root / "fights.csv"
        _write(self.fights,
               ["date", "fighter_a", "fighter_b", "winner"],
               [{"date": FIGHT_DATE, "fighter_a": "Ann Ace",
                 "fighter_b": "Bea Bolt", "winner": "A"},
                {"date": FIGHT_DATE, "fighter_a": "Cal Cruz",
                 "fighter_b": "Dee Dane", "winner": "draw"}])

    def _settle(self, trades):
        _write(self.trades, TRADE_FIELDS, trades)
        settle_completed(trades_path=str(self.trades),
                      settlements_path=str(self.settlements),
                      fights_path=str(self.fights),
                      closing_path=str(self.root / "absent.csv"),
                      captured_closing_path=str(self.root / "absent2.csv"))
        if not self.settlements.exists():
            # Nothing settled, so the file was never written. That is the
            # correct outcome for an open trade, not a missing-file error.
            return pd.DataFrame(columns=SETTLEMENT_FIELDS)
        return pd.read_csv(self.settlements)

    def test_a_won_bet_pays_the_price(self):
        got = self._settle([_trade("t1", "Ann Ace", "Bea Bolt")])
        self.assertEqual(got.iloc[0]["result"], "WIN")
        self.assertAlmostEqual(float(got.iloc[0]["pnl"]), 1.5)

    def test_a_lost_bet_loses_the_stake(self):
        got = self._settle([_trade("t1", "Bea Bolt", "Ann Ace")])
        self.assertEqual(got.iloc[0]["result"], "LOSS")
        self.assertAlmostEqual(float(got.iloc[0]["pnl"]), -1.0)

    def test_a_draw_settles_as_a_push_rather_than_staying_open(self):
        got = self._settle([_trade("t1", "Cal Cruz", "Dee Dane")])
        self.assertEqual(len(got), 1)
        self.assertEqual(got.iloc[0]["result"], "PUSH")
        self.assertAlmostEqual(float(got.iloc[0]["pnl"]), 0.0)

    def test_a_draw_is_a_push_for_both_sides(self):
        # The old filter dropped drawn fights wholesale. Had it not, the
        # equality test would have called one side a winner.
        got = self._settle([_trade("t1", "Cal Cruz", "Dee Dane"),
                            _trade("t2", "Dee Dane", "Cal Cruz")])
        self.assertEqual(set(got["result"]), {"PUSH"})
        self.assertEqual(list(got["pnl"]), [0.0, 0.0])

    def test_a_push_shrinks_roi_toward_zero(self):
        """The reason pushes cannot just be dropped.

        ROI divides profit by money staked. A push adds a full stake and no
        profit, so omitting pushes overstates the magnitude of whatever the
        ledger reports.
        """
        self._settle([_trade("t1", "Ann Ace", "Bea Bolt"),
                      _trade("t2", "Cal Cruz", "Dee Dane")])
        settled = pd.read_csv(self.settlements).merge(
            pd.read_csv(self.trades)[["trade_id", "stake", "staking_policy"]],
            on="trade_id", how="left")
        with_push = paper_ledger._ledger_metrics(pd.read_csv(self.trades), settled)
        without = paper_ledger._ledger_metrics(
            pd.read_csv(self.trades), settled[settled["result"] != "PUSH"])
        self.assertAlmostEqual(with_push["roi"], 0.75)
        self.assertAlmostEqual(without["roi"], 1.5)

    def test_an_unparseable_price_leaves_the_trade_open(self):
        got = self._settle([_trade("t1", "Ann Ace", "Bea Bolt", price="n/a")])
        self.assertEqual(len(got), 0)

    def test_an_unparseable_price_is_never_recorded_as_a_loss(self):
        # The old behaviour: pnl 0.00 with result LOSS, which counts against
        # the win rate while contributing nothing to P&L.
        got = self._settle([_trade("t1", "Ann Ace", "Bea Bolt", price=""),
                            _trade("t2", "Bea Bolt", "Ann Ace")])
        self.assertEqual(list(got["trade_id"]), ["t2"])
        self.assertNotIn("t1", set(got["trade_id"]))

    def test_a_fixed_price_settles_on_the_next_run(self):
        # Leaving it open is only correct if it is recoverable.
        self._settle([_trade("t1", "Ann Ace", "Bea Bolt", price="bad")])
        got = self._settle([_trade("t1", "Ann Ace", "Bea Bolt", price="+150")])
        self.assertEqual(list(got["trade_id"]), ["t1"])
        self.assertAlmostEqual(float(got.iloc[0]["pnl"]), 1.5)

    def test_settling_twice_does_not_double_count(self):
        self._settle([_trade("t1", "Ann Ace", "Bea Bolt")])
        got = self._settle([_trade("t1", "Ann Ace", "Bea Bolt")])
        self.assertEqual(len(got), 1)

    def test_a_trade_locked_after_the_bell_is_refused(self):
        late = _trade("t1", "Ann Ace", "Bea Bolt")
        late["locked_at"] = f"{FIGHT_DATE}T23:30:00+00:00"
        self.assertEqual(len(self._settle([late])), 0)

    def test_a_fight_with_no_result_yet_stays_open(self):
        got = self._settle([_trade("t1", "Eve East", "Fay Frost")])
        self.assertEqual(len(got), 0)

    def test_every_settlement_field_is_written(self):
        self._settle([_trade("t1", "Cal Cruz", "Dee Dane")])
        with open(self.settlements) as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, SETTLEMENT_FIELDS)


if __name__ == "__main__":
    unittest.main()
