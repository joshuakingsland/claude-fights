import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import notify_card

TRADE_FIELDS = ["trade_id", "date", "pick", "opp", "price", "execution_price",
                "execution_book", "consensus_price", "net_edge", "stake"]


def _trade(trade_id, date, pick="Ann Ace", opp="Bea Bolt",
           execution_price="+140", consensus_price="+130"):
    return {"trade_id": trade_id, "date": date, "pick": pick, "opp": opp,
            "price": execution_price, "execution_price": execution_price,
            "execution_book": "DraftKings", "consensus_price": consensus_price,
            "net_edge": 4.8, "stake": 1}


class NotifyCardTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.trades = self.root / "trades.csv"
        self.notified = self.root / "notified.csv"
        self.settlements = self.root / "settlements.csv"
        self.future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=6)
                       ).strftime("%Y-%m-%d")
        self.past = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=6)
                     ).strftime("%Y-%m-%d")

    def _write(self, rows):
        with self.trades.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRADE_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _run(self, deliver_returns="resend"):
        with mock.patch.object(notify_card.notify_email, "deliver",
                               return_value=deliver_returns) as sent:
            notify_card.run(str(self.trades), str(self.notified),
                            str(self.settlements))
        return sent

    def test_a_new_upcoming_trade_is_announced(self):
        self._write([_trade("t1", self.future)])
        sent = self._run()
        sent.assert_called_once()
        self.assertIn("Ann Ace", sent.call_args[0][1])

    def test_the_same_trade_is_not_announced_twice(self):
        """The snapshot job runs six-hourly; without this it would announce
        the same locked trade until the fight happened."""
        self._write([_trade("t1", self.future)])
        self._run()
        self.assertEqual(self._run().call_count, 0)

    def test_a_trade_for_a_fight_already_run_is_never_announced(self):
        self._write([_trade("t1", self.past)])
        self.assertEqual(self._run().call_count, 0)

    def test_a_first_run_backlog_is_absorbed_without_a_blast(self):
        # Six historical trades and one upcoming: only the upcoming one goes.
        self._write([_trade(f"old{i}", self.past) for i in range(6)]
                    + [_trade("new", self.future)])
        sent = self._run()
        sent.assert_called_once()
        body = sent.call_args[0][1]
        self.assertEqual(body.count("Ann Ace vs Bea Bolt"), 1)

    def test_a_suppressed_trade_is_recorded_so_it_stays_suppressed(self):
        self._write([_trade("t1", self.past)])
        self._run()
        with self.notified.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["trade_id"] for r in rows], ["t1"])
        self.assertEqual(rows[0]["channel"], "suppressed")

    def test_the_shopping_saving_is_included_when_it_is_real(self):
        self._write([_trade("t1", self.future,
                            execution_price="+140", consensus_price="+130")])
        self.assertIn("shopping:", self._run().call_args[0][1])

    def test_no_shopping_line_when_the_best_price_is_the_consensus(self):
        self._write([_trade("t1", self.future,
                            execution_price="-150", consensus_price="-150")])
        self.assertNotIn("shopping:", self._run().call_args[0][1])

    def test_every_notice_says_it_is_not_a_recommendation(self):
        """The most important assertion here.

        The rule that flags these fights failed its pre-registered test, and
        an alert that reads like a tip would be the most harmful thing this
        repository could send.
        """
        self._write([_trade("t1", self.future)])
        body = self._run().call_args[0][1]
        self.assertIn("not a recommendation", body)
        self.assertIn("H1", body)

    def test_a_dry_run_sends_nothing(self):
        self._write([_trade("t1", self.future)])
        with mock.patch.object(notify_card.notify_email, "deliver") as sent:
            notify_card.run(str(self.trades), str(self.notified),
                            str(self.settlements), dry_run=True)
        self.assertEqual(sent.call_count, 0)
        self.assertFalse(self.notified.exists())

    def test_a_missing_trades_file_does_not_raise(self):
        self.assertEqual(
            notify_card.run(str(self.root / "absent.csv"), str(self.notified),
                            str(self.settlements)), 0)

    def test_an_empty_ledger_does_not_raise(self):
        self._write([])
        self.assertEqual(self._run().call_count, 0)

    def test_a_dead_channel_still_records_the_attempt(self):
        # Otherwise a notifier outage would replay every trade afterwards.
        self._write([_trade("t1", self.future)])
        self._run(deliver_returns=None)
        self.assertEqual(self._run().call_count, 0)


class TestNotificationTests(unittest.TestCase):
    def test_the_test_send_uses_the_same_body_a_real_signal_would(self):
        with mock.patch.object(notify_card.notify_email, "deliver",
                               return_value="resend") as sent:
            notify_card.send_test()
        body = sent.call_args[0][1]
        self.assertIn("TEST NOTIFICATION", body)
        self.assertIn("not a recommendation", body)

    def test_it_reports_failure_when_no_channel_is_configured(self):
        with mock.patch.object(notify_card.notify_email, "deliver",
                               return_value=None):
            self.assertEqual(notify_card.send_test(), 1)


if __name__ == "__main__":
    unittest.main()
