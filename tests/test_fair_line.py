import unittest

import numpy as np
import pandas as pd

import fair_line as fl


def _snapshot(rows, fight="f1", date="2024-05-04", a="Ann Ace", b="Bea Bolt",
              winner="ann ace"):
    """rows: (book_key, odds_a, odds_b)."""
    return pd.DataFrame([
        {"api_event_id": fight, "event_date": date, "fighter_a": a,
         "fighter_b": b, "book_key": book, "odds_a": oa, "odds_b": ob,
         "winner_name": winner}
        for book, oa, ob in rows])


class DevigTests(unittest.TestCase):
    def test_a_balanced_book_devigs_to_a_coin_flip(self):
        self.assertAlmostEqual(float(fl._devig(np.array([-110]), np.array([-110]))[0]),
                               0.5)

    def test_the_margin_is_removed_not_ignored(self):
        # -120/-120 implies 54.5% each, 109% total. De-vigged it is 50/50.
        self.assertAlmostEqual(float(fl._devig(np.array([-120]), np.array([-120]))[0]),
                               0.5)

    def test_implied_keeps_the_vig(self):
        """The asymmetry that makes the test honest.

        p_fair is de-vigged because it estimates truth. p_offered is not,
        because it is a price you pay. De-vigging both would compare two
        opinions and delete the margin the bet has to beat.
        """
        self.assertGreater(float(fl._implied([-120])[0]), 0.5)
        self.assertAlmostEqual(float(fl._implied([-120])[0]), 120 / 220)


class FairTableTests(unittest.TestCase):
    def test_the_reference_book_is_excluded_from_the_pool(self):
        """Otherwise the reference predicts itself.

        Pinnacle here has the longest price on A. If it were left in the pool
        it would be its own best price, the edge would be its own margin, and
        every fight would look mispriced against nothing.
        """
        frame = _snapshot([("pinnacle", +200, -260), ("draftkings", +150, -180),
                           ("fanduel", +140, -170), ("betmgm", +145, -175),
                           ("bovada", +142, -172)])
        row = fl.fair_table(frame).iloc[0]
        self.assertNotEqual(row["book_a"], "pinnacle")
        self.assertEqual(row["odds_a"], 150)

    def test_a_fight_the_reference_does_not_quote_is_dropped(self):
        # Substituting the consensus would quietly make this a different
        # hypothesis, so the fight is dropped instead.
        frame = _snapshot([("draftkings", +150, -180), ("fanduel", +140, -170),
                           ("betmgm", +145, -175), ("bovada", +142, -172),
                           ("betus", +141, -171)])
        self.assertTrue(fl.fair_table(frame).empty)

    def test_a_thin_pool_is_dropped(self):
        frame = _snapshot([("pinnacle", +200, -260), ("draftkings", +150, -180)])
        self.assertTrue(fl.fair_table(frame).empty)

    def test_edge_is_fair_minus_what_is_offered(self):
        frame = _snapshot([("pinnacle", -200, +170), ("draftkings", -150, +130),
                           ("fanduel", -155, +135), ("betmgm", -152, +132),
                           ("bovada", -151, +131)])
        row = fl.fair_table(frame).iloc[0]
        expected = row["fair_a"] - float(fl._implied([-150])[0])
        self.assertAlmostEqual(row["edge_a"], expected)
        self.assertGreater(row["edge_a"], 0)  # -150 is longer than fair -200

    def test_the_winner_is_resolved_onto_side_a(self):
        frame = _snapshot([("pinnacle", -200, +170), ("draftkings", -150, +130),
                           ("fanduel", -155, +135), ("betmgm", -152, +132),
                           ("bovada", -151, +131)], winner="bea bolt")
        self.assertFalse(bool(fl.fair_table(frame).iloc[0]["a_won"]))


class ValueBetTests(unittest.TestCase):
    def _table(self, rows, **kwargs):
        return fl.fair_table(_snapshot(rows, **kwargs))

    def test_it_bets_the_side_offered_longer_than_fair(self):
        table = self._table([("pinnacle", -200, +170), ("draftkings", -150, +130),
                             ("fanduel", -155, +135), ("betmgm", -152, +132),
                             ("bovada", -151, +131)])
        bet = fl.value_bets(table, threshold=0.02).iloc[0]
        self.assertEqual(bet["side"], "a")
        self.assertEqual(bet["bet_odds"], -150)

    def test_nothing_qualifies_when_the_pool_agrees_with_the_reference(self):
        table = self._table([("pinnacle", -200, +170), ("draftkings", -200, +170),
                             ("fanduel", -205, +172), ("betmgm", -202, +171),
                             ("bovada", -201, +170)])
        self.assertEqual(len(fl.value_bets(table, threshold=0.02)), 0)

    def test_a_higher_threshold_takes_fewer_bets(self):
        table = self._table([("pinnacle", -200, +170), ("draftkings", -150, +130),
                             ("fanduel", -155, +135), ("betmgm", -152, +132),
                             ("bovada", -151, +131)])
        loose = len(fl.value_bets(table, threshold=0.01))
        tight = len(fl.value_bets(table, threshold=0.40))
        self.assertGreaterEqual(loose, tight)
        self.assertEqual(tight, 0)

    def test_profit_follows_the_result(self):
        table = self._table([("pinnacle", -200, +170), ("draftkings", -150, +130),
                             ("fanduel", -155, +135), ("betmgm", -152, +132),
                             ("bovada", -151, +131)], winner="bea bolt")
        self.assertAlmostEqual(float(fl.value_bets(table, 0.02).iloc[0]["profit"]), -1.0)


class FavouriteBetTests(unittest.TestCase):
    def _table(self, rows, **kwargs):
        return fl.fair_table(_snapshot(rows, **kwargs))

    def test_the_favourite_is_chosen_by_the_reference_not_by_the_pool(self):
        """Selection and pricing must not come from the same numbers.

        Picking the favourite by whichever side the pool prices shorter would
        make the bet a function of the prices being bet.
        """
        table = self._table([("pinnacle", -400, +320), ("draftkings", -300, +250),
                             ("fanduel", -310, +255), ("betmgm", -305, +252),
                             ("bovada", -302, +251)])
        bet = fl.favourite_bets(table, minimum=0.70).iloc[0]
        self.assertEqual(bet["side"], "a")
        self.assertEqual(bet["bet_odds"], -300)  # best price in the pool

    def test_a_fight_below_the_threshold_is_not_bet(self):
        table = self._table([("pinnacle", -110, -110), ("draftkings", -105, -105),
                             ("fanduel", -108, -108), ("betmgm", -106, -106),
                             ("bovada", -107, -107)])
        self.assertEqual(len(fl.favourite_bets(table, minimum=0.70)), 0)

    def test_it_backs_side_b_when_b_is_the_favourite(self):
        table = self._table([("pinnacle", +320, -400), ("draftkings", +250, -300),
                             ("fanduel", +255, -310), ("betmgm", +252, -305),
                             ("bovada", +251, -302)])
        bet = fl.favourite_bets(table, minimum=0.70).iloc[0]
        self.assertEqual(bet["side"], "b")
        self.assertEqual(bet["bet_odds"], -300)

    def test_a_winning_favourite_pays_the_best_price_taken(self):
        table = self._table([("pinnacle", -400, +320), ("draftkings", -300, +250),
                             ("fanduel", -310, +255), ("betmgm", -305, +252),
                             ("bovada", -302, +251)])
        self.assertAlmostEqual(
            float(fl.favourite_bets(table, 0.70).iloc[0]["profit"]), 100 / 300)

    def test_an_empty_selection_returns_a_scoreable_frame(self):
        table = self._table([("pinnacle", -110, -110), ("draftkings", -105, -105),
                             ("fanduel", -108, -108), ("betmgm", -106, -106),
                             ("bovada", -107, -107)])
        empty = fl.favourite_bets(table, minimum=0.70)
        self.assertEqual(len(empty), 0)
        self.assertIn("profit", empty.columns)


if __name__ == "__main__":
    unittest.main()
