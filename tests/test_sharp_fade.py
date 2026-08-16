import unittest

import numpy as np
import pandas as pd

import sharp_fade as sf


def _quotes(rows, date="2024-05-04", fight="f1", a="Ann Ace", b="Bea Bolt"):
    """rows: (book_key, odds_a, odds_b)."""
    return pd.DataFrame([
        {"api_event_id": fight, "event_date": date, "fighter_a": a,
         "fighter_b": b, "book_key": book, "odds_a": oa, "odds_b": ob,
         "winner_name": "ann ace"}
        for book, oa, ob in rows])


class PayoutTests(unittest.TestCase):
    def test_payout_is_profit_per_unit(self):
        self.assertAlmostEqual(float(sf._payout([+150])[0]), 1.5)
        self.assertAlmostEqual(float(sf._payout([-200])[0]), 0.5)

    def test_the_better_price_wins_across_the_sign_boundary(self):
        """The reason best price is ranked on payout, not on the number.

        Sorting the printed American price would put every underdog above
        every favourite, so the "best price in the world" would just be
        whichever book had the longest dog.
        """
        self.assertGreater(float(sf._payout([+105])[0]), float(sf._payout([-105])[0]))
        self.assertGreater(float(sf._payout([-105])[0]), float(sf._payout([-120])[0]))


class BestPriceTests(unittest.TestCase):
    def test_it_picks_the_highest_payout_on_each_side(self):
        frame = _quotes([("pinnacle", +140, -160),
                         ("draftkings", +120, -130),
                         ("fanduel", +110, -125)])
        table = sf.best_price_table(frame, min_books=3)
        row = table.iloc[0]
        self.assertEqual(row["book_a"], "pinnacle")
        self.assertEqual(row["odds_a"], 140)
        self.assertEqual(row["book_b"], "fanduel")
        self.assertEqual(row["odds_b"], -125)

    def test_a_thin_market_is_not_evidence(self):
        # Best of three is close to a coin flip on which book holds it.
        frame = _quotes([("pinnacle", +140, -160), ("draftkings", +120, -130)])
        self.assertTrue(sf.best_price_table(frame, min_books=5).empty)

    def test_the_winner_is_carried_through(self):
        frame = _quotes([("a", +140, -160), ("b", +120, -130), ("c", +110, -125)])
        self.assertTrue(bool(sf.best_price_table(frame, min_books=3).iloc[0]["a_won"]))


class FadeTests(unittest.TestCase):
    """The direction of the bet is the hypothesis. Getting it backwards
    would invert every number in the report."""

    def _table(self, rows):
        return sf.best_price_table(_quotes(rows), min_books=3)

    def test_it_backs_the_side_the_trigger_book_is_not_generous_on(self):
        # Pinnacle holds the best price on A, so the bet is B.
        table = self._table([("pinnacle", +140, -160),
                             ("draftkings", +120, -130),
                             ("fanduel", +110, -125)])
        bet = sf.fade(table, "pinnacle").iloc[0]
        self.assertEqual(bet["side"], "b")
        self.assertEqual(bet["bet_book"], "fanduel")
        self.assertEqual(bet["bet_odds"], -125)

    def test_the_operators_example_bets_jones(self):
        """Circa best on Gane at +135, DraftKings best on Jones at -148.

        The worked example from the request, kept as a test so the rule
        cannot silently flip.
        """
        frame = _quotes([("circasports", +135, -170),
                         ("draftkings", +120, -148),
                         ("fanduel", +115, -155)],
                        a="Ciryl Gane", b="Jon Jones")
        bet = sf.fade(sf.best_price_table(frame, min_books=3), "circasports").iloc[0]
        self.assertEqual(bet["side"], "b")            # Jones
        self.assertEqual(bet["bet_odds"], -148)       # DraftKings
        self.assertEqual(bet["bet_book"], "draftkings")

    def test_a_book_holding_both_best_prices_is_not_a_signal(self):
        # A low-margin book is cheap on everything. Counting it would let
        # thin vig masquerade as an opinion about the fight.
        table = self._table([("pinnacle", +140, -120),
                             ("draftkings", +120, -130),
                             ("fanduel", +110, -135)])
        self.assertEqual(table.iloc[0]["book_a"], "pinnacle")
        self.assertEqual(table.iloc[0]["book_b"], "pinnacle")
        self.assertEqual(len(sf.fade(table, "pinnacle")), 0)

    def test_a_book_holding_neither_produces_no_bet(self):
        table = self._table([("pinnacle", +100, -160),
                             ("draftkings", +120, -130),
                             ("fanduel", +110, -125)])
        self.assertEqual(len(sf.fade(table, "pinnacle")), 0)

    def test_profit_follows_the_result(self):
        # A won, and the rule bet B, so the bet loses one unit.
        table = self._table([("pinnacle", +140, -160),
                             ("draftkings", +120, -130),
                             ("fanduel", +110, -125)])
        self.assertAlmostEqual(float(sf.fade(table, "pinnacle").iloc[0]["profit"]), -1.0)

    def test_a_winning_fade_pays_the_price_taken(self):
        table = self._table([("pinnacle", -160, +140),
                             ("draftkings", -130, +120),
                             ("fanduel", -125, +110)])
        bet = sf.fade(table, "pinnacle").iloc[0]
        self.assertEqual(bet["side"], "a")
        self.assertAlmostEqual(float(bet["profit"]), 100 / 125)


class ExchangeTests(unittest.TestCase):
    """Exchanges quote gross of commission and hold most best-price slots.

    Treating their screen price as takeable is what makes a shopping backtest
    look free.
    """

    def test_exchanges_are_removed(self):
        frame = _quotes([("betfair", +200, -300), ("pinnacle", +140, -160),
                         ("draftkings", +120, -130), ("fanduel", +110, -125)])
        self.assertNotIn("betfair", set(sf.drop_exchanges(frame)["book_key"]))

    def test_removing_them_changes_who_holds_the_best_price(self):
        frame = _quotes([("betfair", +200, -300), ("pinnacle", +140, -160),
                         ("draftkings", +120, -130), ("fanduel", +110, -125)])
        self.assertEqual(sf.best_price_table(frame, min_books=3).iloc[0]["book_a"],
                         "betfair")
        self.assertEqual(
            sf.best_price_table(sf.drop_exchanges(frame), min_books=3).iloc[0]["book_a"],
            "pinnacle")

    def test_commission_is_charged_only_on_winners(self):
        bets = pd.DataFrame({"bet_book": ["betfair", "betfair"],
                             "won": [True, False], "profit": [2.0, -1.0]})
        charged = sf.apply_commission(bets, rate=0.05)
        self.assertAlmostEqual(charged.iloc[0]["profit"], 1.9)
        self.assertAlmostEqual(charged.iloc[1]["profit"], -1.0)

    def test_a_real_book_is_never_charged(self):
        bets = pd.DataFrame({"bet_book": ["draftkings"], "won": [True],
                             "profit": [2.0]})
        self.assertAlmostEqual(
            sf.apply_commission(bets, rate=0.05).iloc[0]["profit"], 2.0)


class ControlTests(unittest.TestCase):
    def test_the_baseline_takes_a_side_without_consulting_any_book(self):
        # Control 2 exists to measure line shopping on its own, so it must not
        # look at book identity at all.
        table = sf.best_price_table(
            _quotes([("pinnacle", +140, -160), ("draftkings", +120, -130),
                     ("fanduel", +110, -125)]), min_books=3)
        favourite = sf.baseline_best_price(table, "favourite").iloc[0]
        self.assertEqual(favourite["bet_odds"], -125)
        underdog = sf.baseline_best_price(table, "underdog").iloc[0]
        self.assertEqual(underdog["bet_odds"], 140)

    def test_an_unknown_side_raises(self):
        table = sf.best_price_table(
            _quotes([("a", +140, -160), ("b", +120, -130), ("c", +110, -125)]),
            min_books=3)
        with self.assertRaises(ValueError):
            sf.baseline_best_price(table, "whichever")


class ScoreTests(unittest.TestCase):
    def _bets(self, profits, dates):
        return pd.DataFrame({"profit": profits, "won": [p > 0 for p in profits],
                             "event_date": dates})

    def test_roi_is_mean_profit_per_unit(self):
        got = sf.score(self._bets([1.0, -1.0, 1.0, -1.0], ["a", "a", "b", "b"]),
                       "x", draws=50)
        self.assertAlmostEqual(got["roi"], 0.0)

    def test_the_sample_floor_is_enforced(self):
        # n >= 200 is pre-registered and is what kills the Circa arm.
        got = sf.score(self._bets([1.0] * 10, ["a"] * 10), "x", draws=50)
        self.assertGreater(got["roi"], 0)
        self.assertFalse(got["passes"])

    def test_an_empty_rule_scores_nothing_rather_than_raising(self):
        got = sf.score(pd.DataFrame(columns=["profit", "won", "event_date"]),
                       "x", draws=10)
        self.assertEqual(got["n"], 0)
        self.assertFalse(got["passes"])


class OutcomeTests(unittest.TestCase):
    def test_draws_are_excluded_from_the_outcome_map(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "f.csv"
            pd.DataFrame({
                "date": ["2024-05-04", "2024-05-04"],
                "fighter_a": ["Ann Ace", "Cal Cruz"],
                "fighter_b": ["Bea Bolt", "Dee Dane"],
                "winner": ["A", "draw"],
            }).to_csv(path, index=False)
            got = sf.outcomes(str(path))
        self.assertEqual(len(got), 1)
        self.assertEqual(list(got.values()), ["ann ace"])


if __name__ == "__main__":
    unittest.main()
