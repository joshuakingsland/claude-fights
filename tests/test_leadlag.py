import unittest

import numpy as np
import pandas as pd

import leadlag


def _panel(prices):
    """prices: {(fight, hours_before): {book: devigged probability}}."""
    rows = []
    for (fight, hours), books in prices.items():
        for book, p in books.items():
            rows.append({"api_event_id": fight, "event_date": "2024-01-01",
                         "hours_before": float(hours), "book_key": book, "p": p})
    return pd.DataFrame(rows)


class DevigTests(unittest.TestCase):
    def test_a_balanced_pair_is_a_coin_flip(self):
        got = leadlag._devig(np.array([100.0]), np.array([-100.0]))
        self.assertAlmostEqual(got[0], 0.5)

    def test_the_margin_is_removed(self):
        # -110/-110 is even money once the vig comes out, not 52.4%.
        got = leadlag._devig(np.array([-110.0]), np.array([-110.0]))
        self.assertAlmostEqual(got[0], 0.5)

    def test_a_favourite_reads_above_a_half(self):
        got = leadlag._devig(np.array([-300.0]), np.array([250.0]))
        self.assertGreater(got[0], 0.5)

    def test_exactly_minus_one_hundred_does_not_produce_a_nan(self):
        # Both np.where branches evaluate, so this divides by zero in the
        # branch that gets discarded. The kept value must still be right.
        got = leadlag._devig(np.array([-100.0]), np.array([-100.0]))
        self.assertTrue(np.isfinite(got[0]))
        self.assertAlmostEqual(got[0], 0.5)


class SlopeTests(unittest.TestCase):
    def test_a_perfect_one_to_one_relationship_recovers_slope_one(self):
        x = np.arange(20.0)
        slope, r2, n = leadlag._slope(x, x)
        self.assertAlmostEqual(slope, 1.0)
        self.assertAlmostEqual(r2, 1.0)
        self.assertEqual(n, 20)

    def test_pure_noise_gives_no_explanatory_power(self):
        rng = np.random.default_rng(0)
        slope, r2, _ = leadlag._slope(rng.normal(size=4000), rng.normal(size=4000))
        self.assertLess(abs(slope), 0.1)
        self.assertLess(r2, 0.01)

    def test_a_constant_predictor_is_refused_rather_than_dividing_by_zero(self):
        slope, r2, _ = leadlag._slope(np.ones(10), np.arange(10.0))
        self.assertTrue(np.isnan(slope))


class SelfExclusionTests(unittest.TestCase):
    """A book must never be inside the consensus it is measured against.

    This is the property the whole H3 conclusion rests on. A book is part of
    the market, so regressing the market on a book still inside it recovers
    that book's own weight and reports it as prediction. Every book would look
    like a leader, most strongly at fights where few others are quoted.
    """

    def test_the_tested_book_is_dropped_from_its_own_target(self):
        # One book jumps; the others never move. If the jumper were still in
        # the consensus, the consensus would appear to move with it.
        prices = {
            ("f1", 3.0): {"jumper": 0.50, "a": 0.50, "b": 0.50, "c": 0.50, "d": 0.50},
            ("f1", 2.0): {"jumper": 0.90, "a": 0.50, "b": 0.50, "c": 0.50, "d": 0.50},
            ("f1", 1.0): {"jumper": 0.90, "a": 0.50, "b": 0.50, "c": 0.50, "d": 0.50},
        }
        moves = leadlag.build_moves(_panel(prices), horizon_hours=1.0)
        row = moves[moves.book_key == "jumper"].iloc[0]
        self.assertAlmostEqual(row["lead"], 0.40)
        self.assertAlmostEqual(row["follow"], 0.0)   # others truly did not move

    def test_a_book_that_leads_shows_a_positive_follow(self):
        # The jumper moves first; the others follow next interval.
        prices = {
            ("f1", 3.0): {"jumper": 0.50, "a": 0.50, "b": 0.50, "c": 0.50, "d": 0.50},
            ("f1", 2.0): {"jumper": 0.60, "a": 0.50, "b": 0.50, "c": 0.50, "d": 0.50},
            ("f1", 1.0): {"jumper": 0.60, "a": 0.60, "b": 0.60, "c": 0.60, "d": 0.60},
        }
        moves = leadlag.build_moves(_panel(prices), horizon_hours=1.0)
        row = moves[moves.book_key == "jumper"].iloc[0]
        self.assertAlmostEqual(row["lead"], 0.10)
        self.assertAlmostEqual(row["follow"], 0.10)

    def test_a_thinly_quoted_fight_is_skipped(self):
        # Too few other books to form a consensus worth the name.
        prices = {("f1", 3.0): {"a": 0.5, "b": 0.5},
                  ("f1", 2.0): {"a": 0.6, "b": 0.5},
                  ("f1", 1.0): {"a": 0.6, "b": 0.6}}
        self.assertEqual(len(leadlag.build_moves(_panel(prices))), 0)


class LeaderReportTests(unittest.TestCase):
    def test_books_below_the_observation_floor_are_not_reported(self):
        # winamax_fr scored the highest R^2 in the real run on 510 rows across
        # three cards. Small samples produce the most attractive numbers, which
        # is exactly when a floor matters.
        moves = pd.DataFrame({
            "book_key": ["tiny"] * 20, "event_date": ["2024-01-01"] * 20,
            "lead": np.linspace(-0.1, 0.1, 20), "follow": np.linspace(-0.1, 0.1, 20),
        })
        self.assertEqual(len(leadlag.leaders(moves, draws=20)), 0)

    def test_a_real_leader_is_recovered_with_slope_near_one(self):
        rng = np.random.default_rng(3)
        lead = rng.normal(scale=0.02, size=600)
        moves = pd.DataFrame({
            "book_key": ["sharp"] * 600,
            "event_date": np.repeat([f"2024-01-{d:02d}" for d in range(1, 21)], 30),
            "lead": lead, "follow": lead + rng.normal(scale=0.001, size=600),
        })
        got = leadlag.leaders(moves, draws=200, min_observations=100).iloc[0]
        self.assertAlmostEqual(got["slope"], 1.0, places=1)
        self.assertGreater(got["r2"], 0.9)
        self.assertGreater(got["ci90_low"], 0)


if __name__ == "__main__":
    unittest.main()
