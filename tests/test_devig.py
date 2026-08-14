import unittest

import numpy as np

import devig


class ImpliedTests(unittest.TestCase):
    def test_even_money(self):
        self.assertAlmostEqual(float(devig.implied([+100])[0]), 0.5)

    def test_a_favourite_implies_more_than_half(self):
        self.assertAlmostEqual(float(devig.implied([-200])[0]), 2 / 3)

    def test_a_priced_book_sums_to_more_than_one(self):
        total = float(devig.implied([-110])[0] + devig.implied([-110])[0])
        self.assertGreater(total, 1.0)


class DevigTests(unittest.TestCase):
    """Every method must be a probability pair, whatever else it does."""

    PRICES = [(-110, -110), (-370, +290), (-900, +550), (-150, +130),
              (-2000, +900), (-105, -115), (+200, -250)]

    def test_every_method_sums_to_one_across_both_sides(self):
        for method in devig.METHODS:
            for odds_a, odds_b in self.PRICES:
                with self.subTest(method=method, price=(odds_a, odds_b)):
                    a = float(devig.devig([odds_a], [odds_b], method)[0])
                    b = float(devig.devig([odds_b], [odds_a], method)[0])
                    self.assertAlmostEqual(a + b, 1.0, places=6)

    def test_every_method_agrees_on_a_balanced_price(self):
        # With no asymmetry there is nothing for the methods to disagree about.
        for method in devig.METHODS:
            with self.subTest(method=method):
                self.assertAlmostEqual(
                    float(devig.devig([-110], [-110], method)[0]), 0.5, places=9)

    def test_every_method_stays_inside_zero_and_one(self):
        for method in devig.METHODS:
            for odds_a, odds_b in self.PRICES:
                value = float(devig.devig([odds_a], [odds_b], method)[0])
                self.assertGreater(value, 0.0)
                self.assertLess(value, 1.0)

    def test_proportional_is_the_most_favourable_to_the_hypothesis(self):
        """The confound H24 exists to test.

        Proportional hands the favourite the lowest fair probability of any
        method, which makes "the favourite wins more often than implied"
        easiest to achieve without the market being wrong.
        """
        for odds_a, odds_b in [(-370, +290), (-900, +550), (-2000, +900)]:
            with self.subTest(price=(odds_a, odds_b)):
                proportional = float(devig.devig([odds_a], [odds_b], "proportional")[0])
                for method in ("additive", "power", "shin"):
                    self.assertGreater(
                        float(devig.devig([odds_a], [odds_b], method)[0]),
                        proportional)

    def test_power_is_the_strictest_on_a_heavy_favourite(self):
        for odds_a, odds_b in [(-370, +290), (-900, +550), (-2000, +900)]:
            values = {m: float(devig.devig([odds_a], [odds_b], m)[0])
                      for m in devig.METHODS}
            self.assertEqual(max(values, key=values.get), "power")

    def test_shin_and_additive_coincide_on_a_two_way_price(self):
        """Not a bug, and worth pinning so nobody 'fixes' it.

        Shin only separates from additive with three or more outcomes. On a
        two-way market the solved insider fraction reproduces the additive
        answer exactly, so the family is three estimators, not four, and a
        claim to have survived four methods would overstate by one.
        """
        for odds_a, odds_b in self.PRICES:
            with self.subTest(price=(odds_a, odds_b)):
                self.assertAlmostEqual(
                    float(devig.devig([odds_a], [odds_b], "shin")[0]),
                    float(devig.devig([odds_a], [odds_b], "additive")[0]),
                    places=9)

    def test_an_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            devig.devig([-110], [-110], "vibes")

    def test_it_vectorises(self):
        got = devig.devig([-110, -370], [-110, +290], "power")
        self.assertEqual(len(got), 2)
        self.assertAlmostEqual(float(got[0]), 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
