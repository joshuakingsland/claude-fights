import unittest

import shopping


class PriceParsingTests(unittest.TestCase):
    def test_a_signed_string_is_a_price(self):
        self.assertEqual(shopping._price("+150"), 150.0)
        self.assertEqual(shopping._price("-200"), -200.0)

    def test_blanks_and_nans_are_not_prices(self):
        for value in (None, "", "  ", "nan", "None", float("nan")):
            with self.subTest(value=value):
                self.assertIsNone(shopping._price(value))

    def test_zero_is_not_a_price(self):
        # A 0 in the feed means "unquoted", and treating it as even money
        # would invent a saving out of a missing number.
        self.assertIsNone(shopping._price(0))


class GainTests(unittest.TestCase):
    def test_a_better_favourite_price_saves_points(self):
        # -330 implies 76.74%, -400 implies 80.00%; shopping saves 3.26 pts.
        self.assertAlmostEqual(shopping.gain_points(-400, -330), 3.2558, places=3)

    def test_a_better_underdog_price_saves_points(self):
        self.assertGreater(shopping.gain_points(250, 290), 0)

    def test_an_identical_price_saves_nothing(self):
        self.assertAlmostEqual(shopping.gain_points(-150, -150), 0.0)

    def test_a_worse_best_price_is_reported_not_hidden(self):
        """Consensus is an average, so it can sit above a single quote.

        Flooring that at zero would hide a broken feed behind a plausible
        looking 0.0, so the negative is reported.
        """
        self.assertLess(shopping.gain_points(-330, -400), 0)

    def test_a_missing_price_yields_no_number(self):
        self.assertIsNone(shopping.gain_points(None, -150))
        self.assertIsNone(shopping.gain_points(-150, ""))


class MarginTests(unittest.TestCase):
    def test_a_balanced_book_charges_its_vig(self):
        self.assertAlmostEqual(shopping.margin_points(-110, -110), 4.7619, places=3)

    def test_a_fair_pair_charges_nothing(self):
        self.assertAlmostEqual(shopping.margin_points(100, -100), 0.0)

    def test_crossed_prices_go_negative(self):
        self.assertLess(shopping.margin_points(120, -110), 0)

    def test_a_missing_side_yields_no_margin(self):
        self.assertIsNone(shopping.margin_points(-110, None))


class SideTests(unittest.TestCase):
    def test_it_reports_the_book_and_the_saving(self):
        got = shopping.side("Ann Ace", -400, -330, "FanDuel")
        self.assertEqual(got["price"], "-330")
        self.assertEqual(got["consensus"], "-400")
        self.assertEqual(got["book"], "FanDuel")
        self.assertTrue(got["worth_shopping"])

    def test_a_rounding_level_saving_is_not_worth_shopping(self):
        # One book, or two quoting the same number: nothing to send anyone
        # across the internet for.
        self.assertFalse(shopping.side("Ann", -150, -150, "DK")["worth_shopping"])

    def test_an_unquoted_side_degrades_rather_than_raising(self):
        got = shopping.side("Ann Ace", None, None, None)
        self.assertIsNone(got["price"])
        self.assertFalse(got["worth_shopping"])


class FightTests(unittest.TestCase):
    def _fight(self, **kwargs):
        # -330 / +290 sums to 102.4%, a normal shopped two-sided price. An
        # earlier fixture used +340, which crosses - the arb case belongs in
        # its own test rather than hiding in the default.
        base = dict(pick_name="Ann Ace", opp_name="Bea Bolt",
                    pick_consensus=-400, opp_consensus=270,
                    pick_best=-330, pick_book="FanDuel",
                    opp_best=290, opp_book="BetMGM", books=16)
        base.update(kwargs)
        return shopping.fight(**base)

    def test_both_fighters_are_reported_not_only_the_model_pick(self):
        got = self._fight()
        self.assertEqual(got["pick"]["name"], "Ann Ace")
        self.assertEqual(got["opp"]["name"], "Bea Bolt")
        self.assertIsNotNone(got["opp"]["price"])

    def test_shopping_lowers_the_two_sided_margin(self):
        got = self._fight()
        self.assertLess(got["best_margin_pts"], got["consensus_margin_pts"])

    def test_the_headline_is_the_larger_of_the_two_savings(self):
        got = self._fight()
        self.assertAlmostEqual(
            got["best_gain_pts"],
            max(got["pick"]["gain_pts"], got["opp"]["gain_pts"]), places=6)

    def test_a_one_book_fight_offers_no_value(self):
        got = self._fight(pick_best=-400, opp_best=270, books=1)
        self.assertFalse(got["any_value"])
        self.assertEqual(got["books"], 1)

    def test_crossed_best_prices_are_flagged(self):
        # Best prices that cross: backing both sides wins either way.
        got = self._fight(pick_consensus=-110, opp_consensus=-110,
                          pick_best=120, opp_best=-110)
        self.assertTrue(got["crossed"])

    def test_an_uncrossed_fight_is_not_flagged(self):
        self.assertFalse(self._fight()["crossed"])

    def test_a_fight_with_no_quotes_does_not_raise(self):
        got = self._fight(pick_consensus=None, opp_consensus=None,
                          pick_best=None, opp_best=None, books=None)
        self.assertIsNone(got["best_gain_pts"])
        self.assertIsNone(got["books"])
        self.assertFalse(got["any_value"])
        self.assertFalse(got["crossed"])

    def test_books_survives_a_float_from_the_csv(self):
        self.assertEqual(self._fight(books=16.0)["books"], 16)

    def test_the_real_makhachev_quote(self):
        """The worked example from the session: best -330, consensus -350."""
        got = self._fight(pick_name="Islam Makhachev", pick_consensus=-350,
                          pick_best=-330, pick_book="FanDuel")
        self.assertEqual(got["pick"]["price"], "-330")
        self.assertGreater(got["pick"]["gain_pts"], 0.9)
        self.assertTrue(got["pick"]["worth_shopping"])


if __name__ == "__main__":
    unittest.main()
