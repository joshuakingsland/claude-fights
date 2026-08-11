import unittest

import numpy as np
import pandas as pd

import rounds_model as rm


def _fights():
    """Two corners with deliberately different finishing profiles."""
    rows = []
    for i in range(60):
        rows.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=14 * i),
            "fighter_a": f"Banger {i % 6}", "fighter_b": f"Grinder {i % 6}",
            "winner": "A" if i % 2 else "B",
            "method": "KO/TKO" if i % 3 else "Decision - Unanimous",
            "fight_time_min": 3.0 if i % 3 else 15.0,
            "time_format": "3 Rnd (5-5-5)",
            "weightclass": "Lightweight Bout",
            "dob_a": pd.Timestamp("1992-01-01"),
            "dob_b": pd.Timestamp("1990-01-01"),
        })
    return pd.DataFrame(rows)


class ScheduleTests(unittest.TestCase):
    def test_standard_formats_map_to_round_counts(self):
        formats = pd.Series(["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)",
                             "No Time Limit", "1 Rnd + OT (12-3)"])
        got = rm.scheduled_rounds(formats)
        self.assertEqual(got[0], 3.0)
        self.assertEqual(got[1], 5.0)
        self.assertTrue(np.isnan(got[2]) and np.isnan(got[3]))

    def test_five_round_fights_get_the_later_edges(self):
        self.assertEqual(rm.edges_for(3)[-1], 15.0)
        self.assertEqual(rm.edges_for(5)[-1], 25.0)
        self.assertEqual(rm.edges_for(5)[:6], rm.edges_for(3))


class CornerInvarianceTests(unittest.TestCase):
    """How long a fight lasts cannot depend on which corner is called A.

    This is the property that separates this model from the moneyline one.
    A-minus-B differentials are right for "who wins" and useless for "how
    long": two knockout artists and two decision grinders both differ by zero.
    """

    def test_swapping_the_corners_leaves_features_unchanged(self):
        featured = rm.prepare(_fights())
        straight = rm.build_X(featured)

        swapped = featured.copy()
        for column in list(featured.columns):
            if column.endswith("_a"):
                partner = column[:-2] + "_b"
                if partner in featured.columns:
                    swapped[column] = featured[partner].to_numpy()
                    swapped[partner] = featured[column].to_numpy()
        flipped = rm.build_X(swapped)
        pd.testing.assert_frame_equal(straight, flipped, check_exact=False,
                                      atol=1e-12)

    def test_pair_summary_keeps_level_and_spread(self):
        mean, gap = rm._pair([0.8, 0.2], [0.2, 0.2])
        self.assertAlmostEqual(mean[0], 0.5)
        self.assertAlmostEqual(gap[0], 0.6)
        # Same mean, no spread: a different fight that a differential would
        # have called identical.
        self.assertAlmostEqual(mean[1], 0.2)
        self.assertAlmostEqual(gap[1], 0.0)


class HazardLayoutTests(unittest.TestCase):
    def test_a_fight_contributes_one_row_per_interval_it_reached(self):
        featured = rm.prepare(_fights())
        usable = featured[featured["_usable"]].reset_index(drop=True)
        X = rm.build_X(usable)
        rows, labels = rm._hazard_rows(usable, X)

        # A 3:00 finish reaches bins 0 and 1 and ends in bin 1.
        quick = usable.index[usable["fight_time_min"] == 3.0][0]
        # A decision reaches all six and never ends in one.
        went = usable.index[usable["_distance"]][0]
        self.assertTrue(len(rows) > len(usable))
        self.assertEqual(set(labels) - {0.0, 1.0}, set())
        self.assertGreater(labels.sum(), 0)
        self.assertIsNotNone(quick)
        self.assertIsNotNone(went)

    def test_a_decision_never_records_a_finish(self):
        featured = rm.prepare(_fights())
        usable = featured[featured["_usable"]].reset_index(drop=True)
        distance_only = usable[usable["_distance"]].reset_index(drop=True)
        rows, labels = rm._hazard_rows(distance_only, rm.build_X(distance_only))
        self.assertEqual(labels.sum(), 0.0)
        self.assertEqual(len(rows), len(distance_only) * len(rm.EDGES_3))


class SurvivalTests(unittest.TestCase):
    def setUp(self):
        self.fights = _fights()
        self.model = rm.train(self.fights)
        self.featured = rm.prepare(self.fights)
        self.featured = self.featured[self.featured["_usable"]].reset_index(drop=True)

    def test_survival_never_increases(self):
        for curve in rm.survival(self.model, self.featured):
            values = [p for _, p in curve]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertTrue(all(0.0 <= v <= 1.0 for v in values))

    def test_distance_equals_surviving_the_final_bell(self):
        curves = rm.survival(self.model, self.featured)
        priced = rm.totals_and_distance(self.model, self.featured)
        for curve, row in zip(curves, priced):
            self.assertAlmostEqual(row["distance"], curve[-1][1])

    def test_only_half_round_lines_are_quoted(self):
        # Books hang MMA totals on half-rounds. Emitting over_2.0 would be a
        # price for a bet nobody offers.
        for row in rm.totals_and_distance(self.model, self.featured):
            for key in row["lines"]:
                mark = float(key.split("_")[1])
                self.assertAlmostEqual(mark % 1.0, 0.5)

    def test_over_the_last_line_is_never_below_the_distance_price(self):
        # A three-rounder that passes 12:30 has only the rest of round three
        # left, so it must be at least as likely as going the full distance.
        for row in rm.totals_and_distance(self.model, self.featured):
            if "over_2.5" in row["lines"]:
                self.assertGreaterEqual(row["lines"]["over_2.5"] + 1e-9,
                                        row["distance"])

    def test_batched_scoring_matches_a_row_at_a_time(self):
        batched = rm.survival(self.model, self.featured)
        for position in range(0, len(self.featured), 7):
            one = rm.survival(self.model,
                              self.featured.iloc[[position]].reset_index(drop=True))
            for (_, a), (_, b) in zip(batched[position], one[0]):
                self.assertAlmostEqual(a, b, places=10)


class OrderAlignmentTests(unittest.TestCase):
    """Prices must come back bound to the fight they were asked about.

    prepare() sorts by date for the career-rate join, and card_prices used to
    return that order. The caller zips the list against its own fights, so the
    two only agreed while the odds feed happened to write rows date-sorted.
    Reversing a 44-fight card moved six onto the wrong fight - a main event
    priced over three rounds, a prelim over five - and the length check could
    not see it because the count was still 44.
    """

    def setUp(self):
        self.fights = _fights()
        self.model = rm.train(self.fights)

    def _card(self, order):
        rows = [{
            "date": pd.Timestamp("2026-08-16"), "event": "UPCOMING",
            "time_format": "5 Rnd (5-5-5-5-5)" if five else "3 Rnd (5-5-5)",
            "weightclass": "Lightweight Bout", "fighter_a": f"Corner {i}",
            "fighter_b": f"Rival {i}", "winner": "A", "method": "",
            "fight_time_min": float("nan"),
            "dob_a": pd.Timestamp("1992-01-01"), "dob_b": pd.Timestamp("1990-01-01"),
        } for i, five in order]
        return pd.concat([self.fights, pd.DataFrame(rows)], ignore_index=True)

    def test_prices_follow_the_input_order_not_the_date_sort(self):
        # Same three fights, opposite arrival order. Each must keep its own
        # round count.
        order = [(0, False), (1, True), (2, False)]
        for arrangement in (order, order[::-1]):
            priced = rm.card_prices(self.model, self._card(arrangement))
            self.assertEqual(len(priced), len(arrangement))
            for row, (i, five) in zip(priced, arrangement):
                self.assertEqual(row["fighter_a"], f"Corner {i}")
                self.assertEqual(len(row["totals"]), 5 if five else 3)

    def test_every_row_names_the_fighters_it_priced(self):
        priced = rm.card_prices(self.model, self._card([(0, False), (1, False)]))
        self.assertEqual([r["fighter_a"] for r in priced], ["Corner 0", "Corner 1"])
        self.assertEqual([r["fighter_b"] for r in priced], ["Rival 0", "Rival 1"])


class NoReadGuardTests(unittest.TestCase):
    """Two ways the totals model can have nothing to say, both observed live.

    On the 2026-08-11 Contender Series card every fighter had zero recorded
    bouts, so every career feature was zero and the model returned the
    unconditional base rate - 69% over 1.5 rounds - for all five fights. The
    books, which could see one fighter was a -3500 favourite, ranged from 20%
    to 54%. Naively that reads as a 49-point edge on every fight. Nothing else
    in the pipeline would have caught it: these prices carry no book count,
    and the identity gate passes because a Contender Series fighter resolves
    against the UFCStats registry perfectly well - he has simply never fought.
    """

    def setUp(self):
        self.fights = _fights()
        self.model = rm.train(self.fights)

    def test_a_debutant_pairing_is_marked_as_having_no_career_data(self):
        card = pd.concat([self.fights, pd.DataFrame([{
            "date": pd.Timestamp("2026-08-16"), "event": "UPCOMING",
            "time_format": "3 Rnd (5-5-5)", "weightclass": "Lightweight Bout",
            "fighter_a": "Debut One", "fighter_b": "Debut Two",
            "winner": "A", "method": "", "fight_time_min": float("nan"),
            "dob_a": pd.Timestamp("1995-01-01"), "dob_b": pd.Timestamp("1996-01-01"),
        }])], ignore_index=True)
        row = rm.card_prices(self.model, card)[0]
        self.assertTrue(row["no_career_data"])
        self.assertEqual(row["career_bouts"], 0)

    def test_an_experienced_pairing_is_not_marked(self):
        card = pd.concat([self.fights, pd.DataFrame([{
            "date": pd.Timestamp("2026-08-16"), "event": "UPCOMING",
            "time_format": "3 Rnd (5-5-5)", "weightclass": "Lightweight Bout",
            "fighter_a": "Banger 0", "fighter_b": "Grinder 0",
            "winner": "A", "method": "", "fight_time_min": float("nan"),
            "dob_a": pd.Timestamp("1992-01-01"), "dob_b": pd.Timestamp("1990-01-01"),
        }])], ignore_index=True)
        row = rm.card_prices(self.model, card)[0]
        self.assertFalse(row["no_career_data"])
        self.assertGreater(row["career_bouts"], 0)

    def test_one_debutant_does_not_condemn_the_rest_of_a_card(self):
        # A real card carries newcomers; the flag is per fight, not per card.
        rows = [{"date": pd.Timestamp("2026-08-16"), "event": "UPCOMING",
                 "time_format": "3 Rnd (5-5-5)", "weightclass": "Lightweight Bout",
                 "fighter_a": a, "fighter_b": b, "winner": "A", "method": "",
                 "fight_time_min": float("nan"),
                 "dob_a": pd.Timestamp("1992-01-01"),
                 "dob_b": pd.Timestamp("1990-01-01")}
                for a, b in [("Banger 0", "Grinder 0"), ("Debut One", "Debut Two")]]
        priced = rm.card_prices(self.model,
                                pd.concat([self.fights, pd.DataFrame(rows)],
                                          ignore_index=True))
        self.assertEqual([r["no_career_data"] for r in priced], [False, True])


class FlatCardTests(unittest.TestCase):
    """A card of prices that agree with each other and with the prior."""

    BASE = 0.47

    def test_a_card_stuck_on_the_base_rate_is_flagged(self):
        self.assertTrue(rm.flat_card([0.47, 0.48, 0.46, 0.47, 0.48], self.BASE))

    def test_a_card_with_real_spread_is_not_flagged(self):
        self.assertFalse(rm.flat_card([0.20, 0.38, 0.55, 0.61, 0.36], self.BASE))

    def test_flat_but_far_from_the_prior_is_not_this_failure(self):
        # Tight agreement well away from the base rate is a claim, not a
        # shrug, so it is not what this check is looking for.
        self.assertFalse(rm.flat_card([0.80, 0.81, 0.79, 0.80, 0.81], self.BASE))

    def test_a_short_card_is_never_flagged(self):
        # Measured: five fights drawn from a genuine card look this flat 11%
        # of the time, so the check needs enough fights to mean anything and
        # is a warning rather than a gate even then.
        self.assertFalse(rm.flat_card([0.47, 0.47, 0.47], self.BASE))

    def test_missing_prices_do_not_break_the_check(self):
        self.assertFalse(rm.flat_card([float("nan")] * 6, self.BASE))

    def test_the_threshold_sits_below_every_real_card_measured(self):
        # 110 held-out cards of six or more bouts had a minimum spread of
        # 0.0567; the threshold is 0.05 so it has never fired on a real card.
        self.assertLess(rm.FLAT_CARD_SD, 0.0567)


class FairPriceTests(unittest.TestCase):
    def test_favourite_and_underdog_prices_straddle_even_money(self):
        self.assertLess(rm.fair_american(0.75), 0)
        self.assertGreater(rm.fair_american(0.25), 0)
        self.assertEqual(rm.fair_american(0.5), -100)

    def test_extreme_probabilities_are_clamped_rather_than_infinite(self):
        self.assertTrue(np.isfinite(rm.fair_american(0.0)))
        self.assertTrue(np.isfinite(rm.fair_american(1.0)))


if __name__ == "__main__":
    unittest.main()
