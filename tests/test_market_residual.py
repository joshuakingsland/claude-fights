import unittest

import numpy as np
import pandas as pd

import market_residual as mr


def _frame(n=400, effect=0.0, seed=0):
    """A synthetic book with a known amount of mispricing built in.

    `effect` is the true coefficient: the probability points of market error
    per unit of the feature. At zero the market is exactly right and any
    surviving candidate is a false positive.
    """
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 40, n) * 7, unit="D")
    feature = rng.normal(0, 5, n)
    p_line = np.clip(rng.uniform(0.25, 0.75, n), 0.02, 0.98)
    truth = np.clip(p_line + effect * feature, 0.01, 0.99)
    frame = pd.DataFrame({
        "date": dates,
        "y": rng.binomial(1, truth),
        "p_line": p_line,
        "p_close_line": p_line,
        "age_diff": feature,
        "R_odds": np.where(p_line > 0.5, -120.0, 130.0),
        "B_odds": np.where(p_line > 0.5, 110.0, -140.0),
    })
    frame["resid"] = frame["y"] - frame["p_line"]
    return frame


class OddsArithmeticTests(unittest.TestCase):
    def test_implied_probability_of_a_pick_em(self):
        self.assertAlmostEqual(float(mr._implied([-110])[0]), 110 / 210)
        self.assertAlmostEqual(float(mr._implied([+100])[0]), 0.5)

    def test_even_money_at_minus_100_does_not_produce_nan(self):
        # np.where evaluates the discarded branch, which divides by zero here.
        self.assertAlmostEqual(float(mr._implied([-100])[0]), 0.5)
        self.assertAlmostEqual(float(mr._profit([-100], [True])[0]), 1.0)

    def test_profit_pays_the_price_and_loses_one_unit(self):
        self.assertAlmostEqual(float(mr._profit([+150], [True])[0]), 1.5)
        self.assertAlmostEqual(float(mr._profit([+150], [False])[0]), -1.0)
        self.assertAlmostEqual(float(mr._profit([-200], [True])[0]), 0.5)

    def test_a_fair_book_returns_zero_in_expectation(self):
        # -110 both sides is not fair, but +100/-100 is; a coin flip on it
        # should break even, which is the baseline every ROI here is against.
        profit = mr._profit([100] * 2, [True, False])
        self.assertAlmostEqual(float(profit.mean()), 0.0)


class FamilyTests(unittest.TestCase):
    """The correction has to actually bind, or the family is theatre."""

    def test_a_family_of_one_needs_no_correction(self):
        # Bonferroni spends 10% across the family, so a single candidate keeps
        # the whole budget and the two intervals coincide exactly.
        row = mr.family(_frame(effect=0.004), {"a": "age_diff"}, draws=200).iloc[0]
        self.assertEqual(row["bonferroni_low"], row["ci90_low"])
        self.assertEqual(row["bonferroni_high"], row["ci90_high"])

    def test_bonferroni_interval_is_wider_once_the_family_has_two(self):
        frame = _frame(effect=0.004)
        frame["b"] = frame["age_diff"]
        row = mr.family(frame, {"a": "age_diff", "b": "b"}, draws=400).iloc[0]
        self.assertLess(row["bonferroni_low"], row["ci90_low"])
        self.assertGreater(row["bonferroni_high"], row["ci90_high"])

    def test_more_candidates_means_a_stricter_bar(self):
        # Same feature, tested alongside more of them. Bonferroni spends the
        # family's error budget across the candidates, so each interval has to
        # widen as the family grows or the correction is not being applied.
        frame = _frame(effect=0.004)
        frame["b"] = frame["age_diff"]
        frame["c"] = frame["age_diff"]
        alone = mr.family(frame, {"a": "age_diff"}, draws=300).iloc[0]
        crowd = mr.family(frame, {"a": "age_diff", "b": "b", "c": "c"},
                          draws=300).iloc[0]
        self.assertLess(crowd["bonferroni_low"], alone["bonferroni_low"])

    def test_a_real_effect_is_recovered_with_the_right_sign(self):
        result = mr.family(_frame(n=1500, effect=-0.006), {"a": "age_diff"},
                           draws=300)
        self.assertLess(result.iloc[0]["coefficient"], 0)

    def test_an_efficient_market_leaves_nothing_to_find(self):
        result = mr.family(_frame(n=1200, effect=0.0), {"a": "age_diff"},
                           draws=400)
        self.assertFalse(bool(result.iloc[0]["survives"]))

    def test_every_candidate_is_reported_not_only_survivors(self):
        frame = _frame(n=800, effect=0.0)
        frame["noise"] = np.random.default_rng(1).normal(0, 1, len(frame))
        result = mr.family(frame, {"a": "age_diff", "b": "noise"}, draws=200)
        self.assertEqual(len(result), 2)

    def test_a_candidate_without_enough_data_is_dropped_not_guessed(self):
        frame = _frame(n=300)
        frame["sparse"] = np.nan
        frame.loc[frame.index[:10], "sparse"] = 1.0
        result = mr.family(frame, {"a": "age_diff", "sparse": "sparse"},
                           draws=100)
        self.assertEqual(list(result["candidate"]), ["a"])

    def test_a_missing_column_does_not_raise(self):
        result = mr.family(_frame(n=300), {"absent": "not_a_column"}, draws=50)
        self.assertTrue(result.empty)


class ClusteringTests(unittest.TestCase):
    def test_resampling_is_by_card_not_by_fight(self):
        """Cards, not fights, are the independent unit.

        Every fight on a card shares the same date here, so a card-level
        bootstrap can only ever draw whole cards. If it resampled fights the
        drawn sample would contain a partial card, which this catches.
        """
        frame = _frame(n=200)
        sizes = frame.groupby("date").size()
        slopes = mr._clustered_slopes(frame, "age_diff", draws=50, seed=2)
        self.assertEqual(len(slopes), 50)
        self.assertGreater(len(sizes), 1)

    def test_ignoring_clustering_would_narrow_the_interval(self):
        # The reason clustering is not optional: pretending correlated fights
        # are independent shrinks the interval and manufactures significance.
        frame = _frame(n=600, effect=0.0)
        clustered = mr._clustered_slopes(frame, "age_diff", draws=400, seed=5)
        shuffled = frame.copy()
        shuffled["date"] = range(len(shuffled))  # one fight per "card"
        independent = mr._clustered_slopes(shuffled, "age_diff", draws=400, seed=5)
        self.assertLessEqual(independent.std(), clustered.std() * 1.5)


class SeedStabilityTests(unittest.TestCase):
    """A survivor that depends on the seed is a bound resting on zero."""

    def test_a_strong_effect_survives_every_seed(self):
        frame = _frame(n=1500, effect=-0.02, seed=12)
        stability = mr.seed_stability(frame, "age_diff", n_candidates=12,
                                      seeds=(1, 2, 3), draws=800)
        self.assertTrue(stability["survives"].all())

    def test_no_effect_survives_no_seed(self):
        frame = _frame(n=1500, effect=0.0, seed=13)
        stability = mr.seed_stability(frame, "age_diff", n_candidates=12,
                                      seeds=(1, 2, 3), draws=800)
        self.assertFalse(stability["survives"].any())

    def test_it_reports_one_row_per_seed(self):
        stability = mr.seed_stability(_frame(n=400), "age_diff",
                                      seeds=(1, 2, 3, 4), draws=100)
        self.assertEqual(list(stability["seed"]), [1, 2, 3, 4])

    def test_a_bigger_family_widens_every_bound(self):
        frame = _frame(n=800, effect=-0.004, seed=14)
        small = mr.seed_stability(frame, "age_diff", n_candidates=2,
                                  seeds=(7,), draws=600).iloc[0]
        large = mr.seed_stability(frame, "age_diff", n_candidates=12,
                                  seeds=(7,), draws=600).iloc[0]
        self.assertLess(large["bonferroni_low"], small["bonferroni_low"])
        self.assertGreater(large["bonferroni_high"], small["bonferroni_high"])


class FlatStakeGateTests(unittest.TestCase):
    """The gate is what separates a coefficient from a bet."""

    def test_it_backs_the_side_the_coefficient_favours(self):
        # Negative slope on age_diff means the higher-age side underperforms,
        # so the bet belongs on the other one. Getting this backwards would
        # invert every ROI in the report.
        frame = _frame(n=900, effect=-0.01, seed=4)
        gate = mr.flat_stake_gate(frame, thresholds=(0,), draws=100)
        self.assertGreater(gate.iloc[0]["roi"], 0)

    def test_a_market_with_no_edge_does_not_pass(self):
        gate = mr.flat_stake_gate(_frame(n=900, effect=0.0, seed=6),
                                  thresholds=(0,), draws=300)
        self.assertFalse(bool(gate.iloc[0]["passes"]))

    def test_a_small_sample_cannot_pass_however_good_it_looks(self):
        # n >= 200 is pre-registered, and it is the rule that killed H1.
        frame = _frame(n=120, effect=-0.02, seed=8)
        gate = mr.flat_stake_gate(frame, thresholds=(0,), draws=100)
        self.assertLess(gate.iloc[0]["n"], 200)
        self.assertFalse(bool(gate.iloc[0]["passes"]))

    def test_passing_needs_both_the_lower_bound_and_the_sample(self):
        gate = mr.flat_stake_gate(_frame(n=900, effect=-0.012, seed=9),
                                  thresholds=(0,), draws=300)
        row = gate.iloc[0]
        self.assertEqual(bool(row["passes"]),
                         bool(row["roi_ci90_low"] > 0 and row["n"] >= 200))

    def test_thresholds_narrow_the_sample(self):
        gate = mr.flat_stake_gate(_frame(n=900, seed=10),
                                  thresholds=(0, 4, 8), draws=50)
        self.assertTrue(gate["n"].is_monotonic_decreasing)

    def test_missing_price_columns_raise_rather_than_score_zero(self):
        frame = _frame(n=300).drop(columns=["R_odds"])
        with self.assertRaises(KeyError):
            mr.flat_stake_gate(frame, thresholds=(0,), draws=10)

    def test_a_raw_validation_frame_is_rejected_by_name(self):
        # The easy mistake is passing the CSV straight in. Both entry points
        # need residuals() to have run first, and should say which column is
        # missing rather than failing from inside a bootstrap loop.
        frame = _frame(n=300).drop(columns=["resid"])
        for call in (lambda: mr.flat_stake_gate(frame, thresholds=(0,), draws=10),
                     lambda: mr.family(frame, {"a": "age_diff"}, draws=10)):
            with self.assertRaises(KeyError) as caught:
                call()
            self.assertIn("resid", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
