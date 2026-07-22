import unittest

import numpy as np
import pandas as pd

from research_entry_models import (OFFSET_FEATURES, fit_market_offset,
                                   select_past_market_weight)


class MarketOffsetTests(unittest.TestCase):
    def test_zero_correction_is_the_entry_market(self):
        train = pd.DataFrame({feature: np.zeros(8) for feature in OFFSET_FEATURES})
        train["line_logit"] = [-1.2, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.2]
        train["y"] = [0, 0, 0, 1, 0, 1, 1, 1]
        model = fit_market_offset(train)
        expected = 1.0 / (1.0 + np.exp(-train["line_logit"].to_numpy()))
        np.testing.assert_allclose(model.predict_proba(train), expected, atol=1e-12)

    def test_corner_swap_is_exactly_symmetric(self):
        rng = np.random.default_rng(14)
        train = pd.DataFrame({
            feature: rng.normal(size=120) for feature in OFFSET_FEATURES
        })
        train["line_logit"] = rng.normal(size=len(train))
        signal = train["line_logit"] + 0.2 * train[OFFSET_FEATURES[0]]
        train["y"] = (signal + rng.normal(size=len(train)) > 0).astype(int)
        model = fit_market_offset(train)

        original = train.iloc[:20].copy()
        flipped = original.copy()
        flipped[list(OFFSET_FEATURES)] *= -1
        flipped["line_logit"] *= -1
        probability = model.predict_proba(original)
        flipped_probability = model.predict_proba(flipped)
        np.testing.assert_allclose(probability, 1.0 - flipped_probability, atol=1e-12)


class PastOnlyBlendTests(unittest.TestCase):
    def test_future_rows_cannot_change_selected_weight(self):
        cutoff = pd.Timestamp("2025-01-10", tz="UTC")
        prior = pd.DataFrame({
            "commence_time": pd.to_datetime([
                "2025-01-01T03:00:00Z",
                "2025-01-02T03:00:00Z",
                "2025-01-03T03:00:00Z",
                "2025-01-04T03:00:00Z",
            ]),
            "y": [1, 0, 1, 0],
            "p_line": [0.75, 0.25, 0.70, 0.30],
            "p_current": [0.55, 0.45, 0.52, 0.48],
        })
        weight, details = select_past_market_weight(
            prior, cutoff, grid=(0.0, 0.5, 1.0), min_history=4
        )

        future = pd.DataFrame({
            "commence_time": pd.to_datetime(["2025-02-01T03:00:00Z"] * 20),
            "y": [0, 1] * 10,
            "p_line": [0.99, 0.01] * 10,
            "p_current": [0.01, 0.99] * 10,
        })
        with_future, future_details = select_past_market_weight(
            pd.concat([prior, future], ignore_index=True),
            cutoff,
            grid=(0.0, 0.5, 1.0),
            min_history=4,
        )
        self.assertEqual(weight, with_future)
        self.assertEqual(weight, 1.0)
        self.assertEqual(details["history_rows"], 4)
        self.assertEqual(future_details["history_rows"], 4)


if __name__ == "__main__":
    unittest.main()
