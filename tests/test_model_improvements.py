import unittest

import numpy as np
import pandas as pd

from audit_model_improvements import (
    apply_symmetric_temperature, score_policy,
    walk_forward_timing_predictions,
)


class SymmetricCalibrationTests(unittest.TestCase):
    def test_temperature_preserves_side_swap_symmetry(self):
        probability = np.array([0.08, 0.25, 0.50, 0.71, 0.93])
        calibrated = apply_symmetric_temperature(probability, 0.82)
        flipped = apply_symmetric_temperature(1.0 - probability, 0.82)
        np.testing.assert_allclose(calibrated, 1.0 - flipped, atol=1e-12)

    def test_uncertainty_transform_is_nonnegative_and_symmetric(self):
        probability = np.array([0.2, 0.7])
        uncertainty = np.array([0.03, 0.04])
        _, transformed = apply_symmetric_temperature(probability, 0.75, uncertainty)
        _, flipped = apply_symmetric_temperature(
            1.0 - probability, 0.75, uncertainty
        )
        self.assertTrue((transformed >= 0).all())
        np.testing.assert_allclose(transformed, flipped, atol=1e-12)


class ImprovementPolicyTests(unittest.TestCase):
    @staticmethod
    def frame():
        return pd.DataFrame({
            "date": ["2025-01-01", "2025-01-01"],
            "R_odds": [-110, 120],
            "B_odds": [-110, -140],
            "p_model": [0.60, 0.30],
            "se": [0.01, 0.02],
            "y": [1, 0],
        })

    def test_locked_pick_never_changes_with_alternate_price(self):
        scored = score_policy(
            self.frame(), odds_a=[-300, -300], odds_b=[240, 240],
            locked_pick=np.array(["A", "B"]),
        )
        self.assertEqual(scored["pick_side"].tolist(), ["A", "B"])

    def test_eligibility_filter_is_applied_before_card_cap(self):
        frame = self.frame()
        scored = score_policy(frame, eligible=np.array([False, True]))
        self.assertEqual(scored.loc[0, "stake"], 0)
        self.assertEqual(scored.loc[1, "stake"], 1)


class TimingLeakageTests(unittest.TestCase):
    def test_future_target_cannot_change_earlier_prediction(self):
        dates = pd.to_datetime([
            *("2023-01-%02d" % day for day in range(1, 9)),
            "2025-01-01",
            "2026-01-01",
        ], utc=True)
        frame = pd.DataFrame({
            "date": dates,
            "fight_id": [f"fight-{index}" for index in range(len(dates))],
            "timing_target": np.linspace(-0.02, 0.02, len(dates)),
        })
        for index, feature in enumerate((
            "gross_edge", "se", "entry_lead_hours", "entry_n_books",
            "model_confidence", "market_confidence", "pick_is_favorite",
            "entry_source_api",
        )):
            frame[feature] = np.linspace(index, index + 1, len(frame))

        baseline = walk_forward_timing_predictions(frame, min_train=8)
        changed = frame.copy()
        changed.loc[changed["date"].dt.year == 2026, "timing_target"] = 999.0
        rerun = walk_forward_timing_predictions(changed, min_train=8)
        first_baseline = baseline.loc[baseline["date"].dt.year == 2025,
                                      "predicted_timing_target"].iloc[0]
        first_changed = rerun.loc[rerun["date"].dt.year == 2025,
                                  "predicted_timing_target"].iloc[0]
        self.assertAlmostEqual(first_baseline, first_changed, places=12)


if __name__ == "__main__":
    unittest.main()
