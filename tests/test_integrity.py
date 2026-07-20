import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paper_ledger import (lock_paper_trades, record_prediction_snapshots,
                          settle_completed)
from production import (DIFF_FEATURES, MODEL_FEATURES, event_seed, fit_ensemble,
                        predict_probabilities)


class SeedAndSymmetryTests(unittest.TestCase):
    def test_event_seed_is_window_independent(self):
        self.assertEqual(event_seed("2026-07-11"),
                         event_seed(pd.Timestamp("2026-07-11")))
        self.assertNotEqual(event_seed("2026-07-11"),
                            event_seed("2026-07-18"))

    def test_prediction_symmetry(self):
        rows = []
        for i in range(12):
            row = {c: 0.0 for c in MODEL_FEATURES}
            row["line_logit"] = (-1) ** i * (0.1 + i / 20)
            row["line_abs"] = abs(row["line_logit"])
            for j, c in enumerate(DIFF_FEATURES):
                row[c] = ((i + j) % 5 - 2) / 4
            row["y"] = i % 2
            rows.append(row)
        train = pd.DataFrame(rows)
        models = fit_ensemble(train, n_models=2, seed=event_seed("test"))
        x = train.iloc[[0]][MODEL_FEATURES].copy()
        flipped = x.copy()
        flipped["line_logit"] *= -1
        flipped[DIFF_FEATURES] *= -1
        pa, _ = predict_probabilities(models, x)
        pb, _ = predict_probabilities(models, flipped)
        self.assertAlmostEqual(float(pa[0] + pb[0]), 1.0, places=10)


class LedgerIntegrityTests(unittest.TestCase):
    def prediction(self, start="2099-01-02T03:00:00Z"):
        return {
            "scheduled_start": start,
            "date": start[:10],
            "pick": "Fighter A",
            "opp": "Fighter B",
            "price": "+120",
            "market": 44.0,
            "model": 52.0,
            "edge": 8.0,
            "se": 1.0,
            "net": 7.0,
            "bet": True,
            "stake": 1,
            "meta": "Test",
            "odds_source": "test",
        }

    def test_retroactive_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "snapshots.csv"
            with self.assertRaises(ValueError):
                record_prediction_snapshots(
                    [self.prediction("2000-01-02T03:00:00Z")], path=path,
                    recorded_at="2000-01-03T00:00:00Z")
            self.assertFalse(path.exists())

    def test_official_trade_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            snapshots = Path(td) / "snapshots.csv"
            trades = Path(td) / "trades.csv"
            pred = self.prediction()
            stamp = "2099-01-01T00:00:00Z"
            self.assertEqual(record_prediction_snapshots(
                [pred], path=snapshots, recorded_at=stamp), 1)
            self.assertEqual(lock_paper_trades(
                [pred], snapshots_path=snapshots, trades_path=trades,
                locked_at=stamp), 1)
            self.assertEqual(lock_paper_trades(
                [pred], snapshots_path=snapshots, trades_path=trades,
                locked_at="2099-01-01T01:00:00Z"), 0)
            self.assertEqual(len(pd.read_csv(trades)), 1)

    def test_unverifiable_trade_does_not_settle(self):
        with tempfile.TemporaryDirectory() as td:
            trades = Path(td) / "trades.csv"
            settlements = Path(td) / "settlements.csv"
            fights = Path(td) / "fights.csv"
            pd.DataFrame([{
                "trade_id": "bad", "snapshot_id": "x",
                "locked_at": "2020-01-02T00:00:00Z",
                "scheduled_start": "2020-01-01T00:00:00Z",
                "timing_precision": "exact", "date": "2020-01-01",
                "fight_key": "x", "pick": "Fighter A", "opp": "Fighter B",
                "price": "+100", "market": 50, "model": 55, "edge": 5,
                "se": 0, "net_edge": 5, "stake": 1, "meta": "",
                "model_version": "test", "manifest_hash": "",
                "odds_source": "test", "odds_fetched_at": "",
            }]).to_csv(trades, index=False)
            pd.DataFrame([{
                "date": "2020-01-01", "fighter_a": "Fighter A",
                "fighter_b": "Fighter B", "winner": "A",
            }]).to_csv(fights, index=False)
            self.assertEqual(settle_completed(
                trades_path=trades, settlements_path=settlements,
                fights_path=fights, closing_path=Path(td) / "missing.csv"), 0)
            self.assertFalse(settlements.exists())


if __name__ == "__main__":
    unittest.main()
