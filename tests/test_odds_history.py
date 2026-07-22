import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest import american_to_prob
from historical_odds import _read_manifest
from prepare_api_odds_history import _consensus
from validate_entry_history import promotion_gate


class OddsConsensusTests(unittest.TestCase):
    def test_consensus_medians_per_book_devig_probabilities(self):
        rows = []
        prices = [(-300, 100), (-200, 500), (-150, 130)]
        for index, (odds_a, odds_b) in enumerate(prices):
            pa = american_to_prob([odds_a])[0]
            pb = american_to_prob([odds_b])[0]
            rows.append({
                "event_uid": "event-1",
                "event_name": "Card",
                "event_date": pd.Timestamp("2025-01-01", tz="UTC"),
                "snapshot_kind": "entry",
                "pair": "a|b",
                "actual_snapshot": pd.Timestamp("2024-12-30", tz="UTC"),
                "commence_time": pd.Timestamp("2025-01-01 03:00", tz="UTC"),
                "fighter_a": "A",
                "fighter_b": "B",
                "odds_a": odds_a,
                "odds_b": odds_b,
                "book_key": f"book-{index}",
                "book_prob_a": pa / (pa + pb),
            })
        quotes = pd.DataFrame(rows)
        result = _consensus(quotes, min_books=3).iloc[0]
        expected = quotes["book_prob_a"].median()
        median_odds_prob = (
            american_to_prob([quotes["odds_a"].median()])[0]
            / (american_to_prob([quotes["odds_a"].median()])[0]
               + american_to_prob([quotes["odds_b"].median()])[0])
        )
        self.assertAlmostEqual(result["consensus_prob_a"], expected)
        self.assertNotAlmostEqual(result["consensus_prob_a"], median_odds_prob)


class ManifestResumeTests(unittest.TestCase):
    def test_manifest_reuses_card_start_and_terminal_attempts(self):
        fields = [
            "event_uid", "event_name", "event_date", "snapshot_kind",
            "requested_snapshot", "actual_snapshot", "event_start_utc",
            "status", "matched_fights", "quote_rows", "response_sha256",
            "response_file", "requests_remaining", "requests_used",
            "requests_last",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "event_uid": "event-1",
                    "snapshot_kind": "discovery_72h",
                    "event_start_utc": "2025-01-02T01:00:00Z",
                    "status": "ok",
                })
                writer.writerow({
                    "event_uid": "event-1",
                    "snapshot_kind": "entry",
                    "event_start_utc": "2025-01-02T01:00:00Z",
                    "status": "no_quotes",
                })
            attempted, starts = _read_manifest(path)
        self.assertIn(("event-1", "discovery_72h"), attempted)
        self.assertIn(("event-1", "entry"), attempted)
        self.assertEqual(str(starts["event-1"]), "2025-01-02 01:00:00+00:00")


class PromotionGateTests(unittest.TestCase):
    @staticmethod
    def _qualifying_report():
        return {
            "events": 80,
            "bets": 250,
            "roi_ci90_event_clustered": [0.01, 0.20],
            "log_loss_model": 0.58,
            "log_loss_entry_market": 0.59,
            "closing_line_value": {
                "active_bets_with_close": 220,
                "mean_clv_prob_points_ci90_event_clustered": [0.05, 0.80],
                "positive_clv_rate_active_bets": 0.56,
            },
        }

    def test_gate_blocks_small_clv_sample(self):
        report = self._qualifying_report()
        report["closing_line_value"]["active_bets_with_close"] = 40
        gate = promotion_gate(report)
        self.assertEqual(gate["status"], "paper_only")
        self.assertFalse(gate["checks"]["minimum_active_clv_bets"]["passed"])

    def test_gate_marks_only_fully_qualified_candidate(self):
        gate = promotion_gate(self._qualifying_report())
        self.assertEqual(gate["status"], "candidate_for_manual_review")
        self.assertTrue(all(item["passed"] for item in gate["checks"].values()))


if __name__ == "__main__":
    unittest.main()
