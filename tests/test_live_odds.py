import csv
import os
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fetch_odds import (LOG_FIELDS, MARKET_QUOTE_FIELDS, UPCOMING_FIELDS,
                        _american_to_prob, append_log, append_quote_log,
                        consensus_quote, main, paired_book_quotes)
from predict_card import execution_ladder, market_probability, quote_age_minutes


class LiveOddsConsensusTests(unittest.TestCase):
    def test_prediction_prefers_supplied_consensus_with_price_fallback(self):
        supplied, _, _ = market_probability(+150, -180, supplied=0.42)
        fallback, pa, pb = market_probability(+150, -180)
        self.assertEqual(supplied, 0.42)
        self.assertAlmostEqual(fallback, pa / (pa + pb))

    def test_consensus_uses_paired_book_devig_probabilities(self):
        event = {
            "home_team": "Fighter A",
            "away_team": "Fighter B",
            "bookmakers": [
                {"key": "one", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Fighter A", "price": -300},
                    {"name": "Fighter B", "price": 100},
                ]}]},
                {"key": "two", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Fighter A", "price": -200},
                    {"name": "Fighter B", "price": 500},
                ]}]},
                {"key": "three", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Fighter A", "price": -150},
                    {"name": "Fighter B", "price": 130},
                ]}]},
                {"key": "missing-side", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Fighter A", "price": -145},
                ]}]},
            ],
        }
        result = consensus_quote(event)
        probabilities = []
        for odds_a, odds_b in [(-300, 100), (-200, 500), (-150, 130)]:
            pa = _american_to_prob(odds_a)
            pb = _american_to_prob(odds_b)
            probabilities.append(pa / (pa + pb))
        self.assertEqual(result["market_books"], 3)
        self.assertEqual(result["odds_a"], -200)
        self.assertEqual(result["odds_b"], 130)
        self.assertEqual(result["best_odds_a"], -150)
        self.assertEqual(result["best_book_a"], "three")
        self.assertEqual(result["best_odds_b"], 500)
        self.assertEqual(result["best_book_b"], "two")
        self.assertGreater(result["market_spread"], 0)
        self.assertAlmostEqual(
            result["market_prob_a"], statistics.median(probabilities), places=8
        )

    def test_old_log_schema_is_migrated_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds_log.csv"
            path.write_text(
                "fetched_at,commence_time,date,fighter_a,fighter_b,odds_a,odds_b,odds_source\n"
                "2025-01-01T00:00:00Z,2025-01-02T00:00:00Z,2025-01-02,A,B,+120,-140,old\n",
                encoding="utf-8",
            )
            append_log(path, [{
                "fetched_at": "2025-01-01T01:00:00Z",
                "commence_time": "2025-01-02T00:00:00Z",
                "date": "2025-01-02",
                "fighter_a": "A",
                "fighter_b": "B",
                "odds_a": "+115",
                "odds_b": "-135",
                "market_prob_a": "0.46",
                "market_books": "8",
                "odds_source": "new",
            }])
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, LOG_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["market_prob_a"], "")
            self.assertEqual(rows[1]["market_books"], "8")

    def test_full_book_quote_log_is_deduplicated(self):
        event = {
            "id": "event-1", "commence_time": "2026-01-02T00:00:00Z",
            "home_team": "A", "away_team": "B", "bookmakers": [{
                "key": "fd", "title": "FanDuel",
                "last_update": "2026-01-01T00:00:00Z",
                "markets": [{"key": "h2h", "outcomes": [
                    {"name": "A", "price": -200},
                    {"name": "B", "price": 170},
                ]}],
            }],
        }
        quotes = paired_book_quotes(event)
        self.assertEqual(quotes[0]["book_title"], "FanDuel")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.csv"
            row = {field: "" for field in MARKET_QUOTE_FIELDS}
            row.update({"snapshot_id": "same", "book_title": "FanDuel"})
            append_quote_log(path, [row, row])
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)

    def test_fixed_consensus_execution_ladder(self):
        ladder = execution_ladder(0.7380193601, 0.0100331085)
        self.assertEqual(ladder["1u"], -220)
        self.assertEqual(ladder["2u_candidate"], -168)
        self.assertAlmostEqual(
            quote_age_minutes("2026-01-01T00:00:00Z",
                              now="2026-01-01T01:00:00Z"),
            60.0,
        )

    def test_required_key_fails_and_manual_template_has_no_fake_fight(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(SystemExit):
                        main(["--require-key"])
                    main([])
                with open("odds_upcoming.csv", newline="", encoding="utf-8") as source:
                    reader = csv.DictReader(source)
                    self.assertEqual(reader.fieldnames, UPCOMING_FIELDS)
                    self.assertEqual(list(reader), [])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
