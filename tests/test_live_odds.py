import csv
import json
import os
import statistics
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from config import PRICED_ODDS_REGIONS
from fetch_odds import (LOG_FIELDS, MARKET_QUOTE_FIELDS, UPCOMING_FIELDS,
                        _american_to_prob, _is_future, append_log,
                        append_quote_log, classify_promotion, collect_events,
                        consensus_quote, main, leader_split, paired_book_quotes,
                        priced_quotes, promotion_matches)
from predict_card import (execution_ladder, filter_upcoming_promotion,
                          market_probability, quote_age_minutes)


def _event(event_id, books, **extra):
    return {
        "id": event_id, "commence_time": "2099-01-02T00:00:00Z",
        "home_team": "A", "away_team": "B",
        "bookmakers": [{
            "key": key, "title": title,
            "last_update": "2026-01-01T00:00:00Z",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "A", "price": odds_a},
                {"name": "B", "price": odds_b},
            ]}],
        } for key, title, odds_a, odds_b in books],
        **extra,
    }


class InPlayRejectionTests(unittest.TestCase):
    """A fight already under way must never reach the card.

    The odds endpoint keeps returning a fight after it starts. Those are
    in-play prices reflecting what has happened in the cage. On 2026-08-07
    one was captured three minutes after its start, reached predict_card,
    and assert_pre_event correctly refused it - taking the scheduled
    snapshot run down with it. Failing closed there was right; capturing
    the row at all was the bug.
    """

    def test_future_start_is_accepted(self):
        ahead = datetime.now(timezone.utc) + timedelta(hours=2)
        self.assertTrue(_is_future(ahead.strftime("%Y-%m-%dT%H:%M:%SZ")))

    def test_fight_already_started_is_rejected(self):
        behind = datetime.now(timezone.utc) - timedelta(minutes=3)
        self.assertFalse(_is_future(behind.strftime("%Y-%m-%dT%H:%M:%SZ")))

    def test_the_exact_failing_row_is_rejected(self):
        now = datetime(2026, 8, 7, 13, 38, 13, tzinfo=timezone.utc)
        self.assertFalse(_is_future("2026-08-07T13:35:00Z", now))

    def test_missing_or_malformed_start_is_rejected(self):
        self.assertFalse(_is_future(""))
        self.assertFalse(_is_future(None))
        self.assertFalse(_is_future("soon"))


class ResearchRegionTests(unittest.TestCase):
    """A captured region must not reach the model until it is priced."""

    US = [("dk", "DraftKings", -200, 170), ("fd", "FanDuel", -210, 175),
          ("betonlineag", "BetOnline.ag", -205, 180)]
    # Deliberately far off the US market and offering a much better B price.
    EU = [("pinnacle", "Pinnacle", -320, 260)]

    def _fetch(self, key, region):
        return {"us": [_event("e1", self.US)],
                "eu": [_event("e1", self.EU)]}.get(region, [])

    def setUp(self):
        self.assertEqual(tuple(PRICED_ODDS_REGIONS), ("us",))

    def test_unpriced_region_is_captured_but_not_consumed(self):
        collected = collect_events("k", ("us", "eu"), fetch=self._fetch)
        self.assertEqual(len(collected), 1)
        _, paired = collected[0]
        self.assertEqual(len(paired), 4)
        self.assertIn("pinnacle", {q["book_key"] for q in paired})
        self.assertEqual(len(priced_quotes(paired)), 3)

    def test_consensus_and_execution_ignore_the_research_region(self):
        _, paired = collect_events("k", ("us", "eu"), fetch=self._fetch)[0]
        with_eu = consensus_quote(None, paired)
        us_only = consensus_quote(None, priced_quotes(paired))
        self.assertEqual(with_eu, us_only)
        self.assertEqual(with_eu["market_books"], 3)
        # Pinnacle's +260 is the best B price on the card and must be ignored.
        self.assertEqual(with_eu["best_odds_b"], 180)
        self.assertNotEqual(with_eu["best_book_b"], "Pinnacle")

    def test_region_order_does_not_change_the_consensus(self):
        forward = collect_events("k", ("us", "eu"), fetch=self._fetch)[0][1]
        reverse = collect_events("k", ("eu", "us"), fetch=self._fetch)[0][1]
        self.assertEqual(consensus_quote(None, forward),
                         consensus_quote(None, reverse))

    def test_a_book_in_two_regions_keeps_its_priced_quote(self):
        def fetch(key, region):
            return {"us": [_event("e1", self.US)],
                    "eu": [_event("e1", [("dk", "DraftKings", -900, 600)])]}[region]
        _, paired = collect_events("k", ("us", "eu"), fetch=fetch)[0]
        dk = [q for q in paired if q["book_key"] == "dk"]
        self.assertEqual(len(dk), 1)
        self.assertEqual(dk[0]["region"], "us")
        self.assertEqual(dk[0]["odds_a"], -200)

    def test_leader_split_counts_unpriced_leaders_but_priced_followers(self):
        _, paired = collect_events("k", ("us", "eu"), fetch=self._fetch)[0]
        split = leader_split(paired)
        # Pinnacle (eu, unpriced) and BetOnline (us) are both market setters.
        self.assertEqual(split["leader_books"], 2)
        # Followers are the priced non-leader books: DraftKings and FanDuel.
        self.assertEqual(split["follower_books"], 2)
        self.assertGreater(split["leader_prob_a"], split["follower_prob_a"])

    def test_leader_split_does_not_touch_the_consensus(self):
        _, paired = collect_events("k", ("us", "eu"), fetch=self._fetch)[0]
        before = consensus_quote(None, paired)
        leader_split(paired)
        self.assertEqual(consensus_quote(None, paired), before)
        self.assertNotIn("leader_prob_a", before)

    def test_leader_split_is_blank_when_no_setter_quoted(self):
        event = _event("e1", [("dk", "DraftKings", -200, 170),
                              ("fd", "FanDuel", -210, 175)])
        split = leader_split(paired_book_quotes(event))
        self.assertEqual(split["leader_prob_a"], "")
        self.assertEqual(split["leader_books"], 0)
        self.assertEqual(split["follower_books"], 2)

    def test_quotes_carry_region_and_priced_provenance(self):
        _, paired = collect_events("k", ("us", "eu"), fetch=self._fetch)[0]
        by_key = {q["book_key"]: q for q in paired}
        self.assertEqual(by_key["pinnacle"]["region"], "eu")
        self.assertEqual(by_key["pinnacle"]["priced"], 0)
        self.assertEqual(by_key["dk"]["priced"], 1)
        for field in ("region", "priced"):
            self.assertIn(field, MARKET_QUOTE_FIELDS)


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

    def test_dwcs_promotion_is_detected_from_event_metadata(self):
        dwcs = _event(
            "dwcs-1", [("fd", "FanDuel", -120, 100)],
            event_title="Dana White's Contender Series: Week 1",
        )
        ufc = _event("ufc-1", [("fd", "FanDuel", -120, 100)],
                     sport_title="UFC")
        unlabeled = _event("mma-1", [("fd", "FanDuel", -120, 100)],
                           sport_title="MMA")
        self.assertEqual(classify_promotion(dwcs), "DWCS")
        self.assertEqual(classify_promotion(ufc), "UFC")
        self.assertTrue(promotion_matches(dwcs, "dwcs"))
        self.assertFalse(promotion_matches(ufc, "dwcs"))
        self.assertTrue(promotion_matches(unlabeled, "ufc"))

    def test_prediction_filter_accepts_tagged_dwcs_rows(self):
        rows = [
            {"date": "2099-01-01", "promotion": "DWCS",
             "event_title": "Dana White's Contender Series", "fighter_a": "A",
             "fighter_b": "B", "odds_a": -120, "odds_b": 100},
            {"date": "2099-01-01", "promotion": "UFC",
             "event_title": "UFC Fight Night", "fighter_a": "C",
             "fighter_b": "D", "odds_a": -140, "odds_b": 120},
            {"date": "2099-01-02", "fighter_a": "E", "fighter_b": "F",
             "odds_a": -110, "odds_b": -110},
        ]
        import pandas as pd
        frame = pd.DataFrame(rows)
        self.assertEqual(len(filter_upcoming_promotion(frame, "dwcs")), 1)
        self.assertEqual(len(filter_upcoming_promotion(frame, "ufc")), 2)
        self.assertEqual(len(filter_upcoming_promotion(frame, "all")), 3)

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

    def test_multi_region_run_prices_us_and_files_eu_as_research(self):
        us = [("dk", "DraftKings", -200, 170), ("fd", "FanDuel", -210, 175),
              ("betonlineag", "BetOnline.ag", -205, 180)]
        eu = [("pinnacle", "Pinnacle", -320, 260)]

        def fetch(key, region, timeout=30):
            return {"us": [_event("e1", us)], "eu": [_event("e1", eu)]}[region]

        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                with patch.dict(os.environ, {"ODDS_API_KEY": "k"}, clear=True):
                    with patch("fetch_odds.fetch_region", fetch):
                        main(["--require-key", "--promotion", "all"])
                with open("odds_upcoming.csv", newline="", encoding="utf-8") as src:
                    card = list(csv.DictReader(src))
                self.assertEqual(len(card), 1)
                # Pinnacle is in the quote file but not in the model's input.
                self.assertEqual(card[0]["market_books"], "3")
                self.assertEqual(card[0]["best_book_b"], "BetOnline.ag")
                self.assertEqual(float(card[0]["best_odds_b"]), 180.0)
                quotes = list(Path("data/market_quotes").glob("quotes_*.csv"))
                self.assertEqual(len(quotes), 1)
                with quotes[0].open(newline="", encoding="utf-8") as src:
                    rows = list(csv.DictReader(src))
                by_book = {row["book_key"]: row for row in rows}
                self.assertEqual(len(rows), 4)
                self.assertEqual(by_book["pinnacle"]["region"], "eu")
                self.assertEqual(by_book["pinnacle"]["priced"], "0")
                self.assertEqual(by_book["betonlineag"]["priced"], "1")
                manifest = json.loads(
                    Path("market_snapshot_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["regions_requested"], ["us", "eu"])
                self.assertEqual(manifest["regions_priced"], ["us"])
                self.assertEqual(manifest["promotion"], "all")
                self.assertEqual(manifest["paired_book_quotes"], 4)
                self.assertEqual(manifest["priced_book_quotes"], 3)
            finally:
                os.chdir(previous)

    def test_dwcs_run_filters_to_dwcs_events(self):
        priced = [("dk", "DraftKings", -200, 170),
                  ("fd", "FanDuel", -210, 175),
                  ("betonlineag", "BetOnline.ag", -205, 180)]

        def fetch(key, region, timeout=30):
            return [_event("dwcs", priced,
                           event_title="Dana White's Contender Series: Week 1"),
                    _event("ufc", priced, event_title="UFC Fight Night")]

        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                with patch.dict(os.environ, {"ODDS_API_KEY": "k"}, clear=True):
                    with patch("fetch_odds.fetch_region", fetch):
                        main(["--require-key", "--promotion", "dwcs"])
                with open("odds_upcoming.csv", newline="", encoding="utf-8") as src:
                    card = list(csv.DictReader(src))
                self.assertEqual(len(card), 1)
                self.assertEqual(card[0]["promotion"], "DWCS")
                self.assertIn("Contender Series", card[0]["event_title"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
