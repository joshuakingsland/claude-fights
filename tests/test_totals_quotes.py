import unittest

import pandas as pd

import historical_odds as ho

WHEN = pd.Timestamp("2024-01-20T23:20:00Z")
START = pd.Timestamp("2024-01-21T00:00:00Z")
EVENT = {"event_uid": "u", "event_name": "e", "event_date": "2024-01-20"}


def _payload(markets):
    return [{"id": "e1", "home_team": "Red", "away_team": "Blue",
             "commence_time": START.isoformat(),
             "bookmakers": [{"key": "bk", "title": "Book", "last_update": "",
                             "markets": markets}]}]


def _parse(markets):
    return ho._quotes(_payload(markets), EVENT, "close", WHEN, "a", START, "sha")


class TotalsParsingTests(unittest.TestCase):
    """Totals return Over/Under with a point, not two fighter names.

    The parser matched outcomes against fighter names, so a totals response
    produced nothing and the manifest recorded no_quotes - a terminal status
    that is never retried. The probe was sent to a scratch directory for
    exactly that reason.
    """

    def test_an_over_under_pair_is_captured_with_its_line(self):
        rows = _parse([{"key": "totals", "last_update": "", "outcomes": [
            {"name": "Over", "point": 1.5, "price": 100},
            {"name": "Under", "point": 1.5, "price": -130}]}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_key"], "totals")
        self.assertEqual(rows[0]["point"], 1.5)
        self.assertEqual((rows[0]["odds_a"], rows[0]["odds_b"]), (100, -130))

    def test_mismatched_lines_are_refused_rather_than_paired(self):
        # Over 1.5 and Under 2.5 are not the same market. De-vigging them
        # against each other would produce a number meaning nothing.
        rows = _parse([{"key": "totals", "last_update": "", "outcomes": [
            {"name": "Over", "point": 1.5, "price": -110},
            {"name": "Under", "point": 2.5, "price": -110}]}])
        self.assertEqual(rows, [])

    def test_a_one_sided_quote_is_skipped(self):
        rows = _parse([{"key": "totals", "last_update": "", "outcomes": [
            {"name": "Over", "point": 1.5, "price": -110}]}])
        self.assertEqual(rows, [])

    def test_moneyline_parsing_is_unchanged(self):
        rows = _parse([{"key": "h2h", "last_update": "", "outcomes": [
            {"name": "Red", "price": -150}, {"name": "Blue", "price": 130}]}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_key"], "h2h")
        self.assertEqual(rows[0]["point"], "")
        self.assertEqual((rows[0]["odds_a"], rows[0]["odds_b"]), (-150, 130))

    def test_both_markets_in_one_response_stay_separable(self):
        rows = _parse([
            {"key": "h2h", "last_update": "", "outcomes": [
                {"name": "Red", "price": -150}, {"name": "Blue", "price": 130}]},
            {"key": "totals", "last_update": "", "outcomes": [
                {"name": "Over", "point": 2.5, "price": 120},
                {"name": "Under", "point": 2.5, "price": -145}]}])
        self.assertEqual({r["market_key"] for r in rows}, {"h2h", "totals"})

    def test_an_unknown_market_is_ignored(self):
        self.assertEqual(_parse([{"key": "double_chance", "last_update": "",
                                  "outcomes": [{"name": "x", "price": 100}]}]), [])


if __name__ == "__main__":
    unittest.main()
