import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import forward_test as ft

FIGHT = pd.Timestamp("2026-09-05T23:00:00Z")


def _quotes(rows, commence=FIGHT, lead=12.0, a="Ann Ace", b="Bea Bolt"):
    """rows: (book_key, odds_a, odds_b). One snapshot `lead` hours out."""
    fetched = commence - pd.Timedelta(hours=lead)
    return pd.DataFrame([
        {"book_key": book, "odds_a": oa, "odds_b": ob,
         "commence_time": commence, "fetched_at": fetched,
         "lead_hours": lead, "date": str(commence.date()),
         "fighter_a": a, "fighter_b": b}
        for book, oa, ob in rows])


def _heavy_favourite(a_side=True, extra=6):
    """Pinnacle at ~78% on one side, plus a pool of shorter-priced books."""
    pin = (-360, +290) if a_side else (+290, -360)
    rows = [("pinnacle", *pin)]
    for i in range(extra):
        price = (-300 - i, +250 + i) if a_side else (+250 + i, -300 - i)
        rows.append((f"book{i}", *price))
    return rows


class FightKeyTests(unittest.TestCase):
    def test_the_key_does_not_depend_on_which_corner_is_listed_first(self):
        """A fight logged under two orderings would be bet twice."""
        one = ft._fight_key({"date": "2026-09-05", "fighter_a": "Ann Ace",
                             "fighter_b": "Bea Bolt"})
        other = ft._fight_key({"date": "2026-09-05", "fighter_a": "Bea Bolt",
                               "fighter_b": "Ann Ace"})
        self.assertEqual(one, other)


class CandidateTests(unittest.TestCase):
    def test_a_heavy_favourite_is_selected_at_the_best_pool_price(self):
        got = ft.candidates(_quotes(_heavy_favourite()))
        self.assertEqual(len(got), 1)
        row = got.iloc[0]
        self.assertEqual(row["pick"], "Ann Ace")
        self.assertEqual(row["bet_odds"], -300)      # best in the pool
        self.assertEqual(row["bet_book"], "book0")
        self.assertNotEqual(row["bet_book"], "pinnacle")

    def test_it_backs_side_b_when_b_is_the_favourite(self):
        got = ft.candidates(_quotes(_heavy_favourite(a_side=False)))
        self.assertEqual(got.iloc[0]["pick"], "Bea Bolt")
        self.assertEqual(got.iloc[0]["side"], "b")

    def test_a_coin_flip_is_not_selected(self):
        rows = [("pinnacle", -110, -110)] + [(f"b{i}", -105, -105) for i in range(6)]
        self.assertEqual(len(ft.candidates(_quotes(rows))), 0)

    def test_a_fight_without_the_reference_book_is_skipped(self):
        rows = [(f"book{i}", -300, +250) for i in range(7)]
        self.assertEqual(len(ft.candidates(_quotes(rows))), 0)

    def test_a_thin_pool_is_skipped(self):
        rows = [("pinnacle", -360, +290), ("book0", -300, +250)]
        self.assertEqual(len(ft.candidates(_quotes(rows))), 0)

    def test_a_snapshot_outside_the_window_is_ignored(self):
        self.assertEqual(len(ft.candidates(_quotes(_heavy_favourite(), lead=72.0))), 0)
        self.assertEqual(len(ft.candidates(_quotes(_heavy_favourite(), lead=1.0))), 0)

    def test_only_one_bet_per_fight_even_with_many_snapshots(self):
        """Logging a fight again as its line moves would double-count one view."""
        frames = [_quotes(_heavy_favourite(), lead=lead) for lead in (8.0, 12.0, 20.0)]
        got = ft.candidates(pd.concat(frames, ignore_index=True))
        self.assertEqual(len(got), 1)

    def test_it_uses_the_snapshot_nearest_the_tested_horizon(self):
        frames = [_quotes(_heavy_favourite(), lead=lead) for lead in (7.0, 12.5, 23.0)]
        got = ft.candidates(pd.concat(frames, ignore_index=True))
        self.assertAlmostEqual(float(got.iloc[0]["lead_hours"]), 12.5)

    def test_a_fight_inside_the_holdout_is_refused(self):
        """It is already counted once out of sample."""
        old = pd.Timestamp("2026-08-01T23:00:00Z")
        self.assertEqual(len(ft.candidates(_quotes(_heavy_favourite(), commence=old))), 0)

    def test_a_fight_after_the_holdout_is_allowed(self):
        fresh = pd.Timestamp("2026-08-09T23:00:00Z")
        self.assertEqual(
            len(ft.candidates(_quotes(_heavy_favourite(), commence=fresh))), 1)

    def test_an_exchange_never_supplies_the_price(self):
        rows = _heavy_favourite() + [("betfair", -150, +130)]
        self.assertNotEqual(ft.candidates(_quotes(rows)).iloc[0]["bet_book"], "betfair")

    def test_a_bet_written_after_the_bell_is_marked_as_backfill(self):
        past = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
        got = ft.candidates(_quotes(_heavy_favourite(), commence=past))
        self.assertFalse(bool(got.iloc[0]["logged_before_event"]))

    def test_a_bet_written_before_the_bell_is_marked_forward(self):
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=5)
        got = ft.candidates(_quotes(_heavy_favourite(), commence=future))
        self.assertTrue(bool(got.iloc[0]["logged_before_event"]))

    def test_empty_input_returns_the_right_shape(self):
        got = ft.candidates(pd.DataFrame())
        self.assertEqual(list(got.columns), ft.LOG_FIELDS)


class LogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "log.csv"

    def test_a_bet_is_written_once(self):
        rows = ft.candidates(_quotes(_heavy_favourite()))
        self.assertEqual(ft.append_log(rows, self.path), 1)
        self.assertEqual(ft.append_log(rows, self.path), 0)
        self.assertEqual(len(pd.read_csv(self.path)), 1)

    def test_a_relogged_fight_never_overwrites_the_original_price(self):
        """The record has to be what was knowable then."""
        ft.append_log(ft.candidates(_quotes(_heavy_favourite())), self.path)
        moved = _heavy_favourite()
        moved[1] = ("book0", -500, +400)        # line moved against us
        ft.append_log(ft.candidates(_quotes(moved)), self.path)
        self.assertEqual(float(pd.read_csv(self.path).iloc[0]["bet_odds"]), -300)

    def test_every_field_is_written(self):
        ft.append_log(ft.candidates(_quotes(_heavy_favourite())), self.path)
        self.assertEqual(list(pd.read_csv(self.path).columns), ft.LOG_FIELDS)


class SettleTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.log = self.root / "log.csv"
        self.fights = self.root / "fights.csv"

    def _write_fights(self, winner):
        pd.DataFrame({"date": ["2026-09-05"], "fighter_a": ["Ann Ace"],
                      "fighter_b": ["Bea Bolt"], "winner": [winner]}
                     ).to_csv(self.fights, index=False)

    def test_a_winning_bet_pays_the_logged_price(self):
        ft.append_log(ft.candidates(_quotes(_heavy_favourite())), self.log)
        self._write_fights("A")
        settled = ft.settle(self.log, self.fights)
        self.assertTrue(bool(settled.iloc[0]["won"]))
        self.assertAlmostEqual(float(settled.iloc[0]["profit"]), 100 / 300)

    def test_a_losing_bet_costs_one_unit(self):
        ft.append_log(ft.candidates(_quotes(_heavy_favourite())), self.log)
        self._write_fights("B")
        self.assertAlmostEqual(float(ft.settle(self.log, self.fights).iloc[0]["profit"]),
                               -1.0)

    def test_an_unfought_bet_stays_open(self):
        ft.append_log(ft.candidates(_quotes(_heavy_favourite())), self.log)
        pd.DataFrame({"date": [], "fighter_a": [], "fighter_b": [], "winner": []}
                     ).to_csv(self.fights, index=False)
        self.assertEqual(len(ft.settle(self.log, self.fights)), 0)

    def test_backfilled_bets_are_excluded_from_the_headline(self):
        """Mixing them would let a decision made with the result in hand
        borrow the credibility of one made blind."""
        past = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
        rows = ft.candidates(_quotes(_heavy_favourite(), commence=past))
        rows["date"] = "2026-09-05"                 # settle against the fixture
        rows["fight_key"] = [ft._fight_key(r) for _, r in rows.iterrows()]
        ft.append_log(rows, self.log)
        self._write_fights("A")
        report = ft.summary(self.log, self.fights, draws=50)
        self.assertEqual(report["settled"], 1)
        self.assertEqual(report["settled_backfilled"], 1)
        self.assertNotIn("roi", report)


class PowerTests(unittest.TestCase):
    def test_a_smaller_effect_needs_quadratically_more_bets(self):
        big = ft.bets_needed(0.04, 0.033, 166)
        small = ft.bets_needed(0.02, 0.033, 166)
        self.assertAlmostEqual(small / big, 4.0, delta=0.15)

    def test_a_negative_effect_is_unresolvable_by_more_data(self):
        self.assertIsNone(ft.bets_needed(-0.01, 0.03, 100))

    def test_the_holdout_numbers_imply_a_few_hundred_bets(self):
        # ROI +4.26%, SE 3.30%, n=166 - the actual out-of-sample result.
        self.assertLess(ft.bets_needed(0.0426, 0.0330, 166), 400)


if __name__ == "__main__":
    unittest.main()
