import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import adapter
from capture_close import due_events, run as run_close
from data_quality import audit_fights
from discover_prop_markets import prop_keys
from features import build_features
from freshness import assess_freshness
from identity import fighter_registry, resolve_fighter
from update_data import _regression_errors


class StableIdentityTests(unittest.TestCase):
    @staticmethod
    def physicals():
        return pd.DataFrame([
            {"FIGHTER": "Bruno Silva", "HEIGHT": "5' 4\"", "WEIGHT": "125 lbs.",
             "REACH": "65\"", "STANCE": "Orthodox", "DOB": "Mar 16, 1990",
             "URL": "http://ufcstats.com/fighter-details/aaaaaaaaaaaaaaaa"},
            {"FIGHTER": "Bruno Silva", "HEIGHT": "6' 0\"", "WEIGHT": "185 lbs.",
             "REACH": "74\"", "STANCE": "Orthodox", "DOB": "Jul 13, 1989",
             "URL": "http://ufcstats.com/fighter-details/bbbbbbbbbbbbbbbb"},
            {"FIGHTER": "Fly Opp", "HEIGHT": "5' 7\"", "WEIGHT": "125 lbs.",
             "REACH": "67\"", "STANCE": "Southpaw", "DOB": "Jan 01, 1991",
             "URL": "http://ufcstats.com/fighter-details/cccccccccccccccc"},
            {"FIGHTER": "Mid Opp", "HEIGHT": "6' 1\"", "WEIGHT": "185 lbs.",
             "REACH": "76\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1990",
             "URL": "http://ufcstats.com/fighter-details/dddddddddddddddd"},
        ])

    def test_same_name_resolves_by_division(self):
        registry = fighter_registry(self.physicals())
        fly = resolve_fighter("Bruno Silva", "Flyweight Bout", registry)
        middle = resolve_fighter("Bruno Silva", "Middleweight Bout", registry)
        self.assertEqual(fly["fighter_id"], "aaaaaaaaaaaaaaaa")
        self.assertEqual(middle["fighter_id"], "bbbbbbbbbbbbbbbb")

    def test_adapter_joins_stats_and_physicals_by_validated_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{"EVENT": "Card", "DATE": "Jan 01, 2025"}]).to_csv(
                root / "ufc_event_details.csv", index=False
            )
            pd.DataFrame([
                {"EVENT": "Card", "BOUT": "Bruno Silva vs. Fly Opp", "OUTCOME": "W/L",
                 "METHOD": "DEC", "ROUND": 3, "TIME": "5:00",
                 "TIME FORMAT": "3 Rnd (5-5-5)", "WEIGHTCLASS": "Flyweight Bout"},
                {"EVENT": "Card", "BOUT": "Mid Opp vs. Bruno Silva", "OUTCOME": "L/W",
                 "METHOD": "KO/TKO", "ROUND": 1, "TIME": "2:00",
                 "TIME FORMAT": "3 Rnd (5-5-5)", "WEIGHTCLASS": "Middleweight Bout"},
            ]).to_csv(root / "ufc_fight_results.csv", index=False)
            stats = []
            for bout, values in [
                ("Bruno Silva vs. Fly Opp", [("Bruno Silva", "20 of 40", "1 of 2"),
                                             ("Fly Opp", "10 of 30", "0 of 1")]),
                ("Mid Opp vs. Bruno Silva", [("Mid Opp", "5 of 12", "0 of 0"),
                                             ("Bruno Silva", "18 of 25", "0 of 0")]),
            ]:
                for fighter, sig, td in values:
                    stats.append({"EVENT": "Card", "BOUT": bout, "FIGHTER": fighter,
                                  "SIG.STR.": sig, "TD": td})
            pd.DataFrame(stats).to_csv(root / "ufc_fight_stats.csv", index=False)
            physicals = self.physicals()
            physicals.to_csv(root / "ufc_fighter_tott.csv", index=False)
            details = []
            for row in physicals.itertuples():
                first, last = row.FIGHTER.split(" ", 1)
                details.append({"FIRST": first, "LAST": last, "NICKNAME": "", "URL": row.URL})
            pd.DataFrame(details).to_csv(root / "ufc_fighter_details.csv", index=False)

            fights = adapter.build(str(root))
            self.assertEqual(audit_fights(fights), [])
            fly = fights[fights["weightclass"] == "Flyweight Bout"].iloc[0]
            middle = fights[fights["weightclass"] == "Middleweight Bout"].iloc[0]
            self.assertEqual(fly["fighter_a_id"], "aaaaaaaaaaaaaaaa")
            self.assertEqual(fly["height_a"], 64)
            self.assertEqual(fly["sig_str_landed_a"], 20)
            self.assertEqual(middle["fighter_b_id"], "bbbbbbbbbbbbbbbb")
            self.assertEqual(middle["height_b"], 72)
            self.assertEqual(middle["sig_str_landed_b"], 18)


class PointInTimeFeatureTests(unittest.TestCase):
    @staticmethod
    def fights():
        rows = []
        bouts = [
            ("2024-01-01", "A", "B", "A"),
            ("2024-02-01", "A", "C", "B"),
            ("2024-03-01", "A", "B", "A"),
        ]
        for index, (date, a, b, winner) in enumerate(bouts):
            rows.append({
                "date": date, "fighter_a": a, "fighter_b": b,
                "fighter_a_id": f"id-{a}", "fighter_b_id": f"id-{b}",
                "winner": winner, "method": "DEC", "fight_time_min": 15,
                "reach_a": 70 + index, "reach_b": 68, "height_a": 70,
                "height_b": 69, "dob_a": "1990-01-01", "dob_b": "1991-01-01",
                "sig_str_landed_a": 30 + index, "sig_str_landed_b": 20,
                "sig_str_absorbed_a": 20, "sig_str_absorbed_b": 30 + index,
                "td_landed_a": 1, "td_landed_b": 0,
                "td_attempted_a": 2, "td_attempted_b": 1,
            })
        return pd.DataFrame(rows)

    def test_flipping_winner_cannot_change_same_fight_features(self):
        fights = self.fights()
        baseline, columns = build_features(fights)
        flipped = fights.copy()
        flipped.loc[1, "winner"] = "A"
        changed, _ = build_features(flipped)
        np.testing.assert_allclose(
            baseline.loc[:1, columns].to_numpy(),
            changed.loc[:1, columns].to_numpy(),
            atol=1e-12,
        )
        self.assertNotEqual(baseline.loc[1, "target"], changed.loc[1, "target"])


class FreshnessAndCaptureTests(unittest.TestCase):
    def test_refresh_rejects_shrinking_or_backward_data(self):
        old = pd.DataFrame({
            "date": ["2025-01-01", "2025-02-01"],
            "fighter_a_id": ["a", "b"], "fighter_b_id": ["c", "d"],
        })
        new = pd.DataFrame({
            "date": ["2024-12-01"],
            "fighter_a_id": ["a"], "fighter_b_id": ["c"],
        })
        errors = _regression_errors(new, old)
        self.assertTrue(any("shrank" in error for error in errors))
        self.assertTrue(any("backward" in error for error in errors))
        self.assertTrue(any("fighter IDs" in error for error in errors))

    def test_completed_tracked_fight_marks_results_lagging(self):
        fights = pd.DataFrame([{
            "date": "2025-01-01", "fighter_a": "A", "fighter_b": "B"
        }])
        with tempfile.TemporaryDirectory() as directory:
            odds = Path(directory) / "odds.csv"
            pd.DataFrame([{
                "commence_time": "2025-01-10T03:00:00Z",
                "fighter_a": "C", "fighter_b": "D",
            }]).to_csv(odds, index=False)
            report = assess_freshness(fights, odds, now="2025-01-11T20:00:00Z")
        self.assertEqual(report["status"], "lagging")
        self.assertEqual(len(report["known_completed_missing"]), 1)

    def test_close_dry_run_never_calls_paid_odds_endpoint(self):
        events = [{
            "id": "due", "commence_time": "2025-01-01T01:30:00Z",
            "home_team": "A", "away_team": "B",
        }]
        calls = []

        def fetcher(path, key, **params):
            calls.append(path)
            if path.endswith("/events"):
                return events
            raise AssertionError("paid odds endpoint should not be called")

        with tempfile.TemporaryDirectory() as directory:
            report = run_close(
                "test", Path(directory) / "close.csv",
                now="2025-01-01T01:00:00Z", dry_run=True, fetcher=fetcher,
            )
        self.assertEqual(report["due"], 1)
        self.assertEqual(report["paid_requests"], 0)
        self.assertEqual(calls, ["/sports/mma_mixed_martial_arts/events"])

    def test_due_window_and_prop_key_filter(self):
        events = [
            {"id": "early", "commence_time": "2025-01-01T01:05:00Z"},
            {"id": "due", "commence_time": "2025-01-01T01:30:00Z"},
            {"id": "late", "commence_time": "2025-01-01T03:00:00Z"},
        ]
        self.assertEqual(
            due_events(events, set(), "2025-01-01T01:00:00Z"), [("due", 30.0)]
        )
        payload = {"bookmakers": [{"markets": [
            {"key": "h2h"}, {"key": "method_of_victory"},
            {"key": "fight_to_go_distance"},
        ]}]}
        self.assertEqual(prop_keys(payload), ["method_of_victory"])


if __name__ == "__main__":
    unittest.main()
