import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import adapter
from capture_close import due_events, run as run_close
from data_quality import audit_fights
from discover_prop_markets import market_book_counts, prop_keys
from features import build_features
import freshness
from freshness import assess_freshness
from identity import canonical_name, fighter_registry, resolve_fighter
from update_data import _event_date_recovery, _regression_errors


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

    def test_cross_source_aliases_are_canonical(self):
        self.assertEqual(canonical_name("Stephen Erceg"), "steve erceg")
        self.assertEqual(canonical_name("Ramazonbek Temirov"), "ramazan temirov")

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

            pd.DataFrame(columns=["EVENT", "DATE"]).to_csv(
                root / "ufc_event_details.csv", index=False
            )
            recovered = adapter.build(
                str(root), fallback_event_dates={"Card": "2025-01-01"}
            )
            self.assertEqual(len(recovered), 2)
            self.assertEqual(str(recovered["date"].min().date()), "2025-01-01")


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

    def test_refresh_rejects_a_dropped_fight_even_when_fighter_ids_remain(self):
        old = pd.DataFrame([
            {"date": "2025-01-01", "event": "Card One",
             "fighter_a_id": "a", "fighter_b_id": "b"},
            {"date": "2025-02-01", "event": "Card Two",
             "fighter_a_id": "a", "fighter_b_id": "c"},
        ])
        new = pd.DataFrame([
            {"date": "2025-01-01", "event": "Card One",
             "fighter_a_id": "a", "fighter_b_id": "b"},
            {"date": "2025-02-02", "event": "Card Two",
             "fighter_a_id": "a", "fighter_b_id": "c"},
        ])
        errors = _regression_errors(new, old)
        self.assertTrue(any("historical fights" in error for error in errors))

    def test_missing_event_date_uses_only_prior_validated_event(self):
        previous = pd.DataFrame({
            "event": ["Known Missing Card"], "date": ["2025-08-22"]
        })
        event_details = pd.DataFrame({
            "EVENT": ["Current Card"], "DATE": ["July 25, 2026"]
        })
        results = pd.DataFrame({
            "EVENT": ["Current Card", "Known Missing Card", "Unknown Card"]
        })
        recovered, unresolved = _event_date_recovery(
            previous, event_details, results
        )
        self.assertEqual(str(recovered["Known Missing Card"].date()), "2025-08-22")
        self.assertEqual(unresolved, ["Unknown Card"])

    def test_source_backed_event_date_fallbacks_cover_deduped_aliases(self):
        previous = pd.DataFrame({
            "event": ["Noche UFC: Lopes vs. Silva"], "date": ["2025-09-13"]
        })
        event_details = pd.DataFrame({
            "EVENT": ["Noche UFC: Lopes vs. Silva"], "DATE": ["September 13, 2025"]
        })
        results = pd.DataFrame({
            "EVENT": ["UFC Fight Night: Lopes vs. Silva"]
        })
        recovered, unresolved = _event_date_recovery(
            previous, event_details, results
        )
        self.assertEqual(str(pd.Timestamp(
            recovered["UFC Fight Night: Lopes vs. Silva"]).date()), "2025-09-13")
        self.assertEqual(unresolved, [])

    def test_completed_tracked_fight_marks_results_lagging(self):
        fights = pd.DataFrame([{
            "date": "2025-01-01", "fighter_a": "A", "fighter_b": "B"
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds = root / "odds.csv"
            roster = root / "roster.csv"
            pd.DataFrame([{
                "commence_time": "2025-01-10T03:00:00Z",
                "fighter_a": "A", "fighter_b": "D",
            }]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "A", "LAST": ""},
                {"FIRST": "D", "LAST": ""},
            ]).to_csv(roster, index=False)
            report = assess_freshness(
                fights, odds, now="2025-01-11T20:00:00Z", fighter_roster=roster,
                grace_days=0,
            )
        self.assertEqual(report["status"], "lagging")
        self.assertEqual(len(report["known_completed_missing"]), 1)

    def test_freshness_deduplicates_aliases_and_known_cancellations(self):
        fights = pd.DataFrame([{
            "date": "2026-07-25",
            "fighter_a": "Steve Erceg",
            "fighter_b": "Ramazan Temirov",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds = root / "odds.csv"
            cancellations = root / "cancelled.csv"
            rows = [
                {
                    "commence_time": "2026-07-25T09:00:00Z",
                    "fighter_a": "Stephen Erceg",
                    "fighter_b": "Ramazonbek Temirov",
                },
                {
                    "commence_time": "2026-07-25T09:00:00Z",
                    "fighter_a": "Islam Dulatov",
                    "fighter_b": "Wellington Turman",
                },
            ]
            pd.DataFrame(rows + rows).to_csv(odds, index=False)
            pd.DataFrame([{
                "date": "2026-07-25",
                "fighter_a": "Islam Dulatov",
                "fighter_b": "Wellington Turman",
                "status": "cancelled",
            }]).to_csv(cancellations, index=False)
            report = assess_freshness(
                fights,
                odds,
                now="2026-07-26T20:00:00Z",
                cancelled_fights=cancellations,
                grace_days=0,
            )
        self.assertEqual(report["status"], "current")
        self.assertEqual(report["known_completed_missing"], [])
        self.assertEqual(len(report["known_cancelled"]), 1)

    def test_freshness_excuses_bouts_the_result_source_never_covers(self):
        fights = pd.DataFrame([{
            "date": "2026-07-25",
            "fighter_a": "Islam Dulatov",
            "fighter_b": "Wellington Turman",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds = root / "odds.csv"
            roster = root / "roster.csv"
            pd.DataFrame([
                {
                    "commence_time": "2026-08-01T00:10:00Z",
                    "fighter_a": "Dakota Ditcheva",
                    "fighter_b": "Denise Kielholtz",
                },
                {
                    "commence_time": "2026-08-01T00:10:00Z",
                    "fighter_a": "Islam Dulatov",
                    "fighter_b": "Cody Brundage",
                },
            ]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "Islam", "LAST": "Dulatov"},
                {"FIRST": "Wellington", "LAST": "Turman"},
                {"FIRST": "Cody", "LAST": "Brundage"},
            ]).to_csv(roster, index=False)
            report = assess_freshness(
                fights,
                odds,
                now="2026-08-02T20:00:00Z",
                fighter_roster=roster,
                grace_days=0,
            )
        self.assertEqual(report["status"], "lagging")
        self.assertEqual(len(report["known_out_of_scope"]), 1)
        self.assertEqual(
            report["known_out_of_scope"][0]["fighter_a"], "Dakota Ditcheva"
        )
        self.assertEqual(len(report["known_completed_missing"]), 1)
        self.assertEqual(
            report["known_completed_missing"][0]["fighter_b"], "Cody Brundage"
        )

    def test_freshness_keeps_ufc_debuts_in_scope(self):
        fights = pd.DataFrame([{
            "date": "2026-07-25",
            "fighter_a": "Islam Dulatov",
            "fighter_b": "Wellington Turman",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds = root / "odds.csv"
            roster = root / "roster.csv"
            pd.DataFrame([{
                "commence_time": "2026-08-01T00:10:00Z",
                "fighter_a": "Islam Dulatov",
                "fighter_b": "Newcomer Prospect",
            }]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "Islam", "LAST": "Dulatov"},
                {"FIRST": "Wellington", "LAST": "Turman"},
                {"FIRST": "Newcomer", "LAST": "Prospect"},
            ]).to_csv(roster, index=False)
            report = assess_freshness(
                fights, odds, now="2026-08-02T20:00:00Z", fighter_roster=roster,
                grace_days=0,
            )
        self.assertEqual(report["status"], "lagging")
        self.assertEqual(report["known_out_of_scope"], [])
        self.assertEqual(len(report["known_completed_missing"]), 1)

    def test_freshness_fails_closed_without_a_fighter_roster(self):
        fights = pd.DataFrame([{
            "date": "2025-01-01", "fighter_a": "A", "fighter_b": "B"
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds = root / "odds.csv"
            pd.DataFrame([{
                "commence_time": "2025-01-10T03:00:00Z",
                "fighter_a": "C", "fighter_b": "D",
            }]).to_csv(odds, index=False)
            report = assess_freshness(
                fights,
                odds,
                now="2025-01-11T20:00:00Z",
                fighter_roster=root / "absent.csv",
                grace_days=0,
            )
        self.assertEqual(report["status"], "lagging")
        self.assertEqual(len(report["known_completed_missing"]), 1)
        self.assertEqual(report["known_out_of_scope"], [])

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
        # Everything but the moneyline. The old filter kept only
        # method_of_victory and silently dropped the distance market, so an
        # empty catalogue could not be read as "nothing is offered".
        self.assertEqual(prop_keys(payload),
                         ["fight_to_go_distance", "method_of_victory"])

    def test_catalogue_records_totals_which_no_term_filter_would_match(self):
        payload = {"bookmakers": [
            {"markets": [{"key": "h2h"}, {"key": "totals"}]},
            {"markets": [{"key": "totals"}]},
        ]}
        self.assertEqual(market_book_counts(payload), {"h2h": 1, "totals": 2})
        self.assertIn("totals", prop_keys(payload))

    def test_book_depth_counts_books_not_market_entries(self):
        # One book quoting several lines of the same market is still one book,
        # and the three-paired-book minimum counts books.
        payload = {"bookmakers": [
            {"markets": [{"key": "totals"}, {"key": "totals"}]},
        ]}
        self.assertEqual(market_book_counts(payload)["totals"], 1)


if __name__ == "__main__":
    unittest.main()


class ResultGracePeriodTests(unittest.TestCase):
    """The results feed is a third-party scrape that publishes days late.

    Failing closed the moment a card ended blocked odds capture and the card
    refresh in the same job, so a normal upstream delay froze the published
    page. On 2026-08-16 that took the whole workflow down while the upstream
    repository's newest event was 2026-08-08 - exactly what ours had.
    """

    def _report(self, fight_time, now, grace_days=7):
        # The prior bout sits on its own date, threading three constraints: it
        # must make both fighters UFC veterans (so the bout is in scope) and
        # give neither a result on the night being judged (which would make
        # the booking superseded rather than awaited), predate every booking
        # tested here (scope is judged as of the fight date, so a debut after
        # the bout would put it out of scope), and stay inside 21 days of
        # `now` so the stale-bundle check does not fire first.
        fights = pd.DataFrame([{
            "date": "2026-07-28", "fighter_a": "Ann Ace", "fighter_b": "Bea Bolt",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds, roster = root / "odds.csv", root / "roster.csv"
            pd.DataFrame([{
                "commence_time": fight_time,
                "fighter_a": "Ann Ace", "fighter_b": "Cal Cruz",
            }]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "Ann", "LAST": "Ace"},
                {"FIRST": "Bea", "LAST": "Bolt"},
                {"FIRST": "Cal", "LAST": "Cruz"},
            ]).to_csv(roster, index=False)
            return assess_freshness(fights, odds, now=now, fighter_roster=roster,
                                    grace_days=grace_days)

    def test_a_fight_that_just_ran_waits_rather_than_failing(self):
        report = self._report("2026-08-15T23:00:00Z", "2026-08-16T20:00:00Z")
        self.assertEqual(report["status"], "pending")
        self.assertEqual(report["known_completed_missing"], [])
        self.assertEqual(len(report["known_awaiting_upstream"]), 1)

    def test_a_fight_still_missing_after_the_window_is_a_fault(self):
        report = self._report("2026-08-01T23:00:00Z", "2026-08-16T20:00:00Z")
        self.assertEqual(report["status"], "lagging")
        self.assertEqual(len(report["known_completed_missing"]), 1)
        self.assertEqual(report["known_awaiting_upstream"], [])

    def test_the_boundary_falls_on_the_fault_side(self):
        # Exactly at the window, the wait stops being routine.
        self.assertEqual(
            self._report("2026-08-09T20:00:00Z", "2026-08-16T20:00:00Z")["status"],
            "lagging")
        self.assertEqual(
            self._report("2026-08-09T21:00:00Z", "2026-08-16T20:00:00Z")["status"],
            "pending")

    def test_how_long_each_fight_has_waited_is_reported(self):
        report = self._report("2026-08-14T23:00:00Z", "2026-08-16T20:00:00Z")
        self.assertEqual(report["known_awaiting_upstream"][0]["days_waiting"], 1)

    def test_the_grace_window_is_recorded_in_the_report(self):
        # So a reader can tell a tolerated wait from a suppressed failure.
        self.assertEqual(self._report("2026-08-15T23:00:00Z",
                                      "2026-08-16T20:00:00Z")["grace_days"], 7)

    def test_a_stale_bundle_still_reports_check_regardless_of_grace(self):
        fights = pd.DataFrame([{
            "date": "2026-01-01", "fighter_a": "Ann Ace", "fighter_b": "Bea Bolt",
        }])
        with tempfile.TemporaryDirectory() as directory:
            odds = Path(directory) / "odds.csv"
            pd.DataFrame(columns=["commence_time", "fighter_a", "fighter_b"]
                         ).to_csv(odds, index=False)
            report = assess_freshness(fights, odds, now="2026-08-16T20:00:00Z")
        self.assertEqual(report["status"], "check")


class NotifyEmailTests(unittest.TestCase):
    """The mailer runs as the `if: failure()` step, so it must not raise.

    An expired Gmail app password on 2026-08-16 turned one red step into two,
    and the top one was a stack trace about SMTP rather than the actual cause.
    """

    def _run(self, side_effect):
        import notify_email
        env = {"SMTP_USER": "u@example.com", "SMTP_PASSWORD": "p",
               "BET_EMAIL_TO": "to@example.com"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch("smtplib.SMTP", side_effect=side_effect):
                return notify_email.main()

    def test_a_rejected_login_does_not_mask_the_real_failure(self):
        import smtplib
        code = self._run(smtplib.SMTPAuthenticationError(535, b"BadCredentials"))
        self.assertEqual(code, 0)

    def test_an_unreachable_server_does_not_mask_the_real_failure(self):
        self.assertEqual(self._run(OSError("connection refused")), 0)

    def test_missing_secrets_are_skipped_quietly(self):
        import notify_email
        with mock.patch.dict(os.environ, {"SMTP_USER": "", "SMTP_PASSWORD": "",
                                          "BET_EMAIL_TO": ""}, clear=False):
            self.assertEqual(notify_email.main(), 0)


class SupersededBookingTests(unittest.TestCase):
    """The odds feed quotes bookings that never happen.

    A fighter is announced against one opponent, the opponent changes, and the
    dead pairing keeps being quoted. Books also spell the same man differently.
    On 2026-08-16 the feed held Charles Johnson against Jose Ochoa, Eduardo
    Henrique and Eduardo Chapolin - one bout, three names, and only the last
    matched the result. The other two would have waited for a result that was
    never coming, and failed the workflow seven days later.
    """

    def _report(self, booked_opponent, now="2026-08-23T20:00:00Z", **kwargs):
        # The result on record: Johnson beat Chapolin on the 15th.
        fights = pd.DataFrame([{
            "date": "2026-08-15", "fighter_a": "Charles Johnson",
            "fighter_b": "Eduardo Chapolin", "winner": "A",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds, roster = root / "odds.csv", root / "roster.csv"
            pd.DataFrame([{
                "commence_time": "2026-08-16T02:00:00Z",
                "fighter_a": "Charles Johnson", "fighter_b": booked_opponent,
            }]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "Charles", "LAST": "Johnson"},
                {"FIRST": "Eduardo", "LAST": "Chapolin"},
                {"FIRST": "Jose", "LAST": "Ochoa"},
            ]).to_csv(roster, index=False)
            return assess_freshness(fights, odds, now=now,
                                    fighter_roster=roster, **kwargs)

    def test_a_replaced_opponent_does_not_wait_forever(self):
        report = self._report("Jose Ochoa", grace_days=0)
        self.assertEqual(report["known_completed_missing"], [])
        self.assertEqual(len(report["known_superseded"]), 1)

    def test_the_same_man_under_another_name_is_not_a_missing_result(self):
        report = self._report("Eduardo Henrique", grace_days=0)
        self.assertEqual(report["known_completed_missing"], [])
        self.assertEqual(len(report["known_superseded"]), 1)

    def test_a_superseded_booking_does_not_fail_the_workflow(self):
        self.assertNotEqual(self._report("Jose Ochoa", grace_days=0)["status"],
                            "lagging")

    def test_a_fighter_who_did_not_compete_is_still_awaited(self):
        """The guard must not excuse everything it cannot match.

        Neither of these two has a result that night, so this is a real gap
        and has to survive the new check.
        """
        fights = pd.DataFrame([
            {"date": "2026-08-15", "fighter_a": "Charles Johnson",
             "fighter_b": "Eduardo Chapolin", "winner": "A"},
            # An earlier bout, so Ann Ace counts as a UFC veteran and the
            # booking is in scope. Without it the pair is excused as a bout
            # the result source never covers, and the test would pass for
            # the wrong reason.
            {"date": "2026-08-10", "fighter_a": "Ann Ace",
             "fighter_b": "Bea Bolt", "winner": "A"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds, roster = root / "odds.csv", root / "roster.csv"
            pd.DataFrame([{
                "commence_time": "2026-08-16T02:00:00Z",
                "fighter_a": "Ann Ace", "fighter_b": "Bea Bolt",
            }]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "Charles", "LAST": "Johnson"},
                {"FIRST": "Eduardo", "LAST": "Chapolin"},
                {"FIRST": "Ann", "LAST": "Ace"},
                {"FIRST": "Bea", "LAST": "Bolt"},
            ]).to_csv(roster, index=False)
            report = assess_freshness(fights, odds, now="2026-08-30T20:00:00Z",
                                      fighter_roster=roster, grace_days=0)
        self.assertEqual(report["known_superseded"], [])
        self.assertEqual(report["status"], "lagging")

    def test_a_matched_pairing_is_never_called_superseded(self):
        report = self._report("Eduardo Chapolin", grace_days=0)
        self.assertEqual(report["known_superseded"], [])
        self.assertEqual(report["known_completed_missing"], [])

    def test_a_result_a_week_away_does_not_excuse_the_booking(self):
        # Only the same night counts; a fighter's bout last month says
        # nothing about whether tonight's happened.
        fights = pd.DataFrame([{
            "date": "2026-07-04", "fighter_a": "Charles Johnson",
            "fighter_b": "Someone Else", "winner": "A",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds, roster = root / "odds.csv", root / "roster.csv"
            pd.DataFrame([{
                "commence_time": "2026-08-16T02:00:00Z",
                "fighter_a": "Charles Johnson", "fighter_b": "Jose Ochoa",
            }]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "Charles", "LAST": "Johnson"},
                {"FIRST": "Someone", "LAST": "Else"},
                {"FIRST": "Jose", "LAST": "Ochoa"},
            ]).to_csv(roster, index=False)
            report = assess_freshness(fights, odds, now="2026-08-30T20:00:00Z",
                                      fighter_roster=roster, grace_days=0)
        self.assertEqual(report["known_superseded"], [])
        self.assertEqual(len(report["known_completed_missing"]), 1)


class ScopeAsOfFightDateTests(unittest.TestCase):
    """Scope must be judged as of the bout, not as of today.

    Matt Adams vs Anthony Wint was a Contender Series bout on 2026-08-12, a
    card UFCStats does not carry. It sat correctly out of scope until Wint
    made his UFC debut on the 22nd - which retroactively made him a "veteran"
    and flipped the older bout into scope, already twelve days past the grace
    window, failing the run on a result that was never going to exist.
    """

    def _report(self, booking_date, debut_date, now="2026-08-24T20:00:00Z"):
        fights = pd.DataFrame([{
            "date": debut_date, "fighter_a": "Anthony Wint",
            "fighter_b": "Terrance Chatman", "winner": "A",
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odds, roster = root / "odds.csv", root / "roster.csv"
            pd.DataFrame([{
                "commence_time": f"{booking_date}T02:00:00Z",
                "fighter_a": "Matt Adams", "fighter_b": "Anthony Wint",
            }]).to_csv(odds, index=False)
            pd.DataFrame([
                {"FIRST": "Matt", "LAST": "Adams"},
                {"FIRST": "Anthony", "LAST": "Wint"},
                {"FIRST": "Terrance", "LAST": "Chatman"},
            ]).to_csv(roster, index=False)
            return assess_freshness(fights, odds, now=now,
                                    fighter_roster=roster, grace_days=7)

    def test_a_bout_before_either_fighters_ufc_debut_stays_out_of_scope(self):
        report = self._report(booking_date="2026-08-12", debut_date="2026-08-22")
        self.assertEqual(report["known_completed_missing"], [])
        self.assertEqual(len(report["known_out_of_scope"]), 1)

    def test_a_later_debut_does_not_retroactively_pull_it_in(self):
        self.assertNotEqual(
            self._report("2026-08-12", "2026-08-22")["status"], "lagging")

    def test_a_bout_after_a_ufc_debut_is_still_tracked(self):
        # The guard must keep working: once a fighter is a UFC veteran, their
        # later bouts are genuinely trackable and a missing result is a fault.
        report = self._report(booking_date="2026-08-12", debut_date="2026-07-04")
        self.assertEqual(report["known_out_of_scope"], [])
        self.assertEqual(len(report["known_completed_missing"]), 1)

    def test_a_debut_on_the_same_day_counts_as_in_scope(self):
        report = self._report(booking_date="2026-08-12", debut_date="2026-08-12")
        self.assertEqual(report["known_out_of_scope"], [])


class QuarantineTests(unittest.TestCase):
    """Bookings no result is coming for should stop failing every later run.

    Three separate root causes have produced this same symptom - an upstream
    publishing lag, a replaced opponent, and a bout that only became
    "trackable" once a fighter later debuted - and each needed a code change
    and a human to notice. The guard cannot tell "the result is late" from "no
    result is ever coming", and only the first is fixed by waiting.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.quarantine = self.root / "quarantine.csv"

    def _fixture(self, count=1):
        fights = pd.DataFrame([{
            "date": "2026-08-01", "fighter_a": "Vet One",
            "fighter_b": "Vet Two", "winner": "A",
        }])
        odds = self.root / "odds.csv"
        roster = self.root / "roster.csv"
        rows, names = [], [{"FIRST": "Vet", "LAST": "One"},
                           {"FIRST": "Vet", "LAST": "Two"}]
        for i in range(count):
            rows.append({"commence_time": "2026-08-05T02:00:00Z",
                         "fighter_a": "Vet One", "fighter_b": f"Ghost {i}"})
            names.append({"FIRST": "Ghost", "LAST": str(i)})
        pd.DataFrame(rows).to_csv(odds, index=False)
        pd.DataFrame(names).to_csv(roster, index=False)
        return fights, odds, roster

    def _assess(self, fights, odds, roster):
        return assess_freshness(fights, odds, now="2026-08-20T20:00:00Z",
                                fighter_roster=roster, grace_days=7,
                                quarantine_log=self.quarantine)

    def test_an_unmatchable_booking_is_a_fault_before_quarantine(self):
        fights, odds, roster = self._fixture()
        # Vet One has a result on 08-01, so the 08-05 booking is in scope and
        # unmatched. Nothing excuses it yet.
        report = self._assess(fights, odds, roster)
        self.assertEqual(report["status"], "lagging")
        self.assertEqual(len(report["known_completed_missing"]), 1)

    def test_quarantining_it_clears_the_fault(self):
        fights, odds, roster = self._fixture()
        stuck = self._assess(fights, odds, roster)["known_completed_missing"]
        self.assertEqual(freshness.quarantine(stuck, self.quarantine), 1)
        after = self._assess(fights, odds, roster)
        self.assertEqual(after["known_completed_missing"], [])
        self.assertEqual(len(after["known_quarantined"]), 1)
        self.assertNotEqual(after["status"], "lagging")

    def test_the_audit_trail_records_what_was_excused(self):
        """Excusing is not ignoring: a wrong call has to stay visible."""
        fights, odds, roster = self._fixture()
        stuck = self._assess(fights, odds, roster)["known_completed_missing"]
        freshness.quarantine(stuck, self.quarantine)
        rows = pd.read_csv(self.quarantine)
        self.assertEqual(list(rows.columns), freshness.QUARANTINE_FIELDS)
        self.assertEqual(rows.iloc[0]["fighter_a"], "Vet One")
        self.assertTrue(str(rows.iloc[0]["quarantined_at"]).endswith("Z"))
        self.assertGreater(int(rows.iloc[0]["days_waited"]), 7)

    def test_quarantining_twice_does_not_duplicate_a_row(self):
        fights, odds, roster = self._fixture()
        stuck = self._assess(fights, odds, roster)["known_completed_missing"]
        freshness.quarantine(stuck, self.quarantine)
        self.assertEqual(freshness.quarantine(stuck, self.quarantine), 0)
        self.assertEqual(len(pd.read_csv(self.quarantine)), 1)

    def test_a_pile_of_them_is_a_broken_feed_not_a_straggler(self):
        # The lid. Writing these off would turn a broken pipeline into a
        # clean-looking one, which is the whole risk of automating this.
        fights, odds, roster = self._fixture(count=9)
        stuck = self._assess(fights, odds, roster)["known_completed_missing"]
        self.assertGreater(len(stuck), freshness.MAX_AUTO_QUARANTINE)

    def test_a_quarantined_booking_never_becomes_a_fault_again(self):
        fights, odds, roster = self._fixture()
        stuck = self._assess(fights, odds, roster)["known_completed_missing"]
        freshness.quarantine(stuck, self.quarantine)
        later = assess_freshness(fights, odds, now="2027-01-01T00:00:00Z",
                                 fighter_roster=roster, grace_days=7,
                                 quarantine_log=self.quarantine)
        self.assertEqual(later["known_completed_missing"], [])
