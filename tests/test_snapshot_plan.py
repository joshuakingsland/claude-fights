import unittest

import pandas as pd

from historical_odds import plan_kinds, snapshot_plan

START = pd.Timestamp("2026-08-16T03:30:00Z")


class PlanTests(unittest.TestCase):
    """The sweep is what makes the lead-lag question answerable at all.

    Entry and close sit 24 hours apart, so they show what a price was and
    never when it moved. A regular sweep back from the card shows the moves.
    """

    def test_without_a_cadence_only_entry_and_close_are_planned(self):
        plan = snapshot_plan(START, entry_hours=24, close_minutes=15)
        self.assertEqual([k for k, _ in plan], ["entry", "close"])

    def test_a_cadence_adds_one_snapshot_per_step(self):
        plan = snapshot_plan(START, 24, 15, cadence_hours=1.0, span_hours=6.0)
        self.assertEqual(len(plan), 2 + 6)

    def test_sweep_timestamps_step_back_from_the_card(self):
        plan = dict(snapshot_plan(START, 24, 15, cadence_hours=2.0, span_hours=6.0))
        self.assertEqual(plan["t_minus_002.00h"], START - pd.Timedelta(hours=2))
        self.assertEqual(plan["t_minus_006.00h"], START - pd.Timedelta(hours=6))

    def test_kind_names_are_derivable_without_a_start_time(self):
        # Resume has to know what a finished card looks like before fetching
        # anything, so the two must agree exactly.
        plan = snapshot_plan(START, 24, 15, cadence_hours=1.5, span_hours=9.0)
        self.assertEqual([k for k, _ in plan], plan_kinds(1.5, 9.0))

    def test_kind_names_sort_in_time_order(self):
        # Zero-padded so a plain sort is chronological, which keeps the
        # manifest readable and any later diff of it stable.
        kinds = [k for k in plan_kinds(1.0, 12.0) if k.startswith("t_minus")]
        self.assertEqual(kinds, sorted(kinds))

    def test_a_cadence_without_a_span_changes_nothing(self):
        self.assertEqual(plan_kinds(1.0, 0.0), ["entry", "close"])
        self.assertEqual(plan_kinds(0.0, 48.0), ["entry", "close"])


if __name__ == "__main__":
    unittest.main()
