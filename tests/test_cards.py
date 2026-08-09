import unittest

from cards import event_groups, infer_five_rounds


class EventGroupingTests(unittest.TestCase):
    """The stake cap needs an event, and a UTC date is not one."""

    def test_two_cards_on_one_utc_date_are_separate_events(self):
        # Observed 2026-08-08: a Friday-night US card at 00:20-03:50 UTC and
        # the Saturday card at 21:10-23:40 UTC shared one UTC calendar date.
        groups = event_groups(["2026-08-08T00:20:00Z", "2026-08-08T03:50:00Z",
                               "2026-08-08T21:10:00Z", "2026-08-08T23:40:00Z"])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[0], groups[2])

    def test_one_card_spanning_midnight_utc_stays_one_event(self):
        groups = event_groups(["2026-08-08T22:00:00Z", "2026-08-09T01:30:00Z"])
        self.assertEqual(groups[0], groups[1])

    def test_input_order_does_not_change_the_grouping(self):
        starts = ["2026-08-08T23:40:00Z", "2026-08-08T00:20:00Z",
                  "2026-08-08T21:10:00Z"]
        groups = dict(zip(starts, event_groups(starts)))
        self.assertEqual(groups["2026-08-08T21:10:00Z"],
                         groups["2026-08-08T23:40:00Z"])
        self.assertNotEqual(groups["2026-08-08T00:20:00Z"],
                            groups["2026-08-08T21:10:00Z"])

    def test_missing_start_times_do_not_crash(self):
        self.assertEqual(len(event_groups(["", None, "2026-08-08T21:00:00Z"])), 3)


class FiveRoundInferenceTests(unittest.TestCase):
    """`five_rounds` was hardcoded to 0 for every fight the feed ever wrote.

    That silently mispriced every main event: the moneyline's `five_rd`
    feature was a constant, and totals for a five-rounder need two extra lines
    and carry a structurally lower chance of going the distance.
    """

    def test_the_last_fight_on_a_card_is_the_main_event(self):
        starts = ["2026-08-16T02:00:00Z", "2026-08-16T02:45:00Z",
                  "2026-08-16T03:30:00Z"]
        self.assertEqual(infer_five_rounds(starts), [0, 0, 1])

    def test_each_card_gets_its_own_main_event(self):
        starts = ["2026-08-08T00:20:00Z", "2026-08-08T02:45:00Z",
                  "2026-08-08T21:10:00Z", "2026-08-08T23:40:00Z"]
        self.assertEqual(infer_five_rounds(starts), [0, 1, 0, 1])

    def test_a_card_with_one_placeholder_time_infers_nothing(self):
        # Futures markets arrive with every fight sharing a commence time.
        # Picking a main event from an arbitrary tie would be worse than
        # admitting the running order is unknown.
        starts = ["2027-01-01T03:00:00Z"] * 4
        self.assertEqual(infer_five_rounds(starts), [0, 0, 0, 0])

    def test_a_tie_at_the_top_infers_nothing(self):
        starts = ["2026-08-16T02:00:00Z", "2026-08-16T03:30:00Z",
                  "2026-08-16T03:30:00Z"]
        self.assertEqual(infer_five_rounds(starts), [0, 0, 0])

    def test_a_lone_fight_is_not_assumed_to_be_a_main_event(self):
        self.assertEqual(infer_five_rounds(["2026-08-12T00:00:00Z"]), [0])

    def test_unparseable_times_are_skipped_rather_than_ranked(self):
        starts = ["", "2026-08-16T02:00:00Z", "2026-08-16T03:30:00Z"]
        self.assertEqual(infer_five_rounds(starts), [0, 0, 1])

    def test_result_length_always_matches_the_input(self):
        for starts in ([], [""], ["a", "b"], ["2026-08-16T02:00:00Z"] * 3):
            self.assertEqual(len(infer_five_rounds(starts)), len(starts))


if __name__ == "__main__":
    unittest.main()
