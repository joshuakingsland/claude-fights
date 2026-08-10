import unittest

import numpy as np
import pandas as pd

import promotion_tiers
import rankings
from config import MAX_EXECUTION_DEVIATION
from paper_ledger import SNAPSHOT_FIELDS, _snapshot_row
from predict_card import (_clean_meta, execution_ladder,
                          filter_upcoming_promotion, leader_gap_for_pick,
                          quote_quality)
from production import allocate_stakes


class LeaderGapProvenanceTests(unittest.TestCase):
    """Research column only: recorded next to a snapshot, never fed back in."""

    def test_gap_is_positive_when_setters_like_the_pick(self):
        leader, follower, gap = leader_gap_for_pick(0.62, 0.60, pick_a=True)
        self.assertAlmostEqual(leader, 0.62)
        self.assertAlmostEqual(follower, 0.60)
        self.assertAlmostEqual(gap, 2.0)

    def test_picking_the_other_corner_flips_the_sign(self):
        _, _, on_a = leader_gap_for_pick(0.62, 0.60, pick_a=True)
        leader, follower, on_b = leader_gap_for_pick(0.62, 0.60, pick_a=False)
        self.assertAlmostEqual(leader, 0.38)
        self.assertAlmostEqual(follower, 0.40)
        self.assertAlmostEqual(on_b, -on_a)

    def test_missing_side_yields_no_gap_rather_than_zero(self):
        self.assertIsNone(leader_gap_for_pick(None, 0.60, pick_a=True)[2])
        self.assertIsNone(leader_gap_for_pick(0.62, None, pick_a=False)[2])
        self.assertIsNone(leader_gap_for_pick(None, None, pick_a=True)[0])


class QuoteQualityTests(unittest.TestCase):
    """A single book's broken price must not be read as a tradable edge."""

    SOURCE = "the-odds-api-paired-book-devig-v1"

    def test_normal_vigged_execution_price_passes(self):
        # A real best price implies slightly more probability than consensus.
        ok, reason = quote_quality(8, 12.0, self.SOURCE, 0.645, 0.662)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_price_far_better_than_consensus_is_rejected(self):
        # Observed 2026-08-01: consensus -265, one book posted +126.
        ok, reason = quote_quality(8, 12.0, self.SOURCE, 0.697, 0.442)
        self.assertFalse(ok)
        self.assertEqual(reason, "book price outlier")

    def test_outlier_rejection_starts_past_the_configured_gap(self):
        inside = MAX_EXECUTION_DEVIATION - 0.005
        outside = MAX_EXECUTION_DEVIATION + 0.005
        self.assertTrue(quote_quality(8, 0.0, self.SOURCE, 0.60, 0.60 - inside)[0])
        self.assertFalse(quote_quality(8, 0.0, self.SOURCE, 0.60, 0.60 - outside)[0])

    def test_book_count_and_staleness_still_take_precedence(self):
        self.assertEqual(
            quote_quality(2, 12.0, self.SOURCE, 0.697, 0.442)[1],
            "fewer than 3 paired books",
        )
        self.assertEqual(
            quote_quality(8, 999.0, self.SOURCE, 0.697, 0.442)[1], "price stale"
        )

    def test_manual_rows_without_provenance_are_not_gated_on_age(self):
        ok, reason = quote_quality(None, 999.0, "manual_or_unknown", 0.60, 0.62)
        self.assertTrue(ok)
        self.assertEqual(reason, "")


class IdentityGateTests(unittest.TestCase):
    """A fighter with no UFCStats identity gets neutral career features.

    Pricing a bout on that is pricing it as though both corners were
    debutants. 29% of scored snapshots involve one; none became a trade only
    because non-UFC bouts usually draw fewer than three books, which is
    incidental rather than a control.
    """

    SOURCE = "the-odds-api-paired-book-devig-v1"

    def test_unresolved_identity_is_rejected_outright(self):
        ok, reason = quote_quality(8, 1.0, self.SOURCE, 0.60, 0.62,
                                   identity_resolved=False)
        self.assertFalse(ok)
        self.assertEqual(reason, "unresolved fighter identity")

    def test_identity_outranks_the_other_quality_reasons(self):
        # Thin books AND unresolved: the identity problem is the real one.
        self.assertEqual(
            quote_quality(1, 1.0, self.SOURCE, 0.60, 0.62,
                          identity_resolved=False)[1],
            "unresolved fighter identity")

    def test_resolved_identity_still_passes(self):
        self.assertTrue(quote_quality(8, 1.0, self.SOURCE, 0.60, 0.62,
                                      identity_resolved=True)[0])

    def test_default_is_resolved_so_manual_rows_are_unaffected(self):
        self.assertTrue(quote_quality(None, None, "manual", 0.60, 0.62)[0])


class DeclaredTierTests(unittest.TestCase):
    """Supplied letter grades stay supplied.

    The premise is the operator's: UFC alone at the top, PFL and Bellator a
    tier down, KSW and Rizin named below them, everything else unranked. The
    risk is that a declared letter and a measured gap get confused for one
    another later, so the test pins that they never touch.
    """

    def test_supplied_ladder_is_returned_verbatim(self):
        self.assertEqual(promotion_tiers.declared_tier("UFC"), "S")
        self.assertEqual(promotion_tiers.declared_tier("PFL"), "A")
        self.assertEqual(promotion_tiers.declared_tier("Bellator"), "A")
        self.assertEqual(promotion_tiers.declared_tier("KSW"), "C")
        self.assertEqual(promotion_tiers.declared_tier("Rizin"), "C")

    def test_everything_else_falls_to_the_default(self):
        self.assertEqual(promotion_tiers.declared_tier("KOTC"), "D")
        self.assertEqual(promotion_tiers.declared_tier("Oktagon MMA"), "D")
        self.assertEqual(promotion_tiers.declared_tier(""), "D")
        self.assertEqual(promotion_tiers.declared_tier(None), "D")

    def test_tier_does_not_track_the_measured_gap(self):
        # Rizin is declared C and measures softer than PFL's A; LFA is
        # declared D and measures harder than KOTC's D. A letter must never
        # be back-solved from a gap, so ordering by one must not order the
        # other.
        bouts = [("a", "b", "UFC"), ("b", "a", "UFC"),
                 ("a", "c", "KOTC"), ("a", "d", "KOTC")]
        frame = promotion_tiers.compare(bouts, min_shared=1, min_bouts=2)
        row = frame[frame["promotion"] == "KOTC"].iloc[0]
        self.assertEqual(row["declared_tier"], "D")
        self.assertGreater(row["mean_gap_vs_ufc"], 0.0)


class PromotionFamilyTests(unittest.TestCase):
    """Sherdog names one promotion many ways; a split promotion vanishes.

    Rizin's 65 bouts were spread over four event-family names, none of which
    reached the five-shared-fighter minimum, so it never appeared in the tier
    table at all despite being one of the two the operator named at C.
    """

    def test_event_families_fold_into_one_organisation(self):
        for name in ("Rizin", "Rizin FF", "Rizin Fighting Federation",
                     "Rizin Fighting World Grand Prix"):
            self.assertEqual(rankings.collapse_promotion(name), "Rizin")
        self.assertEqual(rankings.collapse_promotion("PFL Super Fights"), "PFL")
        self.assertEqual(
            rankings.collapse_promotion("Professional Fighters League"), "PFL")
        self.assertEqual(rankings.collapse_promotion("KSW Epic"), "KSW")
        self.assertEqual(rankings.collapse_promotion("UFC Fight Night"), "UFC")

    def test_similar_names_from_other_promotions_are_left_alone(self):
        # Real entries in the crawl that a loose substring match would eat.
        self.assertEqual(
            rankings.collapse_promotion("Professional Fighters Combat"),
            "Professional Fighters Combat")
        self.assertEqual(rankings.collapse_promotion("Alash Pride"),
                         "Alash Pride")
        self.assertEqual(rankings.collapse_promotion("One Pride MMA Fight Night"),
                         "One Pride MMA Fight Night")

    def test_blank_promotion_does_not_crash(self):
        self.assertEqual(rankings.collapse_promotion(None), "")
        self.assertEqual(rankings.collapse_promotion(""), "")


class PromotionFilterTests(unittest.TestCase):
    """The provider labels every event "MMA", so the label cannot filter.

    `promotion` used to be backfilled from the CLI flag, which made the column
    circular - it recorded the request, so a `--promotion ufc` run proved
    itself right about a card of debutants. With the column honest and empty,
    the roster is what separates the cards.
    """

    @staticmethod
    def _card():
        return pd.DataFrame([
            {"commence_time": "2026-08-12T00:00:00Z", "promotion": "",
             "event_title": "MMA", "fighter_a": "Rookie One",
             "fighter_b": "Rookie Two"},
            {"commence_time": "2026-08-12T00:30:00Z", "promotion": "",
             "event_title": "MMA", "fighter_a": "Rookie Three",
             "fighter_b": "Rookie Four"},
            {"commence_time": "2026-08-16T02:00:00Z", "promotion": "",
             "event_title": "MMA", "fighter_a": "Veteran One",
             "fighter_b": "Veteran Two"},
        ])

    @staticmethod
    def _history():
        return pd.DataFrame([{"fighter_a": "Veteran One",
                              "fighter_b": "Veteran Two"}])

    def test_a_card_of_debutants_is_selectable_without_provider_metadata(self):
        picked = filter_upcoming_promotion(self._card(), "dwcs", self._history())
        self.assertEqual(len(picked), 2)
        self.assertTrue(picked["fighter_a"].str.startswith("Rookie").all())

    def test_the_production_path_still_keeps_every_fight(self):
        # Labelling a card correctly and dropping it from the record are two
        # separate decisions; this filter only makes the first.
        card = self._card()
        self.assertEqual(
            len(filter_upcoming_promotion(card, "ufc", self._history())),
            len(card))

    def test_without_history_the_measurement_is_simply_not_made(self):
        self.assertEqual(
            len(filter_upcoming_promotion(self._card(), "dwcs")), 0)

    def test_an_explicit_provider_label_still_wins(self):
        card = self._card()
        card.loc[0, "event_title"] = "Dana White's Contender Series 12"
        picked = filter_upcoming_promotion(card, "dwcs", self._history())
        self.assertIn("Rookie One", picked["fighter_a"].tolist())


class ExecutionPolicyTests(unittest.TestCase):
    def test_event_day_cap_keeps_highest_two_flat_stakes(self):
        stakes = allocate_stakes(
            np.array([0.09, 0.12, 0.08, 0.07]),
            groups=np.array(["A", "A", "A", "B"]),
        )
        self.assertEqual(stakes.tolist(), [1, 1, 0, 1])

    def test_research_two_unit_threshold_is_not_active_allocation(self):
        stakes = allocate_stakes(np.array([0.15]), groups=np.array(["A"]))
        self.assertEqual(stakes.tolist(), [1])
        self.assertIn("2u_candidate", execution_ladder(0.75, 0.01))

    def test_snapshot_preserves_consensus_and_execution_provenance(self):
        item = {
            "pick": "A", "opp": "B", "price": "-200",
            "execution_price": "-200", "execution_book": "FanDuel",
            "consensus_price": "-220", "consensus_opp_price": "+180",
            "execution_implied": 66.7, "market": 65.8, "model": 73.8,
            "edge": 7.1, "se": 1.0, "net": 6.1, "bet": True,
            "stake": 1, "date": "2030-01-01",
        }
        row = _snapshot_row(item, "2029-01-01T00:00:00Z", {
            "model_version": "test", "manifest_hash": "hash",
        })
        self.assertEqual(row["execution_book"], "FanDuel")
        self.assertEqual(row["consensus_price"], "-220")
        self.assertEqual(row["price"], "-200")
        self.assertTrue(set(row).issubset(SNAPSHOT_FIELDS))

    def test_missing_weight_class_renders_as_tbd(self):
        self.assertEqual(_clean_meta(np.nan), "TBD")
        self.assertEqual(_clean_meta(""), "TBD")
        self.assertEqual(_clean_meta("Lightweight"), "Lightweight")


if __name__ == "__main__":
    unittest.main()
