import unittest

import numpy as np
import pandas as pd

from paper_ledger import SNAPSHOT_FIELDS, _snapshot_row
from predict_card import _clean_meta, execution_ladder
from production import allocate_stakes


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
