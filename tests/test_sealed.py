import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import sealed


def _frame():
    return pd.DataFrame({
        "event_date": ["2024-01-01", "2025-08-31", "2025-09-01", "2026-01-01", None],
        "value": [1, 2, 3, 4, 5],
    })


class SealTests(unittest.TestCase):
    """The holdout has to survive being forgotten about, not just agreed to."""

    def test_development_stops_at_the_seal_date(self):
        got = sealed.development(_frame())
        self.assertEqual(got["value"].tolist(), [1, 2])

    def test_the_seal_date_itself_is_inside_the_holdout(self):
        # 2025-09-01 is sealed, not the last development day.
        frame = pd.DataFrame({"event_date": ["2025-09-01"], "value": [1]})
        self.assertEqual(len(sealed.development(frame)), 0)
        self.assertTrue(sealed.is_sealed(frame).iloc[0])

    def test_undated_rows_are_dropped_rather_than_assumed_safe(self):
        # A row that could be from either side leaks on the side that matters.
        self.assertNotIn(5, sealed.development(_frame())["value"].tolist())

    def test_a_frame_with_no_date_column_refuses_rather_than_guesses(self):
        with self.assertRaises(ValueError):
            sealed.development(pd.DataFrame({"value": [1]}))

    def test_report_counts_both_sides_without_returning_holdout_rows(self):
        r = sealed.report(_frame())
        self.assertEqual((r["rows_development"], r["rows_sealed"], r["rows_undated"]),
                         (2, 2, 1))


class UnsealTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._saved = sealed.ACCESS_LOG
        sealed.ACCESS_LOG = Path(self._tmp.name) / "sealed_access.log"

    def tearDown(self):
        sealed.ACCESS_LOG = self._saved
        self._tmp.cleanup()

    def test_unsealing_returns_only_holdout_rows(self):
        got = sealed.unseal(_frame(), reason="final check")
        self.assertEqual(got["value"].tolist(), [3, 4])

    def test_unsealing_without_a_reason_is_refused(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                sealed.unseal(_frame(), reason=bad)

    def test_every_opening_is_logged(self):
        sealed.unseal(_frame(), reason="first")
        sealed.unseal(_frame(), reason="second")
        lines = sealed.ACCESS_LOG.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("first", lines[0])
        self.assertIn("second", lines[1])

    def test_a_refused_unseal_does_not_consume_the_holdout(self):
        with self.assertRaises(ValueError):
            sealed.unseal(_frame(), reason="")
        self.assertFalse(sealed.ACCESS_LOG.exists())


if __name__ == "__main__":
    unittest.main()
