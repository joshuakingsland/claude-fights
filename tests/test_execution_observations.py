import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from execution_observations import FIELDS, capture_target, observe, record, report
from fetch_odds import main as fetch
from paper_ledger import TRADE_FIELDS


class ExecutionObservationTests(unittest.TestCase):
    def setUp(self):
        self.trade = dict(trade_id='one', locked_at='2026-09-14T12:00:00Z',
                          scheduled_start='2026-09-15T20:00:00Z', pick='A', opp='B',
                          execution_book='Book', price=100, execution_price=100, stake=1,
                          odds_fetched_at='2026-09-14T11:59:00Z')
        self.base = dict(event_id='event', fighter_a='A', fighter_b='B', book_key='book',
                         book_title='Book', commence_time=self.trade['scheduled_start'],
                         fetched_at=self.trade['odds_fetched_at'], priced=1,
                         book_updated_at=self.trade['odds_fetched_at'], odds_a=100, odds_b=-110)
        self.later = dict(self.base, fetched_at='2026-09-14T12:05:01Z',
                          book_updated_at='2026-09-14T12:04:00Z', odds_a=-110)
        self.now = '2026-09-14T12:08:00Z'

    def check(self, *quotes, trade=None):
        return observe(trade or self.trade, pd.DataFrame([self.base, *quotes]), self.now)

    def test_first_quote_worse_is_not_replaced_by_later_better_quote(self):
        better = dict(self.later, fetched_at='2026-09-14T12:06:00Z', odds_a=120)
        result = self.check(self.later, better)
        self.assertEqual(result['observed_price'], -110)
        self.assertEqual(result['same_or_better'], 0)
        self.assertGreater(result['slippage_prob_points'], 0)

    def test_corner_swap_and_positive_improvement(self):
        result = self.check(dict(self.later, fighter_a='B', fighter_b='A', odds_b=110))
        self.assertEqual(result['observed_price'], 110)
        self.assertEqual(result['same_or_better'], 1)

    def test_missing_book_at_first_capture_is_not_later_fill(self):
        absent = dict(self.later, book_key='other', book_title='Other')
        returned = dict(self.later, fetched_at='2026-09-14T12:06:00Z')
        self.assertEqual(self.check(absent, returned)['status'], 'book_not_observed')

    def test_window_rejects_early_late_inplay_and_other_event(self):
        for change in [dict(fetched_at='2026-09-14T12:04:59Z'),
                       dict(fetched_at='2026-09-14T12:07:01Z'),
                       dict(commence_time='2026-09-14T12:05:00Z'), dict(event_id='rematch')]:
            with self.subTest(change=change):
                self.assertEqual(self.check(dict(self.later, **change))['status'], 'unobserved')

    def test_stale_invalid_and_research_quotes_are_not_observed(self):
        for change, expected in [(dict(book_updated_at='2026-09-14T11:00:00Z'), 'stale_quote'),
                                 (dict(odds_a=-1), 'invalid_quote'),
                                 (dict(priced=0), 'book_not_observed')]:
            self.assertEqual(self.check(dict(self.later, **change))['status'], expected)

    def test_unverified_source_and_unrelated_pair(self):
        self.assertEqual(self.check(self.later, trade=dict(self.trade, execution_book='Missing'))['status'], 'source_unverified')
        self.assertEqual(observe(self.trade, pd.DataFrame([dict(self.later, fighter_a='C')]), self.now)['status'], 'unobserved')

    def test_waits_until_window_closes(self):
        self.assertIsNone(observe(self.trade, pd.DataFrame(), '2026-09-14T12:06:59Z'))

    def test_capture_only_recent_future_unchecked_locks(self):
        trades = pd.DataFrame([self.trade])
        now = '2026-09-14T12:01:00Z'
        self.assertEqual(capture_target(trades, pd.DataFrame(), now), pd.Timestamp('2026-09-14T12:05:00Z'))
        self.assertIsNone(capture_target(trades, pd.DataFrame([{'trade_id': 'one'}]), now))
        self.assertIsNone(capture_target(trades, pd.DataFrame(), self.now))
        self.assertIsNone(capture_target(trades, pd.DataFrame(), '2026-09-14T11:59:00Z'))
        started = pd.DataFrame([dict(self.trade, scheduled_start='2026-09-14T12:03:00Z')])
        self.assertIsNone(capture_target(started, pd.DataFrame(), now))

    def test_append_only_even_when_new_history_is_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checks.csv'
            trades = pd.DataFrame([self.trade])
            record(trades, pd.DataFrame([self.base, self.later]), path, self.now)
            before = path.read_bytes()
            self.assertEqual(record(trades, pd.DataFrame(), path, self.now), 0)
            self.assertEqual(path.read_bytes(), before)

    def test_report_covers_missing_and_matched_settlement_subset(self):
        check = self.check(self.later)
        trades = pd.DataFrame([self.trade, dict(self.trade, trade_id='two', pick='C', opp='D')])
        settlements = pd.DataFrame([dict(trade_id='one', result='WIN', pnl=1, clv_prob='')])
        result = report(trades, pd.DataFrame([check]), settlements, self.now)['cumulative']
        self.assertEqual(result['locks'], 2)
        self.assertEqual(result['observed'], 1)
        self.assertEqual(result['statuses']['pending'], 1)
        self.assertAlmostEqual(result['delayed_price_pnl'], 100/110)
        self.assertEqual(result['original_price_pnl_same_subset'], 1)
        self.assertEqual(result['entry_market_clv_covered'], 0)
        self.assertEqual(result['delayed_price_close_covered'], 0)

    def test_delayed_close_advantage_uses_the_executable_price(self):
        check = self.check(self.later)
        settlements = pd.DataFrame([dict(trade_id='one', result='LOSS', pnl=-1, clv_prob=2, closing_market=55)])
        result = report(pd.DataFrame([self.trade]), pd.DataFrame([check]), settlements, self.now)['cumulative']
        self.assertAlmostEqual(result['delayed_price_close_advantage_points'], 55 - 100*110/210)
        self.assertEqual(result['delayed_price_pnl'], -1)

    def test_empty_report_and_historical_separation(self):
        result = report(pd.DataFrame(columns=TRADE_FIELDS), pd.DataFrame(), pd.DataFrame(), self.now)
        self.assertEqual(result['cumulative']['locks'], 0)
        old = dict(self.trade, locked_at='2026-09-01T00:00:00Z')
        result = report(pd.DataFrame([old]), pd.DataFrame(columns=FIELDS), pd.DataFrame(), self.now)
        self.assertEqual(result['cumulative']['locks'], 0)
        self.assertEqual(result['historical_diagnostic']['locks'], 1)

    def test_quotes_only_does_not_touch_card_or_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                names = ['odds_upcoming.csv', 'odds_log.csv', 'market_snapshot_manifest.json']
                for name in names:
                    Path(name).write_text('preserve')
                with patch.dict(os.environ, {'ODDS_API_KEY': 'fake'}), patch('fetch_odds.collect_events', return_value=[]):
                    fetch(['--quotes-only', '--require-key'])
                self.assertTrue(all(Path(n).read_text() == 'preserve' for n in names))
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
