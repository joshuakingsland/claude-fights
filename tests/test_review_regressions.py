"""Regressions for the September 2026 model-integrity review."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest import american_to_prob
from features_v3 import build_features_v3
from method_model import career_method_rates, attach_side_features
from prepare_api_odds_history import _consensus
from paper_ledger import lock_paper_trades, summary
from production import event_pnl
import rounds_model


class HistoricalPriceTests(unittest.TestCase):
    def test_even_book_consensus_never_averages_across_even_money(self):
        rows = []
        for i, price in enumerate([-102, 100]):
            pa, pb = american_to_prob([-119, price])
            rows.append(dict(event_uid='x', event_name='Card',
                event_date=pd.Timestamp('2025-03-22', tz='UTC'),
                snapshot_kind='entry', pair='a|b',
                actual_snapshot=pd.Timestamp('2025-03-21', tz='UTC'),
                commence_time=pd.Timestamp('2025-03-22 20:00', tz='UTC'),
                fighter_a='A', fighter_b='B', book_key=str(i),
                odds_a=-119, odds_b=price, book_prob_a=pa/(pa+pb)))
        result = _consensus(pd.DataFrame(rows), 2).iloc[0]
        self.assertIn(result.odds_b, [-102, 100])
        self.assertAlmostEqual(result.consensus_prob_a,
                               pd.DataFrame(rows).book_prob_a.median())

    def test_invalid_price_cannot_generate_a_windfall(self):
        row = pd.DataFrame([dict(pick_side='B', stake=1, y=0,
                                 R_odds=-119, B_odds=-1)])
        with self.assertRaisesRegex(ValueError, 'American'):
            event_pnl(row)


class RescheduledLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trades = self.root / 'trades.csv'
        self.snapshots = self.root / 'snapshots.csv'
        self.settlements = self.root / 'settlements.csv'

    def item(self, date, pick='Jackson McVey', opp='Wes Schultz'):
        return dict(date=date, scheduled_start=date+'T21:00:00Z',
                    pick=pick, opp=opp, price=-166, market=61, model=70,
                    edge=8, se=1, net=7, bet=True, stake=1)

    def lock(self, item, when):
        return lock_paper_trades([item], self.snapshots, self.trades,
                                 locked_at=when)

    def test_midnight_reschedule_and_corner_swap_cannot_lock_twice(self):
        self.assertEqual(self.lock(self.item('2026-08-23'), '2026-08-12'), 1)
        self.assertEqual(self.lock(self.item('2026-08-22', 'Wes Schultz',
                                            'Jackson McVey'), '2026-08-14'), 0)

    def test_actual_rematch_after_first_fight_can_be_locked(self):
        self.assertEqual(self.lock(self.item('2026-08-23'), '2026-08-12'), 1)
        self.assertEqual(self.lock(self.item('2026-09-13'), '2026-09-01'), 1)

    def test_existing_duplicate_is_explicitly_excluded_without_rewriting(self):
        rows = []
        for i, date in enumerate(['2026-08-23', '2026-08-22']):
            rows.append(dict(self.item(date), trade_id=f't{i}',
                locked_at=f'2026-08-{12+i:02d}T00:00:00Z', staking_policy='test'))
        pd.DataFrame(rows).to_csv(self.trades, index=False)
        pd.DataFrame([dict(trade_id='t0', pnl=.6, clv_prob=1),
                      dict(trade_id='t1', pnl=.55, clv_prob=-1)]).to_csv(
                          self.settlements, index=False)
        before = self.trades.read_bytes(), self.settlements.read_bytes()
        report = summary(self.trades, self.settlements)
        self.assertEqual(report['official_trades'], 1)
        self.assertEqual(report['settled'], 1)
        self.assertAlmostEqual(report['pnl'], .6)
        self.assertEqual(report['excluded_duplicate_trades'][0]['trade_id'], 't1')
        self.assertEqual(report['excluded_duplicate_trades'][0]['retained_trade_id'], 't0')
        self.assertEqual(before, (self.trades.read_bytes(), self.settlements.read_bytes()))


class UnfoughtFeatureTests(unittest.TestCase):
    @staticmethod
    def data():
        bouts = [('2024-01-01', 'A', 'B', 'A'),
                 ('2024-02-01', 'A', 'C', 'B'),
                 ('2024-03-01', 'A', 'B', 'A'),
                 ('2024-04-01', 'A', 'C', 'B'),
                 ('2024-05-01', 'A', 'B', 'A'),
                 ('2024-06-01', 'A', 'C', 'A')]
        fights, stats = [], []
        for i, (date, a, b, winner) in enumerate(bouts):
            upcoming = i >= 4
            event = 'UPCOMING' if upcoming else f'Card {i}'
            bout = f'{a} vs. {b}'
            fights.append(dict(date=pd.Timestamp(date), event=event, bout=bout,
                fighter_a=a, fighter_b=b, fighter_a_id=a, fighter_b_id=b,
                winner=winner, method='' if upcoming else 'KO/TKO',
                fight_time_min=np.nan if upcoming else 10,
                time_format='3 Rnd (5-5-5)', weightclass='Lightweight Bout',
                reach_a=72, reach_b=70, height_a=70, height_b=69,
                dob_a='1990-01-01', dob_b='1991-01-01'))
            if not upcoming:
                for name in (a, b):
                    stats.append(dict(EVENT=event, BOUT=bout, FIGHTER=name,
                        sig_l=20+i, sig_a=40, head_l=10, dist_l=15, td_l=1,
                        td_a=2, kd=1, sub_att=0, ctrl_s=60, n_rounds=2, fade=0))
        return pd.DataFrame(fights), pd.DataFrame(stats)

    def test_unfought_rows_do_not_change_later_moneyline_features(self):
        fights, stats = self.data()
        with patch('features_v2.load_round_stats', return_value=stats):
            batch, cols = build_features_v3(fights)
            single, _ = build_features_v3(fights.drop(index=4))
        np.testing.assert_allclose(batch.iloc[-1][cols].to_numpy(float),
                                   single.iloc[-1][cols].to_numpy(float), atol=1e-12)

    def test_placeholder_winner_cannot_change_future_elo(self):
        fights, stats = self.data()
        changed = fights.copy()
        changed.loc[4, 'winner'] = 'B'
        with patch('features_v2.load_round_stats', return_value=stats):
            original, cols = build_features_v3(fights)
            flipped, _ = build_features_v3(changed)
        np.testing.assert_allclose(original.iloc[-1][cols].to_numpy(float),
                                   flipped.iloc[-1][cols].to_numpy(float), atol=1e-12)

    def test_unfought_rows_do_not_change_method_or_rounds_features(self):
        fights, _ = self.data()
        alone = fights.drop(index=4).reset_index(drop=True)
        both = attach_side_features(fights, career_method_rates(fights))
        single = attach_side_features(alone, career_method_rates(alone))
        cols = [c for c in both if c.startswith(('r_', 'n_pre'))]
        np.testing.assert_allclose(both.iloc[-1][cols].to_numpy(float),
                                   single.iloc[-1][cols].to_numpy(float), atol=1e-12)
        np.testing.assert_allclose(rounds_model.build_X(rounds_model.prepare(fights)).iloc[-1],
                                   rounds_model.build_X(rounds_model.prepare(alone)).iloc[-1])


if __name__ == '__main__':
    unittest.main()
