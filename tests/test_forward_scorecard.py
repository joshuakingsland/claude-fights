import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from execution_observations import read
from forward_scorecard import (FIELDS, RESULT_FIELDS, decision, execution_status,
                               record_predictions, settle, summarize)


class ForwardScorecardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'predictions.csv'
        self.results = Path(self.tmp.name) / 'results.csv'
        self.now = pd.Timestamp('2026-09-14T12:00:00Z')
        self.item = dict(fighter_a='A', fighter_b='B', source_a='A', source_b='B',
                         fighter_a_id='id_a', fighter_b_id='id_b', event_id='event',
                         scheduled_start='2026-09-15T20:00:00Z', source_fetched_at='2026-09-14T11:59:00Z',
                         p_market=.5, p_model=.7, price_a=100, price_b=100,
                         book_a='Book A', book_b='Book B', identity_resolved=True, market_books=3)
        base = dict(fighter_a='A', fighter_b='B', event_id='event',
                    commence_time=self.item['scheduled_start'], fetched_at=self.item['source_fetched_at'],
                    priced=1, odds_a=100, odds_b=100, book_updated_at='2026-09-14T11:58:00Z')
        self.quotes = pd.DataFrame([dict(base, book_key='a', book_title='Book A'),
                                    dict(base, book_key='b', book_title='Book B')])
        self.fight = dict(date='2026-09-15', fighter_a='A', fighter_b='B',
                          fighter_a_id='id_a', fighter_b_id='id_b', winner='A')

    def record(self, item=None, now=None, quotes=None):
        return record_predictions([item or self.item], self.path,
                                  self.quotes if quotes is None else quotes, now or self.now,
                                  dict(model_version='production-v3.1', manifest_hash='hash'))

    def test_full_precision_blend_and_uniform_decisions(self):
        item = dict(self.item, p_model=.712345678)
        self.record(item)
        row = read(self.path).iloc[0]
        self.assertAlmostEqual(row.p_model, item['p_model'], places=12)
        self.assertAlmostEqual(row.p_blend, (.5 + item['p_model'])/2, places=12)
        self.assertEqual(row.market_stake, 0)
        self.assertEqual(row.model_stake, 1)
        self.assertEqual(row.blend_stake, 1)
        self.assertEqual(row.execution_status, 'verified_displayed_quotes')

    def test_no_historical_backfill_or_post_start_recording(self):
        self.assertEqual(self.record(now=pd.Timestamp('2026-09-13T12:00:00Z')), 0)
        self.assertEqual(len(read(self.path)), 0)
        with self.assertRaisesRegex(ValueError, 'future'):
            self.record(now=pd.Timestamp(self.item['scheduled_start']))

    def test_first_abstention_is_frozen_and_corner_reschedule_not_duplicated(self):
        self.record(dict(self.item, p_model=.5))
        before = self.path.read_bytes()
        self.assertEqual(self.record(now=self.now + pd.Timedelta(hours=1)), 0)
        swapped = dict(self.item, fighter_a='B', fighter_b='A', event_id='rescheduled',
                       scheduled_start='2026-09-16T20:00:00Z')
        self.assertEqual(self.record(swapped), 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_real_rematch_can_record(self):
        self.record()
        item = dict(self.item, event_id='rematch', scheduled_start='2026-09-20T20:00:00Z')
        self.assertEqual(self.record(item, now=pd.Timestamp('2026-09-17T12:00:00Z')), 1)

    def test_same_identity_with_changed_display_name_is_not_duplicated(self):
        self.record()
        changed = dict(self.item, fighter_a='Alias A', event_id='new-provider-id')
        self.assertEqual(self.record(changed), 0)

    def test_later_identity_resolution_does_not_duplicate_prediction(self):
        self.record(dict(self.item, fighter_a_id='unresolved:a', identity_resolved=False))
        self.assertEqual(self.record(), 0)

    def test_invalid_probabilities_fail_without_writing_rows(self):
        for probability in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                self.record(dict(self.item, p_model=probability))
        self.assertFalse(self.path.exists())

    def test_missing_execution_does_not_drop_probability_prediction(self):
        self.record(quotes=pd.DataFrame())
        row = read(self.path).iloc[0]
        self.assertEqual(row.execution_status, 'unverified_quotes')
        self.assertEqual(row.model_stake, 0)
        self.assertEqual(row.p_model, .7)

    def test_quote_verification_rejects_stale_wrong_event_unpriced_and_conflicting(self):
        for field, value in [('book_updated_at', '2026-09-14T10:00:00Z'),
                             ('event_id', 'wrong'), ('priced', 0), ('odds_a', -110)]:
            q = self.quotes.copy()
            q[field] = value
            self.assertEqual(execution_status(self.item, q, self.now), 'unverified_quotes')
        self.assertEqual(execution_status(dict(self.item, identity_resolved=False), self.quotes, self.now), 'unresolved_identity')
        self.assertEqual(execution_status(self.item, self.quotes, self.now + pd.Timedelta(minutes=16)), 'stale_or_missing_source')

    def test_same_prices_allow_candidates_to_pick_opposite_sides(self):
        self.assertEqual(decision(.7, 100, 100, True), ('A', 1))
        self.assertEqual(decision(.3, 100, 100, True), ('B', 1))
        self.assertEqual(decision(.5, 120, 120, True), ('', 0))

    def test_settlement_uses_ids_and_handles_corner_reversal(self):
        self.record()
        fight = dict(self.fight, fighter_a='Changed Name', fighter_b='Another Name',
                     fighter_a_id='id_b', fighter_b_id='id_a', winner='B')
        predictions = read(self.path)
        self.assertEqual(settle(predictions, pd.DataFrame([fight]), self.results, '2026-09-16'), 1)
        self.assertEqual(read(self.results).iloc[0].winner_a, 1)
        before = self.results.read_bytes()
        settle(predictions, pd.DataFrame([dict(fight, winner='A')]), self.results, '2026-09-17')
        self.assertEqual(self.results.read_bytes(), before)

    def test_ambiguous_and_future_results_remain_pending(self):
        self.record()
        predictions = read(self.path)
        self.assertEqual(settle(predictions, pd.DataFrame([self.fight, self.fight]), self.results, '2026-09-16'), 0)
        self.assertEqual(settle(predictions, pd.DataFrame([self.fight]), self.results, '2026-09-14'), 0)

    def test_report_matches_all_candidates_and_counts_push_stakes(self):
        self.record()
        settle(read(self.path), pd.DataFrame([dict(self.fight, winner='draw')]), self.results, '2026-09-16')
        result = summarize(read(self.path), read(self.results), '2026-09-16')
        self.assertEqual(result['settled_fights'], 1)
        self.assertEqual(result['candidates']['model']['decisive_fights'], 0)
        self.assertEqual(result['candidates']['model']['staked'], 1)
        self.assertEqual(result['candidates']['model']['pnl'], 0)
        self.assertIsNone(result['candidates']['model']['log_loss'])

    def test_paired_metrics_and_uncertainty(self):
        self.record()
        predictions = read(self.path)
        second = predictions.iloc[0].to_dict()
        second.update(score_id='second', scheduled_start='2026-09-22T20:00:00Z')
        predictions = pd.concat([predictions, pd.DataFrame([second])], ignore_index=True)
        results = pd.DataFrame([dict(score_id=predictions.iloc[0].score_id, result='DECISIVE', winner_a=1, fight_date='2026-09-15'),
                                dict(score_id='second', result='DECISIVE', winner_a=0, fight_date='2026-09-22')])
        report = summarize(predictions, results, '2026-09-23')
        model = report['candidates']['model']
        self.assertAlmostEqual(model['log_loss'], -(np.log(.7) + np.log(.3))/2)
        self.assertAlmostEqual(model['log_loss_minus_market'], model['log_loss'] + np.log(.5))
        self.assertEqual(model['pnl'], 0)
        self.assertEqual(report['candidates']['market']['paired_log_loss_delta_ci90'], [0, 0])
        self.assertTrue(all(v is not None for v in model['paired_log_loss_delta_ci90']))
        one = summarize(predictions.iloc[:1], results.iloc[:1], '2026-09-23')
        self.assertEqual(one['candidates']['model']['paired_log_loss_delta_ci90'], [None, None])

    def test_empty_report(self):
        report = summarize(pd.DataFrame(columns=FIELDS), pd.DataFrame(columns=RESULT_FIELDS), self.now)
        self.assertEqual(report['recorded_fights'], 0)
        self.assertIsNone(report['candidates']['market']['roi'])

    def test_pricing_main_records_payload_but_preview_does_not(self):
        from predict_card import main
        previous = os.getcwd()
        try:
            os.chdir(self.tmp.name)
            up = pd.DataFrame([dict(fighter_a='A', fighter_b='B', date='2099-01-01', commence_time='2099-01-01T20:00:00Z')])
            up.to_csv('odds_upcoming.csv', index=False)
            up[['fighter_a', 'fighter_b']].to_csv('fights_v2.csv', index=False)
            for preview in (False, True):
                with self.subTest(preview=preview), ExitStack() as stack:
                    stack.enter_context(patch('sys.argv', ['predict_card.py'] + (['--preview'] if preview else [])))
                    stack.enter_context(patch('predict_card.filter_upcoming_promotion', return_value=up))
                    stack.enter_context(patch('predict_card.predict_upcoming', return_value=[{'_scorecard': self.item}]))
                    stack.enter_context(patch('predict_card.recent_results', return_value=([], {})))
                    build = stack.enter_context(patch('predict_card.build_site'))
                    stack.enter_context(patch('freshness.assess_freshness', return_value={}))
                    stack.enter_context(patch('model_manifest.write_manifest'))
                    stack.enter_context(patch('model_manifest.sha256', return_value='hash'))
                    stack.enter_context(patch('paper_ledger.record_prediction_snapshots', return_value=0))
                    recorder = stack.enter_context(patch('forward_scorecard.record_predictions', return_value=1))
                    main()
                    self.assertNotIn('_scorecard', build.call_args.args[0][0])
                    if preview:
                        recorder.assert_not_called()
                    else:
                        self.assertEqual(recorder.call_args.args[0], [self.item])
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
