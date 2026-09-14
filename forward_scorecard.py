"""Frozen three-way forward comparison; never selects production wagers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import american_payout, american_to_prob
from execution_observations import quotes_from, read, stamp, valid_price
from identity import norm_name
from paper_ledger import _same_booking

POLICY = 'three-way-first-observation-v1'
START = '2026-09-14T06:00:00Z'
CANDIDATES = ('market', 'model', 'blend')
THRESHOLD = 0.04
MAX_QUOTE_AGE = pd.Timedelta(minutes=15)
FIELDS = ['score_id', 'policy', 'locked_at', 'scheduled_start', 'event_id',
          'fighter_a', 'fighter_b', 'fighter_a_id', 'fighter_b_id',
          'source_fetched_at', 'model_version', 'manifest_hash',
          'p_market', 'p_model', 'p_blend', 'price_a', 'price_b', 'book_a', 'book_b',
          'execution_status'] + [f'{c}_{field}' for c in CANDIDATES for field in ('side', 'stake')]
RESULT_FIELDS = ['score_id', 'settled_at', 'fight_date', 'winner_a', 'result',
                 'result_fighter_a_id', 'result_fighter_b_id']


def append(path, fields, rows):
    exists = Path(path).exists() and Path(path).stat().st_size > 0
    if rows or not exists:
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)


def same_booking(left, right):
    use_ids = all(str(row.get(f'fighter_{side}_id', '')) not in ('', 'nan')
                  and not str(row.get(f'fighter_{side}_id', '')).startswith('unresolved:')
                  for row in (left, right) for side in ('a', 'b'))
    def identity(row):
        ids = (str(row.get('fighter_a_id', '')), str(row.get('fighter_b_id', '')))
        return ids if use_ids else (
            norm_name(row['fighter_a']), norm_name(row['fighter_b']))
    def booking(row):
        a, b = identity(row)
        return dict(row, pick=a, opp=b)
    same_pair = set(identity(left)) == set(identity(right))
    if same_pair and left.get('event_id') and left.get('event_id') == right.get('event_id'):
        return True
    return _same_booking(booking(left), booking(right))


def execution_status(item, quotes, now):
    fetched, start = stamp(item['source_fetched_at']), stamp(item['scheduled_start'])
    if pd.isna(fetched) or not fetched <= now < start or now - fetched > MAX_QUOTE_AGE:
        return 'stale_or_missing_source'
    if not item.get('identity_resolved'):
        return 'unresolved_identity'
    books = pd.to_numeric(item.get('market_books'), errors='coerce')
    if pd.isna(books) or books < 3:
        return 'insufficient_books'
    if quotes.empty or not item.get('event_id'):
        return 'unverified_quotes'
    q = quotes[(quotes.event_id == item['event_id']) & (stamp(quotes.fetched_at) == fetched)
               & (stamp(quotes.commence_time) == start)
               & (pd.to_numeric(quotes.priced, errors='coerce') == 1)]
    for side in ('a', 'b'):
        price = item[f'price_{side}']
        if not valid_price(price):
            return 'invalid_price'
        probability = float(item['p_market']) if side == 'a' else 1 - float(item['p_market'])
        if probability - float(american_to_prob(price)) > 0.08:
            return 'price_outlier'
        book = norm_name(item[f'book_{side}'])
        selected = q[(q.book_title.map(norm_name) == book) | (q.book_key.map(norm_name) == book)]
        fighter, opponent = norm_name(item[f'source_{side}']), norm_name(item['source_b' if side == 'a' else 'source_a'])
        matched = []
        for row in selected.to_dict('records'):
            a, b = norm_name(row['fighter_a']), norm_name(row['fighter_b'])
            value = row['odds_a'] if (a, b) == (fighter, opponent) else row['odds_b'] if (b, a) == (fighter, opponent) else None
            updated = stamp(row['book_updated_at'])
            if valid_price(value) and pd.notna(updated) and fetched - MAX_QUOTE_AGE <= updated <= fetched:
                matched.append(float(value))
        if not matched or set(matched) != {float(price)}:
            return 'unverified_quotes'
    return 'verified_displayed_quotes'


def decision(p, price_a, price_b, eligible):
    if not eligible:
        return '', 0
    edges = (p - float(american_to_prob(price_a)), 1 - p - float(american_to_prob(price_b)))
    if max(edges) < THRESHOLD or edges[0] == edges[1]:
        return '', 0
    return ('A' if edges[0] > edges[1] else 'B'), 1


def record_predictions(items, path='forward_predictions.csv', quotes=None, now=None, provenance=None):
    now = stamp(now or pd.Timestamp.now(tz='UTC'))
    old = read(path)
    kept = old.to_dict('records')
    rows = []
    if now >= stamp(START):
        quotes = quotes_from('data/market_quotes') if quotes is None else quotes
        for item in items:
            start = stamp(item['scheduled_start'])
            if pd.isna(start) or not now < start:
                raise ValueError('Forward scorecard requires an exact future scheduled start')
            p_market, p_model = float(item['p_market']), float(item['p_model'])
            if not all(np.isfinite(p) and 0 <= p <= 1 for p in (p_market, p_model)):
                raise ValueError('Forward probabilities must lie in [0, 1]')
            row = {f: item.get(f, '') for f in FIELDS}
            row.update(locked_at=now.isoformat(), scheduled_start=start.isoformat(), policy=POLICY,
                       p_market=p_market, p_model=p_model, p_blend=(p_market + p_model)/2,
                       model_version=(provenance or {}).get('model_version', ''),
                       manifest_hash=(provenance or {}).get('manifest_hash', ''))
            if any(same_booking(prior, row) for prior in kept):
                continue
            identity = '|'.join((POLICY, now.isoformat(), start.isoformat(),
                                 *sorted((norm_name(row['fighter_a']), norm_name(row['fighter_b'])))))
            row['score_id'] = hashlib.sha256(identity.encode()).hexdigest()[:24]
            row['execution_status'] = execution_status(item, quotes, now)
            for candidate in CANDIDATES:
                side, stake = decision(row[f'p_{candidate}'], row['price_a'], row['price_b'],
                                       row['execution_status'] == 'verified_displayed_quotes')
                row[f'{candidate}_side'], row[f'{candidate}_stake'] = side, stake
            kept.append(row)
            rows.append(row)
    append(path, FIELDS, rows)
    return len(rows)


def settle(predictions, fights, path='forward_results.csv', now=None):
    now = stamp(now or pd.Timestamp.now(tz='UTC'))
    old = read(path)
    done = set(old.score_id) if len(old) else set()
    rows = []
    fight_rows = [(fight, stamp(fight['date'])) for fight in fights.to_dict('records')] if len(predictions) else []
    for row in predictions.to_dict('records'):
        if row['score_id'] in done or not stamp(row['locked_at']) < stamp(row['scheduled_start']) <= now:
            continue
        date = stamp(row['scheduled_start']).normalize()
        matches = []
        for fight, fd in fight_rows:
            if pd.isna(fd) or fd > now or abs(fd.normalize() - date) > pd.Timedelta(days=1):
                continue
            if fd.normalize() + pd.Timedelta(days=1) <= stamp(row['locked_at']):
                continue
            ids = (str(row['fighter_a_id']), str(row['fighter_b_id']))
            result_ids = (str(fight.get('fighter_a_id', '')), str(fight.get('fighter_b_id', '')))
            resolved = all(v and not v.startswith('unresolved:') for v in ids)
            pair = ids if resolved else (norm_name(row['fighter_a']), norm_name(row['fighter_b']))
            other = result_ids if resolved else (norm_name(fight['fighter_a']), norm_name(fight['fighter_b']))
            if pair != other and pair != other[::-1]:
                continue
            winner = str(fight['winner'])
            if winner not in ('A', 'B', 'draw', 'NC'):
                continue
            y = '' if winner in ('draw', 'NC') else int((winner == 'A') == (pair == other))
            matches.append(dict(score_id=row['score_id'], settled_at=now.isoformat(), fight_date=fd.date().isoformat(),
                                winner_a=y, result='PUSH' if y == '' else 'DECISIVE',
                                result_fighter_a_id=result_ids[0], result_fighter_b_id=result_ids[1]))
        if len(matches) == 1:
            rows.append(matches[0])
    append(path, RESULT_FIELDS, rows)
    return len(rows)


def interval(frame, numerator, denominator, n=5000):
    """Paired card-date bootstrap, shared fixed resamples across candidates."""
    grouped = frame.groupby('fight_date')[[numerator, denominator]].sum()
    if len(grouped) < 2 or grouped[denominator].sum() <= 0:
        return [None, None]
    rng = np.random.default_rng(301)
    indices = rng.integers(0, len(grouped), size=(n, len(grouped)))
    sums = grouped.to_numpy()[indices].sum(axis=1)
    ratios = sums[sums[:, 1] > 0, 0] / sums[sums[:, 1] > 0, 1]
    return np.quantile(ratios, [.05, .95]).tolist() if len(ratios) else [None, None]


def summarize(predictions, results, now=None):
    now = stamp(now or pd.Timestamp.now(tz='UTC'))
    predictions = predictions if len(predictions) else pd.DataFrame(columns=FIELDS)
    results = results if len(results) else pd.DataFrame(columns=RESULT_FIELDS)
    merged = predictions.merge(results, on='score_id', how='left', validate='one_to_one')
    report = dict(policy=POLICY, forward_start=START, generated_at=now.isoformat(),
                  status='paper_only', recorded_fights=len(predictions), settled_fights=int(merged.result.notna().sum()),
                  pending_fights=int(merged.result.isna().sum()),
                  model_versions=predictions.model_version.value_counts().to_dict(),
                  execution_coverage=predictions.execution_status.value_counts().to_dict(), candidates={})
    decisive = merged[merged.result == 'DECISIVE'].copy()
    y = pd.to_numeric(decisive.winner_a).to_numpy(dtype=float)
    decisive['unit'] = 1
    market = np.clip(pd.to_numeric(decisive.p_market).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    baseline = -(y*np.log(market) + (1-y)*np.log(1-market))
    for candidate in CANDIDATES:
        raw_p = pd.to_numeric(decisive[f'p_{candidate}']).to_numpy(dtype=float)
        p = np.clip(raw_p, 1e-6, 1 - 1e-6)
        loss = -(y*np.log(p) + (1-y)*np.log(1-p))
        decisive['delta'] = loss - baseline
        active = merged[(merged.result.notna()) & (pd.to_numeric(merged[f'{candidate}_stake']) > 0)].copy()
        active['units'] = 1
        pnl = []
        for row in active.to_dict('records'):
            if row['result'] == 'PUSH':
                pnl.append(0.0)
            else:
                a = row[f'{candidate}_side'] == 'A'
                won = (int(float(row['winner_a'])) == 1) == a
                pnl.append(float(american_payout(row['price_a' if a else 'price_b'])) if won else -1.0)
        active['profit'] = pnl
        report['candidates'][candidate] = dict(
            decisive_fights=len(decisive), card_dates=int(decisive.fight_date.nunique()),
            log_loss=float(loss.mean()) if len(loss) else None,
            brier=float(((raw_p-y)**2).mean()) if len(p) else None,
            accuracy=float(((raw_p >= .5) == y).mean()) if len(p) else None,
            log_loss_minus_market=float((loss-baseline).mean()) if len(loss) else None,
            paired_log_loss_delta_ci90=interval(decisive, 'delta', 'unit'),
            settled_bets=len(active), staked=len(active), pnl=float(sum(pnl)),
            roi=float(np.mean(pnl)) if pnl else None, roi_ci90=interval(active, 'profit', 'units'))
    report['interpretation'] = ('First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. '
        'Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. '
        'They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.')
    return report


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not Path('forward_predictions.csv').exists():
        append('forward_predictions.csv', FIELDS, [])
    predictions = read('forward_predictions.csv')
    settle(predictions, read('fights_v2.csv'))
    report = summarize(predictions, read('forward_results.csv'))
    Path('forward_scorecard.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Forward model scorecard', '', report['interpretation'], '',
             f"Recorded: {report['recorded_fights']}; settled: {report['settled_fights']}; pending: {report['pending_fights']}.", '',
             '| Metric | Market | Current model | 50/50 blend |', '|---|---:|---:|---:|']
    for key in report['candidates']['market']:
        def display(value):
            if value is None:
                return '—'
            if isinstance(value, list):
                return ' to '.join(display(v) for v in value)
            return str(value) if isinstance(value, int) else f'{value:.5f}'
        lines.append('| ' + key.replace('_', ' ').capitalize() + ' | ' + ' | '.join(
            display(report['candidates'][c][key]) for c in CANDIDATES) + ' |')
    lines.extend(['', 'Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.', '',
                  f'Forward start: {START}. Policy: {POLICY}.', ''])
    Path('FORWARD_SCORECARD.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
