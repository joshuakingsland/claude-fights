"""Append-only five-minute quote checks; no wagers or accepted-fill claims."""
import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from backtest import american_payout, american_to_prob
from identity import norm_name
from paper_ledger import TRADE_FIELDS, effective_trades

POLICY = 'quote-persistence-5m-v1'
POLICY_START = '2026-09-14T00:00:00Z'
DELAY = pd.Timedelta(minutes=5)
TOLERANCE = pd.Timedelta(minutes=2)
SOURCE_AGE = pd.Timedelta(minutes=15)
FIELDS = ['trade_id', 'policy', 'checked_at', 'status', 'event_id', 'book_key',
          'source_fetched_at', 'observed_at', 'book_updated_at', 'delay_seconds',
          'requested_price', 'observed_price', 'same_or_better', 'slippage_prob_points']


def read(path):
    return pd.read_csv(path, keep_default_na=False, low_memory=False) if Path(path).exists() else pd.DataFrame()


def quotes_from(root):
    parts = sorted(Path(root).glob('quotes_*.csv'))
    return pd.concat([read(p) for p in parts], ignore_index=True).drop_duplicates() if parts else pd.DataFrame()


def stamp(value):
    return pd.to_datetime(value, utc=True, errors='coerce')


def valid_price(value):
    try:
        return pd.notna(float(value)) and float('-inf') < float(value) < float('inf') and abs(float(value)) >= 100
    except (ValueError, TypeError):
        return False


def observe(trade, quotes, now):
    """Use the first post-delay snapshot, never shop across later snapshots."""
    now, locked, start = stamp(now), stamp(trade['locked_at']), stamp(trade['scheduled_start'])
    if now < locked + DELAY + TOLERANCE:
        return None  # wait until the observation window closes
    row = dict.fromkeys(FIELDS, '')
    row.update(trade_id=trade['trade_id'], policy=POLICY, checked_at=now.isoformat(),
               status='unobserved', requested_price=trade.get('execution_price', trade['price']))
    if pd.isna(locked) or pd.isna(start) or locked >= start or not valid_price(row['requested_price']):
        row['status'] = 'invalid_lock'
        return row
    if quotes.empty:
        return row
    q = quotes.copy()
    q['_time'] = stamp(q.fetched_at)
    q['_start'] = stamp(q.commence_time)
    pick, opp = norm_name(trade['pick']), norm_name(trade['opp'])
    a, b = q.fighter_a.map(norm_name), q.fighter_b.map(norm_name)
    q = q[((a == pick) & (b == opp)) | ((a == opp) & (b == pick))].copy()
    if q.empty:
        return row
    q['_price'] = q.odds_a.where(q.fighter_a.map(norm_name) == pick, q.odds_b)
    # Exact provider event identity comes from the source observation, not a
    # loose name-only join across bookings. Missing evidence remains missing.
    source = q[(q['_time'] == stamp(trade.get('odds_fetched_at')))
               & (q['_time'] <= locked) & (q['_time'] >= locked - SOURCE_AGE)
               & (q['_start'] == start) & (q['_time'] < q['_start'])
               & (pd.to_numeric(q.priced, errors='coerce') == 1)
               & ((q.book_title.map(norm_name) == norm_name(trade.get('execution_book', '')))
                  | (q.book_key.map(norm_name) == norm_name(trade.get('execution_book', ''))))]
    source = source.loc[source['_price'].map(valid_price).astype(bool)]
    source = source[source['_price'].astype(float) == float(row['requested_price'])]
    if len(source[['event_id', 'book_key']].drop_duplicates()) != 1 or not str(source.iloc[0].event_id):
        row['status'] = 'source_unverified'
        return row
    first = source.iloc[0]
    row.update(event_id=first.event_id, book_key=first.book_key, source_fetched_at=first.fetched_at)
    later = q[(q.event_id == first.event_id) & (q['_time'] >= locked + DELAY)
              & (q['_time'] <= locked + DELAY + TOLERANCE) & (q['_time'] <= now)
              & (q['_time'] < start) & (q['_time'] < q['_start'])]
    if later.empty:
        return row
    earliest = later['_time'].min()
    chosen = later[(later['_time'] == earliest) & (later.book_key == first.book_key)
                   & (pd.to_numeric(later.priced, errors='coerce') == 1)]
    row.update(observed_at=earliest.isoformat(), delay_seconds=(earliest - locked).total_seconds())
    if chosen.empty:
        row['status'] = 'book_not_observed'
        return row
    if len(chosen['_price'].astype(str).unique()) != 1 or not valid_price(chosen.iloc[0]['_price']):
        row['status'] = 'invalid_quote'
        return row
    chosen = chosen.iloc[0]
    updated = stamp(chosen.book_updated_at)
    if pd.isna(updated) or updated > earliest or earliest - updated > SOURCE_AGE:
        row['status'] = 'stale_quote'
        return row
    price = float(chosen['_price'])
    requested = float(row['requested_price'])
    row.update(status='observed', observed_price=price, book_updated_at=chosen.book_updated_at,
               same_or_better=int(float(american_payout(price)) >= float(american_payout(requested))),
               slippage_prob_points=float(100 * (american_to_prob(price) - american_to_prob(requested))))
    return row


def record(trades, quotes, path, now):
    old = read(path)
    seen = set(old.trade_id.astype(str)) if len(old) else set()
    canonical, _ = effective_trades(trades)
    rows = [observe(t, quotes, now) for t in canonical.to_dict('records') if str(t['trade_id']) not in seen]
    rows = [r for r in rows if r is not None]
    if rows or not Path(path).exists():
        exists = Path(path).exists() and Path(path).stat().st_size > 0
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def report(trades, checks, settlements, now):
    canonical, corrections = effective_trades(trades)
    if checks.empty:
        checks = pd.DataFrame(columns=FIELDS)
    merged = canonical.merge(checks, on='trade_id', how='left', validate='one_to_one')
    if len(settlements):
        merged = merged.merge(settlements, on='trade_id', how='left', validate='one_to_one')
    def metrics(frame):
        seen = frame[frame.status == 'observed']
        settled = seen[seen.get('result', pd.Series('', index=seen.index)).isin(['WIN', 'LOSS', 'PUSH'])]
        stakes = pd.to_numeric(settled.stake, errors='raise')
        pnl = [float(s) * (float(american_payout(float(p))) if r == 'WIN' else -1 if r == 'LOSS' else 0)
               for s, p, r in zip(stakes, settled.observed_price, settled.get('result', []))]
        clv = pd.to_numeric(seen.get('clv_prob', pd.Series(dtype=float)), errors='coerce').dropna()
        close = pd.to_numeric(seen.get('closing_market', pd.Series(index=seen.index, dtype=float)), errors='coerce')
        priced_close = seen[close.notna() & close.between(0, 100)]
        price_advantage = (close.loc[priced_close.index] - 100 * american_to_prob(
            pd.to_numeric(priced_close.observed_price).to_numpy(dtype=float)))
        return dict(locks=len(frame), observed=len(seen), statuses=frame.status.fillna('pending').value_counts().to_dict(),
                    same_or_better_rate=float(pd.to_numeric(seen.same_or_better).mean()) if len(seen) else None,
                    mean_slippage_probability_points=float(pd.to_numeric(seen.slippage_prob_points).mean()) if len(seen) else None,
                    settled_observed=len(settled), delayed_price_pnl=sum(pnl),
                    delayed_price_roi=sum(pnl)/float(stakes.sum()) if stakes.sum() else None,
                    original_price_pnl_same_subset=float(pd.to_numeric(settled.get('pnl', pd.Series(dtype=float))).sum()),
                    entry_market_clv_covered=len(clv), entry_market_clv_mean_points=float(clv.mean()) if len(clv) else None,
                    delayed_price_close_covered=len(price_advantage),
                    delayed_price_close_advantage_points=float(price_advantage.mean()) if len(price_advantage) else None)
    dates = stamp(merged.locked_at)
    historical = merged[dates < stamp(POLICY_START)]
    forward = merged[(dates >= stamp(POLICY_START)) & (dates <= stamp(now))]
    recent = forward[stamp(forward.locked_at) >= stamp(now) - pd.Timedelta(days=7)]
    return dict(policy=POLICY, generated_at=stamp(now).isoformat(), status='paper_only',
                forward_start=POLICY_START,
                interpretation='Displayed quotes only; no accepted tickets, limits, or account eligibility verified. Observed-subset ROI is not a portfolio return or proof of edge.',
                excluded_duplicates=corrections, historical_diagnostic=metrics(historical),
                cumulative=metrics(forward), last_seven_days=metrics(recent))


def capture_target(trades, checks, now):
    canonical, _ = effective_trades(trades)
    seen = set(checks.trade_id.astype(str)) if len(checks) else set()
    pending = canonical[~canonical.trade_id.astype(str).isin(seen)]
    times = stamp(pending.locked_at)
    starts = stamp(pending.scheduled_start)
    eligible = ((times >= stamp(POLICY_START)) & (times <= stamp(now))
                & (times + DELAY + TOLERANCE > stamp(now))
                & (starts > times + DELAY) & (starts > stamp(now)))
    targets = times[eligible] + DELAY
    return targets.max() if len(targets) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true', help='capture once at five minutes for recent pending locks')
    args = parser.parse_args()
    trades = read('paper_trades.csv')
    if trades.empty:
        trades = pd.DataFrame(columns=TRADE_FIELDS)
    path = 'execution_checks.csv'
    if args.capture:
        now = pd.Timestamp.now(tz='UTC')
        target = capture_target(trades, read(path), now)
        if target is not None:
            time.sleep(max(0, (target - now).total_seconds()))
            from fetch_odds import main as capture
            capture(['--require-key', '--quotes-only'])
            # Close the window before finalizing missing observations.
            time.sleep(max(0, (target + TOLERANCE - pd.Timestamp.now(tz='UTC')).total_seconds()))
    now = pd.Timestamp.now(tz='UTC')
    record(trades, quotes_from('data/market_quotes'), path, now)
    output = report(trades, read(path), read('paper_settlements.csv'), now)
    Path('execution_report.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Weekly execution observations', '', output['interpretation'], '',
             'Positive slippage means a worse price. Missing observations are not failed fills.', '',
             '| Metric | Last seven days (lock date) | Cumulative |', '|---|---:|---:|']
    for key in output['cumulative']:
        if key != 'statuses':
            def display(value):
                if value is None:
                    return '—'
                if key.endswith('_rate') or key.endswith('_roi'):
                    return f'{value:.2%}'
                return str(value) if isinstance(value, int) else f'{value:.4f}'
            label = key.replace('_', ' ').capitalize()
            lines.append(f"| {label} | {display(output['last_seven_days'][key])} | {display(output['cumulative'][key])} |")
    lines.extend(['', 'Observation status counts:', '', '```json', json.dumps(output['cumulative']['statuses'], indent=2), '```', '',
                  f'Forward cohort begins {POLICY_START}. Historical locks are diagnostics only (see JSON).', '',
                  'Policy: ' + POLICY + '. Production selection and stakes are unchanged.', ''])
    Path('EXECUTION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
