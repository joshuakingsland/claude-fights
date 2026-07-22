# Production-v3 execution-pricing and staking update

This package was upgraded from the verified v2 prototype to production-v3.

## Fixed

- Production predictions now fail closed on past or unverifiable event times.
- The stale July 2026 card and old mixed ledger are preserved under `archive/`.
- Every model run goes to `prediction_snapshots.csv`; only explicit qualifying
  locks go to `paper_trades.csv`.
- Official trades are idempotent: one locked wager per fight.
- Settlements reject rows that were not demonstrably recorded before the event.
- Settlement rows calculate closing-line value when closing odds are available.
- Every snapshot/trade stores model version, manifest hash, odds source, fetch
  time, event start, and record/lock time.
- Historical bootstrap seeds are stable by event date, so overlapping
  validation windows reproduce identical probabilities, uncertainty, and bets.
- The deployment ensemble seed is stable across repeated snapshots.
- Feature caches are content-addressed by data and code inputs.
- The stale +6.9% dashboard code was removed.
- Routine card updates and the expensive canonical validation now use separate
  GitHub Actions workflows.
- Integrity tests cover timing, idempotency, settlement gating, stable seeds,
  and corner symmetry.
- Fighter careers, Elo, physicals, and method rates now use URL-derived
  UFCStats IDs. Seven duplicated display names no longer blend careers.
- The source refresh is transactional and rejects shrinking, backward, or
  identity-incomplete datasets before replacing live files.
- The dashboard publishes its exact results-through date and fails closed when
  a tracked completed fight is absent from the result source.
- A cost-gated workflow captures one standardized T-30-window H2H snapshot per
  event; paper CLV prefers it over the bundled historical-line fallback.
- Prop-market discovery is manual, capped, and defaults to zero requests.
- CI now covers identity resolution, validated adapter joins, point-in-time
  winner flipping, refresh regressions, T-30 deduplication, and UTC date joins.
- The model input now remains fixed to paired-book consensus while the best
  captured sportsbook quote determines execution edge and settlement P&L.
- Every paired quote is retained in monthly market-history files with book,
  timestamp, event ID, prices, and per-book de-vig probability.
- Active paper staking is flat 1 unit with a 2-unit event-day cap. The old
  automatic 2-unit rule at 8 points is retired; 10 points is tracked only as
  a gated research candidate in `staking_validation.json`.
- Dashboard prices now show the executable book, consensus pair, book count,
  market spread, timestamp, stale state, and separate research threshold.

## Revalidated production-v3 result

The full 2019+ event-by-event audit completed successfully:

- 302 events / 3,218 fights
- 350 allocated paper signals
- 350 units staked
- +24.12 units
- +6.89% ROI
- Event-clustered 90% ROI interval: -2.24% to +16.25%
- Model log loss 0.60253 vs market 0.60387
- Gate remains `paper_only`

The model probabilities are unchanged. The policy result changes because
stakes are flat and only the two strongest qualifying signals per event day
are allocated. The confidence interval still crosses zero, so no live-money
conclusion is justified.
