# Production-v3 stable-identity and collection update

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

## Revalidated production-v3 result

The full 2019+ event-by-event audit completed successfully:

- 302 events / 3,218 fights
- 415 qualifying bets
- 534 units staked
- +29.45 units
- +5.52% ROI
- Event-clustered 90% ROI interval: -3.18% to +14.49%
- Model log loss 0.60253 vs market 0.60387
- Gate remains `paper_only`

The preceding package showed 404 bets and +4.43% ROI. The change comes from
separating same-name fighters and refreshing the source through July 18, 2026.
The confidence interval still crosses zero, so no live-money conclusion is
justified.
