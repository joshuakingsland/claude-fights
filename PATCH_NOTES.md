# Production-v3 integrity update

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

## Revalidated production-v3 result

The full 2019+ event-by-event audit completed successfully:

- 302 events / 3,218 fights
- 404 qualifying bets
- 526 units staked
- +23.30 units
- +4.43% ROI
- Event-clustered 90% ROI interval: -4.37% to +13.42%
- Model log loss 0.60276 vs market 0.60387
- Gate remains `paper_only`

The prior production-v2 report showed 408 bets and +4.94% ROI. The difference
comes from making bootstrap uncertainty deterministic by event, which changes a
small number of threshold decisions. The confidence interval still crosses
zero, so no live-money conclusion is justified.
