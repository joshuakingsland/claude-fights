# Production-v3 execution-pricing and staking update

This package was upgraded from the verified v2 prototype to production-v3.

## August 2026 forward-data review

The first three weeks of six-hour market capture are summarized in
`FORWARD_TEST_REVIEW.md`. Model probabilities, features, edge rule, and stake
sizing are unchanged.

- Execution prices are now sanity-checked against the paired-book consensus.
  On 2026-08-01 one book posted +126 on a side the consensus held at -265, and
  the resulting 26.5-point net edge passed the edge rule as eligible. The
  28-point `market_spread` was recorded but only drove a display warning, not
  eligibility. `predict_card.quote_quality` now rejects a quote implying more
  than `MAX_EXECUTION_DEVIATION` (8 points) less probability than consensus,
  with reason `book price outlier`. Replayed across all 1,641 priced snapshots
  the guard fires on that row alone.
- The quote gate moved out of the prediction loop into `quote_quality`, which
  returns its own rejection reason and is covered by unit tests.
- Removed `.github/workflows/snapshot-market`, an inert byte-for-byte copy of
  `snapshot-market.yml` still pinned to `actions/checkout@v4` and
  `actions/setup-python@v5`. GitHub never loaded it, so the v5/v6 upgrade noted
  below was in force, but the stale copy contradicted it.
- Added the `.gitignore` the README already described. Bytecode caches and the
  three rebuildable `cache_v3_*.pkl` files are no longer tracked.

Open, and deliberately not changed here: official trades lock only on the
Wednesday run, so a signal that qualifies for days but dips at that one
timestamp is never recorded. Two of five eligible fights were captured in the
window, which puts the 200-bet live gate roughly 4.4 years out. Changing the
cadence changes the forward-test policy mid-run against an append-only ledger.

## July 2026 freshness-guard correction

- The result freshness audit now applies the same explicit fighter-name aliases
  as ingestion, so `Stephen Erceg` / `Steve Erceg` and
  `Ramazonbek Temirov` / `Ramazan Temirov` resolve deterministically.
- Repeated odds-log snapshots are collapsed to one date-and-fighter-pair key
  before missing-result counts are calculated.
- Confirmed cancellations live in `cancelled_fights.csv` with status, reason,
  source, and confirmation date. They are reported separately and do not
  weaken the fail-closed behavior for unaccounted completed fights.
- The first registry entry records the UFC-confirmed cancellation of
  Islam Dulatov vs Wellington Turman on July 25, 2026.

## July 2026 refresh recovery

- The upstream scraper removed the `UFC - Road to UFC 4.6` event-details row
  while retaining its official fight results and statistics. That made two
  completed fights disappear during the adapter join and correctly tripped
  the historical-identity regression gate.
- `update_data.py` now recovers a missing event date only when the event still
  appears in the current result source and the previous validated fight table
  provides one unambiguous date. Recoveries are recorded in
  `data_source_manifest.json` and capped at five events per refresh.
- Unknown missing event dates, disappeared results, dropped fights, dropped
  fighter IDs, row shrinkage, and backward result dates still fail closed.
- All workflows now use `actions/checkout@v5` and
  `actions/setup-python@v6`, removing the Node 20 deprecation warning.

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
