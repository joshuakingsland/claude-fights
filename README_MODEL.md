# Fight Ledger model

This repository generates a static UFC moneyline model dashboard. The
production entry point is `predict_card.py`; it uses point-in-time features,
historical closing odds from `raw/ufc-master.csv`, and stable `production-v3`
configuration from `config.py`.

## Included, runnable workflow

From a fresh checkout with Python 3.12.13:

```bash
python -m pip install -r requirements.txt
python -m py_compile *.py
python -m unittest discover -s tests
python predict_card.py
python validate_production.py --start 2025-01-01 --models 3 --event-bootstrap 200
python validate_paper.py
python monitor_drift.py
python freshness.py --require-current
python validate_method.py
```

The short validator command is a smoke test. The canonical audit is:

```bash
python validate_production.py --start 2019-01-01
```

`predict_card.py` writes `docs/index.html`; the validators write historical
and forward-test audit artifacts. An empty `odds_upcoming.csv` is valid and
produces an empty upcoming card rather than inventing sample predictions.

Historical careers, Elo, physicals, and method rates are keyed by UFCStats
fighter IDs derived from fighter URLs. Display names are never used as career
keys. Same-name fighters are resolved by division, and an unresolved upcoming
identity receives neutral history rather than another fighter's record.

The optional baseline commands below can take substantially longer on a fresh
machine:

```bash
python backtest.py --events 50
python backtest_experiments.py --bootstrap 5000
```

`method_model.pkl` is used only for optional method-prop fair prices. Load it
only from a trusted checkout. Moneyline predictions continue if it cannot load.

## Refreshing data and odds

```bash
python update_data.py
python fetch_odds.py       # set ODDS_API_KEY, or edit odds_upcoming.csv manually
python predict_card.py
```

Automatic odds capture now stores the exact API `commence_time`, source, and
fetch timestamp. Manual rows should use these columns:

```text
date,commence_time,promotion,event_title,event_id,fighter_a,fighter_b,odds_a,odds_b,market_prob_a,market_books,market_spread,best_odds_a,best_book_a,best_odds_b,best_book_b,weightclass,five_rounds,odds_source,fetched_at
```

`promotion`, `event_title`, `event_id`, `market_prob_a`, and `market_books`
are optional for manual rows. Automatic capture pairs both fighters within
each book, de-vigs each paired quote, and uses the median per-book probability
as the model's market input. Median American prices remain consensus
provenance. The best captured price on each side is stored separately and is
the only price used for execution edge and payout calculations. A better
sportsbook quote never changes the model's consensus input.

## DWCS and other MMA cards

The production workflow remains UFC-first:

```bash
python fetch_odds.py --require-key
python predict_card.py --lock-paper-trades
```

DWCS can be scored as an explicit card mode:

```bash
python fetch_odds.py --promotion dwcs
python predict_card.py --promotion dwcs
```

If the odds provider does not include card-level promotion metadata, tag
manual rows with `promotion=DWCS` or `event_title=Dana White's Contender
Series...`, then run `python predict_card.py --promotion dwcs`. The model is
still trained on UFC history. DWCS fighters often lack UFCStats identities, so
neutral-history predictions are expected and the paper-trade quality gates
remain strict.

A best price is only accepted when it is consistent with the consensus it was
drawn from. If the executable price implies more than `MAX_EXECUTION_DEVIATION`
less probability than the paired-book consensus for that side, the signal is
rejected as `book price outlier` rather than treated as an edge. Line shopping
is worth about one to two probability points; a larger gap has so far only ever
meant a stale or mis-mapped quote at a single book.

Every paired sportsbook quote is appended to a monthly file under
`data/market_quotes/`, including book, timestamp, both prices, event ID,
region, priced flag, and per-book de-vig probability.
`market_snapshot_manifest.json` summarizes the latest capture.

## Captured regions versus priced regions

`ODDS_REGIONS` lists the regions requested; `PRICED_ODDS_REGIONS` lists the
regions whose books are allowed to reach the model. They are deliberately
separate. Capture is currently `us` and `eu`, priced is `us` only.

The `eu` region is requested so that Pinnacle — the reference sharp MMA market,
absent from the US region — accumulates history in `data/market_quotes/`.
Nothing from an unpriced region enters the consensus median, the book count,
the market spread, or the executable price. Widening capture therefore cannot
move the model's most important feature, and cannot credit an edge to a price
the ledger has no way to take. Promoting `eu` into `PRICED_ODDS_REGIONS` is a
production model change and needs the full validation track, not a config
flip.

Regions are requested one per API call rather than comma-joined. The cost is
identical, because the API bills one credit per region per market, and each
quote gets an unambiguous region of origin. A book appearing in more than one
region keeps its priced quote, so the consensus does not depend on which
region answered first. Quote rows written before this change have an empty
region and are US; `priced_quotes` treats a missing region as priced.

Cost roughly doubles with the second region. The six-hour `Snapshot MMA
Market` workflow moves from about 124 to about 248 current-odds calls in a
31-day month, and the routine update workflow from about 13 to about 26.
Review API quota before adding a third region.

`capture_close.py` still requests `us` only. Extending the standardized T-30
capture to `eu` would give Pinnacle-referenced closing-line value, at the cost
of doubling that workflow's spend; it is a separate decision.

## Leader and follower research columns

`LEADER_BOOK_KEYS` names the books observed to set the MMA market rather than
follow it. Every snapshot and locked trade records, oriented to the picked
side:

- `leader_prob` — median de-vigged probability among market-setting books,
  drawn from every captured region so Pinnacle counts while `eu` stays unpriced.
- `follower_prob` — median among priced non-leader books, so the comparison is
  against the market the ledger actually trades.
- `leader_books`, `follower_books` — how many books backed each side.
- `leader_gap` — `leader_prob - follower_prob` in probability points. Positive
  means the setters like the pick more than the followers do.

The components are stored rather than only the gap, because the definition of
a leader will change as Pinnacle history accumulates and the raw pieces let any
variant be recomputed. Blank means no book on that side quoted; blank is never
silently read as zero.

Nothing reads these back. They are not features, not part of `quality_ok`, and
not part of the edge rule. They exist so that the question of whether sharp
disagreement predicts outcomes can be answered on forward data instead of a
2022 archive of 226 fights. Rows written before this change leave them empty.

Predictions fail closed when a commence time is not in the future. For a
manual date-only row, the event date itself is rejected because the system
cannot prove the prediction preceded the fight.

Historical training and backtest odds prefer `raw/ufc-master.csv`.
`odds_log.csv` is an explicitly labelled fallback for completed fights not
covered by that file; those rows are not treated as closing lines.

## Historical entry-price and CLV research

Historical entry-price research is separate from the production validation
track. To import a sparse consensus archive, fetch missing API snapshots, and
run the strict audit:

```bash
python prepare_odds_history.py C:\path\to\odds-history.zip
python historical_odds.py --dry-run
python historical_odds.py --max-requests 30
python prepare_api_odds_history.py
python validate_entry_history.py
python research_entry_models.py
```

`historical_odds.py` is resumable and capped by `--max-requests`. It requests
entry prices at least 24 hours before a card and a later pre-card close proxy.
The builder takes the median of each book's de-vigged probability instead of
de-vigging independently aggregated prices. `validate_entry_history.py` uses
only entry information for fitting and betting; the later snapshot is used
only for CLV and market benchmarks.

The reported close is a timestamped pre-card proxy, not a claim that it is
each sportsbook's final tradable price. The report includes its actual lead
time, coverage, book count, and an event-clustered CLV interval.

`research_entry_models.py` runs a separate, entry-trained comparison of the
current estimator, a constrained market-offset model, a fixed 50/50 blend,
and a nested blend whose weight is selected from prior out-of-fold cards only.
It writes `entry_model_research.json` and never changes production settings.

Historical API responses, historical quote rows, request manifests, and
generated historical CSVs are local research inputs. Note that
`data/odds_history/` and `raw/odds_api_historical/` are currently tracked in
this repository despite that intent; review the provider terms for the purchased
archive before treating the repository as publishable, and untrack them if it
should stay local. Current forward market snapshots under `data/market_quotes/`
are tracked deliberately.
Audit JSONs and all code remain safe to publish. Keep `ODDS_API_KEY` only in your environment or a GitHub
Actions secret; never commit it.

`capture_close.py` polls the quota-free event list and makes one H2H odds
request only when an uncaptured fight is 10-50 minutes from its scheduled
start. The result is a standardized T-30-window snapshot, not an official
book close. `discover_prop_markets.py` is manual and defaults to zero discovery
requests; it records available market keys but never fetches prop prices.
`validate_staking.py` audits flat staking, the retired 8-point doubling rule,
the research-only 10-point candidate, edge buckets, drawdown, ROI intervals,
and available CLV. It cannot promote a staking tier automatically.

Runtime versions are exact-pinned in `requirements.txt`, and GitHub Actions
uses Python 3.12.13. Upgrade these deliberately and rerun both canonical
validators before accepting a new lock set.

## Two separate validation tracks

`validate_production.py` reconstructs historical event-by-event predictions.
It reports model-versus-line log loss, net-edge stakes, P&L, and a 90%
event-clustered ROI interval. Ensemble seeds are derived from the immutable
event date, so overlapping validation windows reproduce the same event
predictions.

`validate_paper.py` evaluates only real timestamped forward-test wagers. It
reports official trades, settled P&L, and closing-line value when bundled
closing odds become available. Positive forward closing-line value should be
established before paper ROI is treated as meaningful.

The production validator emits a `live_gate` decision. The entry-price
validator adds fixed model/market blend benchmarks and a stricter promotion
gate. It requires adequate events, bets, positive clustered ROI, lower model
log loss than the entry market, and at least 200 CLV-covered bets with a
positive clustered CLV interval. Observed blend winners are research
candidates only. This repository never places wagers automatically.

`update_data.py` stages all upstream files, rebuilds and audits the complete
fight table, rejects row-count/maximum-date/fight/identity regressions, and
only then atomically replaces live inputs. If upstream event metadata
temporarily disappears while the current official result remains, the refresh
may reuse that event's single previously validated date; the recovery is
manifested and capped. Unknown or broad deletions still fail closed.
`data_freshness.json` and the dashboard show the latest result date. A tracked
completed fight missing from the result source fails the scheduled update
closed. The audit canonicalizes only explicit aliases, deduplicates repeated
market snapshots, and exempts only source-backed entries in
`cancelled_fights.csv`; known cancellations remain visible in the report.

## Immutable paper-ledger design

The files now have distinct purposes:

- `prediction_snapshots.csv` — every verified pre-event model/price snapshot.
- `paper_trades.csv` — one official qualifying locked wager per fight.
- `paper_settlements.csv` — append-only outcomes and available closing-line value.
- `paper_validation.json` — forward-test summary.

Each new snapshot and trade stores the model version, model-manifest hash,
consensus prices, executable price and book, book count, market spread, odds
fetch time, event start, and recording/lock time. A repeat run cannot lock a
second official trade for the same fight.

Both scheduled workflows record a snapshot and lock any qualifying wager they
see, so a signal is captured the first run it clears the edge rule rather than
only if it still clears it on one fixed weekday. Locking stays idempotent by
fight, and `lock_paper_trades` counts trades already on the ledger for that
event day so the 2-unit cap holds across runs, not just within one card
scoring. Candidates are considered strongest-first, so an early marginal signal
cannot take a slot from a stronger one that appears later the same day.

This replaced a Wednesday-only lock. Over 2026-07-20 to 2026-08-05 that cadence
recorded two of five eligible fights; both misses qualified for days and read
just below the edge rule at the single Wednesday timestamp. Wagers locked under
the old rule keep `paper-flat-1u-day-cap2-v1`, and `paper_validation.json`
reports each staking policy separately as well as pooled.

The original mixed ledger is preserved unchanged under
`archive/paper_trades_legacy_mixed_predictions.csv`; it is not counted as
verified forward-test evidence.

## Production model and limits

The deployed model is a symmetrized ridge logistic regression on de-vigged
market probability plus the focused differentials in `config.py` and
`ko_recent`. A displayed paper signal requires execution net edge: model
probability minus best-price implied probability and bootstrap uncertainty,
above 4 points. Active allocations are flat 1 unit and capped at 2 units per
event day. The 10-point 2-unit threshold shown by the dashboard is a research
candidate, not an active allocation.

Content-addressed caches include source-data and feature-code hashes, so a
changed dataset or feature implementation cannot silently reuse an old cache.
The monthly `Refresh Canonical Validation` workflow updates the expensive
historical audit separately from routine card updates.

The dashboard's rolling historical results use the available matched dataset
and exact production screen. They are informative, not a promise of future
returns. The model cannot see injuries, camps, weight cuts, or tape. Name
matching is still required at external odds boundaries, but all internal
career histories and physical joins use stable UFCStats fighter IDs. An
unresolved external name receives neutral history and a visible warning.

## Repository map

- `validate_entry_history.py` - strict entry-price, blend, and CLV audit.
- `research_entry_models.py` - leakage-safe entry-trained candidate comparison.
- `capture_close.py` - deduplicated standardized T-30-window H2H capture.
- `validate_staking.py` - active/candidate stake-policy and edge-tier audit.
- `audit_model_improvements.py` - frozen-split calibration, uncertainty,
  book-count, execution-timing, and sportsbook-dispersion challenger audit.
- `discover_prop_markets.py` - explicitly capped MMA prop-market discovery.
- `freshness.py` - visible result-source freshness and contradiction gate.
- `validate_method.py` - probability-only method-model audit.
- `prepare_odds_history.py`, `prepare_api_odds_history.py` - auditable odds builders.
- `historical_odds.py` - capped, resumable historical API gap filler.
- `.github/workflows/harvest.yml` - dry-run cost planning only; paid historical
  batches remain local so their manifest can resume safely.

- `predict_card.py` — train on matched history, snapshot a card, and build the site.
- `production.py` — exact deployed estimator, symmetry, stable seeds, and stakes.
- `validate_production.py` — historical event-clustered production audit.
- `paper_ledger.py`, `validate_paper.py` — immutable forward-test ledger and audit.
- `config.py` — production constants, independent of research scripts.
- `pipeline.py` — matched data and content-addressed feature cache.
- `features*.py`, `elo.py` — point-in-time feature construction.
- `adapter.py`, `update_data.py`, `fetch_odds.py` — data and odds capture.
- `.github/workflows/update.yml` — routine site/snapshot/settlement workflow.
- `.github/workflows/snapshot-market.yml` - six-hour paired-book quote capture.
  It also locks qualifying wagers, so it runs the full test suite and the
  freshness gate rather than the pricing tests alone.
- `.github/workflows/validate.yml` — separate monthly canonical audit.

`research.py` and `research3.py` are archived research harnesses, not production
entry points. `research3.py` still requires the unshipped `research2.py` and is
therefore intentionally not presented as runnable.
