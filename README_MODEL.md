# Fight Ledger model

This repository generates a static UFC moneyline model dashboard. The
production entry point is `predict_card.py`; it uses point-in-time features,
historical closing odds from `raw/ufc-master.csv`, and stable `production-v3`
configuration from `config.py`.

## Included, runnable workflow

From a fresh checkout with Python 3.10+:

```bash
python -m pip install -r requirements.txt
python -m py_compile *.py
python -m unittest discover -s tests
python predict_card.py
python validate_production.py --start 2025-01-01 --models 3 --event-bootstrap 200
python validate_paper.py
python monitor_drift.py
```

The short validator command is a smoke test. The canonical audit is:

```bash
python validate_production.py --start 2019-01-01
```

`predict_card.py` writes `docs/index.html`; the validators write historical
and forward-test audit artifacts. An empty `odds_upcoming.csv` is valid and
produces an empty upcoming card rather than inventing sample predictions.

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
date,commence_time,fighter_a,fighter_b,odds_a,odds_b,weightclass,five_rounds,odds_source,fetched_at
```

Predictions fail closed when a commence time is not in the future. For a
manual date-only row, the event date itself is rejected because the system
cannot prove the prediction preceded the fight.

Historical training and backtest odds prefer `raw/ufc-master.csv`.
`odds_log.csv` is an explicitly labelled fallback for completed fights not
covered by that file; those rows are not treated as closing lines.

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

The historical validator emits a `live_gate` decision. Until the clustered ROI
interval is wholly positive with at least 50 events and 200 bets, the status is
`paper_only`. This repository never places wagers automatically.

## Immutable paper-ledger design

The files now have distinct purposes:

- `prediction_snapshots.csv` — every verified pre-event model/price snapshot.
- `paper_trades.csv` — one official qualifying locked wager per fight.
- `paper_settlements.csv` — append-only outcomes and available closing-line value.
- `paper_validation.json` — forward-test summary.

Each new snapshot and trade stores the model version, model-manifest hash, odds
source, odds fetch time, event start, and recording/lock time. A repeat run
cannot lock a second official trade for the same fight.

The scheduled workflow records snapshots on Sunday and Monday and records a
snapshot plus the official qualifying paper wagers on Wednesday. The original
mixed ledger is preserved unchanged under
`archive/paper_trades_legacy_mixed_predictions.csv`; it is not counted as
verified forward-test evidence.

## Production model and limits

The deployed model is a symmetrized ridge logistic regression on de-vigged
market probability plus the focused differentials in `config.py` and
`ko_recent`. A displayed wager requires net edge—model edge minus bootstrap
uncertainty—above 4 points; it is 2 units above 8 points.

Content-addressed caches include source-data and feature-code hashes, so a
changed dataset or feature implementation cannot silently reuse an old cache.
The monthly `Refresh Canonical Validation` workflow updates the expensive
historical audit separately from routine card updates.

The dashboard's rolling historical results use the available matched dataset
and exact production screen. They are informative, not a promise of future
returns. The model cannot see injuries, camps, weight cuts, or tape. Name
matching remains normalized text matching; stable fighter and fight IDs are
still the appropriate future upgrade.

## Repository map

- `predict_card.py` — train on matched history, snapshot a card, and build the site.
- `production.py` — exact deployed estimator, symmetry, stable seeds, and stakes.
- `validate_production.py` — historical event-clustered production audit.
- `paper_ledger.py`, `validate_paper.py` — immutable forward-test ledger and audit.
- `config.py` — production constants, independent of research scripts.
- `pipeline.py` — matched data and content-addressed feature cache.
- `features*.py`, `elo.py` — point-in-time feature construction.
- `adapter.py`, `update_data.py`, `fetch_odds.py` — data and odds capture.
- `.github/workflows/update.yml` — routine site/snapshot/settlement workflow.
- `.github/workflows/validate.yml` — separate monthly canonical audit.

`research.py` and `research3.py` are archived research harnesses, not production
entry points. `research3.py` still requires the unshipped `research2.py` and is
therefore intentionally not presented as runnable.
