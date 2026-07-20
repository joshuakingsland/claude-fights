# Fight Ledger model

This repository generates a static UFC moneyline model dashboard. The
production entry point is `predict_card.py`; it uses point-in-time features,
historical closing odds from `raw/ufc-master.csv`, and the shared production
configuration in `config.py`.

## Included, runnable workflow

From a fresh checkout with Python 3.10+:

```bash
python -m pip install -r requirements.txt
python -m py_compile *.py
python backtest.py --events 50
python predict_card.py
python validate_production.py --start 2019-01-01
python backtest_experiments.py --bootstrap 5000
python monitor_drift.py
```

`predict_card.py` writes `docs/index.html`; the validator writes the audit
artifacts. The card command uses the bundled `fights_v2.csv`,
`raw/` inputs, `odds_upcoming.csv`, and `method_model.pkl`. Only load the
pickle when it comes from a trusted checkout; it is not needed for moneyline
predictions, and method props are skipped if it cannot load.

To refresh UFCStats-derived results (network required), then rebuild:

```bash
python update_data.py
python fetch_odds.py       # set ODDS_API_KEY, or keep/edit odds_upcoming.csv manually
python predict_card.py
```

`update_data.py` clears feature caches. Historical training and backtest odds
remain limited to `raw/ufc-master.csv`; `odds_log.csv` records newly captured
upcoming lines and is an explicitly labelled fallback for completed fights not
covered by the master file. Those rows are not closing-line rows.

The exact production audit writes `production_validation.json` and
`production_validation.csv`. It reports model-versus-line log loss, net-edge
stakes, P&L, and a 90% event-clustered ROI interval. Use `--models 30` for the
same bootstrap ensemble size as deployment; smaller values are only a smoke
test.

The validator emits a `live_gate` decision. Until the event-clustered ROI
interval is wholly positive with at least 50 events and 200 bets, the correct
status is `paper_only`; this repository never places wagers automatically.

`backtest_experiments.py` is the promotion gate for calibration, threshold,
event-cap, and drawdown changes. The current experiments do not support
promoting a different betting rule: apparent aggregate improvements do not
survive the 2025+ holdout. `monitor_drift.py` reports distribution shifts but
does not tune the model.

## Production model and limits

The deployment model is a symmetrized ridge logistic regression on de-vigged
market probability plus the focused differentials in `config.py` and
`ko_recent`. A displayed wager requires a net edge (model edge minus bootstrap
uncertainty) above 4 points; it is 2 units above 8 points.

The dashboard's historical “recent results” currently uses the available
matched closing-line dataset and the exact net-edge production screen. It is
informative, but it is not a promise of future returns. The event-clustered
validator is the canonical performance report.

This is research, not betting advice. It cannot account for injury, camp,
weight-cut, or tape information. Name matching is normalized text matching;
the code warns about known homonyms, and stable fighter/fight IDs remain the
appropriate future fix.

## Repository map

- `predict_card.py` — train on all matched history and generate the site.
- `config.py` — production constants, deliberately independent of research.
- `pipeline.py` — matched data, walk-forward, and gross-edge utilities.
- `features.py`, `features_v2.py`, `features_v3.py`, `elo.py` — feature work.
- `backtest.py` — separate baseline backtest (defaults to `fights_v2.csv`).
- `adapter.py`, `update_data.py`, `fetch_odds.py` — data refresh and odds capture.
- `.github/workflows/update.yml` — scheduled GitHub Pages update.

`production.py` and `validate_production.py` are the shared deployed model and
event-clustered audit. `paper_ledger.py` records append-only paper predictions
and settlements. `research.py` and `research3.py` are archived research harnesses, not part of
the production workflow. `research3.py` still requires the unshipped
`research2.py`, so it is intentionally not presented as runnable.
