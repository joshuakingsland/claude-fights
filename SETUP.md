# Fight Ledger setup

## GitHub Pages automation

1. Create a GitHub repository and upload this folder, including `.github/`.
2. In repository Settings → Pages, deploy from `main` and `/docs`.
3. Optional automated odds: create an `ODDS_API_KEY` Actions secret from
   the-odds-api.com. Without it, edit `odds_upcoming.csv` manually.
4. Run **Update Fight Ledger** from the Actions tab once. The workflow installs
   dependencies, refreshes results, fetches odds, writes `docs/index.html`,
   and commits changed generated data.

The workflow runs Sunday, Monday, and Wednesday (the Wednesday run captures
an early odds snapshot). GitHub Pages then serves the committed `docs/` site.

## Local use

```bash
python -m pip install -r requirements.txt
python -m py_compile *.py
python predict_card.py
python validate_production.py --start 2019-01-01
python backtest_experiments.py --bootstrap 5000
python monitor_drift.py
```

For a refresh (network required):

```bash
python update_data.py
python fetch_odds.py       # or edit odds_upcoming.csv by hand
python predict_card.py
```

The validator uses strict event-date walk-forward training and writes
`production_validation.json`. Its ROI interval resamples whole events, not
individual fights. Use `--models 30` for deployment-equivalent uncertainty;
`--models 3 --event-bootstrap 200` is a fast smoke test.
The report's `live_gate` remains `paper_only` unless the interval is wholly
positive and there are at least 50 events and 200 bets.

Run the experiment script before changing calibration, thresholds, event caps,
or drawdown rules. The current holdout results do not justify promoting any of
those alternatives.

Each card run appends every prediction to `paper_trades.csv`. The scheduled
workflow settles completed rows into the separate append-only
`paper_settlements.csv`. These files are paper trading only; no wagers are
sent anywhere.

## Important limits

- UFCStats data can lag an event by a day or two.
- Training and rolling-results evaluation prefer closing odds in
  `raw/ufc-master.csv`. `odds_log.csv` is an explicitly labelled fallback for
  completed fights not covered by that file; it is not a closing-line source.
- The displayed staking rule is uncertainty-adjusted net edge: 1 unit above
  4 points and 2 units above 8 points. The rolling ledger uses this same rule,
  but it is not a promise of future returns.
- This is research, not betting advice. Debutants, long layoffs, injuries,
  camps, and weight cuts are outside the model's reliable inputs.
