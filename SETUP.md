# Fight Ledger setup

## GitHub Pages automation

1. Create a GitHub repository and upload this folder, including `.github/`.
2. In repository Settings → Pages, deploy from `main` and `/docs`.
3. Create a fresh The Odds API key and add it as the `ODDS_API_KEY` Actions
   secret. Rotate any key that has appeared in chat, logs, or a commit.
   Without it, maintain `odds_upcoming.csv` manually with an exact future
   `commence_time`.
4. Run **Update Fight Ledger** from Actions once. It refreshes results, settles
   prior official paper wagers, fetches odds, writes the dashboard, records a
   timestamped snapshot, validates the forward ledger, and commits outputs.

The routine workflow runs Sunday, Monday, and Wednesday. Sunday and Monday
record snapshots only. Wednesday also locks at most one official qualifying
paper wager per fight. The separate **Refresh Canonical Validation** workflow
runs monthly because the exact historical walk-forward audit is much heavier.
The **Capture Standardized T-30 Odds** workflow polls event metadata during
the normal Saturday/Sunday UFC window and spends an odds request only when a
not-yet-captured fight is 10-50 minutes away. GitHub schedules can be delayed,
so this is a standardized window rather than a claim of an exact closing line.

## Local use

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

To deliberately lock official paper wagers from the current card:

```bash
python predict_card.py --lock-paper-trades
```

Without that flag, the command records prediction snapshots but does not add
anything to the official trade ledger.

For a network refresh:

```bash
python update_data.py
python fetch_odds.py --require-key
python predict_card.py
```

For historical entry-price and CLV research:

```bash
python historical_odds.py --dry-run
python historical_odds.py --max-requests 30
python prepare_api_odds_history.py
python validate_entry_history.py
python research_entry_models.py
```

Run the dry run before each paid batch. The manifest makes later runs resume
without repeating completed or deterministic failed requests. The real API
key belongs in the local `ODDS_API_KEY` environment variable or the GitHub
Actions secret named `ODDS_API_KEY`; `.env.example` is only a placeholder.

The local historical CSVs and API response material are intentionally ignored
by Git. Do not override that ignore rule for a public repository. The audit
JSONs contain aggregate counts and hashes and can be committed.

The `Plan Historical Odds Backfill` workflow performs a dry run only. Run paid
historical batches locally, where `snapshot_manifest.csv` persists and prevents
duplicate credit spend.

The manual `Discover MMA Prop Markets` workflow defaults to zero event-market
requests. Increase its explicit request cap only after reviewing API quota;
it catalogs market keys and does not download prop prices.

Manual odds use (`market_prob_a` and `market_books` may be blank):

```text
date,commence_time,fighter_a,fighter_b,odds_a,odds_b,market_prob_a,market_books,weightclass,five_rounds,odds_source,fetched_at
```

Use an ISO-8601 UTC time such as `2026-08-01T23:00:00Z`. The code refuses any
row that is not demonstrably pre-event. A date-only row must be dated after the
current UTC date; it cannot be recorded on the event day.

## Validation and ledger interpretation

The canonical historical command is:

```bash
python validate_production.py --start 2019-01-01
```

It uses strict event-date walk-forward training and event-clustered ROI
uncertainty. The bootstrap ensemble seed comes from the event date, so the same
event is reproduced identically across overlapping validation windows.

Forward evidence is separate:

```bash
python paper_ledger.py summary
python validate_paper.py
```

- `prediction_snapshots.csv` contains every pre-event model run.
- `paper_trades.csv` contains only official qualifying locked wagers.
- `paper_settlements.csv` contains immutable results and closing-line value.

Each row carries model and manifest provenance. The legacy mixed file is
preserved in `archive/` and is excluded from verified forward statistics.

The `live_gate` remains `paper_only` unless the historical clustered interval
is wholly positive with at least 50 events and 200 bets. Forward-test closing
line value should also be positive before any real-money conclusion is drawn.

The separate entry-price gate is stricter: it also requires the model to beat
the entry market on log loss and at least 200 CLV-covered bets with a positive
event-clustered CLV interval. A blend that looks best on the audit sample is
reported as a research candidate, not silently deployed.

## Important limits

- UFCStats data can lag an event by a day or two.
- Historical training prefers closing odds in `raw/ufc-master.csv`.
  `odds_log.csv` is labelled fallback data, not assumed closing data.
- The displayed rule is 1 unit above 4 net points and 2 units above 8.
- The model cannot account for injuries, camps, weight cuts, or tape.
- This is research and paper tracking, not betting advice or automated betting.
