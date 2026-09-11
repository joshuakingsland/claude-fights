# September 2026 integrity corrections

Production remains paper-only. These are correctness repairs, not new model
selection or evidence from an untouched holdout. The archived research results
remain retrospective; the preregistered rules and decision gates are unchanged.

## Historical American prices

`prepare_api_odds_history._consensus` previously took the arithmetic median
of American prices. An even book count could interpolate across the -100/+100
boundary: -102 and +100 became -1, and the payout formula then returned 100
units of profit. The Gunnar Nelson–Kevin Holland row on 2025-03-22 exhibited
this failure in `historical_entry_validation.csv`.

The importer now selects an observed upper-median price, matching live
capture. Market probabilities still use the median of paired, per-book
de-vigged probabilities. Entry/close validation and payouts reject invalid
American prices rather than turning malformed prices into profit.

Both derived API history files were rebuilt from the existing quote partitions.
The original audit window, 2020-06-01 through 2025-09-01 exclusive, was rerun:
990 evaluated fights, 135 selected bets, -0.3543 units and -0.2624% ROI, with a
90% event-clustered interval of [-11.4562%, +10.7046%]. The previous report
showed +73.5007% ROI. The same model probabilities and market probabilities
are retained; corrected prices change execution selection and payouts.

## Rescheduled wagers

The McVey–Schultz booking was locked twice after its scheduled start crossed
a UTC date boundary. `effective_trades` retains the first lock and reports the
later entry as an explicit correction. In `paper_validation.json`,
`b8cfb4555007268dad95` is excluded and `6ddbf2c301d5ba1cc5d4` is retained.
Neither `paper_trades.csv` nor `paper_settlements.csv` is rewritten.

Matching is independent of corner order. Locks for the same pair, with starts
within sixty days and overlapping pre-event lock periods, identify one booking.
A rematch booked after the earlier fight is still a separate wager. Existing
first-touch locks keep their original prices and priority. When the current
card carries a revised start date, the existing wager still consumes a slot
on that card. Notifications and official metrics use the same corrected view.

CLV coverage is now explicit. Missing closing prices are excluded from the
positive-CLV fraction rather than being counted as negative CLV.

## Upcoming matchups

`UPCOMING` rows are queries, not observations. They have no training target
and cannot update Elo, career win/method rates, fight counts, layoff dates,
KO history, or rolling three-bout statistics. Moneyline, method and rounds
models all use this rule. Adding another unfought matchup for the same fighter
therefore cannot change the later matchup's career features.

The deployment is versioned `production-v3.1`. Feature caches and manifests
include the shared historical-state helper. A `predict_card.py --preview`
rebuild updates the page and manifest without adding prediction or trade rows.

## Reproduction

```bash
python -m unittest discover -s tests
python prepare_api_odds_history.py
python validate_entry_history.py --start 2020-06-01 --end 2025-09-01
python validate_production.py --start 2019-01-01
python validate_paper.py
python validate_staking.py
python validate_method.py
python validate_rounds.py
python predict_card.py --preview
```

The regression suite covers invalid median prices and payouts, rescheduling,
corner swaps, genuine rematches, card caps, explicit corrections without file
rewrites, CLV coverage, placeholder outcomes and query-batch invariance.
