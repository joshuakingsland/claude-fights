# Forward model scorecard

First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.

Recorded: 327; settled: 78; pending: 249.

| Metric | Market | Current model | 50/50 blend |
|---|---:|---:|---:|
| Decisive fights | 77 | 77 | 77 |
| Card dates | 8 | 8 | 8 |
| Log loss | 0.66677 | 0.64906 | 0.65585 |
| Brier | 0.23511 | 0.22531 | 0.22930 |
| Accuracy | 0.62338 | 0.67532 | 0.64935 |
| Log loss minus market | 0.00000 | -0.01772 | -0.01092 |
| Paired log loss delta ci90 | 0.00000 to 0.00000 | -0.04781 to 0.01322 | -0.02589 to 0.00463 |
| Settled bets | 0 | 1 | 0 |
| Staked | 0 | 1 | 0 |
| Pnl | 0.00000 | 2.20000 | 0.00000 |
| Roi | — | 2.20000 | — |
| Roi ci90 | — to — | — to — | — to — |

Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.

Forward start: 2026-09-14T06:00:00Z. Policy: three-way-first-observation-v1.
