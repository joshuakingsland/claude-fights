# Forward model scorecard

First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.

Recorded: 287; settled: 66; pending: 221.

| Metric | Market | Current model | 50/50 blend |
|---|---:|---:|---:|
| Decisive fights | 65 | 65 | 65 |
| Card dates | 7 | 7 | 7 |
| Log loss | 0.66351 | 0.65882 | 0.65901 |
| Brier | 0.23327 | 0.22863 | 0.23007 |
| Accuracy | 0.63077 | 0.67692 | 0.64615 |
| Log loss minus market | 0.00000 | -0.00469 | -0.00450 |
| Paired log loss delta ci90 | 0.00000 to 0.00000 | -0.03427 to 0.02266 | -0.01933 to 0.00932 |
| Settled bets | 0 | 1 | 0 |
| Staked | 0 | 1 | 0 |
| Pnl | 0.00000 | 2.20000 | 0.00000 |
| Roi | — | 2.20000 | — |
| Roi ci90 | — to — | — to — | — to — |

Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.

Forward start: 2026-09-14T06:00:00Z. Policy: three-way-first-observation-v1.
