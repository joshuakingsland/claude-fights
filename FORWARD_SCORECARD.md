# Forward model scorecard

First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.

Recorded: 101; settled: 19; pending: 82.

| Metric | Market | Current model | 50/50 blend |
|---|---:|---:|---:|
| Decisive fights | 19 | 19 | 19 |
| Card dates | 2 | 2 | 2 |
| Log loss | 0.71125 | 0.70103 | 0.70406 |
| Brier | 0.24808 | 0.23654 | 0.24167 |
| Accuracy | 0.68421 | 0.73684 | 0.68421 |
| Log loss minus market | 0.00000 | -0.01022 | -0.00719 |
| Paired log loss delta ci90 | 0.00000 to 0.00000 | -0.08036 to 0.04079 | -0.04232 to 0.01835 |
| Settled bets | 0 | 1 | 0 |
| Staked | 0 | 1 | 0 |
| Pnl | 0.00000 | 2.20000 | 0.00000 |
| Roi | — | 2.20000 | — |
| Roi ci90 | — to — | — to — | — to — |

Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.

Forward start: 2026-09-14T06:00:00Z. Policy: three-way-first-observation-v1.
