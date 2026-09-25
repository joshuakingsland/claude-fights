# Forward model scorecard

First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.

Recorded: 100; settled: 11; pending: 89.

| Metric | Market | Current model | 50/50 blend |
|---|---:|---:|---:|
| Decisive fights | 11 | 11 | 11 |
| Card dates | 1 | 1 | 1 |
| Log loss | 0.76429 | 0.80508 | 0.78264 |
| Brier | 0.26569 | 0.27362 | 0.26944 |
| Accuracy | 0.63636 | 0.63636 | 0.63636 |
| Log loss minus market | 0.00000 | 0.04079 | 0.01835 |
| Paired log loss delta ci90 | — to — | — to — | — to — |
| Settled bets | 0 | 1 | 0 |
| Staked | 0 | 1 | 0 |
| Pnl | 0.00000 | 2.20000 | 0.00000 |
| Roi | — | 2.20000 | — |
| Roi ci90 | — to — | — to — | — to — |

Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.

Forward start: 2026-09-14T06:00:00Z. Policy: three-way-first-observation-v1.
