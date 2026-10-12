# Forward model scorecard

First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.

Recorded: 165; settled: 34; pending: 131.

| Metric | Market | Current model | 50/50 blend |
|---|---:|---:|---:|
| Decisive fights | 33 | 33 | 33 |
| Card dates | 4 | 4 | 4 |
| Log loss | 0.67825 | 0.65624 | 0.66473 |
| Brier | 0.23848 | 0.22399 | 0.23023 |
| Accuracy | 0.63636 | 0.69697 | 0.66667 |
| Log loss minus market | 0.00000 | -0.02200 | -0.01352 |
| Paired log loss delta ci90 | 0.00000 to 0.00000 | -0.06210 to 0.02060 | -0.03459 to 0.00848 |
| Settled bets | 0 | 1 | 0 |
| Staked | 0 | 1 | 0 |
| Pnl | 0.00000 | 2.20000 | 0.00000 |
| Roi | — | 2.20000 | — |
| Roi ci90 | — to — | — to — | — to — |

Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.

Forward start: 2026-09-14T06:00:00Z. Policy: three-way-first-observation-v1.
