# Forward model scorecard

First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.

Recorded: 126; settled: 24; pending: 102.

| Metric | Market | Current model | 50/50 blend |
|---|---:|---:|---:|
| Decisive fights | 24 | 24 | 24 |
| Card dates | 3 | 3 | 3 |
| Log loss | 0.61770 | 0.60465 | 0.60951 |
| Brier | 0.20832 | 0.19754 | 0.20240 |
| Accuracy | 0.75000 | 0.79167 | 0.75000 |
| Log loss minus market | 0.00000 | -0.01305 | -0.00819 |
| Paired log loss delta ci90 | 0.00000 to 0.00000 | -0.06690 to 0.02883 | -0.03509 to 0.01273 |
| Settled bets | 0 | 1 | 0 |
| Staked | 0 | 1 | 0 |
| Pnl | 0.00000 | 2.20000 | 0.00000 |
| Roi | — | 2.20000 | — |
| Roi ci90 | — to — | — to — | — to — |

Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.

Forward start: 2026-09-14T06:00:00Z. Policy: three-way-first-observation-v1.
