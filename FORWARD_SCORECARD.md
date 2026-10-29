# Forward model scorecard

First-observation predictions on the production UFC-filtered feed; unresolved identities remain in prediction coverage. Paper quote-price returns use a separate uniform 4-point gross-edge, 1-unit-per-fight rule without card caps or uncertainty deductions. They are not production returns or accepted fills. Intervals require at least two card dates; no result promotes a model automatically.

Recorded: 273; settled: 55; pending: 218.

| Metric | Market | Current model | 50/50 blend |
|---|---:|---:|---:|
| Decisive fights | 54 | 54 | 54 |
| Card dates | 6 | 6 | 6 |
| Log loss | 0.69660 | 0.69229 | 0.69194 |
| Brier | 0.24836 | 0.24326 | 0.24481 |
| Accuracy | 0.59259 | 0.64815 | 0.61111 |
| Log loss minus market | 0.00000 | -0.00431 | -0.00466 |
| Paired log loss delta ci90 | 0.00000 to 0.00000 | -0.03918 to 0.02720 | -0.02204 to 0.01118 |
| Settled bets | 0 | 1 | 0 |
| Staked | 0 | 1 | 0 |
| Pnl | 0.00000 | 2.20000 | 0.00000 |
| Roi | — | 2.20000 | — |
| Roi ci90 | — to — | — to — | — to — |

Negative log-loss differences favor the candidate. ROI is a fraction, not a percentage.

Forward start: 2026-09-14T06:00:00Z. Policy: three-way-first-observation-v1.
