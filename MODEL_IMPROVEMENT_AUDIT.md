# Model improvement audit

Production remains unchanged. Candidates use development data before 2024,
a 2024 validation period, and a final 2025+ holdout.

## Verdicts

| Candidate | Production | Entry | Decision |
| --- | --- | --- | --- |
| Symmetric calibration | reject_for_now | reject_for_now | Do not deploy unless entry evidence passes |
| Uncertainty multiplier | reject_for_now | reject_for_now | Fixed grid, development-selected |
| Minimum book count | n/a | reject_for_now | Coverage filter, not a probability model |
| Entry vs close timing | n/a | defer | Close-covered holdout only |
| Sportsbook dispersion | n/a | defer | API-card sample is insufficient |

## Calibration holdout

| Dataset | Scale | Validation log-loss delta | Holdout log-loss delta | Holdout policy ROI: raw / calibrated |
| --- | ---: | ---: | ---: | ---: |
| Production | 1.0086 | 0.000006 | -0.000063 | +0.4% / -1.2% |
| Entry | 1.0471 | 0.000586 | 0.001327 | +11.8% / +10.3% |

## Selection tests

- Production uncertainty: development selected `0.0x`; verdict `reject_for_now`.
- Entry uncertainty: development selected `2.0x`; verdict `reject_for_now`.
- Minimum books: development selected `3`; verdict `reject_for_now`.
- Timing: 168 holdout fights, direction accuracy +54.8%, verdict `defer`.

## Guardrails

- The production model and policy are not modified by this audit.
- The final holdout is not used to choose calibration or filter settings.
- The timing oracle is explicitly non-actionable and only measures the ceiling.
- Actual-ticket execution cannot be backtested because no ticket ledger exists.
- Individual-book dispersion is unavailable for the large consensus archive.
- Calibration promotion requires paired card-clustered improvement intervals
  wholly below zero in both validation and holdout.

See `model_improvement_audit.json` for all metrics and confidence intervals.
