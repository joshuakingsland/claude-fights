# Entry-price research decision

## Decision

Keep `production-v3` unchanged and paper-only. The historical entry-price
audit now has useful CLV evidence, but it does not clear the promotion gate.

## Data audit

- 28,764 source quote rows were normalized and hashed.
- 1,779 fights have an entry consensus from at least three books at least 24
  hours before the scheduled start.
- 885 of those fights have a valid later pre-card close proxy after merging
  the sparse archive and API snapshots.
- The earlier zero-close result was caused by interpreting the 15-minute
  target as 15 hours. The corrected builder finds 680 archive close proxies.
- API consensus probabilities are the median of per-book de-vigged
  probabilities. Closing observations remain benchmark-only.

## Walk-forward result

The stable-identity 2022-2026 audit contains 173 cards and 1,266 predicted fights:

- Model log loss: 0.59050
- Entry-market log loss: 0.59102
- Bets: 203; staked: 255 units
- P&L: +11.91 units; ROI: +4.67%
- 90% event-clustered ROI interval: -4.88% to +13.95%
- Close-covered model bets: 102
- Mean CLV: +0.59 probability points
- 90% event-clustered mean CLV interval: +0.19 to +1.02 points
- Positive CLV rate: 57.8%

The fixed 50% market / 50% model benchmark improves log loss to 0.58808, but
it produces only 21 bets, of which 16 have close coverage. Its historical ROI
interval is positive while its CLV interval crosses zero; both samples are far
below the promotion minimum. It is a research candidate, not a production
change.

## Entry-model experiment

`research_entry_models.py` compares four entry-trained candidates on the same
173-card walk-forward window. Every fit uses only fights completed before the
current card. The nested blend chooses its market weight from prior
out-of-fold predictions only; future outcomes are excluded by code and test.

| Candidate | Log loss | Bets | ROI | 90% card-clustered ROI interval |
| --- | ---: | ---: | ---: | ---: |
| Current entry refit | 0.59050 | 203 | +4.67% | -4.88% to +13.95% |
| Market-offset | 0.59064 | 110 | +13.49% | -0.28% to +27.34% |
| Fixed 50/50 | **0.58808** | 21 | +36.29% | +5.48% to +70.86% |
| Past-only nested blend | 0.58900 | 9 | +33.18% | -12.60% to +75.81% |

The market-offset candidate fixes the entry-market logit coefficient at 1,
uses no intercept, and learns only a symmetric fighter-stat correction. It
does not improve log loss. The nested blend improves log loss versus both the
entry market and current model, but its nine bets are far too few to assess.
All four candidates remain `paper_only`; `production-v3` is unchanged.

## Promotion blockers

- The clustered ROI lower bound is below zero.
- Only 102 active bets have close coverage; the gate requires 200.

Re-run `prepare_api_odds_history.py` and `validate_entry_history.py` as close
coverage grows. Re-run `research_entry_models.py` only after new point-in-time
entry or close data arrives; repeatedly tuning against the same sample would
overfit the audit. No result in this repository enables automated wagering.
