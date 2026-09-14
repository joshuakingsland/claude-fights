# Three-way forward scorecard

Policy `three-way-first-observation-v1` begins at **2026-09-14 06:00 UTC**.
No historical predictions are reconstructed or imported into this test.

The first scored pre-fight observation of each booking records three full-precision
probabilities for fighter A: bookmaker consensus, the current production model,
and their arithmetic 50/50 blend. The deployed model already uses market inputs;
the comparison measures its incremental contribution, not a market-free model.
The model version and manifest hash are recorded. Routine production retraining
continues, while the blend weight and comparison rules remain fixed.

The scope is every scored fight in the existing production UFC-filtered feed,
including abstentions and unresolved identities. This feed can include unlabeled
MMA events; it is not a census of every UFC fight. DWCS/all-mode runs and previews
do not append this ledger. Stable fighter identities and the existing overlapping
booking rule prevent reschedules, corner swaps, and display-name changes from
creating additional observations. A subsequent genuine rematch is separate.
Later prices or stronger signals never replace the first observation.

## Probability comparison

Every decisively settled recorded fight contributes to log loss, Brier score,
and accuracy for all three candidates, whether or not it was eligible for a
paper wager. Log-loss probabilities are clipped to [0.000001, 0.999999] for
numerical stability. The report shows paired log-loss differences from market
consensus on exactly the same fights. Negative differences favor the candidate.
Draws/no contests remain settled pushes but are excluded from binary accuracy
metrics. Unmatched or ambiguous results remain pending and are counted.

## Common price and selection rule

All candidates receive the same captured best price and bookmaker on each side.
Both prices must match the archived provider event and capture timestamp. Only
priced-region quotes qualify; capture and bookmaker update must each be no more
than fifteen minutes old at their respective checks. A price implying over eight
percentage points less probability than consensus is rejected. At least three
consensus books and resolved fighter identities are required for the paper-return
comparison. Failure of these checks does not remove the probability observation.

Each candidate picks the side with the larger probability-minus-price-implied
edge and records one unit only if that edge is at least four percentage points.
Exact ties abstain. There is no uncertainty subtraction, event cap, or later
reselection in this research rule. It isolates probability differences and is
distinct from the official production strategy. Decisions, prices, probabilities,
and abstentions are frozen before outcomes are available.

Return calculations use these displayed source prices, not accepted fills or
five-minute follow-up prices. The execution report remains the separate test of
quote persistence. ROI uses settled stakes, including push stakes. Each
candidate's bet count is explicit; the return comparison covers different
selections from the same initial opportunity set.

## Settlement, uncertainty, and reporting

Results match stable fighter IDs where available, otherwise normalized names,
in either corner order and within one UTC date of the recorded scheduled start.
Only a single unambiguous completed result can settle a prediction. No result
settles before its scheduled start; an older fight whose entire date predates
the recording time is excluded. Longer reschedules remain pending rather than
being assigned a speculative result. Settlements are append-only.

The report uses 5,000 bootstrap resamples with seed 301, clustered by result UTC
card date. Multiple cards on one date are deliberately grouped together.
Paired log-loss differences use the same fights and clusters for every candidate.
ROI intervals resample dates with settled bets for that candidate. Intervals are
not emitted with fewer than two dates, and a small number of dates can still give
unstable intervals. These are descriptive 90% intervals, not sequential testing
or an automatic promotion rule. Weekly examination is not permission to tune
the candidates or stop the test when an interval looks favorable.

`forward_predictions.csv` and `forward_results.csv` are script-owned ledgers.
`forward_scorecard.json` and `FORWARD_SCORECARD.md` are derived reports. Existing
market/update jobs refresh them; the Monday reporting workflow saves dated
copies under `reports/forward/`. A different model version or comparison rule
requires a separately declared evaluation cohort before interpreting it as
confirmation. Version counts are exposed in the JSON.

```bash
python -m unittest discover -s tests
python forward_scorecard.py  # settle and report only; never backfills predictions
```

No scorecard result changes the production model, stakes, or paper-only status.
