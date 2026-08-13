# Pre-registration: the 2026-08 odds archive

Written **before** any of this data was pulled, and committed so the timestamp
is checkable. The point is to fix the questions and the decision rules while
they are still honest, because the failure mode this archive invites is not
finding nothing. It is finding something.

At roughly a million credits we can sample the market densely enough that a
flexible model will locate *some* subset of fights, books, and time windows
where it appears to have beaten the close. Most of those will be shapes in
noise. This session alone produced three of them: a 49-point "edge" on
Contender Series totals that was a model with no features, apparent moneyline
edges on the same card that were the model echoing the price it was fed, and a
promotion label that proved itself right because it recorded the request. None
announced themselves. Each was caught by looking, and the looking only worked
because there was a stated expectation to violate.

So the rules go first.

## The question that matters

The model's own backtest is **non-monotonic in its claimed edge**, which is
the single most important fact we hold and the reason for the whole exercise:

| claimed edge | production ROI | entry-price ROI | mean CLV | CLV n |
|---|---|---|---|---|
| 4-6 pts | +6.1% | +2.2% | +0.80 | 46 |
| 6-8 pts | -2.4% | -4.9% | +0.09 | 23 |
| 8-10 pts | -12.3% | -20.5% | +0.72 | 16 |
| 10+ pts | +25.8% | +51.4% | +0.57 | 17 |

A model with a real edge gets *better* as its claimed edge grows. This one gets
worse, then abruptly and spectacularly better on 23 bets. The entire positive
result rests on that last cell.

**H1 is therefore the primary hypothesis, and everything else is secondary:**
the 10+ bucket's closing-line value is real rather than a handful of longshots
landing.

## Hypotheses, with the tests fixed in advance

**H1 (primary) - the 10+ bucket has genuine CLV.**
Test: mean CLV in probability points among bets claiming a 10+ point edge,
with a 90% event-clustered bootstrap interval, on n >= 200 qualifying bets.
Confirmed only if the lower bound is above zero.

**H2 - CLV is monotone in claimed edge.**
Test: Spearman rank correlation between edge bucket and mean CLV across the
four buckets, each with n >= 100. A real detector of mispricing should show
CLV rising with the size of the disagreement. Current data shows no trend.

**H3 - a leading book's move predicts the consensus move.**
Test: regression of consensus change over horizon h on a candidate leader's
change over the preceding interval, per book, event-clustered. Reported as an
R^2 and a lag in minutes. Confirmed only if a leader's move predicts the
consensus at a horizon long enough to actually place a bet (>= 10 minutes).

**H4 - stale lines at slow books are capturable.**
Test: given H3, the count and size of windows where one book's price implies a
probability more than the vig hurdle away from a consensus that has already
moved. Reported as opportunities per card and mean size, after vig.

## Decision rules, fixed now

- **Bet only if H1 confirms.** Not H2, not H3, not "the backtest looks good."
- If H1 confirms and H2 does not, the policy becomes 10+ only, and the current
  4-point rule is retired as noise. That is a change to how the model bets,
  not a reason to bet more.
- If H1 fails, stop betting. Keep the pipeline as an instrument. This is the
  expected outcome and it is not a failure of the exercise.
- H3 and H4 are a separate programme. Neither authorises a bet on its own, and
  a measured lead-lag pattern is not evidence it survives vig, latency, or
  a book that limits winners.

## The sealed holdout

Every card from **2025-09-01 onward is sealed**. It is not loaded, not
inspected, not summarised, and not used to choose a specification. It is
opened once, after a decision has been reached on the earlier data, and it
is opened exactly once. If the result disagrees with the development set, the
holdout wins.

This costs us the most recent and most relevant year of data for development.
That is the price of having an answer we can trust, and it is worth paying
because the alternative is discovering the specification was fitted only after
money is on it.

## What would make me wrong to have written this

If H1 confirms cleanly on a large sample and survives the holdout, the caution
here was expensive - we could have been betting sooner. I judge that risk
smaller than the reverse. Four settled trades, an ROI interval spanning zero,
and a CLV of -0.13 points is not a base from which to start believing.
