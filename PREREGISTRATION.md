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

---

# Addendum, 2026-08-13: after H1-H3 failed

Written before looking at any movement data, for the same reason as the
original: the instruction now is to find an edge, and that is precisely the
instruction that manufactures one.

## What the failures taught

A diagnostic on the development set, walk-forward by card (n=690):

| specification | log loss |
|---|---|
| market line only | 0.57872 |
| model: line + 10 fundamentals | 0.58479 |
| raw entry consensus | 0.58360 |
| fundamentals only | 0.64443 |

The fundamentals do not merely fail to help, they hurt: recalibrating the
price alone beats the full model. Fundamentals alone are far behind the
market. So the feature family is the problem, not the hyperparameters, and
adding more career statistics is not a plan.

This also explains H1 cleanly. A model whose best component is the price
cannot beat the price.

## H5: line movement, not fight outcomes

The one thing the archive supports that has never been tried is predicting
**where the line goes**, rather than who wins. That target is different in a
way that matters: closing-line value is the thing itself, not a proxy for it,
and being right about direction pays without ever being right about a fight.

Hypothesis: the state of the market at time t - the dispersion across books,
the recent drift, how far individual books sit from consensus - predicts the
consensus move between t and the close.

Test: walk-forward by card on development data only. Predict the sign and
size of the consensus change from t to close. Reported as R^2 against the
realised move and as accuracy on direction, with 90% event-clustered
intervals.

Confirmed only if **both** hold:
- direction accuracy is above 55% with a 90% event-clustered lower bound
  above 52.4%, which is the break-even rate at standard -110 vig
- the R^2 on the size of the move has a lower bound above zero

Anything less is a curiosity. Beating a coin flip on direction is not enough
when the vig is 4.5%.

## What is still off limits

The holdout remains sealed and unopened. If H5 confirms on development, it
gets exactly one holdout test, and the holdout decides. If H5 fails, that is
the third strike on this dataset and I will say so rather than proposing H6.

No result here authorises a bet on its own. The stop rule from the original
pre-registration stands: betting resumes only on a confirmed hypothesis that
has survived the holdout.

---

# Addendum 2, 2026-08-13: H6, after four failures

I said I would not propose an H6. The operator overrode that twice, and the
argument they made is the reason this is worth doing rather than deference:
if the market is an aggregate of models and public money, it should inherit
whatever bias those share. Every hypothesis so far asked whether we can
out-handicap the market. This asks something different, and something that
does not require our model to be any good.

## H6: favourite-longshot bias

The most documented inefficiency in sports betting: longshots are overbet, so
they win less often than their price implies, and favourites correspondingly
more. It is a claim about market structure rather than about handicapping,
which is why it survives H1-H5 failing.

Test: bucket fights by de-vigged closing probability. Compare realised win
rate to implied probability in each bucket.

Confirmed only if **both**:
- the deviation is monotone in price - longshots underperform and favourites
  overperform, rather than one bucket doing something on its own
- a flat-stake rule betting the favoured side above a threshold returns
  positive ROI with a 90% event-clustered lower bound above zero, on n >= 200

The second is what separates a real effect from one too small to clear vig.
A measurable bias that does not pay is a finding, not an edge, and I will
report it as such.

## Prior

Stated before looking, so it can be held against me: I expect a measurable
bias and I expect it to be too small to beat the margin. This is the single
most-studied inefficiency in the field; a version large enough to bet would
have to have survived everyone else looking for it.

Holdout stays sealed regardless of outcome.

---

# Addendum 3, 2026-08-13: H7 result and H8

Running count: this is the eighth hypothesis. A 90% interval clearing zero
happens by chance roughly 57% of the time across eight independent tests, so
the bar rises with the count and the sealed holdout is the only thing that
settles anything. Recorded here so the multiplicity is visible rather than
forgotten.

## H7 result: the fundamentals family is exhausted

Walk-forward on development, n=690, entry prices:

| specification | log loss | vs market |
|---|---|---|
| line only | 0.57872 | -0.00488 |
| line + 10 fundamentals | 0.58479 | +0.00120 |
| line + all 39, boosting | 0.59071 | +0.00711 |
| line + all 39, heavy shrinkage | 0.59787 | +0.01427 |
| line + all 39 | 0.61143 | +0.02784 |

Every fundamental feature makes it worse, monotonically, across three
regularisation strengths and two model classes. This is not undertuning.
Handicapping UFC from public career statistics does not work, and no further
feature engineering on that family is worth the time.

## H8: the totals market

The first thing tested here that is not the moneyline. It matters because
the reason given for the market beating us was aggregation, and totals are
aggregated by far fewer participants. A 2024 probe returned Over/Under 1.5
from eight books including DraftKings, so historical prices exist and were
simply never requested - the puller hardcoded markets=h2h.

rounds_model was validated before any price existed: held out on 2024+ it
scored 0.67552 on distance against a division table's 0.68380, and beat the
base rate at every totals line. That ordering matters. Testing it against
prices now is checking a prediction made in advance, not fishing.

Hypothesis: rounds_model's totals probabilities beat the de-vigged totals
closing line.

Test: for every fight with an archived totals price and a known outcome,
compare the model's over/under probability at the quoted line against the
de-vigged market probability. Walk-forward, event-clustered.

Confirmed only if **both**:
- the model's log loss beats the de-vigged totals line, with a 90%
  event-clustered interval on the difference whose upper bound is below zero
- a flat-stake rule on the model's disagreements returns positive ROI at the
  real quoted prices, 90% event-clustered lower bound above zero, n >= 200

Prior, recorded before pulling: totals are thinly modelled and I think the
first criterion has a real chance. I doubt the second, because thin markets
carry wide margins and the vig hurdle on a -130/+100 pair is steeper than
the moneyline's.

---

# Addendum 4, 2026-08-13: H9-H20 as one corrected family

The operator wants at least twenty hypotheses before the moneyline is
abandoned, and is right that the totals result is weak - the book chooses
which line to hang, so fights it expects to be short get 1.5 and long get
2.5. The O/U 2.5 subset is therefore selected, and "the over hits 62% against
58% implied" is partly line assignment rather than mispricing. Combined with
the 3-round subset's interval already spanning zero, that is parked.

## Why these run together

Twelve hypotheses tested one at a time, stopping when one passes, is a
procedure that returns a false positive with near certainty. Across twenty
independent tests at 90% confidence, roughly seven will clear zero by chance.
Running them as a family with a correction is both faster and the only way a
survivor means anything.

## The framework

The market's error on a fight is `y - p_market`, using the de-vigged entry
consensus. If the market is efficient that residual is unpredictable. If any
candidate feature predicts it, that feature is an edge, and its size is
directly the probability points we would be capturing.

This subsumes the individual hypotheses. Each candidate below is a claim that
the market misprices a particular kind of fight, and each becomes a
coefficient in the same regression against the same residual:

- H9 layoff: the market misprices fighters returning from long absences
- H10 age: the market misprices old or young fighters beyond the line
- H11 stance: southpaw/orthodox mismatches
- H12 reach: extreme reach advantages
- H13 form: fighters on winning or losing runs
- H14 experience: debutants and very green fighters
- H15 division: heavyweight, women's divisions
- H16 championship rounds: five-round bouts
- H17 knockout history: fighters with recent knockout losses
- H18 book disagreement: fights where books disagree most
- H19 line movement: fights whose price moved most between entry and close
- H20 favourite size: how far the price sits from even

## The bar

Reported for every candidate, not only survivors: coefficient, 90%
event-clustered interval, and a Bonferroni-adjusted interval at family-wise
90% across the number of candidates actually tested.

A candidate is a real finding only if its **Bonferroni-adjusted** interval
excludes zero. An uncorrected interval clearing zero in a family this size is
the expected behaviour of noise, not evidence.

Any survivor then has to clear the same second gate as every hypothesis
before it: a flat-stake rule at real prices, positive ROI, 90%
event-clustered lower bound above zero, n >= 200. And then the holdout.

Prior, recorded before running: I expect zero survivors after correction. If
one survives, the most likely explanation is still that it is the tail of a
family of twelve, which is what the holdout exists to adjudicate.
