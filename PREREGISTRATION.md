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

---

# Addendum 5, 2026-08-13: the H9-H20 result

Development set only, n=990 fights, 2022-06-18 to 2025-08-23. Holdout never
opened. Reproducible via `market_residual.py`; the code is under test.

## Every candidate, not only the survivor

Coefficients are probability points of market error per unit of the feature.

| candidate | n | coefficient | 90% CI | Bonferroni CI |
|---|---|---|---|---|
| H9 layoff gap | 990 | -0.00009 | [-0.00020, +0.00002] | [-0.00026, +0.00010] |
| H10 age gap | 990 | **-0.00730** | [-0.01168, -0.00276] | [-0.01422, -0.00023] |
| H11 stance edge | 990 | +0.05709 | [+0.01291, +0.10142] | [-0.01181, +0.12709] |
| H12 reach gap | 990 | +0.00132 | [-0.00505, +0.00800] | [-0.00987, +0.01268] |
| H13 recent form | 990 | -0.02647 | [-0.08692, +0.04027] | [-0.13217, +0.08293] |
| H14 experience gap | 990 | -0.00363 | [-0.00671, -0.00054] | [-0.00776, +0.00160] |
| H15 heavyweight | 990 | +0.04655 | [-0.02752, +0.11680] | [-0.05724, +0.15796] |
| H16 five-round bout | 990 | -0.07135 | [-0.18313, +0.03862] | [-0.25905, +0.09016] |
| H17 ko-loss history | 990 | -0.01115 | [-0.02580, +0.00671] | [-0.03737, +0.02011] |
| H18 book disagreement | 990 | +0.00115 | [-0.00622, +0.00799] | [-0.00976, +0.01102] |
| H19 line movement | 972 | +1.07108 | [-0.33472, +2.47791] | [-1.15592, +3.29678] |
| H20 favourite size | 990 | +0.03181 | [-0.00972, +0.07070] | [-0.02714, +0.09787] |

Three cleared zero uncorrected: H10, H11, and H14 marginally. That is exactly
what the correction exists for. At 90% across twelve tests you expect about
one by chance and the chance of at least one is 72%, so three uncorrected
passes is noise behaving normally, not a signal. Only H10 survived Bonferroni,
and only in some runs.

## H10 does not survive either

Its Bonferroni upper bound was **-0.00023**. The bound is on zero, and a
family-wise bound across twelve candidates is the 0.42nd percentile of the
bootstrap - the third or fourth order statistic even at 5,000 draws. So the
sign of that bound is partly a property of the random seed.

Re-run across eight seeds at 5,000 draws each, the coefficient is rock stable
(-0.00724 to -0.00736) and the bound is a coin flip:

| seeds where the Bonferroni interval excludes zero | 4 of 8 |
|---|---|

Re-running the whole family through `market_residual.py` at a different draw
count puts the bound at **+0.00019** and returns **zero survivors** - the table
above is the run that produced one. Both runs are the same estimate; only the
tail of the bootstrap moved. A finding that depends on the seed is not a
finding, and `seed_stability` exists so this check is never skipped again.

## And it fails the second gate regardless

Betting the younger fighter at real entry prices, event-clustered:

| age gap | n | win rate | implied (with vig) | ROI | 90% CI |
|---|---|---|---|---|---|
| any | 990 | 60.00% | 59.44% | +9.09% | [-4.21%, +28.76%] |
| 2+ years | 694 | 62.25% | 61.99% | +11.10% | [-7.24%, +41.18%] |
| 4+ years | 450 | 64.89% | 64.90% | +18.79% | [-8.11%, +64.06%] |
| 6+ years | 273 | 71.79% | 68.48% | +2.61% | [-4.27%, +9.37%] |
| 8+ years | 135 | 74.81% | 70.68% | +3.77% | [-5.80%, +13.11%] |

Every interval spans zero, so the pre-registered bar - lower bound above zero
on n >= 200 - fails at every threshold. Three things say the ROI is noise:

- **The win rate is the implied probability.** At 4+ years, 64.89% actual
  against 64.90% implied, vig included. The market has already priced age.
- **ROI is non-monotone in the signal**, peaking at +18.79% on 4+ and
  collapsing to +2.61% on 6+. A real edge strengthens as the signal does.
  This is the same shape that discredited H1's 10+ bucket.
- **The intervals are enormous**, up to +64%. A mean and a median that far
  apart means a handful of longshots landed, not that a rule paid.

The last two are the tell. A rule can only post +18.79% ROI while winning at
exactly its implied rate if the winners happened to be the long prices, which
is variance wearing an edge's clothing.

## Where this leaves the moneyline

Twenty hypotheses. H1 and H2 killed the model's own claimed edge, H3 and H4
found no lead-lag across 34 books and 1.68M observations, H5 could not predict
line movement, H6 found a real favourite-longshot bias too small and too
threshold-fragile to bet, H7 showed every fundamental feature makes the
forecast monotonically worse, H8 was confounded by the book choosing which
total to hang, and H9-H20 found nothing in the market's own residual that
survives correction.

The instruction was twenty before giving up on the moneyline. That is twenty,
and the answer is consistent at every granularity we can reach: **this market
is efficient against everything in this dataset.** The stop rule from the
original pre-registration applies - keep the pipeline as an instrument, do not
bet.

The holdout is still sealed. Nothing has earned the right to open it, and that
is the correct outcome rather than a wasted one: we now know the edge is not
there, which is worth considerably more than the ~360,000 credits it cost to
establish.

## What would change this answer

Not more features, and not more hypotheses of this shape - H7 settled the
first and this family settled the second. The two things genuinely untried are
different in kind rather than degree: prices from books that move slower than
the ones archived here, and markets with fewer participants than the moneyline
where a model has something to price that the crowd has not. Both are data
acquisition problems, not modelling ones.

---

# Addendum 6, 2026-08-13: H21, the sharp-book fade

Written before computing a single outcome. The operator proposed this one and
it is the first hypothesis in the sequence that uses **which book** a price
came from, rather than treating the market as one aggregated number. That is a
real gap in H1-H20 and worth closing.

## The claim

Books do not all mean the same thing by a price. Pinnacle and Circa take large
limits, move on sharp money, and hold thin margins; a recreational book prices
to balance action. So when the sharpest book in the world is the one offering
the *most generous* price on a side - a higher payout than all forty-odd other
books - that is not generosity. It is the sharp book saying that side is
overvalued everywhere else.

The rule the operator described, in their example: Circa has Gane at +135 and
that is the best Gane price in the world, so bet **Jones**, at the best Jones
price anywhere (DraftKings -148).

## The rule, fixed now

At one snapshot per fight:

1. Find the best (highest payout) price on each side across every book quoting
   that fight at that moment.
2. If the best price on exactly one side comes from the trigger book, bet the
   **other** side, at its best price anywhere.
3. If the trigger book holds the best price on **both** sides, no bet. That is
   a low-margin book being cheap on everything, which carries no directional
   information, and counting it would let vig masquerade as signal.
4. Flat one unit.

Primary snapshot is **t_minus_12h**, fixed in advance so the horizon cannot be
chosen after seeing which one pays. A sweep across other horizons is reported
as sensitivity, not as the test.

## What has to be true

Confirmed only if, for a trigger book:
- ROI is positive with a 90% event-clustered lower bound above zero, on
  n >= 200 bets, **and**
- it beats the two controls below by more than their intervals overlap.

## The controls, and why they are the whole test

**Circa cannot pass.** It quotes 33 development fights, against Pinnacle's
2,028. This is known before running and is not a result. Circa is reported for
completeness and is incapable of clearing n >= 200; only the Pinnacle arm is a
live test.

**Control 1, placebo triggers.** The identical rule fired on books nobody
calls sharp - DraftKings, FanDuel, BetMGM, Bovada. If the fade pays the same
when DraftKings triggers it, there is no sharp signal and something else is
doing the work.

**Control 2, best price with no trigger at all.** Bet the best price in the
world on a side, chosen without reference to any book's identity. This is the
control that matters most, and here is why: the best of forty noisy prices is
a biased estimate of the fair price. Taking the maximum across a large book
count systematically beats the consensus and can wipe out the margin on its
own. Line shopping is genuinely valuable and entirely separate from the claim
being tested. If the trigger rule does not beat this baseline, what we have
measured is the value of having forty accounts, not the value of reading
Pinnacle.

## Prior

Stated before looking. I expect Control 2 to look good on its own - possibly
close to break-even or better - because taking a maximum over many books is
a real effect and a well-known one. I expect the Pinnacle trigger to add
little or nothing beyond it, and I expect the placebo books to perform
similarly to Pinnacle. If Pinnacle separates cleanly from both controls on
n >= 200, that is the first thing in twenty-one hypotheses to earn a holdout
test.

Holdout stays sealed either way.

---

# Addendum 7, 2026-08-13: the H21 result

Development only, 2,008 fights with at least five books quoting at the primary
snapshot. Reproducible via `sharp_fade.py`; the rule is under test, including
the operator's worked Jones/Gane example.

## A confound found before scoring, and worth stating on its own

Betfair and Betfair EX EU held **2,377 of roughly 4,000 best-price slots**.
They are exchanges: the price on screen is gross of 2-5% commission on
winnings, so it is not a price anyone can take. Any "best price in the world"
that includes them is overstated, and they win the slot *because* they quote a
number nobody pays.

This was not pre-registered - it was found by looking at which books held best
prices, before any outcome was computed. Results below are reported with
exchanges removed, and the pre-registered version including them is in the
commit history. Removing them changes the sample, not the verdict.

## Result: the pre-registered primary fails

At t_minus_12h, real books only:

| rule | n | win rate | ROI | 90% CI |
|---|---|---|---|---|
| **fade pinnacle** | 590 | 54.58% | **+3.63%** | [-4.35%, +11.74%] |
| fade circasports | 12 | 25.00% | -65.33% | [-100%, -44.52%] |
| placebo: fade draftkings | 185 | 42.70% | -22.14% | [-32.73%, -11.47%] |
| placebo: fade fanduel | 427 | 51.29% | -4.82% | [-13.51%, +4.51%] |
| placebo: fade betmgm | 137 | 54.01% | +2.21% | [-13.32%, +18.25%] |
| placebo: fade bovada | 89 | 53.93% | -4.55% | [-23.50%, +13.80%] |
| control: best price, favourite | 2008 | 66.38% | -0.80% | [-3.39%, +1.84%] |
| control: best price, underdog | 2008 | 33.47% | -6.46% | [-11.58%, -1.04%] |

Pinnacle is the best of the trigger arms and clears n >= 200, but its interval
spans zero, and it does not separate from the controls - the pre-registration
required both. The favourite/underdog mix does not explain the spread between
arms: Pinnacle bets favourites 52% of the time, FanDuel 51%.

## The horizon sweep settles it

The primary was fixed at 12h precisely so this could not be chosen afterwards.
Pinnacle across horizons:

| horizon | n | ROI | 90% CI |
|---|---|---|---|
| 1h | 673 | +5.89% | [-1.22%, +13.24%] |
| 3h | 625 | **-2.28%** | [-9.16%, +5.23%] |
| 6h | 617 | **-0.18%** | [-7.91%, +8.01%] |
| **12h (primary)** | 590 | +3.63% | [-4.35%, +11.74%] |
| 24h | 551 | **+11.94%** | [+3.80%, +20.14%] |
| 48h | 496 | **-0.42%** | [-8.71%, +7.52%] |

It changes sign four times and spikes at exactly one horizon. That is the H6
shape again: an effect that exists only at one setting and collapses either
side of it is a coincidence with a good haircut.

The 24h cell is the one that would tempt us, so it was tested rather than
admired. Corrected across the 18 tests actually swept (6 horizons x 3 books),
at 4,000 draws per seed, it excludes zero in **0 of 8 seeds**. Three of those
18 cells cleared zero uncorrected - one positive, two negative - against 1.8
expected by chance. Noise behaving exactly as advertised.

## What is actually worth keeping

The control is the durable finding, and it is not an edge. Taking the best
price across ~17 real books returns **-0.80%** on the favourite, from a
starting margin of roughly 4.5%. Line shopping recovers almost all of the vig
and stops just short of clearing it. That is worth knowing and worth doing, and
it is not a reason to bet: the interval [-3.39%, +1.84%] contains zero, and it
excludes the exchange prices that would otherwise flatter it.

Backing underdogs at the best price is clearly negative (-6.46%, upper bound
below zero), which is the favourite-longshot bias from H6 showing up again and
still not payable.

That is twenty-one. Holdout still sealed.

---

# Addendum 8, 2026-08-13: H22 and H23, priced against the right hurdle

Written before running either. H21's control changed what question is worth
asking, and it is worth being explicit about why.

## The reframing

Every hypothesis from H1 to H20 tested a signal against the **consensus**
price, which carries a margin near 4.5%. H21's control measured the margin at
the **best available** price across ~17 real books: **-0.80%**. That is a
different hurdle, and it is nearly four points lower.

A signal worth two probability points is invisible against consensus and
decisive against best price. So the honest reading of twenty-one failures is
narrower than "the market is efficient": it is that no signal we found beats a
4.5% margin. Two of those signals were real and merely too small - the
favourite-longshot bias in H6, and the favourite side of H21's own control.
Neither has ever been tested at the price you would actually take.

That is not a new hypothesis dressed up. It is the same signals against a
hurdle that is now measured rather than assumed.

## H22: favourite-longshot bias, at the best price

H6 found favourites beat their price and could not clear the margin. H21's
control found favourites at best price return -0.80% pooled. The untested
claim is that the bias is stronger in heavier favourites, which pooling hides.

Test: bet the favourite at the best price in the world, bucketed by de-vigged
consensus favourite probability. Primary bucket fixed in advance at
**p >= 0.70**, because that is where H6 measured the bias largest and because
picking the bucket afterwards is the failure mode this file exists to prevent.

## H23: Pinnacle as the fair line, not as a bargain

H21 found Pinnacle holds the sole best price on 4.5% of fights against ~10%
by chance. It is systematically *not* the generous book. That is the wrong way
to use it, and it is not how anyone who does this for a living uses it.

The standard method is the opposite: treat the sharp book's de-vigged price as
the best available estimate of the true probability, then bet anywhere that
offers a price implying a probability meaningfully below it. The bet is not a
disagreement with Pinnacle. It is agreement with Pinnacle against a slower
book.

Test: at each snapshot, de-vig Pinnacle's two-sided price to get `p_fair` for
each side. Across all other real books, take the best price on each side and
compute its implied probability `p_offered` (single-sided, vig included, since
that is the actual cost). Edge is `p_fair - p_offered`. Bet where edge clears
a threshold, fixed in advance at **2 probability points**.

Pinnacle is excluded from the pool it is being compared against, for the same
reason the H3 lead-lag test excluded the book from its own consensus:
otherwise it predicts itself. Exchanges are excluded throughout.

## The bar, and the correction

Primary snapshot t_minus_12h for both, matching H21 so the horizon is not a
free parameter. Each needs positive ROI, 90% event-clustered lower bound above
zero, n >= 200.

Sweeps over buckets, thresholds and horizons are reported as sensitivity, with
a Bonferroni-adjusted interval across every cell actually swept, and a
seed-stability check on anything that survives. H21's 24h spike cleared zero
on its own and died at 0 of 8 seeds once corrected; that is the standard.

## Priors

**H22:** I expect it to fail, and to fail closer than anything so far. Heavy
favourites at the best price should land within a point or two of break-even.
The reason I doubt it clears: -110 style pricing on heavy favourites is where
books hold their firmest margin, and the best price on a -400 shot varies less
across books than the best price on a +300 dog, so shopping helps least
exactly where the bias is largest.

**H23:** the most likely of anything in twenty-three to work, and still
probably a mirage. Two specific ways it fools you, both of which the test has
to survive rather than assume away. First, a book quoting a price far from
Pinnacle is often not stale but **gone** - the number is one nobody could
take, already pulled, or attached to a limit of fifty dollars. The archive
cannot see limits, so a measured edge here is an upper bound on a real one.
Second, at a fixed hourly snapshot, some of that gap is books updating on
different clocks rather than disagreeing. If H23 confirms, the honest next
step is a forward test at live prices, not a bet.

Holdout stays sealed for both.

---

# Addendum 9, 2026-08-13: H22 and H23 results

Development only. Reproducible via `fair_line.py`, under test.

## H23 first: underpowered, not dead

| horizon | n | win rate | ROI | 90% CI |
|---|---|---|---|---|
| 1h | 146 | 41.78% | +23.91% | [-1.98%, +48.80%] |
| 3h | 115 | 41.74% | +26.32% | [-6.58%, +56.32%] |
| 6h | 101 | 41.58% | +28.02% | [-5.43%, +59.88%] |
| 12h (primary) | 70 | 37.14% | +9.95% | [-24.50%, +47.81%] |
| 24h | 53 | 35.85% | -3.08% | [-34.99%, +35.38%] |

Fails the pre-registered n >= 200 at every horizon, so it does not confirm and
nothing here authorises anything. It is the only hypothesis in twenty-three
that failed on **sample size rather than on effect**, and the effect is large
and consistent at the three short horizons where books disagree most. That is
a reason to collect more, not a reason to bet.

## H22: the pre-registered primary passes

At fair >= 0.70, t_minus_12h, best price across ~16 real books:

**n=574, win rate 81.88%, ROI +3.59%, 90% CI [+0.26%, +6.75%].**

First pre-registered primary to pass in twenty-two hypotheses. Every stability
check that killed the previous candidates, it survives:

| check | H10 | H21 24h spike | H22 |
|---|---|---|---|
| seeds clearing zero | 4 of 8 | 0 of 8 | **8 of 8** |
| behaviour across the sweep | n/a | one spike | positive at all 6 horizons |
| monotone in the signal | no | no | yes, 0.55 to 0.75 |

Buckets rise smoothly: +1.01%, +2.01%, +2.34%, **+3.59%**, +4.61%, then fall to
+1.38% and -0.91% where n collapses to 176 and 57. Horizons run +5.36%, +4.31%,
+4.08%, +3.59%, +3.72%, +3.26%, rising as the line sharpens toward the fight,
which is the direction a real selection effect should move.

## What it is not

**It fails Bonferroni across the 19 cells swept, 0 of 8 seeds.** The primary
was pre-specified and a sensitivity analysis run afterwards does not retro-
actively correct it - but this is the twenty-second hypothesis on this data,
and family-wise across twenty-two at 90% you expect roughly two false
positives. Both readings are defensible and the second is why the holdout
exists.

**No individual year clears zero.** 2020 +0.96%, 2021 **-3.87%**, 2022 +6.08%,
2023 +6.43%, 2024 +5.25%, 2025 +1.76%. Positive in five of six, both halves
positive, none individually significant.

**Most of it is shopping, not the bias.** Same 574 fights, same selection,
priced three ways:

| price taken | ROI | 90% CI |
|---|---|---|
| best of ~16 books | +3.59% | [+0.26%, +6.75%] |
| median book | +1.41% | [-1.74%, +4.30%] |
| worst book | -0.65% | [-3.73%, +2.19%] |

The favourite-longshot bias contributes about **+1.4 points and does not clear
zero on its own**. Line shopping contributes the other **+2.2**. So this is not
a handicapping edge that H1-H20 missed; it is H6's bias, still too small to bet
by itself, sitting on top of a much lower hurdle. Anyone without accounts at a
dozen-plus books gets the median column, which is not a bet.

## The practical objection, which is not small

The rule stakes heavy favourites - mean payout 0.27, so about -370 - to earn
3.6%. That is roughly fourteen units at risk per unit of expected profit, and
the drawdowns are correspondingly ugly. It also requires taking the best price
in the world repeatedly on short favourites, which is the single fastest way
to get an account limited. A measured 3.6% that survives neither limiting nor a
bad month is not income.

## Status

H22 has earned the one thing nothing else has: a holdout test. Per the
original pre-registration the holdout is opened **once**, and `sealed.py`
enforces it - `sealed_access.log` does not exist, so it has never been opened.
Spending it is irreversible, so it is not being spent unilaterally.

Recorded before opening: I expect the holdout to come back positive but with
an interval spanning zero, because a +1.4 point bias plus shopping is real and
roughly this size, and one year of cards is too few fights to resolve it. If it
returns clearly negative, H22 is dead and the answer to the whole programme is
the one from Addendum 5.

---

# Addendum 10, 2026-08-13: the holdout, opened

Opened once, on operator authorisation, against the specification fixed in
Addendum 8 and never altered: favourite at the best price across real books,
selected by Pinnacle's de-vigged line at 0.70, snapshot t_minus_12h.
`sealed_access.log` is now tracked in git rather than ignored, so the record
is checkable rather than merely asserted.

**397 fights, 2025-09-06 to 2026-08-08. 166 qualifying bets.**

| | development | holdout |
|---|---|---|
| n | 574 | 166 |
| win rate | 81.88% | **82.53%** |
| implied by price paid | 78.7% | **78.58%** |
| ROI | +3.59% | **+4.26%** |
| 90% CI | [+0.26%, +6.75%] | **[-1.39%, +9.46%]** |

## What replicated, and what did not

The effect replicated in direction and size. Win rate came back within
0.7 points, the price paid within 0.1 points, and ROI slightly higher than
development rather than lower - which is the opposite of what a fitted result
does out of sample, where the usual signature is a large positive collapsing
toward zero.

The interval spans zero. n=166 against a pre-registered floor of 200, because
406 fights is all the archive holds past the seal and 29% of them qualify.
That is a limit of how much data exists, not a choice.

**The prior recorded before opening was: "positive but with an interval
spanning zero." That is exactly what came back.** Being right about that is
worth less than it sounds - it was a prediction that the test would be
inconclusive, and an inconclusive test is the easiest thing to predict - but it
does mean the result is not a surprise being rationalised after the fact.

## The decomposition holds too

The same 166 bets, repriced. This required a second read of already-opened
data and is logged as such; it introduces no new selection, specification or
decision rule, and it is descriptive rather than a second test.

| price taken | development | holdout |
|---|---|---|
| best of ~16 books | +3.59% | +4.26% |
| median book | +1.41% | +2.05% |
| worst book | -0.65% | -0.59% |

The shape is identical: roughly half the return is line shopping, the rest is
the favourite-longshot bias, and at a single average book the whole thing
disappears into the margin. Nobody without a dozen-plus accounts has this.

## The verdict

H22 **does not confirm**. The pre-registered bar was a 90% lower bound above
zero on n >= 200, and the holdout delivers neither. Under the stop rule from
the original pre-registration, that means no bet.

What it also is not, is dead. Twenty-one hypotheses died because the effect
vanished. This one produced the same effect, the same size, in data it had
never seen, and failed only because a year of cards is not enough fights to
resolve a four-point edge on 166 bets. Those are different failures and
conflating them would be the least honest thing in this document.

The distinction has a consequence: H22 is the only candidate for which
**collecting more data would change the answer**, and that is now a measurable
question rather than a hope. At the observed effect size, resolving it at 90%
needs roughly 500-700 bets, or about three more years of cards at the current
qualifying rate. The archive cannot produce those retroactively - the seal is
spent and the fights have not happened yet - so the only honest route is
forward: log the bets it would make, at live prices, and settle them as the
cards land.

The holdout is now spent. Nothing further from it is out-of-sample, and no
future result from this archive can be treated as a clean check. That was the
price of asking the question, and asking it was right.

---

# Addendum 11, 2026-08-13: the forward test, and what it will take

The archive is finished as evidence. `forward_test.py` logs what H22 would bet
on cards that have not happened, at prices captured before each fight, and
settles them as results land. It runs on the existing six-hourly snapshot job,
which already captures per-book quotes from 22 books including Pinnacle.

## How many bets

Turning "more data" into a number, from the holdout's own figures - ROI +4.26%,
standard error 3.30% at n=166, error falling as 1/sqrt(n):

**About 270 bets** to put a 90% lower bound above zero, *if the effect is
really 4.26%*. At the development estimate of 3.59% it is roughly 390. At half
the observed effect it is over a thousand, because the requirement grows with
the square of how wrong the estimate is.

At the qualifying rate seen in the archive - 166 bets from 397 fights, so about
29% of cards with a Pinnacle line and a real pool - that is **two to three
years** of UFC cards. This is a slow answer and there is no faster honest one.

## What the log refuses to do

Three guards, each pinned by tests, because each failure would look like
progress:

- **Fights on or before 2026-08-08 are refused.** That is the holdout's last
  card. Re-logging them would present evidence already counted as if it were
  new.
- **A bet written after its fight is marked backfill and never pooled with the
  headline.** The price is honest - the snapshot predates the bell - but the
  decision was not made blind, and only blind decisions are evidence. The two
  bets seeding the log are both backfill and are excluded from every number.
- **A fight is logged once, ever, and never rewritten.** Re-logging as the line
  moved would double-count one opinion; rewriting the price would replace what
  was knowable then with what looks right now.

`RULE_VERSION` is recorded on every row. If the rule is edited mid-flight the
log stops being a forward test of the thing that was tested, and the version
makes that visible instead of silent.

## The pooled estimate, and why it is not a result

Development and holdout combined give roughly **+3.7% on 740 bets**, and that
interval does clear zero. It is the best point estimate we have of the effect
size and it is the right number to plan around.

It is **not** a confirmation, and it must never be reported as one. The
holdout's whole value was independence; once pooled, that is gone and the
combined interval is an in-sample interval on data the specification was chosen
against. Anyone quoting +3.7% as a passed test has recreated the exact error
this document was written to prevent.

## The standing recommendation, unchanged

No bet. The stop rule from the original pre-registration holds until the
forward log clears the pre-registered bar on bets written before their fights.
If it does, the remaining objections are practical rather than statistical -
about fourteen units at risk per unit of profit, and heavy favourites at top
price being the fastest route to a limited account - and those are the
operator's call, not this document's.
