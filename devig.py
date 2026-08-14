"""Four ways to strip a bookmaker's margin, and why the choice decides H22.

A two-sided price implies probabilities that sum to more than one. The excess
is the margin, and removing it is not a single well-defined operation - it is
an assumption about *where the book put the margin*. Different assumptions give
materially different fair probabilities for the same quote, and they disagree
most exactly where H22 lives: on heavy favourites.

    -370 / +290  ->  proportional 0.7543   additive 0.7654
                     Shin         0.7654   power    0.7715

Nearly two probability points of spread on the same quote, all of it against
the hypothesis. H22's entire measured bias was about four.

**Shin and additive coincide exactly here, and that is not a coincidence.**
For a two-outcome market they are the same estimator: solving Shin's insider
fraction z for this quote gives z=0.04436, and substituting it reproduces the
additive answer to ten decimal places. Shin only becomes a distinct method with
three or more outcomes. So the family below is three estimators wearing four
names, and `power` - not Shin - is the strictest bar available on a two-way
price. A test that reported "survives under four methods" would be overstating
its own evidence by one.

**Proportional is the one that flatters the hypothesis.** It scales both sides
down by the overround, which removes margin in proportion to each side's
probability. The empirical finding it ignores is that books load more of their
margin onto the longshot, so proportional hands the favourite a fair
probability that is too low - and "the favourite wins more often than implied"
then follows from the arithmetic without the market being wrong at all.

Every number in H22 used proportional. So the mechanism has to survive the
other three, or it was never there.

- **additive** removes the margin equally in probability points from both
  sides. Simple, and the crudest correction in the right direction.
- **power** finds the exponent k with `a**k + b**k = 1`. Multiplicative in
  log-odds rather than in probability.
- **Shin** models the margin as protection against insiders and solves for the
  implied insider fraction z. It is the standard in the literature, and on a
  two-way price it lands exactly on the additive answer - kept because it is
  solved independently, so agreement is a check on both rather than a
  restatement.
"""

import numpy as np

METHODS = ("proportional", "additive", "power", "shin")


def implied(american):
    """Implied probability from American odds, margin included."""
    odds = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def _proportional(a, b):
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, a / total, np.nan)


def _additive(a, b):
    """Split the overround equally in probability points.

    Clipped because a large margin on a lopsided quote can push the longshot
    below zero, which is not a probability.
    """
    excess = (a + b) - 1.0
    return np.clip(a - excess / 2.0, 1e-6, 1 - 1e-6)


def _power(a, b, iterations=60):
    """Find k with a**k + b**k = 1, by bisection on k.

    k > 1 for a book with margin. Bisection rather than Newton because the
    bracket is guaranteed and 60 halvings is exact to far more digits than a
    price carries.
    """
    a = np.clip(np.asarray(a, dtype=float), 1e-9, 1 - 1e-9)
    b = np.clip(np.asarray(b, dtype=float), 1e-9, 1 - 1e-9)
    low = np.ones_like(a)
    high = np.full_like(a, 8.0)
    for _ in range(iterations):
        mid = (low + high) / 2.0
        over = (a ** mid + b ** mid) > 1.0
        low = np.where(over, mid, low)
        high = np.where(over, high, mid)
    k = (low + high) / 2.0
    return a ** k


def _shin(a, b, iterations=80):
    """Shin's method: solve for the insider fraction z, then invert.

    With z the implied proportion of informed money, the fair probability of an
    outcome quoted at q out of a book summing to s is

        p = [sqrt(z**2 + 4(1-z) q**2 / s) - z] / (2(1-z))

    z is found by bisection on the condition that the two fair probabilities
    sum to one. z=0 recovers proportional, which is the useful sanity check:
    Shin is proportional plus an estimate of how much the margin is defensive.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    total = a + b

    def fair(q, z):
        with np.errstate(divide="ignore", invalid="ignore"):
            inner = z ** 2 + 4.0 * (1.0 - z) * (q ** 2) / total
            return (np.sqrt(np.maximum(inner, 0.0)) - z) / (2.0 * (1.0 - z))

    low = np.zeros_like(a)
    high = np.full_like(a, 0.5)
    for _ in range(iterations):
        mid = (low + high) / 2.0
        summed = fair(a, mid) + fair(b, mid)
        # The sum falls as z rises; push z up while the pair still sums high.
        high = np.where(summed > 1.0, high, mid)
        low = np.where(summed > 1.0, mid, low)
    z = (low + high) / 2.0
    out = fair(a, z)
    return np.where(np.isfinite(out) & (total > 0), out, np.nan)


def devig(odds_a, odds_b, method="proportional"):
    """Fair probability of side A under the named method."""
    a, b = implied(odds_a), implied(odds_b)
    if method == "proportional":
        return _proportional(a, b)
    if method == "additive":
        return _additive(a, b)
    if method == "power":
        return _power(a, b)
    if method == "shin":
        return _shin(a, b)
    raise ValueError(f"unknown de-vig method {method!r}; expected one of {METHODS}")
