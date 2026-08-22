"""What line shopping is worth on a given fight, in probability points.

This is the only effect in twenty-four hypotheses that survived contact with
the data, and until now it was invisible on the card. The page showed which
book held the best price for the side the model happened to prefer, and never
said how much that price was worth against the consensus, or what was
available on the other fighter at all.

Measured across the archive, the margin you actually pay depends almost
entirely on how many books you can reach:

    1 book   4.77%      4 books  2.92%      10 books  2.03%
    2 books  3.77%      6 books  2.48%      18 books  1.54%

The first four books capture 57% of everything available. That is a larger,
more certain effect than anything the model itself produces, and it applies to
every wager rather than to the handful that clear an edge rule.

So the numbers here are deliberately not about the model's opinion. They are
about the price: what the best book pays, how many points that saves against
the consensus, and what the two-sided margin looks like before and after
shopping. A fight where shopping is worth nothing says so.
"""

import numpy as np

from devig import implied

# Below this the "saving" is rounding noise on a price quoted in whole cents,
# not something worth sending anyone to another sportsbook for.
MIN_MEANINGFUL_POINTS = 0.05


def _price(value):
    """A usable American price, or None. Blank strings and NaN are not prices."""
    if value is None:
        return None
    text = str(value).strip().replace("+", "")
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number != 0 else None


def gain_points(consensus, best):
    """Probability points saved by taking `best` instead of `consensus`.

    Positive means the better price costs less implied probability, which is
    the thing being bought. Returns None when either price is unusable, and
    can legitimately return a small negative if the consensus is an average
    that happens to sit above the best single quote - reported rather than
    clamped, because silently flooring it at zero would hide a bad feed.
    """
    consensus, best = _price(consensus), _price(best)
    if consensus is None or best is None:
        return None
    return float((implied([consensus])[0] - implied([best])[0]) * 100.0)


def margin_points(price_a, price_b):
    """Two-sided overround in points: what the pair of prices costs."""
    a, b = _price(price_a), _price(price_b)
    if a is None or b is None:
        return None
    return float((implied([a])[0] + implied([b])[0] - 1.0) * 100.0)


def side(name, consensus, best, book):
    """One fighter's best available price and what it saves."""
    best_price = _price(best)
    consensus_price = _price(consensus)
    # With one book quoting, "best" and "consensus" are the same number and
    # there is nothing to shop; say so rather than showing a 0.0 saving that
    # reads like a measurement.
    gain = gain_points(consensus, best)
    return {
        "name": name,
        "price": f"{int(best_price):+d}" if best_price is not None else None,
        "consensus": (f"{int(consensus_price):+d}"
                      if consensus_price is not None else None),
        "book": str(book).strip() or None if book is not None else None,
        "gain_pts": round(gain, 2) if gain is not None else None,
        "worth_shopping": bool(gain is not None and gain >= MIN_MEANINGFUL_POINTS),
    }


def fight(pick_name, opp_name, pick_consensus, opp_consensus,
          pick_best, pick_book, opp_best, opp_book, books=None):
    """The full shopping picture for one fight, ready to render.

    Both fighters, not just the one the model prefers. The model is not
    betting; the person reading the page might be, and they may back either
    side. Withholding the other fighter's best price would make the only
    validated effect in the repository useful in one direction out of two.
    """
    pick = side(pick_name, pick_consensus, pick_best, pick_book)
    opp = side(opp_name, opp_consensus, opp_best, opp_book)
    gains = [s["gain_pts"] for s in (pick, opp) if s["gain_pts"] is not None]
    consensus_margin = margin_points(pick_consensus, opp_consensus)
    best_margin = margin_points(pick_best, opp_best)
    return {
        "pick": pick,
        "opp": opp,
        "best_gain_pts": round(max(gains), 2) if gains else None,
        "consensus_margin_pts": (round(consensus_margin, 2)
                                 if consensus_margin is not None else None),
        "best_margin_pts": (round(best_margin, 2)
                            if best_margin is not None else None),
        "books": int(books) if books not in (None, "") and np.isfinite(
            float(books)) else None,
        # A two-sided margin at or below zero means the best prices on both
        # fighters cross: backing both wins whatever happens. Roughly 9% of
        # fights in the archive did this across 18 books, and the prices were
        # concurrent - books updated within ~5 minutes of one another. It is
        # flagged, not recommended: limits are small and it closes accounts.
        "crossed": bool(best_margin is not None and best_margin <= 0.0),
        "any_value": bool(pick["worth_shopping"] or opp["worth_shopping"]),
    }
