"""H22 and H23: signals priced against the hurdle you actually face.

Pre-registered in PREREGISTRATION.md, Addendum 8. The reframing behind both is
H21's control: the margin at the best available price across ~17 real books is
-0.80%, against roughly 4.5% at consensus. Twenty-one hypotheses were tested
against the wrong hurdle. A two-point signal is invisible at 4.5% and decisive
at 0.8%.

H22 retests the favourite-longshot bias at the best price rather than the
consensus. H23 uses the sharp book the way people who do this for a living use
it: not as a place to find a bargain - H21 showed Pinnacle holds the best price
on only 4.5% of fights, against ~10% by chance - but as the best available
estimate of the true probability. The bet is then placed against a *different*
book that is offering longer than Pinnacle says it should.

Two design points carry most of the weight:

- **Pinnacle is excluded from the pool it is measured against.** Same reason
  the H3 lead-lag test excluded a book from its own consensus: otherwise the
  reference predicts itself and every book looks mispriced against nothing.

- **The offered price is compared vig-inclusive.** `p_fair` is de-vigged
  because it is an estimate of truth; `p_offered` is not, because it is a cost.
  De-vigging both would compare two opinions and quietly delete the margin -
  which is the entire thing a bet has to overcome.

What this cannot see, stated where it will be read: the archive has no limits
and no take-it-or-leave-it flag. A price far from Pinnacle is often not a slow
book but a dead number - already pulled, or good for fifty dollars. Any edge
measured here is an upper bound on a real one.
"""

import numpy as np
import pandas as pd

from sharp_fade import (EXCHANGES, MIN_BETS, _payout, drop_exchanges,  # noqa: F401
                        score)

FAIR_BOOK = "pinnacle"
PRIMARY_FAVOURITE_BUCKET = 0.70
PRIMARY_EDGE_POINTS = 0.02
MIN_POOL_BOOKS = 4


def _implied(american):
    """Implied probability including vig. This is a cost, not an opinion."""
    odds = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def _devig(odds_a, odds_b):
    """Two-sided de-vig: strip the margin to get an estimate of the truth."""
    a, b = _implied(odds_a), _implied(odds_b)
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((total > 0) & np.isfinite(total), a / total, np.nan)


def fair_table(frame, fair_book=FAIR_BOOK, min_pool=MIN_POOL_BOOKS):
    """Per fight: the sharp book's fair line, and the best price elsewhere.

    `frame` is a snapshot from sharp_fade.snapshot_book, exchanges already
    removed. Fights the fair book does not quote are dropped - without a
    reference there is nothing to compare against, and substituting the
    consensus would silently turn this into a different hypothesis.
    """
    rows = []
    for fight, group in frame.groupby("api_event_id"):
        reference = group[group["book_key"] == fair_book]
        pool = group[group["book_key"] != fair_book]
        if len(reference) != 1 or len(pool) < min_pool:
            continue
        reference = reference.iloc[0]
        fair_a = float(_devig(np.array([reference["odds_a"]]),
                              np.array([reference["odds_b"]]))[0])
        if not np.isfinite(fair_a):
            continue
        payout_a = _payout(pool["odds_a"].to_numpy())
        payout_b = _payout(pool["odds_b"].to_numpy())
        best_a, best_b = int(np.argmax(payout_a)), int(np.argmax(payout_b))
        first = group.iloc[0]
        rows.append({
            "api_event_id": fight,
            "event_date": first["event_date"],
            "n_pool": len(pool),
            "a_won": first["a_won"] if "a_won" in first else None,
            "winner_name": first.get("winner_name"),
            "fighter_a": first["fighter_a"],
            "fair_a": fair_a,
            "fair_b": 1.0 - fair_a,
            "odds_a": float(pool["odds_a"].iloc[best_a]),
            "odds_b": float(pool["odds_b"].iloc[best_b]),
            "book_a": pool["book_key"].iloc[best_a],
            "book_b": pool["book_key"].iloc[best_b],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    from backtest import norm_name
    table["a_won"] = [norm_name(a) == w for a, w
                      in zip(table["fighter_a"], table["winner_name"])]
    table["offered_a"] = _implied(table["odds_a"].to_numpy())
    table["offered_b"] = _implied(table["odds_b"].to_numpy())
    table["edge_a"] = table["fair_a"] - table["offered_a"]
    table["edge_b"] = table["fair_b"] - table["offered_b"]
    return table


def value_bets(table, threshold=PRIMARY_EDGE_POINTS):
    """H23: every side offered longer than the sharp book says it should be.

    Both sides of one fight can qualify at once when the pool is wide - that is
    an arbitrage against the reference, not a double bet on the same view, and
    both are kept because excluding them would flatter the result by dropping
    the cases where books disagree most.
    """
    picks = []
    for side in ("a", "b"):
        qualifying = table[table[f"edge_{side}"] >= threshold].copy()
        if qualifying.empty:
            continue
        qualifying["side"] = side
        qualifying["bet_odds"] = qualifying[f"odds_{side}"]
        qualifying["bet_book"] = qualifying[f"book_{side}"]
        qualifying["edge"] = qualifying[f"edge_{side}"]
        qualifying["won"] = (qualifying["a_won"] if side == "a"
                             else ~qualifying["a_won"])
        picks.append(qualifying)
    if not picks:
        return pd.DataFrame(columns=["profit", "won", "event_date"])
    bets = pd.concat(picks, ignore_index=True)
    bets["profit"] = np.where(bets["won"], _payout(bets["bet_odds"]), -1.0)
    return bets


def favourite_bets(table, minimum=PRIMARY_FAVOURITE_BUCKET):
    """H22: back the favourite at the best price, above a fairness threshold.

    The favourite is identified by the sharp book's de-vigged line rather than
    by which side is cheaper in the pool, so the selection does not depend on
    the same prices being bet.
    """
    back_a = table["fair_a"] >= 0.5
    strength = np.where(back_a, table["fair_a"], table["fair_b"])
    bets = table[strength >= minimum].copy()
    if bets.empty:
        return pd.DataFrame(columns=["profit", "won", "event_date"])
    take_a = bets["fair_a"] >= 0.5
    bets["side"] = np.where(take_a, "a", "b")
    bets["bet_odds"] = np.where(take_a, bets["odds_a"], bets["odds_b"])
    bets["bet_book"] = np.where(take_a, bets["book_a"], bets["book_b"])
    bets["fair"] = np.where(take_a, bets["fair_a"], bets["fair_b"])
    bets["won"] = np.where(take_a, bets["a_won"], ~bets["a_won"])
    bets["profit"] = np.where(bets["won"], _payout(bets["bet_odds"]), -1.0)
    return bets
