"""H21: fade the sharp book when it holds the best price in the world.

Pre-registered in PREREGISTRATION.md, Addendum 6, before any outcome was
computed. This is the first hypothesis in the sequence that cares *which* book
a price came from. H1-H20 all collapsed the market into one aggregated number,
which throws away the thing a bettor actually looks at: Pinnacle and Circa take
large limits and hold thin margins, and a recreational book prices to balance
its action. Those are not the same statement about a fight.

The claim is that when the sharpest book in the world offers the most generous
price on a side - a higher payout than all forty-odd others - that is not
generosity but a verdict: the side is overvalued everywhere else. So bet the
other side, at the best price anywhere.

Two things this module is careful about, because they are what separates the
claim from its cheap imitations:

- **Both-sides triggers are excluded.** A low-margin book is cheap on
  everything. If Pinnacle holds the best price on Gane *and* Jones, that says
  Pinnacle has thin vig, not that either fighter is mispriced. Counting those
  would let margin masquerade as signal.

- **Taking the best price is itself worth money, and that is not this
  hypothesis.** The maximum of forty noisy prices is a biased estimate of the
  fair price; shopping forty books beats the consensus on its own and can eat
  the whole margin. `baseline_best_price` measures that separately, and the
  trigger rule has to beat it. Otherwise what has been measured is the value of
  holding forty accounts, not the value of reading Pinnacle.
"""

import numpy as np
import pandas as pd

import historical_odds
import sealed
from backtest import norm_name

SHARP_BOOKS = ("pinnacle", "circasports")
PLACEBO_BOOKS = ("draftkings", "fanduel", "betmgm", "bovada")
PRIMARY_SNAPSHOT = "t_minus_012.00h"
MIN_BOOKS = 5
MIN_BETS = 200

# Exchanges, not books. They quote gross of commission - typically 2% to 5% on
# net winnings - so a price shown here is not a price you can actually get. It
# matters more than it sounds: exchanges hold roughly 60% of all best-price
# slots in this archive, precisely because they are quoting a number nobody
# pays. Any "best price in the world" that includes them is overstated.
EXCHANGES = ("betfair", "betfair_ex_eu", "matchbook", "betfair_ex_uk",
             "betfair_ex_au", "smarkets")
DEFAULT_COMMISSION = 0.05


def _payout(american):
    """Profit per unit staked if the bet wins.

    This is the ordering key for "best price", and it has to be the payout
    rather than the raw American number: -110 and +110 do not compare as
    integers, and sorting by the printed price would rank every underdog above
    every favourite.
    """
    odds = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, 100.0 / -odds, odds / 100.0)


def outcomes(fights_path="fights_v2.csv"):
    """Winner per (date, normalised name pair), for settling a bet.

    Draws are dropped rather than settled. A push is the right treatment in a
    ledger, but here every bet is on a side chosen by a rule, and including
    zero-P&L rows would dilute an ROI toward zero rather than measure it.
    Their count is reported by `snapshot_book` so the exclusion is visible.
    """
    fights = pd.read_csv(fights_path, parse_dates=["date"])
    fights = fights[fights["winner"].isin(["A", "B"])]
    return {(str(pd.Timestamp(row.date).date()),
             frozenset((norm_name(row.fighter_a), norm_name(row.fighter_b)))):
            (norm_name(row.fighter_a) if row.winner == "A"
             else norm_name(row.fighter_b))
            for row in fights.itertuples()}


def snapshot_book(root="raw/odds_api_historical", snapshot=PRIMARY_SNAPSHOT,
                  development_only=True, fights_path="fights_v2.csv"):
    """Every book's price on both sides of every fight, at one moment.

    One snapshot, fixed in advance. Pooling several would quietly multiply the
    sample by re-betting the same fight at correlated prices, and picking the
    horizon after seeing the results is the exact move this whole file exists
    to avoid.
    """
    quotes = historical_odds.load_quotes(root)
    if development_only:
        quotes = sealed.development(quotes, column="event_date")
    frame = quotes[quotes["snapshot_kind"] == snapshot].copy()
    for column in ("odds_a", "odds_b"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["odds_a", "odds_b", "api_event_id"])
    frame = frame.drop_duplicates(subset=["api_event_id", "book_key"], keep="last")

    results = outcomes(fights_path)
    frame["date_key"] = frame["event_date"].astype(str).str[:10]
    frame["pair"] = [frozenset((norm_name(a), norm_name(b)))
                     for a, b in zip(frame["fighter_a"], frame["fighter_b"])]
    frame["winner_name"] = [results.get((d, p))
                            for d, p in zip(frame["date_key"], frame["pair"])]
    return frame.dropna(subset=["winner_name"])


def _best_sides(group):
    """Which book holds the best price on each side, and what that price is."""
    payout_a = _payout(group["odds_a"].to_numpy())
    payout_b = _payout(group["odds_b"].to_numpy())
    best_a, best_b = int(np.argmax(payout_a)), int(np.argmax(payout_b))
    return {
        "book_a": group["book_key"].iloc[best_a],
        "book_b": group["book_key"].iloc[best_b],
        "odds_a": float(group["odds_a"].iloc[best_a]),
        "odds_b": float(group["odds_b"].iloc[best_b]),
        "payout_a": float(payout_a[best_a]),
        "payout_b": float(payout_b[best_b]),
    }


def drop_exchanges(frame, exchanges=EXCHANGES):
    """Remove exchange quotes, which are not prices anyone can actually take."""
    return frame[~frame["book_key"].isin(exchanges)].copy()


def apply_commission(bets, rate=DEFAULT_COMMISSION, exchanges=EXCHANGES):
    """Charge exchange commission on winning bets, where one was taken.

    Commission falls on net winnings, so a loser is unaffected and a winner
    keeps (1 - rate) of its profit. Leaving this out is what makes an
    exchange-inclusive backtest look free.
    """
    out = bets.copy()
    if not len(out) or "bet_book" not in out.columns:
        return out
    charged = out["bet_book"].isin(exchanges) & out["won"]
    out.loc[charged, "profit"] = out.loc[charged, "profit"] * (1.0 - rate)
    return out


def best_price_table(frame, min_books=MIN_BOOKS):
    """One row per fight: the best price in the world on each side, and whose.

    `min_books` matters. With three books quoting, "best price in the world" is
    the best of three, and which book holds it is close to a coin flip. The
    hypothesis is about an outlier among many, so fights with a thin market are
    not evidence either way.
    """
    rows = []
    for fight, group in frame.groupby("api_event_id"):
        if len(group) < min_books:
            continue
        first = group.iloc[0]
        best = _best_sides(group)
        winner = first["winner_name"]
        rows.append({
            "api_event_id": fight,
            "event_date": first["event_date"],
            "fighter_a": first["fighter_a"],
            "fighter_b": first["fighter_b"],
            "n_books": len(group),
            "a_won": norm_name(first["fighter_a"]) == winner,
            **best,
        })
    return pd.DataFrame(rows)


def fade(table, book):
    """Bets produced by fading `book` where it holds the sole best price.

    Returns one row per bet, with the side taken and the profit on one unit.
    """
    holds_a = table["book_a"] == book
    holds_b = table["book_b"] == book
    # Sole best price only. Holding both is a thin margin, not an opinion.
    trigger = holds_a ^ holds_b
    bets = table[trigger].copy()
    if bets.empty:
        return bets.assign(side=[], profit=[])
    # It holds the best price on A, so the bet is B, and the other way round.
    back_a = bets["book_b"] == book
    bets["side"] = np.where(back_a, "a", "b")
    bets["bet_odds"] = np.where(back_a, bets["odds_a"], bets["odds_b"])
    bets["bet_book"] = np.where(back_a, bets["book_a"], bets["book_b"])
    bets["won"] = np.where(back_a, bets["a_won"], ~bets["a_won"])
    bets["profit"] = np.where(bets["won"],
                              _payout(bets["bet_odds"]), -1.0)
    return bets


def baseline_best_price(table, side="favourite"):
    """Control 2: take the best price in the world, ignoring who is offering it.

    This is the number the hypothesis has to beat. Shopping forty books is a
    real edge that has nothing to do with sharp-book inference, and a rule that
    merely rediscovers it has discovered nothing new.
    """
    bets = table.copy()
    if side == "favourite":
        back_a = bets["payout_a"] < bets["payout_b"]
    elif side == "underdog":
        back_a = bets["payout_a"] > bets["payout_b"]
    elif side == "a":
        back_a = pd.Series(True, index=bets.index)
    else:
        raise ValueError(f"unknown side {side!r}")
    bets["bet_odds"] = np.where(back_a, bets["odds_a"], bets["odds_b"])
    bets["bet_book"] = np.where(back_a, bets["book_a"], bets["book_b"])
    bets["won"] = np.where(back_a, bets["a_won"], ~bets["a_won"])
    bets["profit"] = np.where(bets["won"], _payout(bets["bet_odds"]), -1.0)
    return bets


def score(bets, label, draws=2000, seed=3):
    """ROI with a 90% interval bootstrapped over cards, not bets."""
    if not len(bets):
        return {"rule": label, "n": 0, "win_rate": np.nan, "roi": np.nan,
                "ci90_low": np.nan, "ci90_high": np.nan, "passes": False}
    cards = [group for _, group in bets.groupby("event_date")]
    rng = np.random.default_rng(seed)
    boot = [pd.concat([cards[i] for i in rng.integers(0, len(cards), len(cards))])
            ["profit"].mean() for _ in range(draws)]
    low, high = np.percentile(boot, [5, 95])
    return {
        "rule": label,
        "n": len(bets),
        "win_rate": float(bets["won"].mean()),
        "roi": float(bets["profit"].mean()),
        "ci90_low": float(low),
        "ci90_high": float(high),
        "passes": bool(low > 0 and len(bets) >= MIN_BETS),
    }


def report(table, books=SHARP_BOOKS, placebos=PLACEBO_BOOKS, draws=2000, seed=3):
    """The trigger arms and both controls, in one table, all of them reported."""
    rows = [score(fade(table, book), f"fade {book}", draws, seed)
            for book in books]
    rows += [score(fade(table, book), f"placebo: fade {book}", draws, seed)
             for book in placebos]
    rows += [score(baseline_best_price(table, side),
                   f"control: best price, {side}", draws, seed)
             for side in ("favourite", "underdog")]
    return pd.DataFrame(rows)
