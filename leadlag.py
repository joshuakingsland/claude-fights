"""H3: does one book's move predict where the rest of the market goes?

Pre-registered in PREREGISTRATION.md. This is the only live thread left after
H1 failed, and it survives that failure because it does not depend on the
model being any good. The claim is not "we know something the market doesn't".
It is "one book gets there first, and a slower book is briefly stale".

Method, per candidate book B and horizon h:

    consensus excluding B, change over the next h hours
        regressed on
    B's change over the previous interval

The exclusion is the whole ballgame. A book is part of the consensus, so
regressing the consensus on a book that is inside it recovers the book's own
weight and reports it as prediction. Every book would look like a leader, most
strongly the ones quoted at fights where few others are. Dropping B from its
own target is what separates "moves first" from "is included".

Two more guards:

- Probabilities are de-vigged before differencing. A book that widens its
  margin without changing its opinion would otherwise register as a move.
- Inference is clustered on the card. Fights on one card share news, a fighter
  missing weight moves several lines at once, and treating those as
  independent observations would shrink every interval to nothing.

The archive samples hourly, so the finest horizon measurable here is one hour.
That is far longer than the window a stale line actually survives, which means
a positive result here is evidence the effect exists, not evidence it is
capturable. H4 is where capture gets tested, and the pre-registration is
explicit that neither authorises a bet on its own.
"""

import numpy as np
import pandas as pd

import historical_odds
import sealed

MIN_BOOK_OBSERVATIONS = 200
MIN_SHARED_BOOKS = 4


def _devig(odds_a, odds_b):
    """Implied probability with the margin removed.

    Both branches are evaluated before np.where selects, so a price of exactly
    -100 divides by zero in the branch that gets discarded. The result is
    right either way; the guard just stops a real warning from becoming
    background noise that hides a later one.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(odds_a < 0, -odds_a / (-odds_a + 100.0),
                     100.0 / (odds_a + 100.0))
        b = np.where(odds_b < 0, -odds_b / (-odds_b + 100.0),
                     100.0 / (odds_b + 100.0))
        total = a + b
        out = np.where((total > 0) & np.isfinite(total), a / total, np.nan)
    return out


def panel(root="raw/odds_api_historical", development_only=True):
    """One row per (fight, hours before the card, book) with a de-vigged price.

    Only the hourly sweep is used. entry, close and the discovery probes sit at
    irregular offsets, and mixing them in would put unevenly spaced points into
    a regression that assumes its steps are the same size.
    """
    quotes = historical_odds.load_quotes(root)
    if development_only:
        quotes = sealed.development(quotes, column="event_date")
    sweep = quotes[quotes["snapshot_kind"].str.startswith("t_minus", na=False)].copy()
    sweep["hours_before"] = (sweep["snapshot_kind"]
                             .str.extract(r"t_minus_(\d+\.\d+)h")[0].astype(float))
    for column in ("odds_a", "odds_b"):
        sweep[column] = pd.to_numeric(sweep[column], errors="coerce")
    sweep = sweep.dropna(subset=["odds_a", "odds_b", "hours_before", "api_event_id"])
    sweep["p"] = _devig(sweep["odds_a"].to_numpy(), sweep["odds_b"].to_numpy())
    sweep = sweep.dropna(subset=["p"])
    # One quote per (fight, time, book); a re-fetch can leave two.
    return sweep.drop_duplicates(
        subset=["api_event_id", "hours_before", "book_key"], keep="last")


def _slope(x, y):
    """Least-squares slope and R^2, without pulling in a regression package."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0:
        return np.nan, np.nan, len(x)
    slope = np.cov(x, y, bias=True)[0, 1] / x.var()
    fitted = slope * (x - x.mean()) + y.mean()
    residual = ((y - fitted) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return slope, (1 - residual / total if total > 0 else np.nan), len(x)


def build_moves(frame, horizon_hours=1.0):
    """Per (fight, book, time): the book's last move and the market's next one.

    `lead` is how much book B moved into time t. `follow` is how much the
    consensus of every *other* book moves over the following `horizon_hours`.
    Time runs backwards in hours_before, so "next" is a smaller number.
    """
    wide = {}
    for (fight, hours), group in frame.groupby(["api_event_id", "hours_before"]):
        wide[(fight, hours)] = group.set_index("book_key")["p"].to_dict()

    out = []
    fights = frame["api_event_id"].unique()
    dates = frame.drop_duplicates("api_event_id").set_index("api_event_id")["event_date"]
    for fight in fights:
        times = sorted({h for (f, h) in wide if f == fight}, reverse=True)
        for index in range(1, len(times) - int(horizon_hours)):
            previous, now = times[index - 1], times[index]
            future = now - horizon_hours
            if (fight, future) not in wide:
                continue
            at_prev, at_now, at_future = (wide[(fight, previous)], wide[(fight, now)],
                                          wide[(fight, future)])
            shared = set(at_now) & set(at_future)
            if len(shared) < MIN_SHARED_BOOKS:
                continue
            for book in set(at_prev) & set(at_now):
                others = shared - {book}
                if len(others) < MIN_SHARED_BOOKS - 1:
                    continue
                # Consensus EXCLUDING this book, or it predicts itself.
                before = np.median([at_now[o] for o in others])
                after = np.median([at_future[o] for o in others])
                out.append({
                    "event_date": dates.get(fight),
                    "api_event_id": fight,
                    "book_key": book,
                    "hours_before": now,
                    "lead": at_now[book] - at_prev[book],
                    "follow": after - before,
                })
    return pd.DataFrame(out)


def leaders(moves, draws=2000, seed=17, min_observations=MIN_BOOK_OBSERVATIONS):
    """Per book: does its move predict the rest of the market's next move?"""
    rows = []
    for book, group in moves.groupby("book_key"):
        slope, r2, n = _slope(group["lead"], group["follow"])
        if n < min_observations or not np.isfinite(slope):
            continue
        cards = [g for _, g in group.groupby("event_date")]
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(draws):
            pick = pd.concat([cards[i] for i in rng.integers(0, len(cards), len(cards))])
            value, _, count = _slope(pick["lead"], pick["follow"])
            if np.isfinite(value):
                boot.append(value)
        rows.append({
            "book": book,
            "observations": n,
            "cards": len(cards),
            "slope": round(float(slope), 4),
            "r2": round(float(r2), 4),
            "ci90_low": round(float(np.percentile(boot, 5)), 4) if boot else np.nan,
            "ci90_high": round(float(np.percentile(boot, 95)), 4) if boot else np.nan,
        })
    frame = pd.DataFrame(rows)
    return (frame.sort_values("slope", ascending=False).reset_index(drop=True)
            if len(frame) else frame)
