"""H9-H20: does anything predict the market's own error?

Pre-registered in PREREGISTRATION.md, Addendum 4, before it was run. The
framing matters more than any individual candidate. Every hypothesis before
this one asked whether we could out-handicap the market, and H7 settled that:
career statistics make the forecast monotonically worse, so a model whose best
component is the price cannot beat the price.

This asks the complementary question. Take the market's error directly,

    resid = y - p_line

using the de-vigged entry consensus, and ask whether any observable predicts
it. If the market is efficient the residual is unpredictable by construction.
If some feature predicts it, that feature *is* the edge, and the coefficient is
denominated in exactly the thing we would be capturing: probability points.

Twelve candidates become twelve coefficients against that one residual.

Why they run together
---------------------
Testing twelve hypotheses one at a time and stopping at the first pass is a
procedure that returns a false positive with near certainty. At 90% confidence
across twelve independent tests, the chance that at least one clears zero by
luck alone is 1 - 0.9**12, about 72%. So every candidate is reported with a
Bonferroni-adjusted interval at family-wise 90%, and only that interval counts.
An uncorrected interval clearing zero in a family this size is the expected
behaviour of noise.

The bar is deliberately two-stage. A surviving coefficient is a statistical
finding; it is not a bet. `flat_stake_gate` is the second gate, and it is the
one that matters, because a coefficient lives in probability points while a
bankroll lives in prices. A five-point edge on a -400 favourite and a
five-point edge on a +300 underdog are the same coefficient and nothing like
the same wager. Only the gate knows the difference.
"""

import numpy as np
import pandas as pd

import sealed

# Each candidate is a claim that the market misprices a particular kind of
# fight. The comment is the claim; the column is how it gets measured.
CANDIDATES = {
    "H9  layoff gap": "days_off_diff",          # fighters back from long absences
    "H10 age gap": "age_diff",                  # old or young beyond the line
    "H11 stance edge": "stance_edge",           # southpaw/orthodox mismatches
    "H12 reach gap": "reach_diff",              # extreme reach advantages
    "H13 recent form": "r3_won_diff",           # winning or losing runs
    "H14 experience gap": "c_fights_diff",      # debutants and very green fighters
    "H15 heavyweight": "heavy_bout",            # division effects
    "H16 five-round bout": "five_rd_bout",      # championship rounds
    "H17 ko-loss history": "c_ko_loss_n_diff",  # recent knockout losses
    "H18 book disagreement": "entry_n_books",   # fights books disagree on
    "H19 line movement": "line_move",           # price moved most entry to close
    "H20 favourite size": "line_abs",           # distance from even money
}

FAMILY_CONFIDENCE = 90.0
MIN_CANDIDATE_ROWS = 100
MIN_BOOTSTRAP_ROWS = 30


def residuals(path="historical_entry_validation.csv", development_only=True):
    """One row per settled fight, carrying the market's error on it.

    `line_move` is derived rather than stored, and it is the only candidate
    that is not a pre-fight observable: it needs the closing price. That is
    fine for measuring whether the market's own drift predicts its error, but
    it means H19 could never become a bet placed at entry. Recorded here so
    that asymmetry is visible at the point of use rather than discovered later.
    """
    frame = pd.read_csv(path)
    if development_only:
        frame = sealed.development(frame, column="date")
    frame = frame.dropna(subset=["y", "p_line"]).copy()
    frame["resid"] = frame["y"] - frame["p_line"]
    frame["line_move"] = frame["p_close_line"] - frame["p_line"]
    return frame


def _slope(x, y):
    """Least-squares slope of y on x, ignoring rows either side is missing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_BOOTSTRAP_ROWS or x[ok].std() == 0:
        return np.nan
    return np.cov(x[ok], y[ok], bias=True)[0, 1] / x[ok].var()


def _clustered_slopes(frame, column, draws, seed, target="resid"):
    """Bootstrap the slope, resampling whole cards rather than fights.

    Fights on one card share everything that moves a line at short notice: a
    missed weight cut, a late replacement, a venue's altitude, the same
    afternoon of steam. Treating them as independent draws would shrink every
    interval here by roughly the square root of the card size and manufacture
    significance out of correlation.
    """
    cards = [group for _, group in frame.groupby("date")]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        pick = pd.concat([cards[i] for i in rng.integers(0, len(cards), len(cards))])
        slope = _slope(pick[column], pick[target])
        if np.isfinite(slope):
            out.append(slope)
    return np.asarray(out)


def _require_residual(frame):
    """Both entry points regress against the market's error; say so out loud.

    Without this the failure is a bare pandas KeyError raised from inside a
    bootstrap loop, which reads like a bug in the resampling rather than a
    caller who passed a raw validation frame instead of `residuals()`.
    """
    if "resid" not in frame.columns:
        raise KeyError("frame has no 'resid' column; build it with residuals()")


def family(frame, candidates=None, draws=2000, seed=3):
    """Every candidate's coefficient, with and without the correction.

    Returned for all of them, not only the ones that pass. A family where only
    survivors are reported is indistinguishable from a family of one, which is
    the whole thing the correction exists to prevent.
    """
    _require_residual(frame)
    candidates = CANDIDATES if candidates is None else candidates
    usable = {name: column for name, column in candidates.items()
              if column in frame.columns
              and frame[column].notna().sum() >= MIN_CANDIDATE_ROWS}
    if not usable:
        return pd.DataFrame()

    # Bonferroni: spend the family's whole error budget across the candidates
    # actually tested, not the number originally written down. Dropping a
    # candidate for want of data makes the survivors' bar slightly lower, and
    # it should - there were fewer chances to get lucky.
    alpha = (100.0 - FAMILY_CONFIDENCE) / len(usable)

    rows = []
    for name, column in usable.items():
        subset = frame.dropna(subset=[column])
        boot = _clustered_slopes(subset, column, draws, seed)
        if not len(boot):
            continue
        low90, high90 = np.percentile(boot, [5.0, 95.0])
        low, high = np.percentile(boot, [alpha / 2, 100.0 - alpha / 2])
        rows.append({
            "candidate": name,
            "feature": column,
            "n": len(subset),
            "coefficient": _slope(subset[column], subset["resid"]),
            "ci90_low": low90,
            "ci90_high": high90,
            "bonferroni_low": low,
            "bonferroni_high": high,
            "survives": bool(low > 0 or high < 0),
        })
    return pd.DataFrame(rows)


def seed_stability(frame, column, n_candidates=len(CANDIDATES),
                   seeds=(3, 11, 17, 29, 41, 53, 67, 71), draws=5000):
    """Does the survivor survive, or did it survive this random seed?

    A Bonferroni bound at family-wise 90% across twelve candidates is the
    0.42nd percentile of the bootstrap. That is the third or fourth order
    statistic even at five thousand draws, and order statistics that far into
    a tail move around a lot between seeds. When a bound lands near zero, the
    sign of that bound is partly a property of the resampling rather than of
    the data.

    So a survivor is only a survivor if it survives across seeds. Anything
    that flips is a bound sitting on zero, which is not a finding. This is the
    cheapest possible guard against the failure mode the whole pre-registration
    was written for: reporting the tail of a bootstrap as an edge.
    """
    _require_residual(frame)
    alpha = (100.0 - FAMILY_CONFIDENCE) / max(n_candidates, 1)
    subset = frame.dropna(subset=[column])
    rows = []
    for seed in seeds:
        boot = _clustered_slopes(subset, column, draws, seed)
        if not len(boot):
            continue
        low, high = np.percentile(boot, [alpha / 2, 100.0 - alpha / 2])
        rows.append({
            "seed": seed,
            "coefficient": float(boot.mean()),
            "bonferroni_low": low,
            "bonferroni_high": high,
            "survives": bool(low > 0 or high < 0),
        })
    return pd.DataFrame(rows)


def _implied(american):
    """Implied probability from American odds, vig included.

    np.where evaluates both branches before selecting, so a price of exactly
    -100 divides by zero in the branch that gets thrown away. The answer is
    right regardless; silencing it stops a guaranteed warning from becoming
    background noise that hides a real one later.
    """
    odds = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def _profit(american, won):
    """Profit on a one-unit stake: the price if it lands, -1 if it doesn't."""
    odds = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        payout = np.where(odds < 0, 100.0 / -odds, odds / 100.0)
    return np.where(np.asarray(won, dtype=bool), payout, -1.0)


def flat_stake_gate(frame, column="age_diff", thresholds=(0, 2, 4, 6, 8),
                    draws=2000, seed=3):
    """The second gate: does the signal pay at the prices actually quoted?

    Bets one unit on the side the coefficient favours - for a negative
    coefficient on `column`, the side with the lower value - at the real entry
    price, for every fight where the gap clears each threshold.

    A surviving coefficient that fails here is not a near miss. It means the
    mispricing is real in probability space and too small, or too concentrated
    in the wrong prices, to survive the margin. That is a finding about the
    market, not an edge in it, and the pre-registration is explicit that it
    does not authorise a bet.
    """
    _require_residual(frame)
    needed = {column, "R_odds", "B_odds", "y", "date"}
    missing = needed - set(frame.columns)
    if missing:
        raise KeyError(f"flat_stake_gate needs {sorted(missing)}")

    work = frame.dropna(subset=[column, "R_odds", "B_odds", "y"]).copy()
    sign = np.sign(_slope(work[column], work["resid"]))
    if not np.isfinite(sign) or sign == 0:
        raise ValueError(f"no usable slope on {column}; nothing to bet")

    # sign < 0 means a larger gap predicts A underperforming, so back B.
    back_a = (work[column] * sign) > 0
    work["bet_odds"] = np.where(back_a, work["R_odds"], work["B_odds"])
    work["bet_won"] = np.where(back_a, work["y"] == 1, work["y"] == 0)
    work["profit"] = _profit(work["bet_odds"], work["bet_won"])
    work["implied"] = _implied(work["bet_odds"])
    work["gap"] = work[column].abs()

    rows = []
    for threshold in thresholds:
        subset = work[work["gap"] >= threshold]
        if subset.empty:
            continue
        cards = [group for _, group in subset.groupby("date")]
        rng = np.random.default_rng(seed)
        boot = [pd.concat([cards[i] for i in rng.integers(0, len(cards), len(cards))])
                ["profit"].mean() for _ in range(draws)]
        rows.append({
            "threshold": threshold,
            "n": len(subset),
            "win_rate": subset["bet_won"].mean(),
            "implied": subset["implied"].mean(),
            "roi": subset["profit"].mean(),
            "roi_ci90_low": float(np.percentile(boot, 5)),
            "roi_ci90_high": float(np.percentile(boot, 95)),
        })
    result = pd.DataFrame(rows)
    if len(result):
        # Both conditions, exactly as pre-registered. Either alone is a story.
        result["passes"] = (result["roi_ci90_low"] > 0) & (result["n"] >= 200)
    return result
