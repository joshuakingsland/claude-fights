"""Shared pipeline: cached matched-dataset build + walk-forward engine +
evaluation/betting utilities used by all improvement rounds.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import american_payout, american_to_prob, norm_name
from data_quality import audit_fights

rng = np.random.default_rng(0)


def _odds_rows(max_date):
    """Load closing odds, with captured snapshots as an explicit fallback.

    The UFC master file wins whenever it has a pair.  Logged snapshots are
    useful for paper-trading coverage after the master file stops updating,
    but remain labelled ``odds_log`` so reports cannot mistake them for
    closing prices.
    """
    raw = pd.read_csv("raw/ufc-master.csv", low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "R_odds", "B_odds"])
    raw = raw[raw["Winner"].isin(["Red", "Blue"])].copy()
    raw["key_r"] = raw["R_fighter"].map(norm_name)
    raw["key_b"] = raw["B_fighter"].map(norm_name)
    raw["pair"] = [frozenset(t) for t in zip(raw["key_r"], raw["key_b"])]
    raw = raw[["date", "pair", "key_r", "R_odds", "B_odds"]]
    raw["odds_source"] = "ufc-master"
    raw["odds_is_closing"] = True
    raw["odds_fetched_at"] = pd.Series(pd.NaT, index=raw.index,
                                        dtype="datetime64[ns, UTC]")

    path = "odds_log.csv"
    if not os.path.exists(path):
        return raw
    log = pd.read_csv(path, low_memory=False)
    required = {"date", "fighter_a", "fighter_b", "odds_a", "odds_b"}
    if not required.issubset(log.columns):
        return raw
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    fetched = log["fetched_at"] if "fetched_at" in log else pd.NaT
    log["odds_fetched_at"] = pd.to_datetime(fetched, errors="coerce", utc=True)
    log = log[(log["date"] <= max_date)
              & log["odds_a"].notna() & log["odds_b"].notna()].copy()
    if not len(log):
        return raw
    log["key_r"] = log["fighter_a"].map(norm_name)
    log["key_b"] = log["fighter_b"].map(norm_name)
    log["pair"] = [frozenset(t) for t in zip(log["key_r"], log["key_b"])]
    log = log.rename(columns={"odds_a": "R_odds", "odds_b": "B_odds"})
    log["odds_source"] = "odds_log"
    log["odds_is_closing"] = False
    log = log[["date", "pair", "key_r", "R_odds", "B_odds",
               "odds_source", "odds_is_closing", "odds_fetched_at"]]
    # Prefer the true historical closing line; for fallback snapshots retain
    # only the latest captured quote for each otherwise-unmatched fight.
    all_rows = pd.concat([raw, log], ignore_index=True)
    all_rows["_priority"] = all_rows["odds_source"].eq("odds_log").astype(int)
    all_rows = all_rows.sort_values(["date", "pair", "_priority",
                                     "odds_fetched_at"],
                                    ascending=[True, True, True, False])
    return all_rows.drop_duplicates(["date", "pair"], keep="first") \
        .drop(columns="_priority")


def load_matched_cached(builder, tag, bout_cols=()):
    """Build (or load cached) matched fights+odds table in red-corner frame.

    builder: fn(fights_df) -> (feats, fcols). Differential fcols get
    sign-flipped into the red frame; bout_cols do not (bout-level facts).
    """
    cache = f"cache_{tag}_v3.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    fights = pd.read_csv("fights_v2.csv", parse_dates=["date"])
    errors = audit_fights(fights)
    if errors:
        raise ValueError("Fight data quality gate failed: " + "; ".join(errors))
    feats, fcols = builder(fights)
    feats["key_a"] = feats["fighter_a"].map(norm_name)
    feats["key_b"] = feats["fighter_b"].map(norm_name)
    feats["pair"] = [frozenset(t) for t in zip(feats["key_a"], feats["key_b"])]

    od = _odds_rows(pd.Timestamp(fights["date"].max()))

    m = feats.merge(od[["date", "pair", "key_r", "R_odds", "B_odds",
                        "odds_source", "odds_is_closing", "odds_fetched_at"]],
                    on=["date", "pair"], how="inner") \
             .drop_duplicates(["date", "pair"]).reset_index(drop=True)

    a_red = (m["key_a"] == m["key_r"]).values
    m["y"] = np.where(a_red, m["target"], 1 - m["target"]).astype(int)
    sign = np.where(a_red, 1.0, -1.0)
    diff_cols = [c for c in fcols if c not in bout_cols]
    for c in diff_cols:
        m[c] = m[c] * sign
    pr, pb = american_to_prob(m["R_odds"]), american_to_prob(m["B_odds"])
    m["p_line"] = pr / (pr + pb)
    m["pr_raw"], m["pb_raw"] = pr, pb
    from scipy.special import logit as slogit
    m["line_logit"] = slogit(m["p_line"].clip(0.02, 0.98))

    with open(cache, "wb") as f:
        pickle.dump((m, fcols), f)
    return m, fcols


def walk_forward(m, cols, start, end="2100-01-01", C=0.05, min_train=2000,
                 refit_every=1, half_life_years=None):
    """Refit before each event date (or every k-th) on all prior fights.
    half_life_years: if set, exponential recency weights on training rows."""
    window = m[(m["date"] >= start) & (m["date"] < end)]
    preds = pd.Series(np.nan, index=window.index)
    lr = None
    for i, d in enumerate(np.sort(window["date"].unique())):
        tr = m[m["date"] < d]
        if len(tr) < min_train:
            continue
        if lr is None or i % refit_every == 0:
            lr = make_pipeline(StandardScaler(),
                               LogisticRegression(C=C, max_iter=3000))
            if half_life_years:
                age_y = (d - tr["date"]).dt.days / 365.25
                w = 0.5 ** (age_y / half_life_years)
                lr.fit(tr[cols], tr["y"],
                       logisticregression__sample_weight=w)
            else:
                lr.fit(tr[cols], tr["y"])
        idx = window.index[window["date"] == d]
        preds.loc[idx] = lr.predict_proba(window.loc[idx, cols])[:, 1]
    ok = preds.notna()
    return window[ok].reset_index(drop=True), preds[ok].to_numpy()


def bets(df, p_model, edge):
    pay_r = american_payout(df["R_odds"])
    pay_b = american_payout(df["B_odds"])
    bet_r = np.asarray(p_model > df["pr_raw"] + edge)
    bet_b = np.asarray((1 - p_model) > df["pb_raw"] + edge)
    pnl = np.where(bet_r, np.where(df["y"] == 1, pay_r, -1.0), 0.0) \
        + np.where(bet_b, np.where(df["y"] == 0, pay_b, -1.0), 0.0)
    mask = bet_r | bet_b
    # model prob of the side we bet (for Kelly)
    p_side = np.where(bet_r, p_model, 1 - p_model)[mask]
    pay = np.where(bet_r, pay_r, pay_b)[mask]
    return pnl[mask], np.asarray(df["date"])[mask], p_side, pay


def boot_ci(pnl, n=4000):
    r = [rng.choice(pnl, len(pnl), replace=True).mean() for _ in range(n)]
    return np.percentile(r, [5, 95])


def summarize(te, p, label, edges=(0.04,), by_year=True):
    y = te["y"].to_numpy()
    llm, lll = log_loss(y, p), log_loss(y, te["p_line"])
    print(f"{label}: n={len(te)}  line_ll={lll:.4f}  model_ll={llm:.4f} "
          f"({llm - lll:+.4f})  acc={accuracy_score(y, p >= .5):.3f} "
          f"vs line {accuracy_score(y, te['p_line'] >= .5):.3f}")
    for e in edges:
        pnl, dates, _, _ = bets(te, p, e)
        if len(pnl) < 20:
            continue
        lo, hi = boot_ci(pnl)
        line = (f"  edge={e:.2f}: n={len(pnl)} ROI={pnl.mean():+.3f} "
                f"CI90=[{lo:+.3f},{hi:+.3f}]")
        if by_year:
            yr = pd.DatetimeIndex(dates).year
            per = {int(v): round(pnl[yr == v].mean(), 2)
                   for v in sorted(set(yr))}
            line += f"  by-year={per}"
        print(line)
    return llm - lll
