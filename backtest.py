"""Backtest the model against closing betting lines.

Matches our fight table to a dataset with closing odds (ufc-master.csv,
American odds), takes the most recent N events, trains ONLY on fights
before that window, and compares model probabilities to de-vigged
closing-line probabilities on log loss / Brier / accuracy, plus a simple
flat-stake betting simulation.

Usage:
    python backtest.py --fights fights_v2.csv --odds raw/ufc-master.csv --events 50
"""

import argparse
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features, symmetrize


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace(".", "").replace("-", " ").split())


def american_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(odds < 0, -odds / (-odds + 100.0),
                        100.0 / (odds + 100.0))


def american_payout(odds):
    """Profit on a 1-unit winning stake."""
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, 100.0 / -odds, odds / 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", default="fights_v2.csv")
    ap.add_argument("--odds", default="raw/ufc-master.csv")
    ap.add_argument("--events", type=int, default=50)
    args = ap.parse_args()

    # ---- our features ----------------------------------------------------
    fights = pd.read_csv(args.fights, parse_dates=["date"])
    feats, fcols = build_features(fights)
    feats["key_a"] = feats["fighter_a"].map(norm_name)
    feats["key_b"] = feats["fighter_b"].map(norm_name)

    # ---- odds data --------------------------------------------------------
    od = pd.read_csv(args.odds, low_memory=False)
    od["date"] = pd.to_datetime(od["date"])
    od = od.dropna(subset=["R_odds", "B_odds"])
    od = od[od["Winner"].isin(["Red", "Blue"])].copy()
    od["key_r"] = od["R_fighter"].map(norm_name)
    od["key_b_"] = od["B_fighter"].map(norm_name)

    # ---- match on (date, unordered fighter pair) --------------------------
    feats["pair"] = [frozenset(t) for t in zip(feats["key_a"], feats["key_b"])]
    od["pair"] = [frozenset(t) for t in zip(od["key_r"], od["key_b_"])]
    merged = feats.merge(
        od[["date", "pair", "key_r", "R_odds", "B_odds", "Winner"]],
        on=["date", "pair"], how="inner",
    ).drop_duplicates(subset=["date", "pair"])
    print(f"Matched {len(merged)} fights to closing lines "
          f"({len(merged) / len(od):.1%} of odds rows).")

    # ---- last N events -----------------------------------------------------
    event_dates = np.sort(merged["date"].unique())[-args.events:]
    window = merged[merged["date"].isin(event_dates)].copy()
    cutoff = event_dates[0]
    print(f"Backtest window: {pd.Timestamp(cutoff).date()} -> "
          f"{pd.Timestamp(event_dates[-1]).date()}  "
          f"({args.events} events, {len(window)} fights)")

    # ---- train strictly before the window ---------------------------------
    train = symmetrize(feats[feats["date"] < cutoff], fcols)
    print(f"Training fights (pre-window, symmetrized): {len(train)}")

    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=2000))
    logit.fit(train[fcols], train["target"])

    gbm = HistGradientBoostingClassifier(
        learning_rate=0.03, max_leaf_nodes=15, max_iter=500,
        early_stopping=True, validation_fraction=0.15, random_state=0)
    gbm.fit(train[fcols], train["target"])

    X = window[fcols]
    p_logit = (logit.predict_proba(X)[:, 1]
               + (1 - logit.predict_proba(-X)[:, 1])) / 2
    p_gbm = (gbm.predict_proba(X)[:, 1]
             + (1 - gbm.predict_proba(-X)[:, 1])) / 2

    # ---- align everything to "probability RED wins" ------------------------
    # Our target is P(fighter_a wins); map to red corner.
    a_is_red = (window["key_a"] == window["key_r"]).values
    y_red = np.where(
        a_is_red, window["target"], 1 - window["target"]).astype(int)
    p_logit_red = np.where(a_is_red, p_logit, 1 - p_logit)
    p_gbm_red = np.where(a_is_red, p_gbm, 1 - p_gbm)

    pr_raw = american_to_prob(window["R_odds"])
    pb_raw = american_to_prob(window["B_odds"])
    p_odds_red = pr_raw / (pr_raw + pb_raw)  # de-vigged
    print(f"Average bookmaker vig: {(pr_raw + pb_raw).mean() - 1:.2%}\n")

    def report(name, p):
        print(f"{name:26s} logloss={log_loss(y_red, p):.4f}  "
              f"brier={brier_score_loss(y_red, p):.4f}  "
              f"acc={accuracy_score(y_red, p >= 0.5):.3f}")

    report("Closing line (de-vig)", p_odds_red)
    report("Logistic regression", p_logit_red)
    report("Gradient boosting", p_gbm_red)
    report("50/50 model+line blend", (p_logit_red + p_odds_red) / 2)

    agree = ((p_logit_red >= 0.5) == (p_odds_red >= 0.5)).mean()
    corr = np.corrcoef(p_logit_red, p_odds_red)[0, 1]
    print(f"\nModel vs line: picks agree {agree:.1%} of fights, "
          f"probability correlation {corr:.3f}")

    # When they disagree, who's right?
    dis = (p_logit_red >= 0.5) != (p_odds_red >= 0.5)
    if dis.sum():
        model_right = ((p_logit_red[dis] >= 0.5) == y_red[dis]).mean()
        print(f"Disagreements: {dis.sum()} fights — model right "
              f"{model_right:.1%}, line right {1 - model_right:.1%}")

    # ---- flat-stake betting simulation -------------------------------------
    print("\nFlat 1-unit betting sim (bet side where model prob exceeds "
          "raw implied prob by edge):")
    pay_r = american_payout(window["R_odds"])
    pay_b = american_payout(window["B_odds"])
    for edge in (0.00, 0.05, 0.10):
        bet_red = p_logit_red > pr_raw + edge
        bet_blue = (1 - p_logit_red) > pb_raw + edge
        profit = np.where(
            bet_red, np.where(y_red == 1, pay_r, -1.0), 0.0
        ) + np.where(
            bet_blue, np.where(y_red == 0, pay_b, -1.0), 0.0
        )
        n = int(bet_red.sum() + bet_blue.sum())
        if n == 0:
            print(f"  edge>{edge:.0%}: no bets")
            continue
        wins = int(np.where(bet_red, y_red == 1, False).sum()
                   + np.where(bet_blue, y_red == 0, False).sum())
        print(f"  edge>{edge:>3.0%}: {n:4d} bets, {wins:3d} wins, "
              f"P&L {profit.sum():+7.1f}u, ROI {profit.sum() / n:+7.2%}")

    # Per-event cumulative P&L at 5% edge for inspection
    window = window.assign(p_model=p_logit_red, p_line=p_odds_red, y=y_red)
    window.to_csv("backtest_predictions.csv", index=False)
    print("\nWrote per-fight predictions to backtest_predictions.csv")


if __name__ == "__main__":
    main()
