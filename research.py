"""Research harness — iterate on VALIDATION only (2022-01-01 .. 2025-01-17).

Protocol:
  train  : matched fights < 2022-01-01
  val    : 2022-01-01 .. 2025-01-17   (iterate here freely)
  test   : last 50 events (2025-01-18 ..)  — NOT touched by this script.

Two questions:
  1. Does any feature carry signal BEYOND the closing line?
     -> logistic regression on [logit(line), features]; if features get
        weight and val log loss beats line alone, there's residual signal.
  2. Can a selective betting rule be profitable on val?
     -> grid over edge thresholds / segments, flat stakes, report ROI + N.
"""

import numpy as np
import pandas as pd
from scipy.special import logit as slogit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import american_to_prob, american_payout, norm_name
from features_v2 import build_features_v2

VAL_START, TEST_START = "2022-01-01", "2025-01-18"


def load_matched():
    fights = pd.read_csv("fights_v2.csv", parse_dates=["date"])
    feats, fcols = build_features_v2(fights)
    feats["key_a"] = feats["fighter_a"].map(norm_name)
    feats["key_b"] = feats["fighter_b"].map(norm_name)
    feats["pair"] = [frozenset(t) for t in zip(feats["key_a"], feats["key_b"])]

    od = pd.read_csv("raw/ufc-master.csv", low_memory=False)
    od["date"] = pd.to_datetime(od["date"])
    od = od.dropna(subset=["R_odds", "B_odds"])
    od = od[od["Winner"].isin(["Red", "Blue"])].copy()
    od["key_r"] = od["R_fighter"].map(norm_name)
    od["pair"] = [frozenset(t) for t in
                  zip(od["key_r"], od["B_fighter"].map(norm_name))]

    m = feats.merge(od[["date", "pair", "key_r", "R_odds", "B_odds"]],
                    on=["date", "pair"], how="inner") \
             .drop_duplicates(["date", "pair"]).reset_index(drop=True)

    # everything in RED-corner frame
    a_red = (m["key_a"] == m["key_r"]).values
    m["y"] = np.where(a_red, m["target"], 1 - m["target"]).astype(int)
    sign = np.where(a_red, 1.0, -1.0)
    for c in fcols:
        m[c] = m[c] * sign
    pr, pb = american_to_prob(m["R_odds"]), american_to_prob(m["B_odds"])
    m["p_line"] = pr / (pr + pb)
    m["pr_raw"], m["pb_raw"] = pr, pb
    m["line_logit"] = slogit(m["p_line"].clip(0.02, 0.98))
    return m, fcols


def roi_grid(df, p_model, label, edges=(0.03, 0.05, 0.08, 0.12)):
    pay_r = american_payout(df["R_odds"])
    pay_b = american_payout(df["B_odds"])
    rows = []
    segs = {
        "all": np.ones(len(df), bool),
        "dogs_only": None,  # filled per-bet below
    }
    for edge in edges:
        for seg in ("all", "dogs"):
            bet_r = np.asarray(p_model > df["pr_raw"] + edge).copy()
            bet_b = np.asarray((1 - p_model) > df["pb_raw"] + edge).copy()
            if seg == "dogs":  # only bet sides the market prices < 50%
                bet_r &= np.asarray(df["pr_raw"] < 0.5)
                bet_b &= np.asarray(df["pb_raw"] < 0.5)
            pnl = np.where(bet_r, np.where(df["y"] == 1, pay_r, -1.0), 0.0) \
                + np.where(bet_b, np.where(df["y"] == 0, pay_b, -1.0), 0.0)
            n = int(bet_r.sum() + bet_b.sum())
            if n == 0:
                continue
            rows.append({"model": label, "edge": edge, "seg": seg,
                         "n": n, "pnl": pnl.sum(), "roi": pnl.sum() / n})
    return pd.DataFrame(rows)


def main():
    m, fcols = load_matched()
    tr = m[m["date"] < VAL_START]
    va = m[(m["date"] >= VAL_START) & (m["date"] < TEST_START)]
    print(f"matched={len(m)}  train={len(tr)}  val={len(va)}  "
          f"(test untouched: {len(m) - len(tr) - len(va)})\n")

    ll_line = log_loss(va["y"], va["p_line"])
    print(f"VAL closing line logloss: {ll_line:.4f}  "
          f"acc={( (va['p_line']>=.5)==va['y'] ).mean():.3f}\n")

    # ---- 1) residual model: line + features -----------------------------
    Xtr = tr[["line_logit"] + fcols]
    Xva = va[["line_logit"] + fcols]
    for C in (0.02, 0.05, 0.1, 0.3):
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(C=C, max_iter=3000))
        lr.fit(Xtr, tr["y"])
        p = lr.predict_proba(Xva)[:, 1]
        print(f"line+features C={C:<5} logloss={log_loss(va['y'], p):.4f} "
              f"({log_loss(va['y'], p) - ll_line:+.4f} vs line)")

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(C=0.05, max_iter=3000))
    lr.fit(Xtr, tr["y"])
    coefs = pd.Series(lr[-1].coef_[0], index=["line_logit"] + fcols)
    print("\nLargest non-line coefficients (residual signal candidates):")
    print(coefs.drop("line_logit").abs().sort_values(ascending=False)
          .head(12).round(3).to_string())
    print(f"\n(line_logit coef = {coefs['line_logit']:.3f})")

    # ---- 2) betting-rule grid on val ------------------------------------
    p_resid = lr.predict_proba(Xva)[:, 1]

    # pure-features model too (no line), for reference
    lr2 = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=3000))
    lr2.fit(tr[fcols], tr["y"])
    p_pure = lr2.predict_proba(va[fcols])[:, 1]

    grid = pd.concat([
        roi_grid(va, p_resid, "line+feat"),
        roi_grid(va, p_pure, "pure-feat"),
    ])
    print("\nVAL betting grid (flat stakes):")
    print(grid.sort_values("roi", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
