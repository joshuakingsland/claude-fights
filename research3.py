"""Research round 3 — isolate the source of the edge (VALIDATION only).

Ablations:
  A. line-only recalibration          (is it all shrinkage?)
  B. line + focused top features      (sparse, lower variance)
  C. B trained on 2016+ only          (modern market regime)
  D. B with interaction age x layoff, ko-loss recency

Pick the locked config for the one-shot final test by val log loss,
CI-checked ROI, and 2024 sub-period behavior.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research import load_matched, VAL_START, TEST_START
from research2 import bets_pnl, boot_ci

FOCUS = ["age_diff", "c_apm_diff", "c_ctrld_pm_diff", "c_won_diff",
         "reach_diff", "c_tdd_diff", "c_ko_loss_n_diff", "elo_slow_diff",
         "r3_lpm_diff"]


def fit_eval(tr, va, cols, C=0.05, label=""):
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(C=C, max_iter=3000))
    lr.fit(tr[cols], tr["y"])
    p = lr.predict_proba(va[cols])[:, 1]
    ll = log_loss(va["y"], p)
    out = f"{label:34s} logloss={ll:.4f}"
    for edge in (0.04, 0.05):
        pnl, dates, _ = bets_pnl(va, p, edge)
        if len(pnl) >= 30:
            lo, _, hi = boot_ci(pnl)
            d24 = pd.DatetimeIndex(dates) >= "2024-01-01"
            roi24 = pnl[d24].mean() if d24.sum() > 10 else np.nan
            out += (f" | e{edge:.2f}: n={len(pnl)} roi={pnl.mean():+.3f} "
                    f"CI[{lo:+.2f},{hi:+.2f}] '24={roi24:+.3f}")
    print(out)
    return p


def main():
    m, fcols = load_matched()
    tr = m[m["date"] < VAL_START]
    tr16 = m[(m["date"] >= "2016-01-01") & (m["date"] < VAL_START)]
    va = m[(m["date"] >= VAL_START) & (m["date"] < TEST_START)] \
        .reset_index(drop=True)

    ll_line = log_loss(va["y"], va["p_line"])
    print(f"line alone: logloss={ll_line:.4f}\n")

    fit_eval(tr, va, ["line_logit"], label="A: line-only recalibration")
    fit_eval(tr, va, ["line_logit"] + fcols, label="full: line + all features")
    fit_eval(tr, va, ["line_logit"] + FOCUS, label="B: line + focused features")
    fit_eval(tr16, va, ["line_logit"] + FOCUS, label="C: B, trained 2016+")

    # D: interactions
    for d in (tr, tr16, va):
        d["age_x_off"] = d["age_diff"] * d["days_off_diff"] / 365
        d["ko_recent"] = ((d["days_since_ko_diff"] != 0)
                          & (d["days_since_ko_diff"].abs() < 365)) \
            * np.sign(d["days_since_ko_diff"])
    fit_eval(tr16, va, ["line_logit"] + FOCUS + ["age_x_off", "ko_recent"],
             label="D: C + interactions")

    # feature contribution check for B on tr16
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(C=0.05, max_iter=3000))
    lr.fit(tr16[["line_logit"] + FOCUS], tr16["y"])
    print("\nC coefficients (standardized):")
    print(pd.Series(lr[-1].coef_[0], index=["line_logit"] + FOCUS)
          .round(3).to_string())


if __name__ == "__main__":
    main()
