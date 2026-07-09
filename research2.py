"""Research round 2 — stress-test the residual edge on VALIDATION."""

import numpy as np
import pandas as pd

from backtest import american_payout

rng = np.random.default_rng(0)


def bets_pnl(df, p_model, edge):
    pay_r = american_payout(df["R_odds"])
    pay_b = american_payout(df["B_odds"])
    bet_r = np.asarray(p_model > df["pr_raw"] + edge)
    bet_b = np.asarray((1 - p_model) > df["pb_raw"] + edge)
    pnl = np.where(bet_r, np.where(df["y"] == 1, pay_r, -1.0), 0.0) \
        + np.where(bet_b, np.where(df["y"] == 0, pay_b, -1.0), 0.0)
    mask = bet_r | bet_b
    return pnl[mask], np.asarray(df["date"])[mask], \
        np.asarray(bet_b & (df["pb_raw"] < 0.5))[mask] | \
        np.asarray(bet_r & (df["pr_raw"] < 0.5))[mask]


def boot_ci(pnl, n=4000):
    rois = [rng.choice(pnl, size=len(pnl), replace=True).mean()
            for _ in range(n)]
    return np.percentile(rois, [5, 95])
