"""Backtest proposed production changes before promoting any of them.

This script consumes strict out-of-fold predictions from
``validate_production.py``. It never refits on the rows it evaluates. The
calibration experiments fit only before ``--calibration-train-end`` and all
stake controls are fixed before the evaluated event.

Usage:
    python backtest_experiments.py
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from backtest import american_payout


def event_ci(event_rows, n=5000, seed=0):
    event_rows = event_rows.loc[event_rows["staked"] > 0].reset_index(drop=True)
    if not len(event_rows):
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n)):
        x = event_rows.iloc[rng.integers(0, len(event_rows), len(event_rows))]
        draws.append(x["pnl"].sum() / x["staked"].sum())
    return [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]


def _metrics(df, p, label, bootstrap=5000):
    y = df["y"].to_numpy()
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return {"label": label, "fights": int(len(df)),
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "accuracy": float(((p >= 0.5) == y).mean())}


def _pnl_for(row, stake):
    if not stake:
        return 0.0
    pick_a = row.pick_side == "A"
    won = (row.y == 1) if pick_a else (row.y == 0)
    odds = row.R_odds if pick_a else row.B_odds
    return float(stake * (american_payout(odds) if won else -1.0))


def simulate(df, threshold=0.04, high_threshold=0.08,
             event_cap=None, drawdown_stop=None, bootstrap=5000):
    """Simulate a fixed net-edge rule without future information."""
    rows = []
    equity, peak, halted = 0.0, 0.0, False
    for date in sorted(df["date"].unique()):
        event = df[df["date"] == date].sort_values("net_edge", ascending=False)
        allocations = []
        remaining = float(event_cap) if event_cap is not None else float("inf")
        if halted:
            event = event.iloc[0:0]
        for row in event.itertuples():
            net = float(row.net_edge)
            if net < threshold:
                continue
            desired = 2 if net >= high_threshold else 1
            stake = int(min(desired, max(0.0, remaining)))
            if stake <= 0:
                continue
            allocations.append((row, stake))
            remaining -= stake
        event_pnl = sum(_pnl_for(row, stake) for row, stake in allocations)
        event_staked = sum(stake for _, stake in allocations)
        equity += event_pnl
        peak = max(peak, equity)
        if drawdown_stop is not None and peak - equity >= drawdown_stop:
            halted = True
        rows.append({"date": date, "pnl": event_pnl, "staked": event_staked,
                     "bets": len(allocations), "equity": equity,
                     "halted": halted})
    events = pd.DataFrame(rows)
    staked = float(events["staked"].sum())
    pnl = float(events["pnl"].sum())
    return {"bets": int(events["bets"].sum()), "staked": staked,
            "pnl": pnl, "roi": pnl / staked if staked else None,
            "max_drawdown": float((events["equity"].cummax()
                                    - events["equity"]).max()),
            "halted": bool(events["halted"].iloc[-1]) if len(events) else False,
            "roi_ci90_event_clustered": event_ci(events, bootstrap),
            "events": events}


def fit_calibrators(df, train_end):
    train = df[df["date"] < train_end]
    x = logit(np.clip(train["p_model"].to_numpy(), 1e-4, 1 - 1e-4)).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=2000).fit(x, train["y"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(train["p_model"], train["y"])
    return platt, iso


def calibrate(df, platt, iso):
    x = logit(np.clip(df["p_model"].to_numpy(), 1e-4, 1 - 1e-4)).reshape(-1, 1)
    return (platt.predict_proba(x)[:, 1],
            iso.predict(df["p_model"].to_numpy()))


def with_probability(df, p):
    """Recompute side, gross edge, and net edge from alternate probabilities."""
    out = df.copy()
    p = np.asarray(p, dtype=float)
    edge_a = p - out["pr_raw"].to_numpy()
    edge_b = (1.0 - p) - out["pb_raw"].to_numpy()
    out["p_model"] = p
    out["pick_side"] = np.where(edge_a >= edge_b, "A", "B")
    out["edge"] = np.maximum(edge_a, edge_b)
    out["net_edge"] = out["edge"] - out["se"].to_numpy()
    return out


def run(args):
    df = pd.read_csv(args.predictions, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df["net_edge"] = df["edge"] - df["se"]
    if not len(df):
        raise SystemExit("No predictions found; run validate_production.py first.")

    dev = df[df["date"] < args.holdout_start]
    holdout = df[df["date"] >= args.holdout_start]
    platt, iso = fit_calibrators(df, args.calibration_train_end)
    platt_all, iso_all = calibrate(df, platt, iso)
    report = {
        "predictions": len(df),
        "dev_cutoff": args.holdout_start,
        "calibration_train_end": args.calibration_train_end,
        "calibration": [],
        "calibration_stability": [],
        "calibrated_staking": [],
        "periods": [],
        "stake_controls": [],
        "odds_sources": df.groupby("odds_source").size().astype(int).to_dict()
    }

    for label, p in (("raw", df["p_model"]),
                     ("platt", platt_all), ("isotonic", iso_all)):
        p_series = pd.Series(np.asarray(p), index=df.index)
        for period, part in (("dev", dev), ("holdout", holdout)):
            p_part = p_series.loc[part.index].to_numpy()
            report["calibration"].append(_metrics(part, p_part,
                                                   f"{label}:{period}"))
    for period, part in (("dev", dev), ("holdout", holdout), ("all", df)):
        result = _metrics(part, part["p_model"], f"raw:{period}")
        result["line_log_loss"] = float(log_loss(part["y"], part["p_line"]))
        report["periods"].append(result)

    for label, p in (("raw", df["p_model"]),
                     ("platt", platt_all), ("isotonic", iso_all)):
        calibrated = with_probability(df, p)
        for period, part in (("dev", dev), ("holdout", holdout), ("all", df)):
            sim = simulate(calibrated.loc[part.index], threshold=0.04,
                           high_threshold=0.08, event_cap=2,
                           bootstrap=args.bootstrap)
            report["calibrated_staking"].append({
                "calibration": label, "period": period,
                "bets": sim["bets"], "staked": sim["staked"],
                "pnl": sim["pnl"], "roi": sim["roi"],
                "roi_ci90_event_clustered": sim["roi_ci90_event_clustered"],
            })

    for train_end in ("2022-01-01", "2023-01-01", "2024-01-01"):
        p_cal, i_cal = fit_calibrators(df, train_end)
        _, iso_holdout = calibrate(holdout, p_cal, i_cal)
        calibrated = with_probability(holdout, iso_holdout)
        sim = simulate(calibrated, threshold=0.04, high_threshold=0.08,
                       event_cap=2, bootstrap=args.bootstrap)
        report["calibration_stability"].append({
            "method": "isotonic", "train_end": train_end,
            **_metrics(holdout, iso_holdout, f"isotonic:{train_end}:holdout"),
            "bets": sim["bets"], "staked": sim["staked"],
            "pnl": sim["pnl"], "roi": sim["roi"],
            "roi_ci90_event_clustered": sim["roi_ci90_event_clustered"],
        })

    for threshold in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        for cap in (None, 2, 4, 6):
            sim = simulate(df, threshold=threshold,
                           high_threshold=max(0.08, threshold * 2),
                           event_cap=cap, bootstrap=args.bootstrap)
            report["stake_controls"].append({
                "threshold": threshold, "event_cap": cap,
                "bets": sim["bets"], "staked": sim["staked"],
                "pnl": sim["pnl"], "roi": sim["roi"],
                "max_drawdown": sim["max_drawdown"],
                "halted": sim["halted"],
                "roi_ci90_event_clustered": sim["roi_ci90_event_clustered"],
            })

    for stop in (None, 5, 10, 15, 20):
        sim = simulate(df, threshold=0.04, high_threshold=0.08,
                       drawdown_stop=stop, bootstrap=args.bootstrap)
        report.setdefault("drawdown_controls", []).append({
            "stop": stop, "bets": sim["bets"], "staked": sim["staked"],
            "pnl": sim["pnl"], "roi": sim["roi"],
            "max_drawdown": sim["max_drawdown"], "halted": sim["halted"],
            "roi_ci90_event_clustered": sim["roi_ci90_event_clustered"],
        })

    report["period_stake_controls"] = []
    for label, part in (("dev", dev), ("holdout", holdout), ("all", df)):
        for threshold, cap in ((0.02, 2), (0.03, 2), (0.04, 2), (0.08, 2)):
            sim = simulate(part, threshold=threshold,
                           high_threshold=max(0.08, threshold * 2),
                           event_cap=cap, bootstrap=args.bootstrap)
            report["period_stake_controls"].append({
                "period": label, "threshold": threshold, "event_cap": cap,
                "bets": sim["bets"], "staked": sim["staked"],
                "pnl": sim["pnl"], "roi": sim["roi"],
                "roi_ci90_event_clustered": sim["roi_ci90_event_clustered"],
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    flat = pd.DataFrame(report["stake_controls"])
    flat.to_csv(args.table, index=False)
    print(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.output} and {args.table}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="production_validation.csv")
    ap.add_argument("--holdout-start", default="2025-01-01")
    ap.add_argument("--calibration-train-end", default="2024-01-01")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--output", default="backtest_experiments.json")
    ap.add_argument("--table", default="backtest_stake_controls.csv")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
