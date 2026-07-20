"""Validate the exact deployed model with event-clustered uncertainty.

Usage:
    python validate_production.py
    python validate_production.py --start 2025-01-01 --models 10 --event-bootstrap 2000

Every event is predicted using only rows dated before that event.  The model,
symmetry handling, bootstrap SE, and 1u/2u net-edge stakes are the same code
used by ``predict_card.py``.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from config import BOOTSTRAP_MODELS, EDGE_RULE, MODEL_VERSION
from features_v3 import build_features_v3
from pipeline import load_matched_cached
from production import (event_pnl, event_seed, fit_ensemble,
                        predict_probabilities, score_bets)


def clustered_ci(events, n=10000, seed=0):
    """Cluster-bootstrap ROI by event, preserving within-event dependence."""
    events = events.loc[events["staked"] > 0].reset_index(drop=True)
    if not len(events):
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n)):
        sample = events.iloc[rng.integers(0, len(events), len(events))]
        draws.append(sample["pnl"].sum() / sample["staked"].sum())
    return [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]


def run(args):
    matched, _ = load_matched_cached(build_features_v3, "v3", bout_cols=[])
    matched = matched.sort_values(["date"]).reset_index(drop=True)
    window = matched[(matched["date"] >= args.start)
                     & (matched["date"] < args.end)]
    rows = []
    event_rows = []
    dates = sorted(window["date"].dropna().unique())
    for date in dates:
        train = matched[matched["date"] < date]
        test = window[window["date"] == date].copy()
        if len(train) < args.min_train:
            continue
        models = fit_ensemble(train, n_models=args.models, seed=event_seed(date))
        p, se = predict_probabilities(models, test)
        scored = score_bets(test, p, se)
        pnl = event_pnl(scored)
        scored["p_line"] = test["p_line"].to_numpy()
        scored["pnl"] = pnl
        scored["date"] = date
        rows.append(scored)
        event_rows.append({
            "date": str(pd.Timestamp(date).date()),
            "n_fights": int(len(scored)),
            "n_bets": int((scored["stake"] > 0).sum()),
            "staked": float(scored["stake"].sum()),
            "pnl": float(pnl.sum()),
            "source": ",".join(sorted(scored["odds_source"].astype(str).unique())),
        })
        if args.verbose:
            print(f"{pd.Timestamp(date).date()}  fights={len(scored):3d}  "
                  f"bets={(scored['stake'] > 0).sum():2d}  pnl={pnl.sum():+.2f}")

    if not rows:
        raise SystemExit("No validation events met --min-train; widen the window.")
    pred = pd.concat(rows, ignore_index=True)
    events = pd.DataFrame(event_rows)
    bet_mask = pred["stake"] > 0
    y = pred["y"].to_numpy()
    p = pred["p_model"].to_numpy()
    staked = float(pred["stake"].sum())
    pnl = float(pred["pnl"].sum())
    source_stats = {}
    for source, group in pred.groupby("odds_source", dropna=False):
        active = group["stake"] > 0
        source_stats[str(source)] = {
            "fights": int(len(group)),
            "bets": int(active.sum()),
            "staked": float(group["stake"].sum()),
            "pnl": float(group["pnl"].sum()),
            "roi": (float(group["pnl"].sum() / group["stake"].sum())
                    if group["stake"].sum() else None),
        }
    report = {
        "model": MODEL_VERSION,
        "start": args.start,
        "end": args.end,
        "events": int(len(events)),
        "fights": int(len(pred)),
        "bets": int(bet_mask.sum()),
        "staked": staked,
        "pnl": pnl,
        "roi": pnl / staked if staked else None,
        "roi_ci90_event_clustered": clustered_ci(events, args.event_bootstrap),
        "log_loss_model": float(log_loss(y, p)),
        "log_loss_line": float(log_loss(y, pred["p_line"])),
        "accuracy_model": float(accuracy_score(y, p >= 0.5)),
        "accuracy_line": float(accuracy_score(y, pred["p_line"] >= 0.5)),
        "edge_rule": EDGE_RULE,
        "bootstrap_models_per_event": args.models,
        "min_train": args.min_train,
        "odds_sources": source_stats,
    }
    ci = report["roi_ci90_event_clustered"]
    report["live_gate"] = {
        "status": ("tiny_stakes_only"
                    if len(events) >= 50 and int(bet_mask.sum()) >= 200
                    and ci[0] is not None and ci[0] > 0
                    else "paper_only"),
        "reason": ("The event-clustered ROI interval must be wholly positive "
                   "with at least 50 events and 200 bets before live stakes."),
    }
    pred.to_csv(args.predictions, index=False)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    from model_manifest import write_manifest
    write_manifest()
    print(json.dumps(report, indent=2))
    print(f"wrote {args.predictions} and {args.report}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--models", type=int, default=BOOTSTRAP_MODELS,
                    help="bootstrap models per event (defaults to config.py)")
    ap.add_argument("--event-bootstrap", type=int, default=10000)
    ap.add_argument("--min-train", type=int, default=2000)
    ap.add_argument("--predictions", default="production_validation.csv")
    ap.add_argument("--report", default="production_validation.json")
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
