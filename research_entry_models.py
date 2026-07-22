"""Leakage-safe entry-price model research.

This module compares the current entry-trained candidate with a constrained
market-offset model and two market/model blends. It never changes the model
used by ``predict_card.py``.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from config import BOOTSTRAP_MODELS, EDGE_RULE, MODEL_VERSION
from production import (DIFF_FEATURES, event_pnl, event_seed, fit_ensemble,
                        predict_probabilities, score_bets)
from validate_entry_history import (clustered_ci, clustered_mean_ci,
                                    load_entry_matched, promotion_gate,
                                    _prob_metrics)


OFFSET_FEATURES = tuple(DIFF_FEATURES)
DEFAULT_OFFSET_L2 = 0.02
DEFAULT_BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 0.90, 1.0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MarketOffsetModel:
    """A symmetric logistic correction around the entry market logit."""

    scale: np.ndarray
    coefficients: np.ndarray
    l2: float

    def predict_proba(self, frame):
        x = frame.loc[:, list(OFFSET_FEATURES)].to_numpy(dtype=float)
        offset = frame["line_logit"].to_numpy(dtype=float)
        return expit(offset + (x / self.scale) @ self.coefficients)


def _offset_objective(coefficients, x, y, offset, l2):
    eta = offset + x @ coefficients
    loss = np.mean(np.logaddexp(0.0, eta) - y * eta)
    loss += 0.5 * l2 * float(coefficients @ coefficients)
    probability = expit(eta)
    gradient = x.T @ (probability - y) / len(y) + l2 * coefficients
    return float(loss), gradient


def fit_market_offset(train, l2=DEFAULT_OFFSET_L2):
    """Fit a no-intercept correction while fixing the market coefficient at 1."""
    x = train.loc[:, list(OFFSET_FEATURES)].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    offset = train["line_logit"].to_numpy(dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()
            and np.isfinite(offset).all()):
        raise ValueError("market-offset training data contains non-finite values")

    # RMS scaling, without centering, preserves f(-line, -x) = 1 - f(line, x).
    scale = np.sqrt(np.mean(np.square(x), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    scaled = x / scale
    result = minimize(
        _offset_objective,
        np.zeros(scaled.shape[1], dtype=float),
        args=(scaled, y, offset, float(l2)),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"market-offset fit failed: {result.message}")
    return MarketOffsetModel(scale, np.asarray(result.x, dtype=float), float(l2))


def fit_market_offset_ensemble(train, n_models=BOOTSTRAP_MODELS, seed=0,
                               l2=DEFAULT_OFFSET_L2):
    """Fit the candidate plus row-bootstrap models for uncertainty."""
    train = train.reset_index(drop=True)
    models = [fit_market_offset(train, l2=l2)]
    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(n_models))):
        sample = train.iloc[rng.integers(0, len(train), len(train))]
        models.append(fit_market_offset(sample, l2=l2))
    return models


def predict_market_offset(models, frame):
    values = np.asarray([model.predict_proba(frame) for model in models])
    uncertainty = (
        values.std(axis=0, ddof=0) if len(values) > 1 else np.zeros(len(frame))
    )
    return values[0], uncertainty


def select_past_market_weight(history, event_start, grid=DEFAULT_BLEND_GRID,
                              min_history=200, default=1.0):
    """Choose a blend on completed prior-card OOF predictions only."""
    event_start = pd.Timestamp(event_start)
    prior = history[pd.to_datetime(history["commence_time"], utc=True) < event_start]
    prior = prior.dropna(subset=["y", "p_line", "p_current"])
    if len(prior) < int(min_history):
        return float(default), {
            "history_rows": int(len(prior)),
            "selection": "default_insufficient_history",
            "selected_log_loss": None,
        }

    rows = []
    for weight in grid:
        probability = (
            float(weight) * prior["p_line"].to_numpy()
            + (1.0 - float(weight)) * prior["p_current"].to_numpy()
        )
        rows.append((float(_prob_metrics(prior["y"], probability)["log_loss"]),
                     float(weight)))
    loss, weight = min(rows, key=lambda item: (item[0], -item[1]))
    return weight, {
        "history_rows": int(len(prior)),
        "selection": "prior_oof_log_loss",
        "selected_log_loss": loss,
    }


def _candidate_report(pred, probability_col, uncertainty_col, name,
                      event_bootstrap):
    probability = pred[probability_col].to_numpy(dtype=float)
    uncertainty = pred[uncertainty_col].to_numpy(dtype=float)
    scored = score_bets(pred, probability, uncertainty)
    scored["pnl"] = event_pnl(scored)
    cards = scored.groupby("date", as_index=False).agg(
        staked=("stake", "sum"), pnl=("pnl", "sum")
    )
    staked = float(scored["stake"].sum())
    pnl = float(scored["pnl"].sum())
    metrics = _prob_metrics(scored["y"], probability)
    market = _prob_metrics(scored["y"], scored["p_line"])
    report = {
        "name": name,
        "events": int(scored["date"].nunique()),
        "fights": int(len(scored)),
        "bets": int((scored["stake"] > 0).sum()),
        "staked": staked,
        "pnl": pnl,
        "roi": pnl / staked if staked else None,
        "roi_ci90_event_clustered": clustered_ci(cards, event_bootstrap),
        "log_loss": metrics["log_loss"],
        "log_loss_model": metrics["log_loss"],
        "log_loss_entry_market": market["log_loss"],
        "model_minus_market_log_loss": metrics["log_loss"] - market["log_loss"],
        "brier": metrics["brier"],
        "accuracy": metrics["accuracy"],
    }

    if "p_close_line" in scored:
        active_close = (scored["stake"] > 0) & scored["p_close_line"].notna()
        entry_pick = np.where(
            scored["pick_side"].eq("A"), scored["p_line"], 1.0 - scored["p_line"]
        )
        close_pick = np.where(
            scored["pick_side"].eq("A"),
            scored["p_close_line"],
            1.0 - scored["p_close_line"],
        )
        scored["clv_prob_points"] = (close_pick - entry_pick) * 100.0
        clv_rows = scored.loc[active_close, ["date", "clv_prob_points"]]
        clv = clv_rows["clv_prob_points"]
        report["closing_line_value"] = {
            "active_bets_with_close": int(len(clv)),
            "mean_clv_prob_points_active_bets": (
                float(clv.mean()) if len(clv) else None
            ),
            "mean_clv_prob_points_ci90_event_clustered": clustered_mean_ci(
                clv_rows, "clv_prob_points", n=event_bootstrap
            ),
            "positive_clv_rate_active_bets": (
                float((clv > 0).mean()) if len(clv) else None
            ),
        }

    report["promotion_gate"] = promotion_gate(report)
    report["promotion"] = report["promotion_gate"]["status"]
    audit = scored[["pick_side", "stake", "edge", "net_edge", "pnl"]].copy()
    audit.columns = [f"{name}_{column}" for column in audit.columns]
    return report, audit


def _parse_grid(value):
    try:
        grid = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("blend grid must contain numbers") from exc
    if not grid or any(weight < 0 or weight > 1 for weight in grid):
        raise argparse.ArgumentTypeError("blend weights must be between 0 and 1")
    return grid


def run(args):
    matched = load_entry_matched(
        args.fights, args.history, args.min_entry_hours, args.max_close_hours
    )
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    prior_oof = []
    evaluated = []
    weight_trace = []

    for date in sorted(matched["date"].dropna().unique()):
        test = matched[matched["date"] == date].copy()
        event_start = test["commence_time"].min()
        if event_start >= end:
            continue
        train = matched[matched["commence_time"] < event_start]
        if len(train) < args.min_train:
            continue

        current_models = fit_ensemble(
            train, n_models=args.models, seed=event_seed(event_start)
        )
        p_current, se_current = predict_probabilities(current_models, test)
        history = (
            pd.concat(prior_oof, ignore_index=True)
            if prior_oof else pd.DataFrame(
                columns=["commence_time", "y", "p_line", "p_current"]
            )
        )
        weight, selection = select_past_market_weight(
            history,
            event_start,
            grid=args.blend_grid,
            min_history=args.blend_min_history,
            default=args.blend_default,
        )
        prior_oof.append(pd.DataFrame({
            "commence_time": test["commence_time"].to_numpy(),
            "y": test["y"].to_numpy(),
            "p_line": test["p_line"].to_numpy(),
            "p_current": p_current,
        }))

        if event_start < start:
            continue
        offset_models = fit_market_offset_ensemble(
            train,
            n_models=args.models,
            seed=event_seed(event_start, namespace="entry-market-offset"),
            l2=args.offset_l2,
        )
        p_offset, se_offset = predict_market_offset(offset_models, test)
        test["p_current"] = p_current
        test["se_current"] = se_current
        test["p_market_offset"] = p_offset
        test["se_market_offset"] = se_offset
        test["p_fixed_50"] = 0.5 * test["p_line"] + 0.5 * p_current
        test["se_fixed_50"] = 0.5 * se_current
        test["p_nested_blend"] = weight * test["p_line"] + (1.0 - weight) * p_current
        test["se_nested_blend"] = (1.0 - weight) * se_current
        test["nested_market_weight"] = weight
        test["nested_history_rows"] = selection["history_rows"]
        evaluated.append(test)
        weight_trace.append({
            "date": str(pd.Timestamp(date).date()),
            "event_start": str(event_start),
            "market_weight": weight,
            **selection,
        })

    if not evaluated:
        raise SystemExit("No events met the research window and --min-train.")

    pred = pd.concat(evaluated, ignore_index=True)
    candidates = [
        ("current_entry_refit", "p_current", "se_current"),
        ("market_offset", "p_market_offset", "se_market_offset"),
        ("fixed_50_50", "p_fixed_50", "se_fixed_50"),
        ("nested_past_only_blend", "p_nested_blend", "se_nested_blend"),
    ]
    results = []
    audits = []
    for name, probability_col, uncertainty_col in candidates:
        result, audit = _candidate_report(
            pred, probability_col, uncertainty_col, name, args.event_bootstrap
        )
        results.append(result)
        audits.append(audit)
    for audit in audits:
        pred = pd.concat([pred.reset_index(drop=True), audit.reset_index(drop=True)], axis=1)

    market_metrics = _prob_metrics(pred["y"], pred["p_line"])
    weight_counts = (
        pd.Series([row["market_weight"] for row in weight_trace])
        .value_counts().sort_index()
    )
    report = {
        "experiment": "entry-market-offset-v1",
        "production_model_unchanged": MODEL_VERSION,
        "status": "research_only",
        "entry_only_training": True,
        "close_is_benchmark_only": "p_close_line" in pred,
        "start": args.start,
        "end": args.end,
        "min_train": args.min_train,
        "events": int(pred["date"].nunique()),
        "fights": int(len(pred)),
        "source_sha256": {
            "fights": _sha256(args.fights),
            "history": _sha256(args.history),
            "research_code": _sha256(__file__),
            "requirements": _sha256("requirements.txt"),
        },
        "entry_market": market_metrics,
        "offset": {
            "fixed_market_logit_coefficient": 1.0,
            "intercept": 0.0,
            "features": list(OFFSET_FEATURES),
            "l2": args.offset_l2,
            "symmetry": "f(-line_logit, -features) = 1 - f(line_logit, features)",
        },
        "nested_blend": {
            "objective": "prior out-of-fold fight log loss",
            "grid": list(args.blend_grid),
            "minimum_prior_rows": args.blend_min_history,
            "default_market_weight": args.blend_default,
            "future_rows_available_to_selector": 0,
            "event_weight_counts": {
                str(float(weight)): int(count) for weight, count in weight_counts.items()
            },
            "trace": weight_trace,
        },
        "candidates": results,
        "ranking_by_log_loss": [
            row["name"] for row in sorted(results, key=lambda item: item["log_loss"])
        ],
        "decision_rule": (
            "No candidate changes production automatically. A candidate must pass its "
            "promotion gate and then receive manual review."
        ),
    }
    pred.to_csv(args.predictions, index=False)
    with open(args.report, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument(
        "--history",
        default="data/odds_history/odds_history_entry_with_api_close.csv",
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--min-entry-hours", type=float, default=24.0)
    parser.add_argument("--max-close-hours", type=float, default=12.0)
    parser.add_argument("--models", type=int, default=BOOTSTRAP_MODELS)
    parser.add_argument("--event-bootstrap", type=int, default=10000)
    parser.add_argument("--offset-l2", type=float, default=DEFAULT_OFFSET_L2)
    parser.add_argument("--blend-grid", type=_parse_grid, default=DEFAULT_BLEND_GRID)
    parser.add_argument("--blend-min-history", type=int, default=200)
    parser.add_argument("--blend-default", type=float, default=1.0)
    parser.add_argument("--predictions", default="entry_model_research_predictions.csv")
    parser.add_argument("--report", default="entry_model_research.json")
    args = parser.parse_args()
    if not 0 <= args.blend_default <= 1:
        parser.error("--blend-default must be between 0 and 1")
    if args.offset_l2 < 0:
        parser.error("--offset-l2 cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
