"""Walk-forward audit using only verified pre-event entry snapshots.

This validator is intentionally separate from ``validate_production.py``.
Entry prices are the only market inputs. Later API snapshots are benchmark-
only close proxies used for CLV diagnostics and never enter model training.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from backtest import american_to_prob
from config import BOOTSTRAP_MODELS, EDGE_RULE, MODEL_VERSION
from features_v3 import build_features_v3
from identity import norm_name
from production import (event_pnl, event_seed, fit_ensemble,
                        predict_probabilities, score_bets)


DEFAULT_BLEND_MARKET_WEIGHTS = (0.90, 0.75, 0.50)
PROMOTION_MIN_EVENTS = 50
PROMOTION_MIN_BETS = 200
PROMOTION_MIN_CLV_BETS = 200


def _pair(a, b):
    return "|".join(sorted((norm_name(a), norm_name(b))))


def _prob_metrics(y, probability):
    p = np.clip(np.asarray(probability, dtype=float), 0.001, 0.999)
    target = np.asarray(y, dtype=int)
    return {
        "log_loss": float(log_loss(target, p)),
        "brier": float(brier_score_loss(target, p)),
        "accuracy": float(accuracy_score(target, p >= 0.5)),
    }


def clustered_ci(events, n=10000, seed=0):
    """Cluster-bootstrap ROI by card, preserving within-card dependence."""
    events = events.loc[events["staked"] > 0].reset_index(drop=True)
    if not len(events):
        return [None, None]
    rng = np.random.default_rng(seed)
    roi = []
    for _ in range(int(n)):
        sample = events.iloc[rng.integers(0, len(events), len(events))]
        roi.append(sample["pnl"].sum() / sample["staked"].sum())
    return [float(np.percentile(roi, 5)), float(np.percentile(roi, 95))]


def clustered_mean_ci(rows, value_col, cluster_col="date", n=10000, seed=0):
    """Cluster-bootstrap a row-level mean by resampling whole cards."""
    usable = rows.dropna(subset=[cluster_col, value_col])
    if not len(usable):
        return [None, None]
    cards = usable.groupby(cluster_col)[value_col].agg(["sum", "count"]).reset_index()
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(n)):
        sample = cards.iloc[rng.integers(0, len(cards), len(cards))]
        means.append(sample["sum"].sum() / sample["count"].sum())
    return [float(np.percentile(means, 5)), float(np.percentile(means, 95))]


def _devig_prob_a(odds_a, odds_b):
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    return pa / (pa + pb)


def _parse_market_weights(value):
    try:
        weights = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("blend weights must be comma-separated numbers") from exc
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise argparse.ArgumentTypeError("blend weights must be between 0 and 1")
    return tuple(dict.fromkeys(weights))


def load_entry_matched(fights_path, history_path, min_entry_hours=24.0,
                       max_close_hours=12.0):
    fights = pd.read_csv(fights_path, parse_dates=["date"])
    feats, _ = build_features_v3(fights)
    feats["pair"] = [_pair(a, b) for a, b in zip(feats["fighter_a"], feats["fighter_b"])]
    feats["date"] = pd.to_datetime(feats["date"], utc=True)

    history = pd.read_csv(history_path)
    required = {
        "date", "fighter_a", "fighter_b", "commence_time",
        "entry_snapshot_ts", "entry_lead_hours", "entry_n_books",
        "entry_odds_a", "entry_odds_b",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    for column in ["date", "commence_time", "entry_snapshot_ts", "close_snapshot_ts"]:
        if column in history:
            history[column] = pd.to_datetime(history[column], utc=True, errors="coerce")
    numeric = [
        "entry_lead_hours", "entry_n_books", "entry_odds_a", "entry_odds_b",
        "entry_prob_a", "close_lead_hours", "close_n_books", "close_odds_a",
        "close_odds_b", "close_prob_a",
    ]
    for column in numeric:
        if column in history:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    if history[list(required - {"fighter_a", "fighter_b"})].isna().any().any():
        raise ValueError("history contains an incomplete entry snapshot")
    if not (history["entry_snapshot_ts"] < history["commence_time"]).all():
        raise ValueError("history contains a non-pre-event entry snapshot")
    if (history["entry_lead_hours"] < min_entry_hours).any():
        raise ValueError(f"history contains an entry less than {min_entry_hours:g} hours pre-event")

    close_required = {
        "close_snapshot_ts", "close_lead_hours", "close_n_books",
        "close_odds_a", "close_odds_b",
    }
    has_close = close_required <= set(history.columns)
    if has_close:
        close_columns = sorted(close_required)
        close_any = history[close_columns].notna().any(axis=1)
        close_complete = history[close_columns].notna().all(axis=1)
        if (close_any & ~close_complete).any():
            raise ValueError("history contains a partial close proxy")
        with_close = close_complete
        valid_order = (
            (history.loc[with_close, "entry_snapshot_ts"]
             < history.loc[with_close, "close_snapshot_ts"])
            & (history.loc[with_close, "close_snapshot_ts"]
               < history.loc[with_close, "commence_time"])
        )
        if not valid_order.all():
            raise ValueError("history contains an invalid close proxy time")
        if (history.loc[with_close, "close_lead_hours"] > max_close_hours).any():
            raise ValueError(
                f"history contains a close proxy more than {max_close_hours:g} hours pre-event"
            )

    history["pair"] = [_pair(a, b) for a, b in zip(history["fighter_a"], history["fighter_b"])]
    history = history.sort_values("entry_snapshot_ts")
    history = history.drop_duplicates(["date", "pair"], keep="last")
    history_columns = [
        "date", "pair", "commence_time", "entry_snapshot_ts",
        "entry_lead_hours", "entry_n_books", "entry_odds_a", "entry_odds_b",
    ]
    if "entry_prob_a" in history:
        history_columns.append("entry_prob_a")
    if "entry_source" in history:
        history_columns.append("entry_source")
    if has_close:
        history_columns.extend(sorted(close_required))
        if "close_prob_a" in history:
            history_columns.append("close_prob_a")
        if "close_source" in history:
            history_columns.append("close_source")

    matched = feats.merge(history[history_columns], on=["date", "pair"], how="inner")
    if not len(matched):
        raise ValueError("no entry snapshots matched the feature table")
    matched = matched.rename(columns={"entry_odds_a": "R_odds", "entry_odds_b": "B_odds"})
    matched["pr_raw"] = american_to_prob(matched["R_odds"])
    matched["pb_raw"] = american_to_prob(matched["B_odds"])
    fallback_entry = matched["pr_raw"] / (matched["pr_raw"] + matched["pb_raw"])
    if "entry_prob_a" in matched:
        matched["p_line"] = matched["entry_prob_a"].fillna(fallback_entry)
    else:
        matched["p_line"] = fallback_entry
    if not matched["p_line"].between(0, 1, inclusive="neither").all():
        raise ValueError("history contains an invalid entry probability")

    if has_close:
        fallback_close = _devig_prob_a(matched["close_odds_a"], matched["close_odds_b"])
        if "close_prob_a" in matched:
            matched["p_close_line"] = matched["close_prob_a"].fillna(
                pd.Series(fallback_close, index=matched.index)
            )
        else:
            matched["p_close_line"] = fallback_close
    matched["line_logit"] = logit(matched["p_line"].clip(0.02, 0.98))
    matched["line_abs"] = matched["line_logit"].abs()
    matched["y"] = matched["target"].astype(int)
    return matched.sort_values(["commence_time", "date"]).reset_index(drop=True)


def blend_benchmarks(pred, market_weights, event_bootstrap):
    """Evaluate fixed market anchors without promoting the observed winner."""
    rows = []
    for market_weight in market_weights:
        model_weight = round(1.0 - market_weight, 10)
        probability = (
            market_weight * pred["p_line"].to_numpy()
            + model_weight * pred["p_model"].to_numpy()
        )
        scored = score_bets(pred, probability, model_weight * pred["se"].to_numpy())
        scored["pnl"] = event_pnl(scored)
        scored["date"] = pred["date"].to_numpy()
        cards = scored.groupby("date", as_index=False).agg(
            staked=("stake", "sum"), pnl=("pnl", "sum")
        )
        staked = float(scored["stake"].sum())
        pnl = float(scored["pnl"].sum())
        metrics = _prob_metrics(pred["y"], probability)
        result = {
            "name": (
                f"market_{round(100 * market_weight):d}_"
                f"model_{round(100 * model_weight):d}"
            ),
            "market_weight": market_weight,
            "model_weight": model_weight,
            **metrics,
            "bets": int((scored["stake"] > 0).sum()),
            "staked": staked,
            "pnl": pnl,
            "roi": pnl / staked if staked else None,
            "roi_ci90_event_clustered": clustered_ci(cards, event_bootstrap),
        }
        if "p_close_line" in scored:
            close_rows = scored["p_close_line"].notna()
            active_close = (scored["stake"] > 0) & close_rows
            entry_pick = np.where(
                scored["pick_side"].eq("A"),
                scored["p_line"], 1.0 - scored["p_line"],
            )
            close_pick = np.where(
                scored["pick_side"].eq("A"),
                scored["p_close_line"], 1.0 - scored["p_close_line"],
            )
            scored["blend_clv_prob_points"] = (close_pick - entry_pick) * 100.0
            clv_rows = scored.loc[
                active_close, ["date", "blend_clv_prob_points"]
            ]
            clv = clv_rows["blend_clv_prob_points"]
            result["closing_line_value"] = {
                "active_bets_with_close": int(len(clv)),
                "mean_clv_prob_points": float(clv.mean()) if len(clv) else None,
                "mean_clv_prob_points_ci90_event_clustered": clustered_mean_ci(
                    clv_rows, "blend_clv_prob_points", n=event_bootstrap
                ),
                "positive_clv_rate": (
                    float((clv > 0).mean()) if len(clv) else None
                ),
            }
        rows.append(result)
    return sorted(rows, key=lambda row: row["log_loss"])


def yearly_metrics(pred):
    rows = []
    for year, frame in pred.groupby(pd.to_datetime(pred["date"]).dt.year):
        model = _prob_metrics(frame["y"], frame["p_model"])
        market = _prob_metrics(frame["y"], frame["p_line"])
        rows.append({
            "year": int(year),
            "fights": int(len(frame)),
            "model_log_loss": model["log_loss"],
            "entry_market_log_loss": market["log_loss"],
            "model_minus_market_log_loss": model["log_loss"] - market["log_loss"],
            "close_proxy_fights": int(frame.get("p_close_line", pd.Series(dtype=float)).notna().sum()),
        })
    return rows


def promotion_gate(report):
    clv = report.get("closing_line_value", {})
    if not isinstance(clv, dict):
        clv = {}
    roi_ci = report.get("roi_ci90_event_clustered", [None, None])
    clv_ci = clv.get("mean_clv_prob_points_ci90_event_clustered", [None, None])
    checks = {
        "minimum_events": {
            "passed": report.get("events", 0) >= PROMOTION_MIN_EVENTS,
            "observed": report.get("events", 0),
            "required": f">= {PROMOTION_MIN_EVENTS}",
        },
        "minimum_bets": {
            "passed": report.get("bets", 0) >= PROMOTION_MIN_BETS,
            "observed": report.get("bets", 0),
            "required": f">= {PROMOTION_MIN_BETS}",
        },
        "positive_roi_interval": {
            "passed": roi_ci[0] is not None and roi_ci[0] > 0,
            "observed": roi_ci,
            "required": "90% event-clustered ROI lower bound > 0",
        },
        "model_beats_entry_market": {
            "passed": report["log_loss_model"] < report["log_loss_entry_market"],
            "observed": report["log_loss_model"] - report["log_loss_entry_market"],
            "required": "model-minus-market log loss < 0",
        },
        "minimum_active_clv_bets": {
            "passed": clv.get("active_bets_with_close", 0) >= PROMOTION_MIN_CLV_BETS,
            "observed": clv.get("active_bets_with_close", 0),
            "required": f">= {PROMOTION_MIN_CLV_BETS}",
        },
        "positive_clv_interval": {
            "passed": clv_ci[0] is not None and clv_ci[0] > 0,
            "observed": clv_ci,
            "required": "90% event-clustered mean CLV lower bound > 0",
        },
        "positive_clv_rate": {
            "passed": (clv.get("positive_clv_rate_active_bets") is not None
                       and clv["positive_clv_rate_active_bets"] > 0.55),
            "observed": clv.get("positive_clv_rate_active_bets"),
            "required": "> 0.55",
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "candidate_for_manual_review" if passed else "paper_only",
        "checks": checks,
        "note": "No gate result enables automated wagering.",
    }


def run(args):
    matched = load_entry_matched(
        args.fights, args.history, args.min_entry_hours, args.max_close_hours
    )
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    window = matched[
        (matched["commence_time"] >= start) & (matched["commence_time"] < end)
    ]
    rows = []
    events = []
    for date in sorted(window["date"].dropna().unique()):
        test = window[window["date"] == date].copy()
        event_start = test["commence_time"].min()
        train = matched[matched["commence_time"] < event_start]
        if len(train) < args.min_train:
            continue
        models = fit_ensemble(train, n_models=args.models, seed=event_seed(event_start))
        probability, se = predict_probabilities(models, test)
        scored = score_bets(test, probability, se)
        scored["pnl"] = event_pnl(scored)
        scored["date"] = date
        rows.append(scored)
        events.append({
            "date": str(pd.Timestamp(date).date()),
            "fights": int(len(scored)),
            "bets": int((scored["stake"] > 0).sum()),
            "staked": float(scored["stake"].sum()),
            "pnl": float(scored["pnl"].sum()),
        })
    if not rows:
        raise SystemExit("No events met --min-train; lower it or move --start later.")

    pred = pd.concat(rows, ignore_index=True)
    cards = pd.DataFrame(events)
    active = pred["stake"] > 0
    staked = float(pred["stake"].sum())
    pnl = float(pred["pnl"].sum())
    model_metrics = _prob_metrics(pred["y"], pred["p_model"])
    market_metrics = _prob_metrics(pred["y"], pred["p_line"])
    blends = blend_benchmarks(pred, args.blend_market_weights, args.event_bootstrap)

    report = {
        "model": MODEL_VERSION + "-entry-history-candidate",
        "entry_only": True,
        "close_is_benchmark_only": "p_close_line" in pred,
        "close_definition": (
            "latest verified pre-card archive/API proxy, not an official sportsbook close"
        ),
        "start": args.start,
        "end": args.end,
        "events": int(len(cards)),
        "fights": int(len(pred)),
        "bets": int(active.sum()),
        "staked": staked,
        "pnl": pnl,
        "roi": pnl / staked if staked else None,
        "roi_ci90_event_clustered": clustered_ci(cards, args.event_bootstrap),
        "log_loss_model": model_metrics["log_loss"],
        "log_loss_entry_market": market_metrics["log_loss"],
        "brier_model": model_metrics["brier"],
        "brier_entry_market": market_metrics["brier"],
        "accuracy_model": model_metrics["accuracy"],
        "accuracy_entry_market": market_metrics["accuracy"],
        "median_entry_lead_hours": float(pred["entry_lead_hours"].median()),
        "median_entry_books": float(pred["entry_n_books"].median()),
        "entry_sources": (
            pred["entry_source"].fillna("unknown").value_counts().astype(int).to_dict()
            if "entry_source" in pred else {"unknown": int(len(pred))}
        ),
        "edge_rule": EDGE_RULE,
        "blend_benchmarks": blends,
        "best_observed_blend": blends[0] if blends else None,
        "blend_note": (
            "Fixed research benchmarks only. The best observed blend is selected on this "
            "audit sample and is not automatically promoted."
        ),
        "yearly_metrics": yearly_metrics(pred),
    }

    close_report = {
        "fights_with_close": 0,
        "close_coverage_rate": 0.0,
        "closing_line_value": "unavailable: history has no later close proxy",
    }
    if "p_close_line" in pred:
        close_rows = pred["p_close_line"].notna()
        pred["entry_pick_prob"] = np.where(
            pred["pick_side"].eq("A"), pred["p_line"], 1.0 - pred["p_line"]
        )
        pred["close_pick_prob"] = np.where(
            pred["pick_side"].eq("A"), pred["p_close_line"], 1.0 - pred["p_close_line"]
        )
        pred["clv_prob_points"] = (
            pred["close_pick_prob"] - pred["entry_pick_prob"]
        ) * 100.0
        active_clv_rows = pred.loc[active & close_rows, ["date", "clv_prob_points"]]
        active_clv = active_clv_rows["clv_prob_points"]
        close_metrics = (
            _prob_metrics(pred.loc[close_rows, "y"], pred.loc[close_rows, "p_close_line"])
            if close_rows.any() else {"log_loss": None, "brier": None, "accuracy": None}
        )
        close_report = {
            "log_loss_close_market": close_metrics["log_loss"],
            "brier_close_market": close_metrics["brier"],
            "accuracy_close_market": close_metrics["accuracy"],
            "fights_with_close": int(close_rows.sum()),
            "close_coverage_rate": float(close_rows.mean()),
            "median_close_lead_minutes": (
                float(pred.loc[close_rows, "close_lead_hours"].median() * 60.0)
                if close_rows.any() else None
            ),
            "median_close_books": (
                float(pred.loc[close_rows, "close_n_books"].median())
                if close_rows.any() else None
            ),
            "close_sources": (
                pred.loc[close_rows, "close_source"].fillna("unknown")
                .value_counts().astype(int).to_dict()
                if "close_source" in pred else {"unknown": int(close_rows.sum())}
            ),
            "closing_line_value": {
                "mean_clv_prob_points_active_bets": (
                    float(active_clv.mean()) if len(active_clv) else None
                ),
                "mean_clv_prob_points_ci90_event_clustered": clustered_mean_ci(
                    active_clv_rows, "clv_prob_points", n=args.event_bootstrap
                ),
                "positive_clv_rate_active_bets": (
                    float((active_clv > 0).mean()) if len(active_clv) else None
                ),
                "active_bets_with_close": int(len(active_clv)),
            },
        }
    report.update(close_report)
    report["promotion_gate"] = promotion_gate(report)
    report["promotion"] = report["promotion_gate"]["status"]

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
    parser.add_argument(
        "--blend-market-weights",
        type=_parse_market_weights,
        default=DEFAULT_BLEND_MARKET_WEIGHTS,
        help="comma-separated fixed market weights for research blends",
    )
    parser.add_argument("--predictions", default="historical_entry_validation.csv")
    parser.add_argument("--report", default="historical_entry_validation.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
