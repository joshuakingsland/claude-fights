"""Leakage-resistant audit of proposed model and execution improvements.

The audit consumes existing out-of-fold prediction ledgers. It does not refit
or modify production-v3. Candidate choices are made on development data only,
then reported separately on a 2024 validation period and a 2025+ holdout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import american_to_prob
from config import EDGE_RULE
from identity import norm_name
from production import allocate_stakes, event_pnl
from validate_entry_history import clustered_ci, clustered_mean_ci


DEVELOPMENT_END = "2024-01-01"
HOLDOUT_START = "2025-01-01"
UNCERTAINTY_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)
MIN_BOOKS_GRID = (3, 5, 7, 9)
TIMING_FEATURES = (
    "gross_edge", "se", "entry_lead_hours", "entry_n_books",
    "model_confidence", "market_confidence", "pick_is_favorite",
    "entry_source_api",
)


def _float(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _period(frame, label):
    dates = pd.to_datetime(frame["date"], utc=True)
    development_end = pd.Timestamp(DEVELOPMENT_END, tz="UTC")
    holdout_start = pd.Timestamp(HOLDOUT_START, tz="UTC")
    if label == "development":
        return frame[dates < development_end].copy()
    if label == "validation":
        return frame[(dates >= development_end) & (dates < holdout_start)].copy()
    if label == "holdout":
        return frame[dates >= holdout_start].copy()
    if label == "post_development":
        return frame[dates >= development_end].copy()
    raise ValueError(f"unknown period: {label}")


def calibration_error(y, probability, bins=10):
    """Fixed-bin expected calibration error."""
    y = np.asarray(y, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bucket = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        selected = bucket == index
        if selected.any():
            error += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(error) if total else None


def probability_metrics(frame, probability):
    if not len(frame):
        return {"fights": 0, "log_loss": None, "brier": None, "ece_10": None}
    y = frame["y"].to_numpy(dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5)
    return {
        "fights": int(len(frame)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10": calibration_error(y, p),
    }


def paired_probability_delta_ci(frame, candidate, baseline, metric,
                                bootstrap=5000, seed=0):
    """Card-clustered CI for candidate-minus-baseline probability loss."""
    if not len(frame):
        return [None, None]
    y = frame["y"].to_numpy(dtype=float)
    candidate = np.clip(np.asarray(candidate, dtype=float), 1e-5, 1 - 1e-5)
    baseline = np.clip(np.asarray(baseline, dtype=float), 1e-5, 1 - 1e-5)
    if metric == "log_loss":
        candidate_loss = -(y * np.log(candidate) + (1.0 - y) * np.log(1.0 - candidate))
        baseline_loss = -(y * np.log(baseline) + (1.0 - y) * np.log(1.0 - baseline))
    elif metric == "brier":
        candidate_loss = np.square(y - candidate)
        baseline_loss = np.square(y - baseline)
    else:
        raise ValueError(metric)
    rows = pd.DataFrame({
        "date": frame["date"].astype(str).to_numpy(),
        "delta": candidate_loss - baseline_loss,
    })
    cards = rows.groupby("date")["delta"].agg(["sum", "count"]).reset_index()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(bootstrap)):
        sample = cards.iloc[rng.integers(0, len(cards), len(cards))]
        values.append(sample["sum"].sum() / sample["count"].sum())
    return [float(np.percentile(values, 5)), float(np.percentile(values, 95))]


def fit_symmetric_temperature(frame, probability_col="p_model"):
    """Fit p'=sigmoid(scale*logit(p)); this preserves side-swap symmetry."""
    p = np.clip(frame[probability_col].to_numpy(dtype=float), 1e-5, 1 - 1e-5)
    y = frame["y"].to_numpy(dtype=float)
    z = logit(p)

    def objective(scale):
        return log_loss(y, expit(float(scale) * z), labels=[0, 1])

    result = minimize_scalar(objective, bounds=(0.25, 2.5), method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(result.x)


def apply_symmetric_temperature(probability, scale, uncertainty=None):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5)
    calibrated = expit(float(scale) * logit(probability))
    if uncertainty is None:
        return calibrated
    derivative = (
        float(scale) * calibrated * (1.0 - calibrated)
        / (probability * (1.0 - probability))
    )
    return calibrated, np.asarray(uncertainty, dtype=float) * np.abs(derivative)


def score_policy(frame, probability=None, uncertainty=None,
                 uncertainty_multiplier=1.0, eligible=None, odds_a=None,
                 odds_b=None, locked_pick=None):
    """Apply the current flat 1u/card-cap policy to alternate inputs."""
    out = frame.reset_index(drop=True).copy()
    probability = (
        out["p_model"].to_numpy(dtype=float)
        if probability is None else np.asarray(probability, dtype=float)
    )
    uncertainty = (
        out["se"].to_numpy(dtype=float)
        if uncertainty is None else np.asarray(uncertainty, dtype=float)
    )
    odds_a = (
        out["R_odds"].to_numpy(dtype=float)
        if odds_a is None else np.asarray(odds_a, dtype=float)
    )
    odds_b = (
        out["B_odds"].to_numpy(dtype=float)
        if odds_b is None else np.asarray(odds_b, dtype=float)
    )
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    edge_a = probability - pa
    edge_b = (1.0 - probability) - pb
    choose_a = edge_a >= edge_b if locked_pick is None else np.asarray(locked_pick) == "A"
    gross = np.where(choose_a, edge_a, edge_b)
    penalty = float(uncertainty_multiplier) * uncertainty
    net = gross - penalty
    eligible = np.ones(len(out), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    allocation_edge = np.where(eligible, net, -np.inf)
    groups = out["date"].astype(str).to_numpy()
    stake = allocate_stakes(allocation_edge, groups=groups)

    out["R_odds"] = odds_a
    out["B_odds"] = odds_b
    out["pr_raw"] = pa
    out["pb_raw"] = pb
    out["p_model"] = probability
    out["se"] = uncertainty
    out["uncertainty_penalty"] = penalty
    out["edge"] = gross
    out["net_edge"] = net
    out["pick_side"] = np.where(choose_a, "A", "B")
    out["qualified"] = eligible & (net >= EDGE_RULE)
    out["stake"] = stake
    out["pnl"] = event_pnl(out)
    return out


def policy_metrics(scored, bootstrap=5000, include_clv=True):
    active = scored["stake"] > 0
    cards = scored.groupby("date", as_index=False).agg(
        staked=("stake", "sum"), pnl=("pnl", "sum")
    )
    staked = float(scored["stake"].sum())
    pnl = float(scored["pnl"].sum())
    result = {
        "events": int(scored["date"].nunique()),
        "fights": int(len(scored)),
        "bets": int(active.sum()),
        "staked": staked,
        "pnl": pnl,
        "roi": pnl / staked if staked else None,
        "roi_ci90_event_clustered": clustered_ci(cards, n=bootstrap),
    }
    if include_clv and "p_close_line" in scored:
        close = scored["p_close_line"].notna()
        entry_pick = np.where(
            scored["pick_side"].eq("A"), scored["p_line"], 1.0 - scored["p_line"]
        )
        close_pick = np.where(
            scored["pick_side"].eq("A"),
            scored["p_close_line"], 1.0 - scored["p_close_line"],
        )
        clv_rows = scored.loc[active & close, ["date"]].copy()
        clv_rows["clv"] = (close_pick - entry_pick)[active & close] * 100.0
        result["clv"] = {
            "bets": int(len(clv_rows)),
            "mean_prob_points": _float(clv_rows["clv"].mean()) if len(clv_rows) else None,
            "ci90_event_clustered": clustered_mean_ci(
                clv_rows, "clv", n=bootstrap
            ),
            "positive_rate": _float((clv_rows["clv"] > 0).mean()) if len(clv_rows) else None,
        }
    return result


def _policy_periods(frame, scorer, bootstrap):
    return {
        period: policy_metrics(scorer(_period(frame, period)), bootstrap)
        for period in ("development", "validation", "holdout")
    }


def calibration_audit(frame, bootstrap):
    development = _period(frame, "development")
    scale = fit_symmetric_temperature(development)
    report = {
        "fit_period": f"date < {DEVELOPMENT_END}",
        "scale": scale,
        "interpretation": "scale < 1 shrinks probabilities toward 50%",
        "periods": {},
    }
    for period in ("development", "validation", "holdout"):
        part = _period(frame, period)
        calibrated, calibrated_se = apply_symmetric_temperature(
            part["p_model"], scale, part["se"]
        )
        raw_metrics = probability_metrics(part, part["p_model"])
        calibrated_metrics = probability_metrics(part, calibrated)
        report["periods"][period] = {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            "delta_calibrated_minus_raw": {
                key: calibrated_metrics[key] - raw_metrics[key]
                for key in ("log_loss", "brier", "ece_10")
            },
            "delta_ci90_event_clustered": {
                key: paired_probability_delta_ci(
                    part, calibrated, part["p_model"], key, bootstrap=bootstrap
                )
                for key in ("log_loss", "brier")
            },
            "raw_policy": policy_metrics(score_policy(part), bootstrap),
            "calibrated_policy": policy_metrics(
                score_policy(part, calibrated, calibrated_se), bootstrap
            ),
        }

    validation_ci = report["periods"]["validation"]["delta_ci90_event_clustered"]
    holdout_ci = report["periods"]["holdout"]["delta_ci90_event_clustered"]
    improves_both = all(
        validation_ci[key][1] < 0 and holdout_ci[key][1] < 0
        for key in ("log_loss", "brier")
    )
    report["verdict"] = "keep_as_challenger" if improves_both else "reject_for_now"
    report["gate"] = {
        "validation_and_holdout_log_loss_ci_below_zero": (
            validation_ci["log_loss"][1] < 0 and holdout_ci["log_loss"][1] < 0
        ),
        "validation_and_holdout_brier_ci_below_zero": (
            validation_ci["brier"][1] < 0 and holdout_ci["brier"][1] < 0
        ),
    }
    return report


def uncertainty_audit(frame, bootstrap):
    rows = {}
    for multiplier in UNCERTAINTY_MULTIPLIERS:
        rows[str(multiplier)] = _policy_periods(
            frame,
            lambda part, value=multiplier: score_policy(
                part, uncertainty_multiplier=value
            ),
            bootstrap,
        )
    development_roi = {
        multiplier: rows[str(multiplier)]["development"]["roi"]
        for multiplier in UNCERTAINTY_MULTIPLIERS
    }
    selected = max(
        UNCERTAINTY_MULTIPLIERS,
        key=lambda value: (
            -np.inf if development_roi[value] is None else development_roi[value],
            -abs(value - 1.0),
        ),
    )
    baseline = rows["1.0"]
    challenger = rows[str(selected)]
    improves_later = all(
        challenger[period]["roi"] is not None
        and baseline[period]["roi"] is not None
        and challenger[period]["roi"] > baseline[period]["roi"]
        for period in ("validation", "holdout")
    )
    enough_later_bets = all(
        challenger[period]["bets"] >= 50 for period in ("validation", "holdout")
    )
    return {
        "selection_rule": "highest development ROI; ties prefer multiplier nearest 1",
        "multipliers": rows,
        "development_selected_multiplier": selected,
        "gate": {
            "beats_multiplier_1_roi_in_validation_and_holdout": improves_later,
            "at_least_50_bets_in_validation_and_holdout": enough_later_bets,
        },
        "verdict": (
            "keep_as_challenger" if selected != 1.0 and improves_later and enough_later_bets
            else "reject_for_now"
        ),
    }


def book_count_audit(frame, bootstrap):
    rows = {}
    for minimum in MIN_BOOKS_GRID:
        rows[str(minimum)] = _policy_periods(
            frame,
            lambda part, value=minimum: score_policy(
                part, eligible=part["entry_n_books"].to_numpy(dtype=float) >= value
            ),
            bootstrap,
        )
    development_roi = {
        minimum: rows[str(minimum)]["development"]["roi"] for minimum in MIN_BOOKS_GRID
    }
    selected = max(
        MIN_BOOKS_GRID,
        key=lambda value: (
            -np.inf if development_roi[value] is None else development_roi[value],
            -value,
        ),
    )
    baseline = rows["3"]
    challenger = rows[str(selected)]
    improves_later = all(
        challenger[period]["roi"] is not None
        and baseline[period]["roi"] is not None
        and challenger[period]["roi"] > baseline[period]["roi"]
        for period in ("validation", "holdout")
    )
    enough_later_bets = all(
        challenger[period]["bets"] >= 50 for period in ("validation", "holdout")
    )
    return {
        "selection_rule": "highest development ROI; ties prefer broader coverage",
        "minimum_books": rows,
        "development_selected_minimum": selected,
        "gate": {
            "beats_three_book_baseline_in_validation_and_holdout": improves_later,
            "at_least_50_bets_in_validation_and_holdout": enough_later_bets,
        },
        "verdict": (
            "keep_as_challenger" if selected != 3 and improves_later and enough_later_bets
            else "reject_for_now"
        ),
    }


def _timing_frame(frame):
    usable = frame.dropna(subset=[
        "p_close_line", "close_odds_a", "close_odds_b", "entry_lead_hours",
        "entry_n_books",
    ]).copy()
    entry_scored = score_policy(usable)
    choose_a = entry_scored["pick_side"].eq("A").to_numpy()
    entry_pick = np.where(choose_a, usable["p_line"], 1.0 - usable["p_line"])
    close_pick = np.where(
        choose_a, usable["p_close_line"], 1.0 - usable["p_close_line"]
    )
    usable["locked_pick"] = entry_scored["pick_side"].to_numpy()
    usable["gross_edge"] = entry_scored["edge"].to_numpy()
    usable["model_confidence"] = np.abs(usable["p_model"] - 0.5)
    usable["market_confidence"] = np.abs(usable["p_line"] - 0.5)
    usable["pick_is_favorite"] = (entry_pick > 0.5).astype(float)
    usable["entry_source_api"] = usable.get(
        "entry_source", pd.Series("", index=usable.index)
    ).eq("odds_api_book_consensus").astype(float)
    usable["timing_target"] = close_pick - entry_pick
    return usable.sort_values(["date", "fight_id"]).reset_index(drop=True)


def walk_forward_timing_predictions(frame, min_train=200, alpha=10.0):
    """Predict signed CLV using only earlier close-covered fights."""
    rows = []
    dates = pd.to_datetime(frame["date"], utc=True)
    for date in sorted(dates.unique()):
        train = frame[dates < date]
        test = frame[dates == date].copy()
        if len(train) < int(min_train):
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(train[list(TIMING_FEATURES)], train["timing_target"])
        test["predicted_timing_target"] = model.predict(test[list(TIMING_FEATURES)])
        test["timing_train_rows"] = len(train)
        rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _timing_policy(frame, mode):
    entry_odds_a = frame["R_odds"].to_numpy(dtype=float)
    entry_odds_b = frame["B_odds"].to_numpy(dtype=float)
    close_odds_a = frame["close_odds_a"].to_numpy(dtype=float)
    close_odds_b = frame["close_odds_b"].to_numpy(dtype=float)
    if mode == "entry":
        use_entry = np.ones(len(frame), dtype=bool)
    elif mode == "close":
        use_entry = np.zeros(len(frame), dtype=bool)
    elif mode == "ridge":
        use_entry = frame["predicted_timing_target"].to_numpy() >= 0
    elif mode == "oracle":
        choose_a = frame["locked_pick"].eq("A").to_numpy()
        entry_pick_odds = np.where(choose_a, entry_odds_a, entry_odds_b)
        close_pick_odds = np.where(choose_a, close_odds_a, close_odds_b)
        use_entry = entry_pick_odds >= close_pick_odds
    else:
        raise ValueError(mode)
    scored = score_policy(
        frame,
        odds_a=np.where(use_entry, entry_odds_a, close_odds_a),
        odds_b=np.where(use_entry, entry_odds_b, close_odds_b),
        locked_pick=frame["locked_pick"].to_numpy(),
    )
    scored["execution_time"] = np.where(use_entry, "entry", "close")
    return scored


def _timing_metrics(scored, bootstrap):
    """Report CLV from the selected execution time, not always from entry."""
    result = policy_metrics(scored, bootstrap, include_clv=False)
    active = scored["stake"] > 0
    choose_a = scored["pick_side"].eq("A").to_numpy()
    entry_pick = np.where(
        choose_a, scored["p_line"], 1.0 - scored["p_line"]
    )
    close_pick = np.where(
        choose_a, scored["p_close_line"], 1.0 - scored["p_close_line"]
    )
    used_entry = scored["execution_time"].eq("entry").to_numpy()
    execution_pick = np.where(used_entry, entry_pick, close_pick)
    clv_rows = scored.loc[active, ["date"]].copy()
    clv_rows["clv"] = (close_pick - execution_pick)[active] * 100.0
    result["execution"] = {
        "entry_actions": int((active & used_entry).sum()),
        "close_actions": int((active & ~used_entry).sum()),
        "clv_bets": int(len(clv_rows)),
        "mean_clv_prob_points": (
            _float(clv_rows["clv"].mean()) if len(clv_rows) else None
        ),
        "clv_ci90_event_clustered": clustered_mean_ci(
            clv_rows, "clv", n=bootstrap
        ),
    }
    return result


def timing_audit(frame, bootstrap, min_train=200):
    prepared = _timing_frame(frame)
    predicted = walk_forward_timing_predictions(prepared, min_train=min_train)
    if not len(predicted):
        return {"verdict": "defer", "reason": "insufficient prior close-covered rows"}
    predicted = _period(predicted, "holdout")
    if not len(predicted):
        return {"verdict": "defer", "reason": "no timing predictions in final holdout"}
    policies = {
        mode: _timing_metrics(_timing_policy(predicted, mode), bootstrap)
        for mode in ("entry", "close", "ridge", "oracle")
    }
    direction_accuracy = float(
        ((predicted["predicted_timing_target"] >= 0)
         == (predicted["timing_target"] >= 0)).mean()
    )
    ridge = policies["ridge"]
    entry = policies["entry"]
    improves = (
        ridge["roi"] is not None and entry["roi"] is not None
        and ridge["roi"] > entry["roi"]
    )
    enough_bets = ridge["bets"] >= 100
    return {
        "definition": (
            "Fixed ridge model predicts whether the entry-model side will become "
            "more expensive by the close; positive means bet entry, otherwise wait."
        ),
        "training": "strict prior-card close-covered rows only",
        "minimum_training_rows": min_train,
        "evaluated_fights": int(len(predicted)),
        "evaluated_events": int(predicted["date"].nunique()),
        "first_evaluated_date": str(pd.to_datetime(predicted["date"]).min().date()),
        "direction_accuracy": direction_accuracy,
        "policies": policies,
        "gate": {
            "ridge_roi_beats_always_entry": improves,
            "at_least_100_ridge_bets": enough_bets,
        },
        "verdict": "keep_as_challenger" if improves and enough_bets else "defer",
        "oracle_note": (
            "Oracle uses future prices to choose the better observed entry/close odds. "
            "It is a non-actionable execution-price ceiling, not an ROI guarantee."
        ),
    }


def _pair(a, b):
    return "|".join(sorted((norm_name(a), norm_name(b))))


def dispersion_audit(entry, quotes_path, bootstrap):
    path = Path(quotes_path)
    if not path.exists():
        return {"verdict": "defer", "reason": "individual-book quote file is absent"}
    quotes = pd.read_csv(path)
    quotes = quotes[quotes["snapshot_kind"].eq("entry")].copy()
    quotes["odds_a"] = pd.to_numeric(quotes["odds_a"], errors="coerce")
    quotes["odds_b"] = pd.to_numeric(quotes["odds_b"], errors="coerce")
    quotes = quotes.dropna(subset=["odds_a", "odds_b", "fighter_a", "fighter_b"])
    pa = american_to_prob(quotes["odds_a"].to_numpy(dtype=float))
    pb = american_to_prob(quotes["odds_b"].to_numpy(dtype=float))
    quotes["book_probability"] = pa / (pa + pb)
    quotes["pair"] = [_pair(a, b) for a, b in zip(quotes["fighter_a"], quotes["fighter_b"])]
    quotes["date_key"] = pd.to_datetime(quotes["event_date"], errors="coerce").dt.date.astype(str)
    grouped = quotes.groupby(["date_key", "pair"], as_index=False).agg(
        quote_books=("book_key", "nunique"),
        probability_spread=("book_probability", lambda values: float(values.max() - values.min())),
        probability_std=("book_probability", "std"),
    )
    grouped["probability_std"] = grouped["probability_std"].fillna(0.0)

    sample = entry.copy()
    sample["date_key"] = pd.to_datetime(sample["date"], utc=True).dt.date.astype(str)
    matched = sample.merge(grouped, on=["date_key", "pair"], how="inner")
    if not len(matched):
        return {"verdict": "defer", "reason": "individual-book quotes did not match predictions"}
    matched["squared_model_error"] = (matched["y"] - matched["p_model"]) ** 2
    matched["absolute_close_move"] = (
        matched["p_close_line"] - matched["p_line"]
    ).abs()
    scored = score_policy(matched)
    active = scored["stake"] > 0
    holdout = _period(matched, "holdout")
    holdout_scored = score_policy(holdout) if len(holdout) else holdout
    return {
        "api_cards": int(quotes["event_uid"].nunique()),
        "matched_fights": int(len(matched)),
        "matched_events": int(matched["date"].nunique()),
        "matched_active_bets": int(active.sum()),
        "holdout_fights": int(len(holdout)),
        "holdout_active_bets": (
            int((holdout_scored["stake"] > 0).sum()) if len(holdout_scored) else 0
        ),
        "median_probability_spread_points": float(matched["probability_spread"].median() * 100),
        "spearman_spread_vs_squared_model_error": _float(
            matched["probability_spread"].corr(matched["squared_model_error"], method="spearman")
        ),
        "spearman_spread_vs_absolute_close_move": _float(
            matched.dropna(subset=["absolute_close_move"])["probability_spread"].corr(
                matched.dropna(subset=["absolute_close_move"])["absolute_close_move"],
                method="spearman",
            )
        ),
        "baseline_policy_on_matched_rows": policy_metrics(scored, bootstrap),
        "verdict": "defer",
        "reason": "individual-book history covers too few cards and bets for a promotion test",
    }


def _identity_check(frame):
    rescored = score_policy(frame)
    stake_equal = np.array_equal(
        rescored["stake"].to_numpy(dtype=int), frame["stake"].to_numpy(dtype=int)
    )
    pnl_equal = np.allclose(
        rescored["pnl"].to_numpy(dtype=float), frame["pnl"].to_numpy(dtype=float),
        atol=1e-12,
    )
    return {"stake_exact": bool(stake_equal), "pnl_exact": bool(pnl_equal)}


def _load_predictions(path):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame


def render_markdown(report):
    def number(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"

    def percent(value):
        return "n/a" if value is None else f"{100 * value:+.1f}%"

    lines = [
        "# Model improvement audit",
        "",
        "Production remains unchanged. Candidates use development data before 2024,",
        "a 2024 validation period, and a final 2025+ holdout.",
        "",
        "## Verdicts",
        "",
        "| Candidate | Production | Entry | Decision |",
        "| --- | --- | --- | --- |",
    ]
    prod_cal = report["production"]["calibration"]
    entry_cal = report["entry"]["calibration"]
    lines.append(
        f"| Symmetric calibration | {prod_cal['verdict']} | "
        f"{entry_cal['verdict']} | Do not deploy unless entry evidence passes |"
    )
    lines.append(
        f"| Uncertainty multiplier | {report['production']['uncertainty']['verdict']} | "
        f"{report['entry']['uncertainty']['verdict']} | Fixed grid, development-selected |"
    )
    lines.append(
        f"| Minimum book count | n/a | {report['entry']['book_count']['verdict']} | "
        "Coverage filter, not a probability model |"
    )
    lines.append(
        f"| Entry vs close timing | n/a | {report['entry']['timing']['verdict']} | "
        "Close-covered holdout only |"
    )
    lines.append(
        f"| Sportsbook dispersion | n/a | {report['entry']['dispersion']['verdict']} | "
        "API-card sample is insufficient |"
    )
    lines.extend([
        "",
        "## Calibration holdout",
        "",
        "| Dataset | Scale | Validation log-loss delta | Holdout log-loss delta | Holdout policy ROI: raw / calibrated |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for label, section in (("Production", prod_cal), ("Entry", entry_cal)):
        validation = section["periods"]["validation"]
        holdout = section["periods"]["holdout"]
        lines.append(
            f"| {label} | {number(section['scale'])} | "
            f"{number(validation['delta_calibrated_minus_raw']['log_loss'], 6)} | "
            f"{number(holdout['delta_calibrated_minus_raw']['log_loss'], 6)} | "
            f"{percent(holdout['raw_policy']['roi'])} / "
            f"{percent(holdout['calibrated_policy']['roi'])} |"
        )
    prod_uncertainty = report["production"]["uncertainty"]
    entry_uncertainty = report["entry"]["uncertainty"]
    book_count = report["entry"]["book_count"]
    timing = report["entry"]["timing"]
    lines.extend([
        "",
        "## Selection tests",
        "",
        f"- Production uncertainty: development selected "
        f"`{prod_uncertainty['development_selected_multiplier']}x`; "
        f"verdict `{prod_uncertainty['verdict']}`.",
        f"- Entry uncertainty: development selected "
        f"`{entry_uncertainty['development_selected_multiplier']}x`; "
        f"verdict `{entry_uncertainty['verdict']}`.",
        f"- Minimum books: development selected "
        f"`{book_count['development_selected_minimum']}`; "
        f"verdict `{book_count['verdict']}`.",
        f"- Timing: {timing.get('evaluated_fights', 0)} holdout fights, "
        f"direction accuracy {percent(timing.get('direction_accuracy'))}, "
        f"verdict `{timing['verdict']}`.",
    ])
    lines.extend([
        "",
        "## Guardrails",
        "",
        "- The production model and policy are not modified by this audit.",
        "- The final holdout is not used to choose calibration or filter settings.",
        "- The timing oracle is explicitly non-actionable and only measures the ceiling.",
        "- Actual-ticket execution cannot be backtested because no ticket ledger exists.",
        "- Individual-book dispersion is unavailable for the large consensus archive.",
        "- Calibration promotion requires paired card-clustered improvement intervals",
        "  wholly below zero in both validation and holdout.",
        "",
        "See `model_improvement_audit.json` for all metrics and confidence intervals.",
    ])
    return "\n".join(lines) + "\n"


def run(args):
    production = _load_predictions(args.production_predictions)
    entry = _load_predictions(args.entry_predictions)
    bootstrap = int(args.bootstrap)
    report = {
        "experiment": "model-improvement-audit-v1",
        "status": "research_only",
        "production_unchanged": True,
        "splits": {
            "development": f"date < {DEVELOPMENT_END}",
            "validation": f"{DEVELOPMENT_END} <= date < {HOLDOUT_START}",
            "holdout": f"date >= {HOLDOUT_START}",
        },
        "baseline_identity": {
            "production": _identity_check(production),
            "entry": _identity_check(entry),
        },
        "production": {
            "calibration": calibration_audit(production, bootstrap),
            "uncertainty": uncertainty_audit(production, bootstrap),
        },
        "entry": {
            "calibration": calibration_audit(entry, bootstrap),
            "uncertainty": uncertainty_audit(entry, bootstrap),
            "book_count": book_count_audit(entry, bootstrap),
            "timing": timing_audit(entry, bootstrap, args.timing_min_train),
            "dispersion": dispersion_audit(entry, args.api_quotes, bootstrap),
        },
        "not_historically_testable": {
            "actual_ticket_execution": "No immutable accepted-ticket ledger exists yet.",
            "full_archive_book_dispersion": (
                "The large archive stores consensus quotes, not per-book observations."
            ),
            "future_entry_trained_model": (
                "Requires additional point-in-time entry/close pairs before retuning."
            ),
        },
        "decision_rule": (
            "No audit result changes production automatically. Holdout improvement must "
            "also satisfy sample and forward-paper gates."
        ),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output} and {args.markdown}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-predictions", default="production_validation.csv")
    parser.add_argument("--entry-predictions", default="historical_entry_validation.csv")
    parser.add_argument(
        "--api-quotes",
        default="raw/odds_api_historical/historical_h2h_quotes.csv",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--timing-min-train", type=int, default=200)
    parser.add_argument("--output", default="model_improvement_audit.json")
    parser.add_argument("--markdown", default="MODEL_IMPROVEMENT_AUDIT.md")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
