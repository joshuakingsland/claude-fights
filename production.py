"""The single production moneyline model used by cards and validation.

All callers use the same symmetrized residual model and uncertainty-aware
selection rule.  Keeping this here prevents the dashboard, backtest, and
paper ledger from silently drifting apart.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (BOOTSTRAP_MODELS, EDGE_RULE, EVENT_DAY_STAKE_CAP, FOCUS,
                    PRODUCTION_MAX_STAKE)

MODEL_FEATURES = ["line_logit", "line_abs"] + FOCUS + ["ko_recent"]
DIFF_FEATURES = FOCUS + ["ko_recent"]


def event_seed(value, namespace="production"):
    """Stable 32-bit seed derived from immutable event identity.

    Validation results for the same event are therefore identical regardless
    of the requested validation window or enumeration order.
    """
    if isinstance(value, (list, tuple, set)):
        value = "|".join(sorted(str(v) for v in value))
    try:
        normalized = str(pd.Timestamp(value).date())
    except Exception:
        normalized = str(value)
    digest = hashlib.sha256(f"{namespace}|{normalized}".encode()).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _with_line_abs(frame):
    out = frame.copy()
    if "line_abs" not in out:
        out["line_abs"] = out["line_logit"].abs()
    return out


def _symmetrize(frame):
    """Add the corner-swapped copy used for every production fit."""
    frame = _with_line_abs(frame)
    flip = frame.copy()
    flip[DIFF_FEATURES] = -flip[DIFF_FEATURES]
    flip["line_logit"] = -flip["line_logit"]
    flip["y"] = 1 - flip["y"]
    return pd.concat([frame, flip], ignore_index=True)


def fit_ensemble(train, n_models=BOOTSTRAP_MODELS, seed=0):
    """Fit the deployed model and its bootstrap uncertainty ensemble."""
    train = _with_line_abs(train)
    sym = _symmetrize(train)
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=3000))
    base.fit(sym[MODEL_FEATURES], sym["y"])
    rng = np.random.default_rng(seed)
    models = [base]
    for _ in range(max(0, int(n_models))):
        idx = rng.choice(sym.index, len(sym), replace=True)
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.05, max_iter=1500))
        model.fit(sym.loc[idx, MODEL_FEATURES], sym.loc[idx, "y"])
        models.append(model)
    return models


def predict_probabilities(models, frame):
    """Return symmetry-averaged red-frame probability and bootstrap SE."""
    frame = _with_line_abs(frame)
    xa = frame[MODEL_FEATURES].copy()
    xb = xa.copy()
    xb["line_logit"] = -xb["line_logit"]
    xb[DIFF_FEATURES] = -xb[DIFF_FEATURES]
    values = []
    for model in models:
        pa = model.predict_proba(xa)[:, 1]
        pb = model.predict_proba(xb)[:, 1]
        values.append((pa + 1.0 - pb) / 2.0)
    arr = np.asarray(values)
    return arr[0], arr.std(axis=0, ddof=0) if len(arr) > 1 else np.zeros(len(frame))


def allocate_stakes(net_edge, groups=None, threshold=EDGE_RULE,
                    max_stake=PRODUCTION_MAX_STAKE,
                    group_cap=EVENT_DAY_STAKE_CAP):
    """Allocate a deterministic flat-stake paper policy by event day."""
    net_edge = np.asarray(net_edge, dtype=float)
    groups = (np.zeros(len(net_edge), dtype=int) if groups is None
              else np.asarray(groups))
    stakes = np.zeros(len(net_edge), dtype=int)
    if max_stake <= 0 or group_cap <= 0:
        return stakes
    for group in pd.unique(groups):
        indices = np.flatnonzero(groups == group)
        eligible = indices[net_edge[indices] >= threshold]
        eligible = eligible[np.argsort(-net_edge[eligible], kind="stable")]
        remaining = int(group_cap)
        for index in eligible:
            stake = min(int(max_stake), remaining)
            if stake <= 0:
                break
            stakes[index] = stake
            remaining -= stake
    return stakes


def score_bets(frame, p_model, se):
    """Score both sides using net edge and return a row-level ledger."""
    frame = frame.reset_index(drop=True).copy()
    p_model = np.asarray(p_model, dtype=float)
    se = np.asarray(se, dtype=float)
    pa = np.asarray(frame["pr_raw"], dtype=float)
    pb = np.asarray(frame["pb_raw"], dtype=float)
    edge_a, edge_b = p_model - pa, (1.0 - p_model) - pb
    choose_a = edge_a >= edge_b
    gross = np.where(choose_a, edge_a, edge_b)
    net = gross - se
    groups = (frame["date"].astype(str).to_numpy()
              if "date" in frame else np.zeros(len(frame), dtype=int))
    stake = allocate_stakes(net, groups=groups)
    frame["p_model"] = p_model
    frame["se"] = se
    frame["edge"] = gross
    frame["net_edge"] = net
    frame["pick_side"] = np.where(choose_a, "A", "B")
    frame["qualified"] = net >= EDGE_RULE
    frame["stake"] = stake
    return frame


def event_pnl(frame):
    """Return per-row P&L under the exact production stakes."""
    from backtest import american_payout

    frame = frame.reset_index(drop=True)
    choose_a = frame["pick_side"].eq("A").to_numpy()
    active = frame["stake"].to_numpy() > 0
    winner_a = frame["y"].to_numpy() == 1
    odds_a = np.asarray(frame["R_odds"], dtype=float)
    odds_b = np.asarray(frame["B_odds"], dtype=float)
    pay_a, pay_b = american_payout(odds_a), american_payout(odds_b)
    stake = frame["stake"].to_numpy().astype(float)
    payout = np.where(choose_a, pay_a, pay_b)
    won = np.where(choose_a, winner_a, ~winner_a)
    return np.where(active, np.where(won, stake * payout, -stake), 0.0)
