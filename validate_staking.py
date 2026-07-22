"""Audit active and candidate staking policies from walk-forward predictions."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import american_payout
from config import (EDGE_RULE, EVENT_DAY_STAKE_CAP, PRODUCTION_MAX_STAKE,
                    RESEARCH_TWO_UNIT_RULE, STAKING_POLICY_VERSION)


def _unit_pnl(row):
    pick_a = row.pick_side == "A"
    won = (row.y == 1) if pick_a else (row.y == 0)
    odds = float(row.R_odds if pick_a else row.B_odds)
    return float(american_payout(odds) if won else -1.0)


def _event_roi_ci(events, draws=10000, seed=20260721):
    events = events.loc[events["staked"] > 0, ["pnl", "staked"]]
    if not len(events):
        return [None, None]
    values = events.to_numpy(float)
    rng = np.random.default_rng(seed)
    roi = []
    for _ in range(int(draws)):
        sample = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        roi.append(float(sample[0] / sample[1]))
    return [float(x) for x in np.quantile(roi, [0.05, 0.95])]


def _event_mean_ci(frame, column, draws=10000, seed=20260721):
    groups = [
        pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
        for _, group in frame.groupby("date", sort=True)
    ]
    groups = [group for group in groups if len(group)]
    if not groups:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(draws)):
        sample = [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        means.append(float(np.concatenate(sample).mean()))
    return [float(x) for x in np.quantile(means, [0.05, 0.95])]


def simulate_policy(frame, high_threshold=None, max_stake=1,
                    event_cap=EVENT_DAY_STAKE_CAP, draws=10000):
    allocations = []
    for date, event in frame.groupby("date", sort=True):
        event = event.sort_values("net_edge", ascending=False, kind="stable")
        remaining = int(event_cap)
        event_pnl = 0.0
        event_staked = 0
        event_bets = 0
        for row in event.itertuples():
            if float(row.net_edge) < EDGE_RULE or remaining <= 0:
                continue
            desired = (2 if max_stake >= 2 and high_threshold is not None
                       and float(row.net_edge) >= high_threshold else 1)
            stake = min(desired, remaining)
            event_pnl += stake * _unit_pnl(row)
            event_staked += stake
            event_bets += 1
            remaining -= stake
        allocations.append({
            "date": str(date), "pnl": event_pnl,
            "staked": event_staked, "bets": event_bets,
        })
    events = pd.DataFrame(allocations)
    staked = float(events["staked"].sum()) if len(events) else 0.0
    pnl = float(events["pnl"].sum()) if len(events) else 0.0
    equity = events["pnl"].cumsum() if len(events) else pd.Series(dtype=float)
    drawdown = float((equity.cummax() - equity).max()) if len(equity) else 0.0
    return {
        "bets": int(events["bets"].sum()) if len(events) else 0,
        "staked": staked,
        "pnl": pnl,
        "roi": pnl / staked if staked else None,
        "roi_ci90_event_clustered": _event_roi_ci(events, draws=draws),
        "max_drawdown_units": drawdown,
    }


def edge_bucket(frame, lower, upper=None, draws=10000):
    mask = frame["net_edge"] >= lower
    if upper is not None:
        mask &= frame["net_edge"] < upper
    rows = frame.loc[mask].copy()
    if not len(rows):
        return {"bets": 0, "pnl": 0.0, "roi": None,
                "roi_ci90_event_clustered": [None, None]}
    rows["unit_pnl"] = [_unit_pnl(row) for row in rows.itertuples()]
    events = rows.groupby("date", as_index=False).agg(
        pnl=("unit_pnl", "sum"), staked=("unit_pnl", "size")
    )
    pnl = float(events["pnl"].sum())
    staked = float(events["staked"].sum())
    result = {
        "bets": int(len(rows)), "pnl": pnl, "roi": pnl / staked,
        "roi_ci90_event_clustered": _event_roi_ci(events, draws=draws),
    }
    if "clv_prob_points" in rows:
        clv = pd.to_numeric(rows["clv_prob_points"], errors="coerce").dropna()
        result["clv_bets"] = int(len(clv))
        result["mean_clv_prob_points"] = float(clv.mean()) if len(clv) else None
        result["mean_clv_prob_points_ci90_event_clustered"] = _event_mean_ci(
            rows, "clv_prob_points", draws=draws
        )
    return result


def audit_file(path, draws=10000):
    frame = pd.read_csv(path, parse_dates=["date"])
    policies = {
        "active_flat_1u_day_cap2": simulate_policy(
            frame, max_stake=PRODUCTION_MAX_STAKE,
            event_cap=EVENT_DAY_STAKE_CAP, draws=draws,
        ),
        "legacy_2u_at_8_day_cap2": simulate_policy(
            frame, high_threshold=0.08, max_stake=2,
            event_cap=EVENT_DAY_STAKE_CAP, draws=draws,
        ),
        "candidate_2u_at_10_day_cap2": simulate_policy(
            frame, high_threshold=RESEARCH_TWO_UNIT_RULE, max_stake=2,
            event_cap=EVENT_DAY_STAKE_CAP, draws=draws,
        ),
    }
    buckets = {}
    for label, lower, upper in (
        ("4_to_6", 0.04, 0.06), ("6_to_8", 0.06, 0.08),
        ("8_to_10", 0.08, 0.10), ("10_plus", 0.10, None),
    ):
        buckets[label] = edge_bucket(frame, lower, upper, draws=draws)
    return {"rows": int(len(frame)), "policies": policies,
            "edge_buckets": buckets}


def build_report(production_path, entry_path, draws=10000):
    production = audit_file(production_path, draws=draws)
    entry = audit_file(entry_path, draws=draws)
    prod_candidate = production["edge_buckets"]["10_plus"]
    entry_candidate = entry["edge_buckets"]["10_plus"]
    prod_policy = production["policies"]["candidate_2u_at_10_day_cap2"]
    entry_policy = entry["policies"]["candidate_2u_at_10_day_cap2"]
    clv_ci = entry_candidate.get(
        "mean_clv_prob_points_ci90_event_clustered", [None, None]
    )
    checks = {
        "production_candidate_bets_at_least_100": prod_candidate["bets"] >= 100,
        "entry_candidate_bets_at_least_100": entry_candidate["bets"] >= 100,
        "entry_candidate_clv_bets_at_least_100":
            entry_candidate.get("clv_bets", 0) >= 100,
        "production_candidate_roi_ci_lower_positive":
            (prod_candidate["roi_ci90_event_clustered"][0] or -1) > 0,
        "entry_candidate_roi_ci_lower_positive":
            (entry_candidate["roi_ci90_event_clustered"][0] or -1) > 0,
        "production_candidate_policy_roi_ci_lower_positive":
            (prod_policy["roi_ci90_event_clustered"][0] or -1) > 0,
        "entry_candidate_policy_roi_ci_lower_positive":
            (entry_policy["roi_ci90_event_clustered"][0] or -1) > 0,
        "entry_candidate_clv_ci_lower_positive":
            (clv_ci[0] or -1) > 0,
    }
    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "active_policy": STAKING_POLICY_VERSION,
        "active_status": "paper_only",
        "candidate_two_unit_threshold": RESEARCH_TWO_UNIT_RULE,
        "candidate_status": "eligible_for_forward_paper_test" if all(checks.values())
                            else "paper_only",
        "candidate_gate": checks,
        "production": production,
        "entry_price": entry,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", default="production_validation.csv")
    parser.add_argument("--entry", default="historical_entry_validation.csv")
    parser.add_argument("--output", default="staking_validation.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args(argv)
    for path in (args.production, args.entry):
        if not Path(path).exists():
            raise SystemExit(f"Missing {path}; run the corresponding validation first.")
    report = build_report(args.production, args.entry, draws=args.bootstrap)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
