"""Validate totals and distance probabilities against a division baseline.

The benchmark here is deliberately harsh. Comparing against the global
distance rate flatters any model carrying a `heavy` flag, because weight class
alone moves the rate from 37% at light heavyweight to 68% at women's
strawweight. A model that only rediscovers that has learned nothing a
sportsbook does not already price, so the number worth reporting is the one
against a division-and-format lookup table fitted on the same training data.

Nothing here is an edge, and it is not trying to be. It measures whether the
probabilities are any good. Historical totals prices do now exist in the
archive and were tested separately as H8; that is a different question from
this one and lives in PREREGISTRATION.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import rounds_model as model

SHRINKAGE = 25.0


def _context(frame):
    weight = frame["weightclass"].astype(str).str.replace(" Bout", "", regex=False)
    rounds = np.where(model.scheduled_rounds(frame["time_format"]) == 5, "5", "3")
    return weight + "|" + rounds


def _division_table(train, test):
    """Shrunk per-division distance rate, fitted on training fights only."""
    overall = float(train["_distance"].mean())
    grouped = train.groupby(_context(train))["_distance"]
    rate, size = grouped.mean(), grouped.size()
    shrunk = (rate * size + overall * SHRINKAGE) / (size + SHRINKAGE)
    return _context(test).map(shrunk).fillna(overall).to_numpy(), overall


def run(fights_path="fights_v2.csv", cutoff="2024-01-01",
        output="rounds_validation.json"):
    fights = pd.read_csv(fights_path, parse_dates=["date"])
    stamp = pd.Timestamp(cutoff)
    fitted = model.train(fights[fights["date"] < stamp])

    featured = model.prepare(fights)
    usable = featured[featured["_usable"]]
    train = usable[usable["date"] < stamp].copy()
    test = usable[usable["date"] >= stamp].reset_index(drop=True)

    priced = model.totals_and_distance(fitted, test)
    distance = np.array([row["distance"] for row in priced])
    actual = test["_distance"].to_numpy().astype(int)
    baseline, overall = _division_table(train, test)

    lines = {}
    minutes = test["fight_time_min"].to_numpy(dtype=float)
    for mark in (0.5, 1.5, 2.5, 3.5, 4.5):
        key = f"over_{mark}"
        quoted = np.array([key in row["lines"] for row in priced])
        if quoted.sum() < 50:
            continue
        probability = np.array([row["lines"].get(key, np.nan)
                                for row in priced])[quoted]
        went_over = (minutes[quoted] > mark * model.ROUND_MINUTES).astype(int)
        lines[key] = {
            "fights": int(quoted.sum()),
            "actual_over_rate": round(float(went_over.mean()), 4),
            "mean_prediction": round(float(probability.mean()), 4),
            "log_loss_model": round(float(log_loss(went_over, probability,
                                                   labels=[0, 1])), 5),
            "log_loss_test_base_rate": round(float(log_loss(
                went_over, np.full(len(went_over), went_over.mean()),
                labels=[0, 1])), 5),
        }

    model_loss = float(log_loss(actual, distance, labels=[0, 1]))
    table_loss = float(log_loss(actual, baseline, labels=[0, 1]))
    report = {
        "model": "rounds-hazard-v1",
        "cutoff": cutoff,
        "test_fights": int(len(test)),
        "distance": {
            "actual_rate": round(float(actual.mean()), 4),
            "mean_prediction": round(float(distance.mean()), 4),
            "log_loss_model": round(model_loss, 5),
            "log_loss_division_table": round(table_loss, 5),
            "log_loss_global_rate": round(float(log_loss(
                actual, np.full(len(test), overall), labels=[0, 1])), 5),
            "improvement_over_division_table": round(
                (table_loss - model_loss) / table_loss, 4),
        },
        "totals": lines,
        "status": "probability_only",
        "prop_edge_validated": False,
        "note": ("Probability quality only. Historical totals prices have "
                 "since been archived and tested (H8 in PREREGISTRATION.md); "
                 "that test was parked because the book chooses which total to "
                 "hang, so the O/U 2.5 subset is selected rather than "
                 "mispriced. No method or distance market was quoted in any "
                 "region. These remain fair prices to shop by hand, not a "
                 "tradable signal."),
    }
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--cutoff", default="2024-01-01")
    parser.add_argument("--output", default="rounds_validation.json")
    args = parser.parse_args()
    run(args.fights, args.cutoff, args.output)


if __name__ == "__main__":
    main()
