"""Validate method probabilities without claiming a sportsbook prop edge."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import method_model as model


def run(fights_path="fights_v2.csv", cutoff="2024-01-01",
        output="method_validation.json"):
    fights = pd.read_csv(fights_path, parse_dates=["date"])
    fights = fights.sort_values("date", kind="stable").reset_index(drop=True)
    cutoff_stamp = pd.Timestamp(cutoff)
    train = fights[fights["date"] < cutoff_stamp].copy()
    classifier = model.train(train)

    rates = model.career_method_rates(fights)
    featured = model.attach_side_features(fights, rates)
    classes = list(classifier[-1].classes_)
    method_class = pd.Series(model.method_class(featured["method"]), index=featured.index)
    eligible = (
        (featured["date"] >= cutoff_stamp)
        & featured["winner"].isin(["A", "B"])
        & method_class.isin(classes)
    )
    test = featured[eligible].copy()
    winner_side = np.where(test["winner"] == "A", "a", "b")
    probability = classifier.predict_proba(model.build_X(test, winner_side))
    target = method_class[eligible].to_numpy()

    train_class = model.method_class(train["method"])
    prior = pd.Series(train_class)[pd.Series(train_class).isin(classes)] \
        .value_counts(normalize=True).reindex(classes).fillna(0).to_numpy()
    baseline = np.tile(prior, (len(test), 1))
    report = {
        "model": "method-probability-v1-stable-identities",
        "cutoff": cutoff,
        "training_fights": int(len(train)),
        "test_fights": int(len(test)),
        "classes": classes,
        "log_loss_model": float(log_loss(target, probability, labels=classes)),
        "log_loss_global_prior": float(log_loss(target, baseline, labels=classes)),
        "accuracy_model": float(accuracy_score(target, np.asarray(classes)[probability.argmax(1)])),
        "status": "probability_only",
        "prop_edge_validated": False,
        "note": "No historical sportsbook prop prices are used in this audit.",
    }
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fights", default="fights_v2.csv")
    parser.add_argument("--cutoff", default="2024-01-01")
    parser.add_argument("--output", default="method_validation.json")
    args = parser.parse_args()
    run(args.fights, args.cutoff, args.output)


if __name__ == "__main__":
    main()
