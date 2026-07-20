"""Report distribution drift; this does not tune or place bets."""

import argparse
import json

import numpy as np
import pandas as pd


def _summary(frame):
    fields = ["p_model", "p_line", "edge", "se", "net_edge", "stake"]
    out = {"rows": int(len(frame))}
    for field in fields:
        if field in frame:
            out[field] = {"mean": float(frame[field].mean()),
                          "std": float(frame[field].std(ddof=0)),
                          "missing": int(frame[field].isna().sum())}
    if "stake" in frame:
        out["bet_rate"] = float((frame["stake"] > 0).mean())
    return out


def run(args):
    df = pd.read_csv(args.predictions, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    reference = df[df["date"] < args.reference_end]
    recent = df[df["date"] >= args.reference_end]
    report = {"reference_end": args.reference_end,
              "reference": _summary(reference), "recent": _summary(recent),
              "warnings": []}
    for field in ("p_model", "p_line", "edge", "se", "net_edge", "stake"):
        if field not in reference or field not in recent or not len(recent):
            continue
        ref_mean = float(reference[field].mean())
        cur_mean = float(recent[field].mean())
        ref_std = float(reference[field].std(ddof=0))
        z = (cur_mean - ref_mean) / ref_std if ref_std > 1e-12 else 0.0
        if abs(z) >= args.z_threshold:
            report["warnings"].append({"field": field, "z_mean_shift": z})
    if "odds_source" in recent:
        report["recent_odds_sources"] = recent["odds_source"].value_counts().to_dict()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="production_validation.csv")
    ap.add_argument("--reference-end", default="2025-01-01")
    ap.add_argument("--z-threshold", type=float, default=2.0)
    ap.add_argument("--output", default="drift_report.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
