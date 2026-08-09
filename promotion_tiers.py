"""Rate promotions by how the same fighter performs in each.

The previous approach summarised each promotion by the ratings of everyone who
competed there, and it failed twice in opposite directions: PFL above the UFC
when the sample was PFL-heavy, then the UFC 22nd of 111 once every real UFC
bout was added. Neither was about strength. A mean over participants measures
roster depth, and the UFC's roster includes every fighter who lost a debut and
left, while a regional promotion's collected roster is whoever happened to be
crawled.

This measures something the composition of a roster cannot distort. Take the
fighters who competed in both the UFC and promotion X, and compare each one's
win rate in each place. If X is softer, the same fighter wins more there. The
comparison is within-fighter, so ability cancels and only the difference in
opposition remains.

Two details keep it honest:

- Win rates are Laplace-smoothed. A fighter who went 1-0 in a promotion has a
  win rate of 1.0 and an infinite log-odds, which would dominate any average.
- A promotion needs a minimum number of shared fighters and bouts before it is
  reported at all. Most of the 639 promotions sharing a fighter with the UFC
  share exactly one, which supports no conclusion.

The result is a log-odds gap: how much more likely the same fighter is to win
in that promotion than in the UFC. Positive means softer. It is descriptive,
uses whole careers, and is not point-in-time, so it is not a model feature as
it stands.

Separately, and deliberately not mixed in, there is a declared tier ladder
supplied by the operator. It is an assumption, not a measurement: no letter
grade is ever derived from a measured gap, and no measured gap is ever
adjusted to agree with a letter. They are carried side by side so the two can
be checked against each other, which is the only reason to have both.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import rankings

MIN_SHARED_FIGHTERS = 5
MIN_BOUTS_EACH_SIDE = 2

# Operator-declared, not measured. The stated premise is that the UFC champion
# beats every other promotion's champion roughly 99 times in 100, so the UFC is
# its own tier; PFL and Bellator were peers until they merged; KSW and Rizin are
# the named regional majors; everything else is unranked below them.
#
# Only promotions named here get their declared letter. Anything else falls to
# DEFAULT_TIER, which means "not distinguished from the rest of the field"
# rather than "measured and found to be at this level".
DECLARED_TIERS = {
    "UFC": "S",
    "PFL": "A",
    "Bellator": "A",
    "KSW": "C",
    "Rizin": "C",
}
DEFAULT_TIER = "D"
TIER_SOURCE = "operator-declared"


def declared_tier(promotion):
    """The supplied letter for a promotion. Never inferred from the data."""
    return DECLARED_TIERS.get(str(promotion or "").strip(), DEFAULT_TIER)


def _smoothed_logit(wins, losses):
    """Laplace-smoothed log-odds; finite for an unbeaten short record."""
    rate = (wins + 1.0) / (wins + losses + 2.0)
    return float(np.log(rate / (1.0 - rate)))


def fighter_records(bouts):
    """Per fighter, per promotion: wins and losses."""
    record = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for winner, loser, promotion in bouts:
        record[winner][promotion][0] += 1
        record[loser][promotion][1] += 1
    return record


def compare(bouts, anchor="UFC", min_shared=MIN_SHARED_FIGHTERS,
            min_bouts=MIN_BOUTS_EACH_SIDE):
    record = fighter_records(bouts)
    gaps = defaultdict(list)
    for fighter, promotions in record.items():
        if anchor not in promotions:
            continue
        anchor_w, anchor_l = promotions[anchor]
        if anchor_w + anchor_l < min_bouts:
            continue
        anchor_logit = _smoothed_logit(anchor_w, anchor_l)
        for promotion, (wins, losses) in promotions.items():
            if promotion == anchor or wins + losses < min_bouts:
                continue
            gaps[promotion].append({
                "fighter": fighter,
                "gap": _smoothed_logit(wins, losses) - anchor_logit,
                "bouts_here": wins + losses,
                "bouts_anchor": anchor_w + anchor_l,
            })
    rows = []
    for promotion, entries in gaps.items():
        values = np.array([e["gap"] for e in entries])
        weights = np.array([min(e["bouts_here"], e["bouts_anchor"])
                            for e in entries], dtype=float)
        rows.append({
            "promotion": promotion,
            "shared_fighters": len(entries),
            "mean_gap_vs_ufc": round(float(np.average(values, weights=weights)), 4),
            "median_gap_vs_ufc": round(float(np.median(values)), 4),
            "reportable": len(entries) >= min_shared,
            # Carried alongside the measurement, never computed from it.
            "declared_tier": declared_tier(promotion),
        })
    frame = pd.DataFrame(rows)
    if not len(frame):
        return frame
    return frame.sort_values("mean_gap_vs_ufc").reset_index(drop=True)


def bootstrap_gap(bouts, promotion, anchor="UFC", draws=2000, seed=5,
                  min_bouts=MIN_BOUTS_EACH_SIDE):
    """Interval over shared fighters, which are the independent units here."""
    record = fighter_records(bouts)
    values = []
    for fighter, promotions in record.items():
        if anchor not in promotions or promotion not in promotions:
            continue
        aw, al = promotions[anchor]
        pw, pl = promotions[promotion]
        if aw + al < min_bouts or pw + pl < min_bouts:
            continue
        values.append(_smoothed_logit(pw, pl) - _smoothed_logit(aw, al))
    if len(values) < 3:
        return None
    values = np.array(values)
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean()
             for _ in range(draws)]
    return [round(float(np.percentile(means, 5)), 4),
            round(float(np.percentile(means, 95)), 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bouts", default="data/sherdog_bouts.csv")
    parser.add_argument("--ufc", default="fights_v2.csv")
    parser.add_argument("--out", default="data/promotion_tiers.csv")
    parser.add_argument("--report", default="promotion_tiers.json")
    parser.add_argument("--min-shared", type=int, default=MIN_SHARED_FIGHTERS)
    args = parser.parse_args()

    bouts = rankings.load_bouts(args.bouts)
    seen = {tuple(sorted(b[:2])) for b in bouts if b[2] == "UFC"}
    for winner, loser, promotion in rankings.load_ufc_bouts(args.ufc):
        if tuple(sorted((winner, loser))) not in seen:
            bouts.append((winner, loser, promotion))
    print(f"{len(bouts)} bouts")

    frame = compare(bouts, min_shared=args.min_shared)
    reportable = frame[frame["reportable"]].copy()
    reportable["ci90"] = [bootstrap_gap(bouts, p) for p in reportable["promotion"]]
    frame.to_csv(args.out, index=False)

    print(f"\n{len(frame)} promotions share a fighter with the UFC; "
          f"{len(reportable)} clear the {args.min_shared}-fighter minimum\n")
    print(f"{'promotion':30s}{'shared':>8}{'gap':>8}{'median':>8}  {'90% interval':16s}tier")
    print("  (gap > 0 means the same fighter wins MORE there than in the UFC)")
    print("  (tier is declared, not measured; the two columns are independent)")
    for row in reportable.itertuples(index=False):
        interval = (f"[{row.ci90[0]:+.2f}, {row.ci90[1]:+.2f}]"
                    if row.ci90 else "n/a")
        print(f"  {row.promotion[:28]:28s}{row.shared_fighters:>8}"
              f"{row.mean_gap_vs_ufc:>8.3f}{row.median_gap_vs_ufc:>8.3f}  "
              f"{interval:16s}{row.declared_tier}")

    # Declared promotions with no measured gap are the interesting hole: the
    # letter is an assumption nothing in the data has yet checked.
    measured = set(reportable["promotion"])
    unchecked = [p for p in DECLARED_TIERS if p not in measured and p != "UFC"]
    if unchecked:
        print(f"\n  declared but not yet measurable: {', '.join(sorted(unchecked))}")

    Path(args.report).write_text(json.dumps({
        "bouts": len(bouts),
        "promotions_sharing_a_fighter": int(len(frame)),
        "promotions_reportable": int(len(reportable)),
        "min_shared_fighters": args.min_shared,
        "anchor": "UFC",
        "reading": ("Positive gap means the same fighter wins more often in "
                    "that promotion than in the UFC, i.e. softer opposition."),
        "caveat": ("Whole-career, not point-in-time. Descriptive only; not a "
                   "model feature in this form."),
        "declared_tiers": dict(DECLARED_TIERS),
        "default_tier": DEFAULT_TIER,
        "tier_source": TIER_SOURCE,
        "tier_caveat": ("Tiers are supplied, not estimated. No letter is "
                        "derived from mean_gap_vs_ufc and no gap is adjusted "
                        "to match a letter."),
        "declared_without_measurement": sorted(unchecked),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
