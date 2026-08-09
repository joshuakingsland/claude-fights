"""Rate fighters across promotions, then rate the promotions themselves.

The premise, supplied rather than derived: the UFC is the strongest promotion,
PFL and Bellator sit a tier below it, and everything else is unknown. The
question is what "everything else" is worth, and the answer has to come from
who actually fought whom rather than from which country a show runs in.

A single Bradley-Terry fit over every collected bout does that. Each fighter
gets one strength, estimated from wins and losses against opponents whose own
strengths are estimated the same way, so a win over a future PFL fighter
counts for more than a win over someone who went 3-105. Promotions are then
rated by the fighters who competed in them, not by reputation.

This is only possible because the graph connects. All 111 promotions with 25
or more bouts have at least one fighter linked to the UFC pool. After one hop
that was not true and the promotions looked like isolated islands; the paths
ran through opponents whose records had not been fetched yet.

Two guards matter:

- Ratings are regularised toward the mean. A fighter with three bouts should
  not outrank a fighter with thirty on the strength of a small sample, and an
  undefeated regional record against weak opposition should pull toward
  average rather than toward the top.
- A promotion with too few bouts gets no tier at all. Roughly 2,300 of the
  2,400 promotions here appear once or twice, and inventing a letter grade for
  them would be exactly the kind of authoritative-looking guess this file
  exists to avoid.

The output is a promotion rating relative to the UFC, expressed in the same
log-odds units as the fighter ratings. It is descriptive, not a betting
signal; using it as a model feature would require point-in-time construction,
since a fighter's rating here uses their whole career including bouts that had
not happened at the time of any given fight.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from identity import norm_name

# Sherdog splits promotions across event families. Left uncollapsed they become
# separate organisations and the strongest anchor is fragmented three ways.
# Rizin is the worst case: 65 bouts split four ways, none of which clears the
# shared-fighter minimum alone, so the promotion silently disappears from the
# tier table. The patterns are anchored to avoid false folds - "Professional
# Fighters Combat" is not the PFL, and "Alash Pride" is not Pride.
FAMILY_PATTERNS = (
    ("UFC", re.compile(r"^ufc\b|ultimate fighting championship", re.I)),
    ("Bellator", re.compile(r"^bellator\b", re.I)),
    ("PFL", re.compile(r"^pfl\b|^professional fighters league\b", re.I)),
    ("Rizin", re.compile(r"^rizin\b", re.I)),
    ("KSW", re.compile(r"^ksw\b", re.I)),
)
DECISIVE = {"win", "loss"}

# Pseudo-bouts against an average opponent. Higher values pull short records
# harder toward the mean.
PRIOR_STRENGTH = 2.0
MIN_BOUTS_FOR_TIER = 25


def collapse_promotion(name):
    """Fold event families into one organisation."""
    text = str(name or "").strip()
    for canonical, pattern in FAMILY_PATTERNS:
        if pattern.search(text):
            return canonical
    return text


def load_ufc_bouts(path="fights_v2.csv"):
    """Every UFC bout on record, as the anchor population.

    Without this the UFC enters the graph only through whichever of its
    fighters happened to face someone in the Sherdog crawl. That crawl was
    seeded from fighters with no UFCStats history - overwhelmingly PFL - so
    the PFL arrives with full careers and the UFC with a thin, unrepresentative
    slice. The first fit put PFL above the UFC as a result, which is a
    statement about the sample rather than about the sport.
    """
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["fighter_a", "fighter_b", "winner"])
    rows = []
    for row in frame.itertuples():
        side = str(row.winner).strip().upper()
        if side not in ("A", "B"):
            continue
        a, b = norm_name(row.fighter_a), norm_name(row.fighter_b)
        if not a or not b:
            continue
        rows.append((a, b, "UFC") if side == "A" else (b, a, "UFC"))
    return rows


def load_bouts(path="data/sherdog_bouts.csv"):
    """Return deduplicated decisive bouts as (winner, loser, promotion)."""
    frame = pd.read_csv(path)
    frame["promotion"] = frame["promotion"].map(collapse_promotion)
    frame = frame[frame["result"].isin(DECISIVE)]
    frame["fighter"] = frame["fighter_name"].map(norm_name)
    frame["rival"] = frame["opponent"].map(lambda x: norm_name(str(x)))
    frame = frame[(frame["fighter"] != "") & (frame["rival"] != "")]

    # A bout appears once in each fighter's record. Key on the unordered pair
    # plus date so it is counted once, otherwise every bout between two
    # collected fighters carries double weight.
    seen, rows = set(), []
    for row in frame.itertuples():
        key = (tuple(sorted((row.fighter, row.rival))), str(row.event_date))
        if key in seen:
            continue
        seen.add(key)
        if row.result == "win":
            rows.append((row.fighter, row.rival, row.promotion))
        else:
            rows.append((row.rival, row.fighter, row.promotion))
    return rows


def fit_bradley_terry(bouts, iterations=300, prior=PRIOR_STRENGTH, tol=1e-7):
    """Regularised Bradley-Terry strengths via minorisation-maximisation.

    Each fighter is given `prior` pseudo-wins and pseudo-losses against a
    hypothetical average opponent, which keeps undefeated short records finite
    and pulls thin samples toward the middle rather than to the extremes.
    """
    fighters = sorted({f for bout in bouts for f in bout[:2]})
    index = {name: i for i, name in enumerate(fighters)}
    wins = np.zeros(len(fighters))
    pairs = defaultdict(int)
    for winner, loser, _ in bouts:
        wins[index[winner]] += 1
        pairs[(index[winner], index[loser])] += 1

    # Standard minorisation-maximisation update:
    #   p_i <- W_i / sum_j [ N_ij / (p_i + p_j) ]
    # extended with `prior` pseudo-wins and `prior` pseudo-losses against a
    # fixed average opponent of strength 1, which contributes `prior` to the
    # numerator and 2*prior/(p_i + 1) to the denominator. Without it an
    # undefeated fighter has an unbounded maximum-likelihood strength.
    strength = np.ones(len(fighters))
    for _ in range(iterations):
        previous = strength.copy()
        numerator = wins + prior
        denominator = 2.0 * prior / (strength + 1.0)
        for (i, j), count in pairs.items():
            total = strength[i] + strength[j]
            denominator[i] += count / total
            denominator[j] += count / total
        strength = numerator / np.maximum(denominator, 1e-12)
        strength /= np.exp(np.mean(np.log(np.maximum(strength, 1e-12))))
        if np.max(np.abs(strength - previous)) < tol:
            break
    rating = np.log(np.maximum(strength, 1e-12))
    return {name: float(rating[index[name]]) for name in fighters}


def rate_promotions(bouts, ratings, min_bouts=MIN_BOUTS_FOR_TIER):
    """Rate each promotion by the strength of the fighters who competed there."""
    by_promotion = defaultdict(list)
    fighters = defaultdict(set)
    for winner, loser, promotion in bouts:
        for name in (winner, loser):
            by_promotion[promotion].append(ratings.get(name, 0.0))
            fighters[promotion].add(name)
    rows = []
    for promotion, values in by_promotion.items():
        values = np.asarray(values, dtype=float)
        rows.append({
            "promotion": promotion,
            "bouts": len(values) // 2,
            "fighters": len(fighters[promotion]),
            "mean_rating": round(float(values.mean()), 4),
            "median_rating": round(float(np.median(values)), 4),
            "p75_rating": round(float(np.percentile(values, 75)), 4),
        })
    frame = pd.DataFrame(rows)
    anchor = frame.loc[frame["promotion"] == "UFC", "mean_rating"]
    reference = float(anchor.iloc[0]) if len(anchor) else 0.0
    frame["vs_ufc"] = (frame["mean_rating"] - reference).round(4)
    frame["ratable"] = frame["bouts"] >= min_bouts
    return frame.sort_values("mean_rating", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bouts", default="data/sherdog_bouts.csv")
    parser.add_argument("--ufc", default="fights_v2.csv")
    parser.add_argument("--out", default="data/promotion_ratings.csv")
    parser.add_argument("--report", default="promotion_ratings.json")
    parser.add_argument("--min-bouts", type=int, default=MIN_BOUTS_FOR_TIER)
    args = parser.parse_args()

    bouts = load_bouts(args.bouts)
    sherdog_count = len(bouts)
    # UFC bouts come from the tracked UFCStats table, not from whatever the
    # Sherdog crawl happened to touch, so the anchor is the real population.
    seen = {(tuple(sorted(b[:2]))) for b in bouts if b[2] == "UFC"}
    for winner, loser, promotion in load_ufc_bouts(args.ufc):
        if tuple(sorted((winner, loser))) in seen:
            continue
        bouts.append((winner, loser, promotion))
    print(f"{sherdog_count} Sherdog bouts + {len(bouts) - sherdog_count} "
          f"UFCStats bouts = {len(bouts)} total")
    ratings = fit_bradley_terry(bouts)
    print(f"fitted strengths for {len(ratings)} fighters")
    frame = rate_promotions(bouts, ratings, args.min_bouts)
    frame.to_csv(args.out, index=False)

    ratable = frame[frame["ratable"]]
    payload = {
        "bouts": len(bouts),
        "fighters_rated": len(ratings),
        "promotions_total": int(len(frame)),
        "promotions_ratable": int(len(ratable)),
        "min_bouts_for_tier": args.min_bouts,
        "anchors": {"UFC": "S (given)", "PFL/Bellator": "A (given)"},
        "caveat": ("Descriptive only. Ratings use each fighter's whole career, "
                   "so this is not point-in-time and must not be used as a "
                   "model feature without rebuilding it as of each fight."),
    }
    Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{'promotion':30s}{'bouts':>7}{'fght':>6}{'mean':>8}{'vs UFC':>8}")
    for row in ratable.head(28).itertuples(index=False):
        print(f"  {row.promotion[:28]:28s}{row.bouts:>7}{row.fighters:>6}"
              f"{row.mean_rating:>8.3f}{row.vs_ufc:>8.3f}")
    print(f"\n{len(frame) - len(ratable)} promotions below the "
          f"{args.min_bouts}-bout threshold get no tier")


if __name__ == "__main__":
    main()
