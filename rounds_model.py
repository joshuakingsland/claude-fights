"""How long a fight lasts: totals lines and the distance market from one fit.

Distance and over/under are the same question asked at different cut points,
so fitting them separately invites the two answers to disagree - a model that
says 60% to go the distance and 55% to go over 2.5 rounds in a three-round
fight is claiming something impossible. Here a single discrete-time hazard
over half-round intervals produces both, and they agree by construction.

The intervals are half-rounds because that is where the market puts its
lines. An MMA total of 2.5 rounds settles on elapsed time past 12:30, so
every line the books hang falls exactly on a bin edge, and the survival
function evaluated at that edge *is* the fair over price. Going to the
distance is survival past the final bin.

Two things drive the design:

Features are corner-invariant. The moneyline model is built from A-minus-B
differentials, which is right for "who wins" and wrong for "how long". Two
knockout artists and two decision grinders both have a differential of zero
and could hardly be less alike. So each career rate enters as a pair mean and
an absolute gap, which carries the same information as {min, max} while
staying symmetric: swapping the corners cannot change the prediction.

The benchmark is a division table, not the global rate. Weight class alone
moves the distance rate from 37% at light heavyweight to 68% at women's
strawweight, and any model with a `heavy` flag will clear a global baseline
without having learned anything a book doesn't have. Beating the division
table is the weakest result worth reporting.

IMPORTANT: as with method_model, no historical prices exist for these
markets, so everything here is a fair price validated for probability
quality. It is not a measured edge against any book, and a 2026-08-09 sweep
of the odds feed found no US book exposing a totals or distance market at
all. Treat the output as a view to shop by hand, not as a tradable signal.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from method_model import attach_side_features, career_method_rates

ROUND_MINUTES = 5.0
# Half-round edges, in minutes, for three- and five-round fights. A fight
# surviving every edge went to a decision.
EDGES_3 = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
EDGES_5 = EDGES_3 + (17.5, 20.0, 22.5, 25.0)

FEATS = [
    "finish_mean", "finish_gap",      # career finish rate as winner
    "finished_mean", "finished_gap",  # career rate of being finished
    "ko_mean", "sub_mean",
    "experience_mean", "experience_gap",
    "age_mean", "age_gap",
    "heavy", "light_heavy", "women", "flyweight", "five_rd",
]


def scheduled_rounds(time_format):
    """3 or 5 for the standard formats; NaN for anything exotic."""
    text = time_format.astype(str)
    return np.where(text.str.startswith("5 Rnd (5-5-5-5-5)"), 5.0,
                    np.where(text.str.startswith("3 Rnd (5-5-5)"), 3.0, np.nan))


def edges_for(rounds):
    return EDGES_5 if int(rounds) == 5 else EDGES_3


def _pair(a, b):
    """Corner-invariant summary of a two-sided quantity."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (a + b) / 2.0, np.abs(a - b)


def build_X(featured):
    """Corner-invariant features. Swapping a and b must not change a row."""
    F = featured
    ko_w = _pair(F["r_ko_w_a"], F["r_ko_w_b"])
    sub_w = _pair(F["r_sub_w_a"], F["r_sub_w_b"])
    # Finish rate as a winner: KO plus submission. Being finished: the rates
    # recorded on the losing side.
    fin_a = F["r_ko_w_a"].to_numpy() + F["r_sub_w_a"].to_numpy()
    fin_b = F["r_ko_w_b"].to_numpy() + F["r_sub_w_b"].to_numpy()
    lost_a = F["r_ko_l_a"].to_numpy() + F["r_sub_l_a"].to_numpy()
    lost_b = F["r_ko_l_b"].to_numpy() + F["r_sub_l_b"].to_numpy()
    finish_mean, finish_gap = _pair(fin_a, fin_b)
    finished_mean, finished_gap = _pair(lost_a, lost_b)
    exp_mean, exp_gap = _pair(F["n_pre_a"], F["n_pre_b"])

    age_a = (F["date"] - pd.to_datetime(F["dob_a"], errors="coerce")).dt.days / 365.25
    age_b = (F["date"] - pd.to_datetime(F["dob_b"], errors="coerce")).dt.days / 365.25
    age_mean, age_gap = _pair(age_a, age_b)

    weight = F["weightclass"].astype(str)
    X = pd.DataFrame({
        "finish_mean": finish_mean, "finish_gap": finish_gap,
        "finished_mean": finished_mean, "finished_gap": finished_gap,
        "ko_mean": ko_w[0], "sub_mean": sub_w[0],
        "experience_mean": exp_mean, "experience_gap": exp_gap,
        "age_mean": age_mean, "age_gap": age_gap,
        "heavy": weight.str.contains("Heavyweight", case=False).to_numpy(float),
        "light_heavy": weight.str.contains("Light Heavyweight",
                                           case=False).to_numpy(float),
        "women": weight.str.contains("Women", case=False).to_numpy(float),
        "flyweight": weight.str.contains("Flyweight", case=False).to_numpy(float),
        "five_rd": (scheduled_rounds(F["time_format"]) == 5).astype(float),
    }, index=F.index)
    return X[FEATS].fillna(0.0)


def _hazard_rows(featured, X):
    """Explode each fight into one row per interval it actually reached.

    A fight contributes a row for every bin it entered, labelled 1 if it ended
    in that bin and 0 if it survived it. That is the standard person-period
    layout for a discrete-time hazard, and it is what lets one fit serve every
    line at once.
    """
    rounds = scheduled_rounds(featured["time_format"])
    minutes = featured["fight_time_min"].to_numpy(dtype=float)
    went_distance = featured["_distance"].to_numpy()
    frames, labels = [], []
    for position in range(len(featured)):
        if not np.isfinite(rounds[position]):
            continue
        for order, edge in enumerate(edges_for(rounds[position])):
            reached = minutes[position] > edge - 2.5 + 1e-9
            if not reached:
                break
            # Ended inside this bin, unless it was still going at the bell.
            ended = (minutes[position] <= edge + 1e-9) and not went_distance[position]
            frames.append((position, order))
            labels.append(1.0 if ended else 0.0)
            if ended:
                break
    if not frames:
        return pd.DataFrame(columns=list(X.columns) + ["bin"]), np.array([])
    positions = [p for p, _ in frames]
    orders = [o for _, o in frames]
    rows = X.iloc[positions].reset_index(drop=True).copy()
    rows["bin"] = orders
    return rows, np.asarray(labels)


def prepare(fights):
    """Point-in-time features for every fight, in the caller's order."""
    fights = fights.sort_values("date", kind="stable").reset_index(drop=True)
    featured = attach_side_features(fights, career_method_rates(fights))
    method = featured["method"].astype(str).str.upper()
    featured["_distance"] = method.str.contains("DEC").to_numpy()
    featured["_usable"] = (
        np.isfinite(scheduled_rounds(featured["time_format"]))
        & featured["fight_time_min"].notna().to_numpy()
        & ~method.str.contains("DQ|OVERTURNED|OTHER").to_numpy()
    )
    return featured


def train(fights):
    """Fit the interval hazard. `bin` enters as a feature, so the baseline
    hazard is free to differ by round rather than being held flat."""
    featured = prepare(fights)
    usable = featured[featured["_usable"]].reset_index(drop=True)
    X = build_X(usable)
    rows, labels = _hazard_rows(usable, X)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=3000))
    model.fit(rows, labels)
    return model


def survival(model, featured, rounds=None):
    """P(fight passes each half-round edge), one list per fight.

    Each entry is (rounds elapsed, probability of still going). The last entry
    sits on the final bell, so it is the probability of going the distance.

    Scored in one batched call: every (fight, bin) pair is stacked, predicted
    together, then the per-fight hazards are multiplied back up. Looping a
    predict_proba per bin makes a full backtest take minutes instead of
    seconds.
    """
    X = build_X(featured)
    scheduled = scheduled_rounds(featured["time_format"]) if rounds is None \
        else np.full(len(featured), float(rounds))

    positions, orders, edge_of = [], [], []
    for position in range(len(featured)):
        if not np.isfinite(scheduled[position]):
            continue
        for order, edge in enumerate(edges_for(scheduled[position])):
            positions.append(position)
            orders.append(order)
            edge_of.append(edge)
    out = [[] for _ in range(len(featured))]
    if not positions:
        return out

    probe = X.iloc[positions].reset_index(drop=True).copy()
    probe["bin"] = orders
    hazards = model.predict_proba(probe)[:, 1]

    alive = {}
    for position, edge, hazard in zip(positions, edge_of, hazards):
        alive[position] = alive.get(position, 1.0) * (1.0 - float(hazard))
        out[position].append((edge / ROUND_MINUTES, alive[position]))
    return out


def totals_and_distance(model, featured):
    """Fair probabilities for the market's lines, plus the distance market.

    Books hang MMA totals on half-rounds only, so the whole-round survival
    points are dropped rather than reported as lines nobody quotes. `over_2.5`
    is the chance of passing 12:30, which is exactly how the bet settles.
    """
    rows = []
    for curve in survival(model, featured):
        if not curve:
            rows.append({"distance": np.nan, "lines": {}})
            continue
        lines = {f"over_{mark}": probability for mark, probability in curve[:-1]
                 if abs(mark - round(mark)) > 0.25}
        rows.append({"distance": curve[-1][1], "lines": lines})
    return rows


def fair_american(p):
    p = min(max(float(p), 0.005), 0.995)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else \
        int(round(100 * (1 - p) / p))


def card_prices(model, fights_all, tag="UPCOMING"):
    """Fair prices for the rows tagged as upcoming.

    Features come from the whole table so an upcoming fight sees each corner's
    full career to date, then only the tagged rows are scored. Scheduled
    fights carry no result, so `_usable` is not applied here; it gates
    training, where an outcome is required, not serving.
    """
    featured = prepare(fights_all)
    selected = featured[featured["event"] == tag].reset_index(drop=True)
    if not len(selected):
        return []
    priced = totals_and_distance(model, selected)
    out = []
    for row in priced:
        distance = row["distance"]
        if not np.isfinite(distance):
            out.append(None)
            continue
        out.append({
            "distance_pct": round(float(distance) * 100, 1),
            "distance": fair_american(distance),
            "finish": fair_american(1.0 - distance),
            "totals": [
                {
                    "line": float(key.split("_")[1]),
                    "over": fair_american(probability),
                    "under": fair_american(1.0 - probability),
                    "over_pct": round(float(probability) * 100, 1),
                }
                for key, probability in sorted(
                    row["lines"].items(), key=lambda kv: float(kv[0].split("_")[1]))
            ],
        })
    return out
