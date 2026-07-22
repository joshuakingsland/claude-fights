"""Method-of-victory model: P(KO / SUB / DEC | this fighter wins).

Winner-frame multinomial logistic. Combined with the moneyline model:
    P(fighter by KO) = P(fighter wins) x P(KO | fighter wins)
giving fair prices for the six method-prop outcomes.

Validated after the stable-identity migration (train <2024, test 2024+,
n=1,310): 3-way log loss 0.95188 vs 1.01469 global frequencies.

IMPORTANT HONESTY NOTE: unlike the moneyline model, there is no
historical prop-odds dataset here, so these are fair prices validated
for probability quality only — never a verified edge against any book.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from identity import fighter_keys

FEATS = ["w_ko_rate", "w_sub_rate", "w_dec_rate", "w_n", "l_ko_l",
         "l_sub_l", "l_age", "w_age", "heavy", "women", "five_rd"]


def method_class(method_series):
    m = method_series.astype(str).str.upper()
    return np.where(m.str.contains("KO"), "KO",
           np.where(m.str.contains("SUB"), "SUB",
           np.where(m.str.contains("DEC"), "DEC", "OTHER")))


def career_method_rates(fights):
    """Point-in-time per-(fighter,fight) method-rate table (shift(1)).
    Uses the caller's index AS-IS. Never re-sorts internally: the default
    pandas sort is unstable and permutes same-date fights, which silently
    breaks every (index, fighter) join downstream."""
    mcls = method_class(fights["method"])
    frames = []
    for s, o in (("a", "b"), ("b", "a")):
        frames.append(pd.DataFrame({
            "idx": fights.index, "date": fights["date"],
            "f": fighter_keys(fights, s),
            "ko_w": ((fights["winner"] == s.upper()) & (mcls == "KO")).astype(float),
            "sub_w": ((fights["winner"] == s.upper()) & (mcls == "SUB")).astype(float),
            "dec_w": ((fights["winner"] == s.upper()) & (mcls == "DEC")).astype(float),
            "ko_l": ((fights["winner"] == o.upper()) & (mcls == "KO")).astype(float),
            "sub_l": ((fights["winner"] == o.upper()) & (mcls == "SUB")).astype(float),
        }))
    L = pd.concat(frames).sort_values(["f", "date"])
    g = L.groupby("f", sort=False)
    for c in ["ko_w", "sub_w", "dec_w", "ko_l", "sub_l"]:
        L[f"r_{c}"] = g[c].transform(
            lambda s: s.shift(1).expanding().mean()).fillna(0)
    L["n_pre"] = g.cumcount()
    return L


def attach_side_features(fights, L):
    """Attach each side's PIT method rates to the (full) fights table.
    Simple, verifiable: one merge per side keyed on (original row index,
    normalized name) — no remapping, no filtered reindex."""
    fights = fights.copy()
    sub = L.set_index(["idx", "f"])[
        ["r_ko_w", "r_sub_w", "r_dec_w", "r_ko_l", "r_sub_l", "n_pre"]]
    for side in ("a", "b"):
        key = fighter_keys(fights, side)
        idx = pd.MultiIndex.from_arrays(
            [fights.index.to_numpy(), key.to_numpy()])
        got = sub.reindex(idx).reset_index(drop=True)
        for c in got.columns:
            fights[f"{c}_{side}"] = got[c].to_numpy()
    return fights


def build_X(fights_with_sides, w_side_arr):
    F, w = fights_with_sides, w_side_arr
    l = np.where(w == "a", "b", "a")

    def col(side_arr, tmpl):
        va = F[tmpl.format("a")].to_numpy()
        vb = F[tmpl.format("b")].to_numpy()
        return np.where(side_arr == "a", va, vb)

    X = pd.DataFrame({
        "w_ko_rate": col(w, "r_ko_w_{}"), "w_sub_rate": col(w, "r_sub_w_{}"),
        "w_dec_rate": col(w, "r_dec_w_{}"), "w_n": col(w, "n_pre_{}"),
        "l_ko_l": col(l, "r_ko_l_{}"), "l_sub_l": col(l, "r_sub_l_{}"),
        "l_age": (F["date"] - pd.to_datetime(
            np.where(l == "a", F["dob_a"], F["dob_b"]))).dt.days / 365.25,
        "w_age": (F["date"] - pd.to_datetime(
            np.where(w == "a", F["dob_a"], F["dob_b"]))).dt.days / 365.25,
        "heavy": F["weightclass"].astype(str).str.contains(
            "Heavyweight", case=False).astype(float).to_numpy(),
        "women": F["weightclass"].astype(str).str.contains(
            "Women", case=False).astype(float).to_numpy(),
        "five_rd": F["time_format"].astype(str).str.contains(
            "5 Rnd").astype(float).to_numpy(),
    })
    return X[FEATS].fillna(0)


def train(fights):
    fights = fights.sort_values("date", kind="stable").reset_index(drop=True)
    L = career_method_rates(fights)
    F = attach_side_features(fights, L)
    mcls = method_class(F["method"])
    ok = F["winner"].isin(["A", "B"]) & (mcls != "OTHER")
    W = F[ok]
    X = build_X(W.reset_index(drop=True),
                np.where(W["winner"] == "A", "a", "b"))
    y = mcls[ok.to_numpy()]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.3, max_iter=3000))
    clf.fit(X, y)
    return clf


def fair_american(p):
    p = min(max(p, 0.005), 0.995)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else \
        int(round(100 * (1 - p) / p))


def method_props(clf, fights_all, fight_rows, p_win_a):
    """Fair method odds for both sides of each fight in fight_rows.
    fights_all must contain fight_rows; features computed on the full
    table then selected by index."""
    fights_all = fights_all.sort_values("date", kind="stable").reset_index(drop=True)
    L = career_method_rates(fights_all)
    F = attach_side_features(fights_all, L)
    sel = F[F["event"] == "UPCOMING"].reset_index(drop=True)
    Pa = clf.predict_proba(build_X(sel, np.array(["a"] * len(sel))))
    Pb = clf.predict_proba(build_X(sel, np.array(["b"] * len(sel))))
    classes = list(clf[-1].classes_)
    out = []
    for i in range(len(sel)):
        pw = p_win_a[i]
        props = {}
        for tag, P, w in (("a", Pa, pw), ("b", Pb, 1 - pw)):
            for c in classes:
                props[f"{tag}_{c}"] = round(
                    fair_american(w * P[i][classes.index(c)]))
        out.append(props)
    return out


if __name__ == "__main__":
    fights = pd.read_csv("fights_v2.csv", parse_dates=["date"])
    clf = train(fights)
    pickle.dump(clf, open("method_model.pkl", "wb"))
    print("method_model.pkl trained on all data")
