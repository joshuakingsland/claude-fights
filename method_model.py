"""Method-of-victory model: P(KO / SUB / DEC | this fighter wins).

Winner-frame multinomial logistic. Combined with the moneyline model:
    P(fighter by KO) = P(fighter wins) x P(KO | fighter wins)
giving fair prices for the six method-prop outcomes.

Validated (train <2024, test 2024+, n=1,228): 3-way log loss 0.951 vs
0.990 weight-class baseline and 1.010 global frequencies.

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

from backtest import norm_name

FEATS = ["w_ko_rate", "w_sub_rate", "w_dec_rate", "w_n", "l_ko_l",
         "l_sub_l", "l_age", "w_age", "heavy", "women", "five_rd"]


def method_class(method_series):
    m = method_series.astype(str).str.upper()
    return np.where(m.str.contains("KO"), "KO",
           np.where(m.str.contains("SUB"), "SUB",
           np.where(m.str.contains("DEC"), "DEC", "OTHER")))


def career_method_rates(fights):
    """Point-in-time per-(fighter,fight) method-rate table (shift(1))."""
    fights = fights.sort_values("date").reset_index(drop=True)
    mcls = method_class(fights["method"])
    frames = []
    for s, o in (("a", "b"), ("b", "a")):
        frames.append(pd.DataFrame({
            "idx": fights.index, "date": fights["date"],
            "f": fights[f"fighter_{s}"].map(norm_name),
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


def frame_features(fights, L, w_side, l_side):
    """Feature rows with `w_side` as the hypothetical winner."""
    fights = fights.reset_index(drop=True)
    sub = L.set_index(["idx", "f"])

    def side(side_name, cols):
        key = fights[f"fighter_{side_name}"].map(norm_name)
        idx = pd.MultiIndex.from_arrays([fights.index, key])
        return sub.reindex(idx)[cols].reset_index(drop=True)

    Wp = side(w_side, ["r_ko_w", "r_sub_w", "r_dec_w", "n_pre"])
    Lp = side(l_side, ["r_ko_l", "r_sub_l"])
    X = pd.DataFrame({
        "w_ko_rate": Wp["r_ko_w"], "w_sub_rate": Wp["r_sub_w"],
        "w_dec_rate": Wp["r_dec_w"], "w_n": Wp["n_pre"],
        "l_ko_l": Lp["r_ko_l"], "l_sub_l": Lp["r_sub_l"],
        "l_age": (fights["date"] - pd.to_datetime(fights[f"dob_{l_side}"])).dt.days / 365.25,
        "w_age": (fights["date"] - pd.to_datetime(fights[f"dob_{w_side}"])).dt.days / 365.25,
        "heavy": fights["weightclass"].astype(str).str.contains(
            "Heavyweight", case=False).astype(float),
        "women": fights["weightclass"].astype(str).str.contains(
            "Women", case=False).astype(float),
        "five_rd": fights["time_format"].astype(str).str.contains(
            "5 Rnd").astype(float),
    })
    return X[FEATS].fillna(0)


def train(fights):
    L = career_method_rates(fights)
    mcls = method_class(fights["method"])
    ok = fights["winner"].isin(["A", "B"]) & (mcls != "OTHER")
    W = fights[ok].reset_index(drop=True)
    Lok = L[L["idx"].isin(fights.index[ok])]
    # remap idx to positional
    remap = {old: new for new, old in enumerate(fights.index[ok])}
    Lok = Lok.assign(idx=Lok["idx"].map(remap))
    wside = np.where(W["winner"] == "A", "a", "b")
    Xa = frame_features(W, Lok, "a", "b")
    Xb = frame_features(W, Lok, "b", "a")
    X = pd.DataFrame(np.where(wside[:, None] == "a", Xa, Xb), columns=FEATS)
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
    """For each fight row, return fair method odds for both sides.
    p_win_a: model P(fighter_a wins) per row."""
    L = career_method_rates(fights_all)
    # positions of fight_rows within fights_all
    remap = {old: new for new, old in enumerate(fight_rows.index)}
    Lr = L[L["idx"].isin(fight_rows.index)].assign(
        idx=lambda d: d["idx"].map(remap))
    fr = fight_rows.reset_index(drop=True)
    out = []
    Pa = clf.predict_proba(frame_features(fr, Lr, "a", "b"))
    Pb = clf.predict_proba(frame_features(fr, Lr, "b", "a"))
    classes = list(clf[-1].classes_)
    for i in range(len(fr)):
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
