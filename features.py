"""Point-in-time (leakage-free) feature engineering.

Strategy
--------
1. Explode the fight table to "long" format: one row per (fighter, fight),
   containing that fighter's in-fight stats and outcome.
2. Sort by fighter + date, compute expanding career aggregates, then
   .shift(1) within each fighter so a fight NEVER contributes to its own
   features. This is the leakage firewall.
3. Merge the shifted career stats back onto the wide fight table for both
   corners, and build A-minus-B differentials.

Expected input schema (wide, one row per fight)
-----------------------------------------------
date, fighter_a, fighter_b, winner ('A'/'B'/'draw'),
dob_a, dob_b, reach_a, reach_b, height_a, height_b,
sig_str_landed_a/b   : significant strikes landed in that fight
sig_str_absorbed_a/b : significant strikes absorbed in that fight
td_landed_a/b, td_attempted_a/b
fight_time_min       : fight duration in minutes
method               : e.g. 'KO/TKO', 'SUB', 'DEC'

Any missing stat columns are simply skipped.
"""

import numpy as np
import pandas as pd

from elo import compute_elo
from identity import fighter_keys

# Per-fight stats we accumulate into career rates. (name, per-minute?)
STAT_COLS = [
    ("sig_str_landed", True),
    ("sig_str_absorbed", True),
    ("td_landed", False),
    ("td_attempted", False),
]


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (fighter, fight) with that fighter's stats + result."""
    frames = []
    for side, opp in (("a", "b"), ("b", "a")):
        cols = {
            "date": df["date"],
            "fight_id": df["fight_id"],
            "fighter": fighter_keys(df, side),
            "won": (df["winner"] == side.upper()).astype(float),
            "finish": df.get(
                "method", pd.Series("", index=df.index)
            ).astype(str).str.upper().str.contains("KO|SUB").astype(float),
            "fight_time_min": df.get(
                "fight_time_min", pd.Series(np.nan, index=df.index)
            ),
        }
        for stat, _ in STAT_COLS:
            c = f"{stat}_{side}"
            if c in df.columns:
                cols[stat] = df[c]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def _career_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Expanding career aggregates, shifted so current fight is excluded."""
    g = long_df.sort_values(["fighter", "date"], kind="stable").copy()
    grp = g.groupby("fighter", sort=False)

    # --- everything below uses shift(1): strictly pre-fight information ---
    g["career_fights"] = grp.cumcount()  # fights BEFORE this one
    g["career_wins"] = grp["won"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)
    g["career_winrate"] = (
        g["career_wins"] / g["career_fights"].replace(0, np.nan)
    ).fillna(0.5)  # unknown fighters get a neutral prior

    g["career_minutes"] = grp["fight_time_min"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    g["career_finish_rate"] = grp["finish"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(0)

    # Win streak going into the fight
    def _streak(s: pd.Series) -> pd.Series:
        out, run = [], 0
        for w in s.shift(1):
            out.append(run)
            if pd.isna(w):
                continue
            run = run + 1 if w == 1.0 else 0
        return pd.Series(out, index=s.index)

    g["win_streak"] = grp["won"].transform(_streak)

    # Layoff: days since previous fight
    g["days_since_last"] = grp["date"].transform(
        lambda s: (s - s.shift(1)).dt.days
    ).fillna(365)

    # Per-minute career rates for volume stats
    for stat, per_min in STAT_COLS:
        if stat not in g.columns:
            continue
        cum = grp[stat].transform(lambda s: s.shift(1).expanding().sum())
        if per_min:
            g[f"career_{stat}_pm"] = (
                cum / g["career_minutes"].replace(0, np.nan)
            ).fillna(0)
        else:
            g[f"career_{stat}_total"] = cum.fillna(0)

    # Takedown accuracy
    if {"career_td_landed_total", "career_td_attempted_total"} <= set(g.columns):
        g["career_td_acc"] = (
            g["career_td_landed_total"]
            / g["career_td_attempted_total"].replace(0, np.nan)
        ).fillna(0)

    keep = ["fight_id", "fighter"] + [
        c for c in g.columns
        if c.startswith(("career_", "win_streak", "days_since_last"))
    ]
    return g[keep]


def build_features(fights: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (feature table, feature column names).

    Output has one row per fight with A-minus-B differential features and
    the binary target `target` (1 = fighter A won). Draws are dropped.
    """
    df = fights.sort_values("date", kind="stable").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["fight_id"] = np.arange(len(df))
    df["_fighter_key_a"] = fighter_keys(df, "a")
    df["_fighter_key_b"] = fighter_keys(df, "b")

    # 1) Elo (pre-fight by construction)
    df = compute_elo(df)

    # 2) Point-in-time career stats
    career = _career_stats(_to_long(df))
    stat_cols = [c for c in career.columns if c not in ("fight_id", "fighter")]

    for side in ("a", "b"):
        merged = df[["fight_id", f"_fighter_key_{side}"]].merge(
            career,
            left_on=["fight_id", f"_fighter_key_{side}"],
            right_on=["fight_id", "fighter"],
            how="left",
        )
        for c in stat_cols:
            df[f"{c}_{side}"] = merged[c].values

    # 3) Physical / age differentials
    for c in ("reach", "height"):
        if {f"{c}_a", f"{c}_b"} <= set(df.columns):
            df[f"{c}_diff"] = df[f"{c}_a"] - df[f"{c}_b"]
    if {"dob_a", "dob_b"} <= set(df.columns):
        age_a = (df["date"] - pd.to_datetime(df["dob_a"])).dt.days / 365.25
        age_b = (df["date"] - pd.to_datetime(df["dob_b"])).dt.days / 365.25
        df["age_a"], df["age_b"] = age_a, age_b
        df["age_diff"] = age_a - age_b

    # 4) Career-stat differentials
    for c in stat_cols:
        df[f"{c}_diff"] = df[f"{c}_a"] - df[f"{c}_b"]

    # 5) Assemble
    feature_cols = ["elo_diff"] + [
        c for c in df.columns
        if c.endswith("_diff") and c != "elo_diff"
    ]
    df = df[df["winner"].isin(["A", "B"])].copy()
    df["target"] = (df["winner"] == "A").astype(int)

    identity_columns = [
        column for column in (
            "fighter_a_id", "fighter_b_id", "fighter_a_url", "fighter_b_url"
        ) if column in df
    ]
    out = df[["fight_id", "date", "fighter_a", "fighter_b"]
             + identity_columns + ["target"] + feature_cols].copy()
    out[feature_cols] = out[feature_cols].fillna(0)
    return out, feature_cols


def symmetrize(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Duplicate every fight with corners swapped (all differentials negated,
    target flipped). Removes corner-assignment bias and enforces
    P(A beats B) ≈ 1 - P(B beats A)."""
    flipped = X.copy()
    flipped[feature_cols] = -flipped[feature_cols]
    flipped["target"] = 1 - flipped["target"]
    return pd.concat([X, flipped], ignore_index=True)
