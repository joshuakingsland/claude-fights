"""Features v3 = v2 + weight-class context + interactions + bout flags.

Post-hoc additions (joined back from fights_v2.csv by fight_id, which is
the date-sorted row order used inside build_features_v2):

  reach_z_diff : reach z-scored within weight class (a 76" reach means
                 something different at flyweight vs heavyweight)
  heavy_bout   : heavyweight/LHW bout flag (bout-level, higher variance)
  five_rd_bout : scheduled 5-rounder (bout-level)
  age_x_off    : age_diff x layoff_diff (ring rust hits older fighters)
  ko_recent    : fighting within 365d of being KO'd (signed diff)

BOUT_COLS are bout-level facts and must NOT be sign-flipped when moving
to the red-corner frame.
"""

import numpy as np
import pandas as pd

from features_v2 import build_features_v2

BOUT_COLS = ["heavy_bout", "five_rd_bout"]


def build_features_v3(fights):
    feats, fcols = build_features_v2(fights)

    raw = fights.sort_values("date", kind="stable").reset_index(drop=True)
    raw["fight_id"] = np.arange(len(raw))
    aux = raw[["fight_id", "weightclass", "time_format",
               "reach_a", "reach_b"]].copy()
    feats = feats.merge(aux, on="fight_id", how="left")

    # ---- weight-class-normalized reach ---------------------------------
    wc = feats["weightclass"].astype(str)
    reach_long = pd.concat([
        pd.DataFrame({"wc": wc, "reach": feats["reach_a"]}),
        pd.DataFrame({"wc": wc, "reach": feats["reach_b"]}),
    ])
    stats = reach_long.groupby("wc")["reach"].agg(["mean", "std"])
    mu = wc.map(stats["mean"])
    sd = wc.map(stats["std"]).replace(0, np.nan)
    za = (feats["reach_a"] - mu) / sd
    zb = (feats["reach_b"] - mu) / sd
    feats["reach_z_diff"] = (za - zb).fillna(0)

    # ---- bout-level flags ------------------------------------------------
    feats["heavy_bout"] = wc.str.contains(
        "Heavyweight", case=False).astype(float)
    feats["five_rd_bout"] = feats["time_format"].astype(str) \
        .str.contains("5 Rnd").astype(float)

    # ---- interactions ----------------------------------------------------
    feats["age_x_off"] = feats["age_diff"] * feats["days_off_diff"] / 365.0
    dsk = feats["days_since_ko_diff"]
    feats["ko_recent"] = (((dsk != 0) & (dsk.abs() < 365))
                          * np.sign(dsk)).astype(float)

    new = ["reach_z_diff", "age_x_off", "ko_recent"] + BOUT_COLS
    feats[new] = feats[new].fillna(0)
    return feats.drop(columns=["weightclass", "time_format",
                               "reach_a", "reach_b"]), fcols + new
