"""Adapter: raw UFCStats scrape (Greco1899/scrape_ufc_stats CSVs, the same
source behind most Kaggle UFC datasets) -> fights_v2.csv in the schema
expected by the feature builders.

Usage:
    python adapter.py --raw-dir raw --out fights_v2.csv

Inputs expected in --raw-dir:
    ufc_event_details.csv   (EVENT, DATE, ...)
    ufc_fight_results.csv   (EVENT, BOUT, OUTCOME, METHOD, ROUND, TIME, ...)
    ufc_fight_stats.csv     (per-round per-fighter stats)
    ufc_fighter_tott.csv    (HEIGHT, REACH, DOB per fighter)
"""

import argparse
import re

import numpy as np
import pandas as pd


def parse_of(s):
    """'23 of 38' -> (23, 38); missing -> (nan, nan)."""
    if isinstance(s, str):
        m = re.match(r"\s*(\d+)\s+of\s+(\d+)", s)
        if m:
            return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan


def parse_height(s):
    """5' 11" -> inches."""
    if isinstance(s, str):
        m = re.match(r"(\d+)'\s*(\d+)", s)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
    return np.nan


def parse_reach(s):
    if isinstance(s, str):
        m = re.match(r"(\d+)", s.strip('" '))
        if m:
            return float(m.group(1))
    return np.nan


def fight_minutes(round_no, time_str, time_format):
    """Total fight duration. Completed rounds are 5 min each in modern UFC;
    very old events had different formats — we approximate with 5."""
    try:
        mm, ss = time_str.split(":")
        return (int(round_no) - 1) * 5 + int(mm) + int(ss) / 60.0
    except Exception:
        return np.nan


def build(raw_dir: str) -> pd.DataFrame:
    def load(name):
        df = pd.read_csv(f"{raw_dir}/{name}")
        for c in df.columns:
            if not pd.api.types.is_string_dtype(df[c]):
                continue
            df[c] = df[c].str.strip()
        return df

    ev = load("ufc_event_details.csv")
    res = load("ufc_fight_results.csv")
    stats = load("ufc_fight_stats.csv")
    tott = load("ufc_fighter_tott.csv")

    # --- events: name -> date ------------------------------------------
    ev["date"] = pd.to_datetime(ev["DATE"], format="mixed", errors="coerce")
    ev = ev[["EVENT", "date"]].drop_duplicates("EVENT")

    # --- results: one row per fight -------------------------------------
    res = res.merge(ev, on="EVENT", how="left")
    ab = res["BOUT"].str.split(r"\s+vs\.?\s+", n=1, regex=True, expand=True)
    res["fighter_a"], res["fighter_b"] = ab[0].str.strip(), ab[1].str.strip()
    res["winner"] = res["OUTCOME"].map(
        {"W/L": "A", "L/W": "B", "D/D": "draw"}
    )  # NC/NC -> NaN, dropped below
    res["fight_time_min"] = [
        fight_minutes(r, t, f)
        for r, t, f in zip(res["ROUND"], res["TIME"], res["TIME FORMAT"])
    ]
    res = res.dropna(subset=["date", "winner", "fighter_a", "fighter_b"])
    res["method"] = res["METHOD"].fillna("")
    res["event"] = res["EVENT"]
    res["bout"] = res["BOUT"]
    res["time_format"] = res["TIME FORMAT"].fillna("")
    res["weightclass"] = res["WEIGHTCLASS"].fillna("")

    # --- per-round stats -> per-fight per-fighter totals -----------------
    landed_att = stats["SIG.STR."].map(parse_of)
    stats["sig_landed"] = [x[0] for x in landed_att]
    td = stats["TD"].map(parse_of)
    stats["td_landed"] = [x[0] for x in td]
    stats["td_att"] = [x[1] for x in td]

    agg = (
        stats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)
        .agg(sig_landed=("sig_landed", "sum"),
             td_landed=("td_landed", "sum"),
             td_att=("td_att", "sum"))
    )

    def side_stats(side_col, prefix):
        m = res[["EVENT", "BOUT", side_col]].merge(
            agg, left_on=["EVENT", "BOUT", side_col],
            right_on=["EVENT", "BOUT", "FIGHTER"], how="left",
        )
        return m[["sig_landed", "td_landed", "td_att"]].rename(
            columns={
                "sig_landed": f"sig_str_landed_{prefix}",
                "td_landed": f"td_landed_{prefix}",
                "td_att": f"td_attempted_{prefix}",
            }
        )

    sa = side_stats("fighter_a", "a").reset_index(drop=True)
    sb = side_stats("fighter_b", "b").reset_index(drop=True)
    res = pd.concat([res.reset_index(drop=True), sa, sb], axis=1)

    # Strikes absorbed = opponent's strikes landed
    res["sig_str_absorbed_a"] = res["sig_str_landed_b"]
    res["sig_str_absorbed_b"] = res["sig_str_landed_a"]

    # --- fighter physicals ----------------------------------------------
    tott = tott.drop_duplicates("FIGHTER", keep="first").copy()
    tott["height_in"] = tott["HEIGHT"].map(parse_height)
    tott["reach_in"] = tott["REACH"].map(parse_reach)
    tott["dob"] = pd.to_datetime(tott["DOB"], format="mixed", errors="coerce")
    tott["stance"] = tott["STANCE"]
    phys = tott[["FIGHTER", "height_in", "reach_in", "dob", "stance"]]

    for side in ("a", "b"):
        res = res.merge(
            phys.rename(columns={
                "FIGHTER": f"fighter_{side}",
                "height_in": f"height_{side}",
                "reach_in": f"reach_{side}",
                "dob": f"dob_{side}",
                "stance": f"stance_{side}",
            }),
            on=f"fighter_{side}", how="left",
        )

    cols = [
        "date", "event", "bout", "time_format", "weightclass",
        "stance_a", "stance_b", "fighter_a", "fighter_b", "winner", "method",
        "fight_time_min",
        "dob_a", "dob_b", "reach_a", "reach_b", "height_a", "height_b",
        "sig_str_landed_a", "sig_str_landed_b",
        "sig_str_absorbed_a", "sig_str_absorbed_b",
        "td_landed_a", "td_landed_b",
        "td_attempted_a", "td_attempted_b",
    ]
    out = res[cols].sort_values("date").reset_index(drop=True)

    # Known caveat: UFCStats lists the winner first in most historical
    # bouts, so 'A' is heavily the winner. symmetrize() in training and
    # sign-flipped evaluation handle this; do NOT interpret raw corner
    # frequencies as signal.
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out", default="fights_v2.csv")
    args = ap.parse_args()

    df = build(args.raw_dir)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} fights to {args.out}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Winner balance: {df['winner'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Missing reach: {(df['reach_a'].isna() | df['reach_b'].isna()).mean():.1%}")
    print(f"Missing stats: {df['sig_str_landed_a'].isna().mean():.1%}")
