"""Feature engineering v2 — hunting for signal the market may underprice.

New relative to features.py:

In-fight stats mined from round-level data:
  knockdowns scored/absorbed, control time for/against, sub attempts,
  striking accuracy, head-strike share, distance share,
  late-round fade (per-min output in R3+ minus R1).

Career features (all point-in-time via shift(1)):
  recent-3 form (win rate, output), opponent-adjusted striking output
  (own landed rate minus what that opponent's previous foes landed on
  them), KO/sub-loss aftermath (came off KO loss, KO losses absorbed,
  days since last KO'd), fast Elo (K=64) vs slow Elo (K=20) split
  (form vs class), stance matchup, five-round-fight experience,
  age x mileage interaction.

Output: same contract as features.py -> (feature table, feature cols).
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- helpers
def _of(series):
    l, a = [], []
    for s in series:
        m = re.match(r"\s*(\d+)\s+of\s+(\d+)", s) if isinstance(s, str) else None
        l.append(float(m.group(1)) if m else np.nan)
        a.append(float(m.group(2)) if m else np.nan)
    return np.array(l), np.array(a)


def _ctrl_sec(series):
    out = []
    for s in series:
        try:
            mm, ss = str(s).split(":")
            out.append(int(mm) * 60 + int(ss))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


def _shifted(grp, col, how):
    if how == "sum":
        return grp[col].transform(lambda s: s.shift(1).expanding().sum())
    if how == "mean":
        return grp[col].transform(lambda s: s.shift(1).expanding().mean())
    if how == "last3":
        return grp[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    raise ValueError(how)


# ------------------------------------------------------- round-level stats
def load_round_stats(raw_dir="raw"):
    st = pd.read_csv(f"{raw_dir}/ufc_fight_stats.csv")
    for c in st.columns:
        if pd.api.types.is_string_dtype(st[c]):
            st[c] = st[c].str.strip()

    st["round_no"] = st["ROUND"].str.extract(r"(\d+)").astype(float)
    st["sig_l"], st["sig_a"] = _of(st["SIG.STR."])
    st["head_l"], _ = _of(st["HEAD"])
    st["dist_l"], _ = _of(st["DISTANCE"])
    st["td_l"], st["td_a"] = _of(st["TD"])
    st["kd"] = pd.to_numeric(st["KD"], errors="coerce")
    st["sub_att"] = pd.to_numeric(st["SUB.ATT"], errors="coerce")
    st["ctrl_s"] = _ctrl_sec(st["CTRL"])

    g = st.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)
    per = g.agg(
        sig_l=("sig_l", "sum"), sig_a=("sig_a", "sum"),
        head_l=("head_l", "sum"), dist_l=("dist_l", "sum"),
        td_l=("td_l", "sum"), td_a=("td_a", "sum"),
        kd=("kd", "sum"), sub_att=("sub_att", "sum"),
        ctrl_s=("ctrl_s", "sum"), n_rounds=("round_no", "max"),
    )

    # Late-round fade: per-round landed in R3+ minus R1 (needs 3+ rounds).
    r1 = st[st.round_no == 1].groupby(["EVENT", "BOUT", "FIGHTER"])["sig_l"].sum()
    r3p = st[st.round_no >= 3].groupby(["EVENT", "BOUT", "FIGHTER"]).agg(
        l=("sig_l", "sum"), n=("round_no", "nunique"))
    fade = (r3p["l"] / r3p["n"] - r1).rename("fade")
    per = per.merge(fade, on=["EVENT", "BOUT", "FIGHTER"], how="left")
    return per


# --------------------------------------------------------------- main build
def build_features_v2(fights, raw_dir="raw"):
    df = fights.sort_values("date", kind="stable").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["fight_id"] = np.arange(len(df))
    df["is_5rd"] = df.get("time_format", pd.Series("", index=df.index)) \
        .astype(str).str.contains("5 Rnd").astype(float)

    per = load_round_stats(raw_dir)

    # ---- long format: one row per (fighter, fight) ----------------------
    frames = []
    for side, opp in (("a", "b"), ("b", "a")):
        d = pd.DataFrame({
            "fight_id": df["fight_id"], "date": df["date"],
            "event": df["event"], "bout": df["bout"],
            "fighter": df[f"fighter_{side}"],
            "opponent": df[f"fighter_{opp}"],
            "won": (df["winner"] == side.upper()).astype(float),
            "t_min": df["fight_time_min"],
            "is_5rd": df["is_5rd"],
        })
        m = df["method"].astype(str).str.upper()
        d["ko_win"] = ((df["winner"] == side.upper()) & m.str.contains("KO")).astype(float)
        d["ko_loss"] = ((df["winner"] == opp.upper()) & m.str.contains("KO")).astype(float)
        d["sub_loss"] = ((df["winner"] == opp.upper()) & m.str.contains("SUB")).astype(float)
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)

    long = long.merge(per, left_on=["event", "bout", "fighter"],
                      right_on=["EVENT", "BOUT", "FIGHTER"], how="left")
    opp_per = per.rename(columns={c: f"opp_{c}" for c in
                                  ["sig_l", "sig_a", "kd", "ctrl_s", "td_l"]})
    long = long.merge(
        opp_per[["EVENT", "BOUT", "FIGHTER",
                 "opp_sig_l", "opp_kd", "opp_ctrl_s", "opp_td_l"]],
        left_on=["event", "bout", "opponent"],
        right_on=["EVENT", "BOUT", "FIGHTER"], how="left", suffixes=("", "_o"))

    # Per-fight rates
    t = long["t_min"].replace(0, np.nan)
    long["lpm"] = long["sig_l"] / t                       # landed per min
    long["apm"] = long["opp_sig_l"] / t                   # absorbed per min
    long["acc"] = long["sig_l"] / long["sig_a"].replace(0, np.nan)
    long["head_share"] = long["head_l"] / long["sig_l"].replace(0, np.nan)
    long["dist_share"] = long["dist_l"] / long["sig_l"].replace(0, np.nan)
    long["kd_pm"] = long["kd"] / t
    long["kdd_pm"] = long["opp_kd"] / t                   # knockdowns absorbed
    long["ctrl_pm"] = long["ctrl_s"] / 60 / t
    long["ctrld_pm"] = long["opp_ctrl_s"] / 60 / t
    long["sub_pm"] = long["sub_att"] / t
    long["tdd"] = 1 - long["opp_td_l"] / t                # crude TD defense proxy

    # ---------------- career aggregates (ALL shifted: pre-fight only) -----
    long = long.sort_values(["fighter", "date"], kind="stable")
    grp = long.groupby("fighter", sort=False)

    C = {}
    for col in ["lpm", "apm", "acc", "head_share", "dist_share", "kd_pm",
                "kdd_pm", "ctrl_pm", "ctrld_pm", "sub_pm", "fade", "tdd",
                "won", "ko_win", "is_5rd"]:
        C[f"c_{col}"] = _shifted(grp, col, "mean")
    for col in ["ko_loss", "sub_loss"]:
        C[f"c_{col}_n"] = _shifted(grp, col, "sum")
    for col in ["won", "lpm", "kdd_pm"]:
        C[f"r3_{col}"] = _shifted(grp, col, "last3")     # recent-3 form

    C["c_fights"] = grp.cumcount().astype(float)
    C["c_minutes"] = _shifted(grp, "t_min", "sum")
    C["days_off"] = grp["date"].transform(lambda s: (s - s.shift(1)).dt.days)
    C["off_ko"] = grp["ko_loss"].transform(lambda s: s.shift(1))  # last fight KO loss?

    def _days_since_ko(g):
        last, out = None, []
        for d, kol in zip(g["date"], g["ko_loss"]):
            out.append((d - last).days if last is not None else np.nan)
            if kol == 1.0:
                last = d
        return pd.Series(out, index=g.index)
    C["days_since_ko"] = grp[["date", "ko_loss"]].apply(_days_since_ko) \
        .reset_index(level=0, drop=True)

    for k, v in C.items():
        long[k] = v

    # Opponent-adjusted output: my landed rate in fight f minus what the
    # opponent had been absorbing before f (both strictly pre/known-at-f).
    opp_absorb = long[["fight_id", "fighter", "c_apm"]].rename(
        columns={"fighter": "opponent", "c_apm": "opp_c_apm"})
    long = long.merge(opp_absorb, on=["fight_id", "opponent"], how="left")
    long["adj_out"] = long["lpm"] - long["opp_c_apm"]
    # Opponent-adjusted defense: what I absorbed vs what that opponent
    # normally lands (negative = better-than-expected defense).
    opp_output = long[["fight_id", "fighter", "c_lpm"]].rename(
        columns={"fighter": "opponent", "c_lpm": "opp_c_lpm"})
    long = long.merge(opp_output, on=["fight_id", "opponent"], how="left")
    long["adj_def"] = long["apm"] - long["opp_c_lpm"]
    long = long.sort_values(["fighter", "date"], kind="stable")
    g2 = long.groupby("fighter", sort=False)
    long["c_adj_out"] = _shifted(g2, "adj_out", "mean")
    long["c_adj_def"] = _shifted(g2, "adj_def", "mean")

    feat_cols_f = [k for k in C] + ["c_adj_out", "c_adj_def"]

    # ---------------- fast & slow Elo -------------------------------------
    from elo import compute_elo
    for tag, k in (("fast", 64.0), ("slow", 20.0)):
        e = compute_elo(df, k=k)[["fight_id", "elo_a_pre", "elo_b_pre"]]
        df[f"elo_{tag}_a"] = e["elo_a_pre"]
        df[f"elo_{tag}_b"] = e["elo_b_pre"]

    # ---------------- merge fighter-level features back to wide ----------
    keep = long[["fight_id", "fighter"] + feat_cols_f]
    for side in ("a", "b"):
        m = df[["fight_id", f"fighter_{side}"]].merge(
            keep, left_on=["fight_id", f"fighter_{side}"],
            right_on=["fight_id", "fighter"], how="left")
        for c in feat_cols_f:
            df[f"{c}_{side}"] = m[c].values

    # ---------------- stance, physicals, age ------------------------------
    if "stance_a" in df.columns:
        for s in ("a", "b"):
            df[f"southpaw_{s}"] = (df[f"stance_{s}"] == "Southpaw").astype(float)
        df["stance_edge"] = df["southpaw_a"] - df["southpaw_b"]
    for c in ("reach", "height"):
        if {f"{c}_a", f"{c}_b"} <= set(df.columns):
            df[f"{c}_diff"] = df[f"{c}_a"] - df[f"{c}_b"]
    age_a = (df["date"] - pd.to_datetime(df["dob_a"])).dt.days / 365.25
    age_b = (df["date"] - pd.to_datetime(df["dob_b"])).dt.days / 365.25
    df["age_diff"] = age_a - age_b
    df["agemile_a"] = age_a * df["c_minutes_a"].fillna(0) / 100
    df["agemile_b"] = age_b * df["c_minutes_b"].fillna(0) / 100

    # ---------------- differentials ---------------------------------------
    diffs = {
        "elo_fast_diff": df["elo_fast_a"] - df["elo_fast_b"],
        "elo_slow_diff": df["elo_slow_a"] - df["elo_slow_b"],
        "agemile_diff": df["agemile_a"] - df["agemile_b"],
    }
    for c in feat_cols_f:
        diffs[f"{c}_diff"] = df[f"{c}_a"] - df[f"{c}_b"]
    for k, v in diffs.items():
        df[k] = v

    feature_cols = list(diffs.keys()) + ["age_diff", "reach_diff",
                                         "height_diff"]
    if "stance_edge" in df.columns:
        feature_cols.append("stance_edge")

    df = df[df["winner"].isin(["A", "B"])].copy()
    df["target"] = (df["winner"] == "A").astype(int)
    out = df[["fight_id", "date", "fighter_a", "fighter_b", "target"]
             + feature_cols].copy()

    # days_since_ko: NaN means never KO'd -> large value; off_ko NaN -> 0
    for s in ("",):
        pass
    out["days_since_ko_diff"] = out["days_since_ko_diff"].fillna(0)
    out[feature_cols] = out[feature_cols].fillna(0)
    return out, feature_cols
