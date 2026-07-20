"""Predict the upcoming card and generate the site (docs/index.html).

Reads odds_upcoming.csv (from fetch_odds.py or edited by hand), computes
point-in-time features for each matchup, trains the production model on
all matched history, and writes a self-contained HTML page with:
  - the upcoming card: model probability vs market, edge, bet flags
  - a rolling results ledger: how model picks fared at recent events

Usage: python predict_card.py
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import logit as slogit

from adapter import parse_height, parse_reach
from backtest import american_to_prob, american_payout, norm_name
from data_quality import assert_clean, identity_warnings
from features_v3 import build_features_v3
from pipeline import load_matched_cached
from config import EDGE_RULE, FOCUS
from production import (MODEL_FEATURES, event_pnl, fit_ensemble,
                        predict_probabilities, score_bets)
import method_model as MM

# --------------------------------------------------------------- modeling
def resolve_names(up, fights):
    known = {}
    for s in ("a", "b"):
        for n in fights[f"fighter_{s}"]:
            known.setdefault(norm_name(n), set()).add(n)
    ambiguous = ({key for key, values in known.items() if len(values) > 1}
                 | HOMONYMS)
    for s in ("a", "b"):
        fixed = []
        for n in up[f"fighter_{s}"]:
            k = norm_name(n)
            if k in ambiguous:
                # Never blend two fighters behind one normalized name. Keep
                # the display name separately and force neutral features.
                fixed.append(f"__AMBIGUOUS__{k}")
                print(f"  WARNING: '{n}' is ambiguous in historical data "
                      "and will receive neutral features")
            elif k in known:
                fixed.append(next(iter(known[k])))
            else:
                # exact match only — a last-name fuzzy fallback once assigned
                # a debutant an unrelated veteran's entire career. Unknown
                # fighters stay unknown (neutral features).
                fixed.append(n)
                print(f"  note: '{n}' not in data — treated as debutant")
        up[f"fighter_{s}"] = fixed
    return up


HOMONYMS = {"bruno silva", "jean silva", "mike davis", "victor valenzuela"}


def predict_upcoming(up):
    fights = pd.read_csv("fights_v2.csv", parse_dates=["date"])
    assert_clean(fights, up)
    for warning in identity_warnings(fights):
        print(f"  WARNING: {warning}")
    up = up.copy()
    up["display_a"] = up["fighter_a"]
    up["display_b"] = up["fighter_b"]
    up = resolve_names(up.copy(), fights)
    for s in ("a", "b"):
        for n in up[f"fighter_{s}"]:
            if norm_name(n) in HOMONYMS:
                print(f"  WARNING: '{n}' matches multiple distinct UFC "
                      f"fighters — career features are blended; treat this "
                      f"fight's numbers with caution")

    rows = []
    for _, r in up.iterrows():
        rows.append({"date": pd.Timestamp(r["date"]), "event": "UPCOMING",
                     "bout": f"{r['display_a']} vs. {r['display_b']}",
                     "time_format": ("5 Rnd (5-5-5-5-5)"
                                     if int(r.get("five_rounds", 0) or 0)
                                     else "3 Rnd (5-5-5)"),
                     "weightclass": r.get("weightclass", "") or "",
                     "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"],
                     "winner": "A", "method": "", "fight_time_min": np.nan})
    hyp = pd.DataFrame(rows)

    phys = pd.read_csv("raw/ufc_fighter_tott.csv")
    for c in phys.columns:
        if pd.api.types.is_string_dtype(phys[c]):
            phys[c] = phys[c].str.strip()
    phys["height_in"] = phys["HEIGHT"].map(parse_height)
    phys["reach_in"] = phys["REACH"].map(parse_reach)
    phys["dob"] = pd.to_datetime(phys["DOB"], format="mixed", errors="coerce")
    pm = phys.drop_duplicates("FIGHTER").set_index("FIGHTER")
    for s in ("a", "b"):
        hyp[f"height_{s}"] = hyp[f"fighter_{s}"].map(pm["height_in"])
        hyp[f"reach_{s}"] = hyp[f"fighter_{s}"].map(pm["reach_in"])
        hyp[f"dob_{s}"] = hyp[f"fighter_{s}"].map(pm["dob"])
        hyp[f"stance_{s}"] = hyp[f"fighter_{s}"].map(pm["STANCE"])

    feats, _ = build_features_v3(pd.concat([fights, hyp], ignore_index=True))
    new = feats.merge(hyp[["fighter_a", "fighter_b"]].assign(_u=1),
                      on=["fighter_a", "fighter_b"], how="inner")
    new = new[new["date"] >= hyp["date"].min()]

    m, _ = load_matched_cached(build_features_v3, "v3", bout_cols=[])
    ensemble = fit_ensemble(m, n_models=30, seed=0)
    lr = ensemble[0]
    cols = MODEL_FEATURES

    out = []
    for _, r in up.iterrows():
        row = new[(new["fighter_a"] == r["fighter_a"])
                  & (new["fighter_b"] == r["fighter_b"])]
        oa = float(str(r["odds_a"]).replace("+", ""))
        ob = float(str(r["odds_b"]).replace("+", ""))
        pa, pb = american_to_prob(oa), american_to_prob(ob)
        p_line = float(pa / (pa + pb))
        if not len(row):
            continue
        ll = slogit(np.clip(p_line, 0.02, 0.98))
        fv = [row.iloc[0][c] for c in FOCUS + ["ko_recent"]]
        X = pd.DataFrame([[ll, abs(ll)] + fv], columns=cols)
        p, se = predict_probabilities(ensemble, X)
        p, se = float(p[0]), float(se[0])
        ea, eb = p - pa, (1 - p) - pb
        pick_a = ea >= eb
        net = max(ea, eb) - se
        feat_vals = [row.iloc[0][c] for c in FOCUS + ["ko_recent"]]
        feat_pick = feat_vals if pick_a else [-v for v in feat_vals]
        imp_now = pa if pick_a else pb
        ladder = sizing_ladder(lr, cols, feat_pick, float(imp_now), float(pa + pb - 1), se_sub=se)
        out.append({
            "_p_a": p, "_row_idx": int(row.index[0]),
            "ladder": ladder,
            "pick": r["display_a"] if pick_a else r["display_b"],
            "opp": r["display_b"] if pick_a else r["display_a"],
            "price": f"{int(oa):+d}" if pick_a else f"{int(ob):+d}",
            "market": round((pa if pick_a else pb) / (pa + pb) * 100, 1),
            "model": round((p if pick_a else 1 - p) * 100, 1),
            "edge": round(max(ea, eb) * 100, 1),
            "se": round(se * 100, 1),
            "net": round(net * 100, 1),
            "bet": bool(net >= EDGE_RULE),
            "stake": (2 if net >= 2 * EDGE_RULE else
                       (1 if net >= EDGE_RULE else 0)),
            "date": str(pd.Timestamp(r["date"]).date()),
            "meta": f"{r.get('weightclass','') or 'TBD'} · {r['date']}",
        })
    # method props (fair prices; probability-validated, no verified prop edge)
    try:
        import pickle
        clf = pickle.load(open("method_model.pkl", "rb"))
        allf = pd.concat([fights, hyp], ignore_index=True)
        allf["date"] = pd.to_datetime(allf["date"])
        fr = allf[allf["event"] == "UPCOMING"]
        if len(fr) == len(out):
            props = MM.method_props(clf, allf, fr, [o["_p_a"] for o in out])
            for o, pr, (_, r) in zip(out, props, fr.iterrows()):
                pick_is_a = norm_name(o["pick"]) == norm_name(r["fighter_a"])
                o["props"] = {("pick_" if (k[0] == "a") == pick_is_a
                               else "opp_") + k[2:]: v
                              for k, v in pr.items()}
    except Exception as exc:
        print("props skipped:", exc)
    for o in out:
        o.pop("_p_a", None); o.pop("_row_idx", None)
    return out


def recent_results(days=120):
    """Exact production walk-forward picks at recent completed events."""
    m, _ = load_matched_cached(build_features_v3, "v3", bout_cols=[])
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    window = m[m["date"] >= start]
    scored_events = []
    for event_no, date in enumerate(sorted(window["date"].unique())):
        train = m[m["date"] < date]
        test = window[window["date"] == date].copy()
        if len(train) < 2000:
            continue
        models = fit_ensemble(train, n_models=30, seed=event_no)
        p, se = predict_probabilities(models, test)
        scored = score_bets(test, p, se)
        scored["p_line"] = test["p_line"].to_numpy()
        scored["pnl"] = event_pnl(scored)
        scored["date"] = date
        scored_events.append(scored)
    if not scored_events:
        return [], {}
    te = pd.concat(scored_events, ignore_index=True)
    active = te["stake"] > 0
    staked = float(te["stake"].sum())
    summary = {"n": int(active.sum()), "pnl": round(float(te["pnl"].sum()), 2),
               "staked": round(staked, 2),
               "roi": round(float(te["pnl"].sum() / staked * 100), 1)
               if staked else 0.0}

    rows = []
    for i in te.index[::-1][:30]:
        pm = te.loc[i, "p_model"]
        pick_red = te.loc[i, "pick_side"] == "A"
        pick = te.loc[i, "fighter_a"] if (te.loc[i, "key_a"] == te.loc[i, "key_r"]) == pick_red \
            else te.loc[i, "fighter_b"]
        won = bool((te.loc[i, "y"] == 1) == pick_red)
        rows.append({"date": str(pd.Timestamp(te.loc[i, "date"]).date()), "pick": pick,
                     "model": round((pm if pick_red else 1 - pm) * 100),
                     "edge": round(float(te.loc[i, "net_edge"]) * 100, 1),
                     "bet": bool(te.loc[i, "stake"] > 0), "won": won})
    return rows, summary




def american_from_prob(p):
    """Fair American odds for an implied probability."""
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def sizing_ladder(lr, cols, feat_row, imp_now, vig, se_sub=0.0):
    """Worst acceptable price per stake tier, within +/-15 implied points
    of market. A tier is shown only if its qualifying region includes or
    extends to longer odds than the current price — i.e., the ladder
    answers 'how much worse a price can I take', never 'what if the line
    steams toward my pick'."""
    lo, hi = max(0.02, imp_now - 0.15), min(0.95, imp_now + 0.15)
    grid = np.linspace(lo, hi, 160)
    edges = []
    for imp in grid:
        opp = max((1 + vig) - imp, 1e-6)
        ll = slogit(np.clip(imp / (imp + opp), 0.02, 0.98))
        X = pd.DataFrame([[ll, abs(ll)] + feat_row], columns=cols)
        edges.append(float(lr.predict_proba(X)[0, 1]) - imp)
    step = grid[1] - grid[0]
    ths = {}
    for tier, cut in (("2u", 0.08), ("1u", 0.04)):
        regions, start = [], None
        for i, ed in enumerate(edges):
            if ed > cut + se_sub and start is None:
                start = i
            elif ed <= cut + se_sub and start is not None:
                regions.append((start, i - 1)); start = None
        if start is not None:
            regions.append((start, len(grid) - 1))
        for a, b in regions:
            if grid[a] <= imp_now + step:      # touches or extends longer than market
                ths[tier] = american_from_prob(grid[b])
                break
    return ths

# --------------------------------------------------------------- site
def build_site(upcoming, recent, summary):
    with open("site_template.html") as f:
        tpl = f.read()
    stamp = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    html = (tpl.replace("__UPCOMING__", json.dumps(upcoming))
               .replace("__RECENT__", json.dumps(recent))
               .replace("__SUMMARY__", json.dumps(summary))
               .replace("__STAMP__", stamp))
    import os
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(html)
    print(f"docs/index.html written "
          f"({len(upcoming)} upcoming, {len(recent)} recent)")


if __name__ == "__main__":
    up = pd.read_csv("odds_upcoming.csv")
    upcoming = predict_upcoming(up)
    recent, summary = recent_results()
    build_site(upcoming, recent, summary)
    from paper_ledger import record_predictions
    print(f"paper ledger: appended {record_predictions(upcoming)} predictions")
    from model_manifest import write_manifest
    write_manifest()
