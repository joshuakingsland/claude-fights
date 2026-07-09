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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adapter import parse_height, parse_reach
from backtest import american_to_prob, american_payout, norm_name
from features_v3 import build_features_v3
from pipeline import load_matched_cached, walk_forward, bets
from research3 import FOCUS
import method_model as MM

EDGE_RULE = 0.04
RECORD_CHIPS = [("backtest", "841 bets · 2019–2026"),
                ("ROI", "+6.9% flat stakes"),
                ("90% CI", "+0.9% … +13.1%"),
                ("vs closing line log loss", "0.6018 / 0.6039")]


# --------------------------------------------------------------- modeling
def resolve_names(up, fights):
    known = {}
    for s in ("a", "b"):
        for n in fights[f"fighter_{s}"]:
            known[norm_name(n)] = n
    for s in ("a", "b"):
        fixed = []
        for n in up[f"fighter_{s}"]:
            k = norm_name(n)
            if k in known:
                fixed.append(known[k])
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
                     "bout": f"{r['fighter_a']} vs. {r['fighter_b']}",
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
    m["line_abs"] = m["line_logit"].abs()
    cols = ["line_logit", "line_abs"] + FOCUS + ["ko_recent"]
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(C=0.05, max_iter=3000))
    flip = m.copy()
    flip[FOCUS + ["ko_recent"]] = -flip[FOCUS + ["ko_recent"]]
    flip["line_logit"] = -flip["line_logit"]          # line_abs unchanged
    flip["y"] = 1 - flip["y"]
    tr_sym = pd.concat([m, flip], ignore_index=True)
    lr.fit(tr_sym[cols], tr_sym["y"])
    # bootstrap ensemble for per-fight uncertainty (SE)
    rngb = np.random.default_rng(0)
    ensemble = []
    for _ in range(30):
        idx = rngb.choice(tr_sym.index, len(tr_sym), replace=True)
        eb = make_pipeline(StandardScaler(),
                           LogisticRegression(C=0.05, max_iter=1500))
        eb.fit(tr_sym.loc[idx, cols], tr_sym.loc[idx, "y"])
        ensemble.append(eb)

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
        Xa = pd.DataFrame([[ll, abs(ll)] + fv], columns=cols)
        Xb = pd.DataFrame([[-ll, abs(ll)] + [-v for v in fv]], columns=cols)
        p = float((lr.predict_proba(Xa)[0, 1]
                   + 1 - lr.predict_proba(Xb)[0, 1]) / 2)
        se = float(np.std([(e.predict_proba(Xa)[0, 1]
                            + 1 - e.predict_proba(Xb)[0, 1]) / 2
                           for e in ensemble]))
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
            "pick": r["fighter_a"] if pick_a else r["fighter_b"],
            "opp": r["fighter_b"] if pick_a else r["fighter_a"],
            "price": f"{int(oa):+d}" if pick_a else f"{int(ob):+d}",
            "market": round((pa if pick_a else pb) / (pa + pb) * 100, 1),
            "model": round((p if pick_a else 1 - p) * 100, 1),
            "edge": round(max(ea, eb) * 100, 1),
            "se": round(se * 100, 1),
            "net": round(net * 100, 1),
            "bet": bool(net > EDGE_RULE),
            "stake": (2 if net > 0.08 else (1 if net > EDGE_RULE else 0)),
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
                pick_is_a = o["pick"] == r["fighter_a"]
                o["props"] = {("pick_" if (k[0] == "a") == pick_is_a
                               else "opp_") + k[2:]: v
                              for k, v in pr.items()}
    except Exception as exc:
        print("props skipped:", exc)
    for o in out:
        o.pop("_p_a", None); o.pop("_row_idx", None)
    return out


def recent_results(days=120):
    """Walk-forward model picks at recent completed events."""
    from features_v3 import build_features_v3 as b3
    m, _ = load_matched_cached(b3, "v3", bout_cols=[])
    m["line_abs"] = m["line_logit"].abs()
    cols = ["line_logit", "line_abs"] + FOCUS + ["ko_recent"]
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    te, p = walk_forward(m, cols, start)
    if not len(te):
        return [], {}
    pnl, dates, _, _ = bets(te, p, EDGE_RULE)
    summary = {"n": int(len(pnl)), "pnl": round(float(pnl.sum()), 2),
               "roi": round(float(pnl.mean()) * 100, 1) if len(pnl) else 0.0}

    rows = []
    pr = te["pr_raw"].to_numpy()
    for i in te.index[::-1][:30]:
        pm = p[i]
        pick_red = (pm - te.loc[i, "pr_raw"]) >= ((1 - pm) - te.loc[i, "pb_raw"])
        pick = te.loc[i, "fighter_a"] if (te.loc[i, "key_a"] == te.loc[i, "key_r"]) == pick_red \
            else te.loc[i, "fighter_b"]
        won = bool((te.loc[i, "y"] == 1) == pick_red)
        edge = max(pm - te.loc[i, "pr_raw"], (1 - pm) - te.loc[i, "pb_raw"])
        rows.append({"date": str(te.loc[i, "date"].date()), "pick": pick,
                     "model": round((pm if pick_red else 1 - pm) * 100),
                     "edge": round(edge * 100, 1),
                     "bet": bool(edge > EDGE_RULE), "won": won})
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
