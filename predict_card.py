"""Predict the upcoming card and generate the site (docs/index.html).

Reads odds_upcoming.csv (from fetch_odds.py or edited by hand), computes
point-in-time features for each matchup, trains the production model on
all matched history, and writes a self-contained HTML page with:
  - the upcoming card: model probability vs market, edge, bet flags
  - a rolling results ledger: how model picks fared at recent events

Usage: python predict_card.py [--lock-paper-trades]
"""

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import logit as slogit

from adapter import parse_height, parse_reach
from backtest import american_to_prob, american_payout
from data_quality import assert_clean, identity_warnings
from features_v3 import build_features_v3
from identity import assign_fighter_identities, fighter_registry
from pipeline import load_matched_cached
from config import (BOOTSTRAP_MODELS, EDGE_RULE, EVENT_DAY_STAKE_CAP, FOCUS,
                    MARKET_DISAGREEMENT_WARNING, MAX_ODDS_AGE_MINUTES,
                    MIN_MARKET_BOOKS, MODEL_VERSION, RESEARCH_TWO_UNIT_RULE)
from production import (MODEL_FEATURES, allocate_stakes, event_pnl, event_seed,
                        fit_ensemble, predict_probabilities, score_bets)
import method_model as MM

# --------------------------------------------------------------- modeling
def resolve_identities(up, physicals, details=None):
    """Attach stable IDs; unresolved sportsbook names receive neutral history."""
    registry = fighter_registry(physicals, details)
    resolved = assign_fighter_identities(up, registry, strict=False)
    for side in ("a", "b"):
        unresolved = resolved[f"fighter_{side}_id"].str.startswith("unresolved:")
        for name in resolved.loc[unresolved, f"fighter_{side}"]:
            print(f"  WARNING: '{name}' has no unambiguous UFCStats identity; "
                  "using neutral career features")
    return resolved, registry


def market_probability(odds_a, odds_b, supplied=np.nan):
    """Prefer a paired-book consensus, with a de-vigged price fallback."""
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    supplied = pd.to_numeric(supplied, errors="coerce")
    probability = (float(supplied)
                   if pd.notna(supplied) and 0 < supplied < 1
                   else float(pa / (pa + pb)))
    return probability, float(pa), float(pb)


def _optional_number(value, fallback=None):
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if pd.notna(value) else fallback


def execution_ladder(p_model, se):
    """Return fixed-consensus execution thresholds for each stake tier."""
    ladder = {}
    for label, threshold in (
        ("1u", EDGE_RULE),
        ("2u_candidate", RESEARCH_TWO_UNIT_RULE),
    ):
        probability = float(p_model) - float(se) - threshold
        if 0.02 <= probability <= 0.98:
            exact = (-100.0 * probability / (1.0 - probability)
                     if probability >= 0.5
                     else 100.0 * (1.0 - probability) / probability)
            ladder[label] = int(math.ceil(exact))
    return ladder


def quote_age_minutes(value, now=None):
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = (current.tz_localize("UTC") if current.tzinfo is None
               else current.tz_convert("UTC"))
    return max(0.0, float((current - timestamp).total_seconds() / 60.0))


def _clean_meta(value, fallback="TBD"):
    if value is None or pd.isna(value) or not str(value).strip():
        return fallback
    return str(value).strip()


def predict_upcoming(up):
    fights = pd.read_csv("fights_v2.csv", parse_dates=["date"])
    assert_clean(fights, up)
    for warning in identity_warnings(fights):
        print(f"  WARNING: {warning}")
    up = up.copy()
    up["display_a"] = up["fighter_a"]
    up["display_b"] = up["fighter_b"]
    phys = pd.read_csv("raw/ufc_fighter_tott.csv")
    details = pd.read_csv("raw/ufc_fighter_details.csv")
    up, registry = resolve_identities(up.copy(), phys, details)

    rows = []
    for _, r in up.iterrows():
        rows.append({"date": pd.Timestamp(r["date"]), "event": "UPCOMING",
                     "bout": f"{r['display_a']} vs. {r['display_b']}",
                     "time_format": ("5 Rnd (5-5-5-5-5)"
                                     if int(r.get("five_rounds", 0) or 0)
                                     else "3 Rnd (5-5-5)"),
                     "weightclass": r.get("weightclass", "") or "",
                     "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"],
                     "fighter_a_id": r["fighter_a_id"],
                     "fighter_b_id": r["fighter_b_id"],
                     "fighter_a_url": r["fighter_a_url"],
                     "fighter_b_url": r["fighter_b_url"],
                     "winner": "A", "method": "", "fight_time_min": np.nan})
    hyp = pd.DataFrame(rows)

    registry["height_in"] = registry["HEIGHT"].map(parse_height)
    registry["reach_in"] = registry["REACH"].map(parse_reach)
    registry["dob"] = pd.to_datetime(registry["DOB"], format="mixed", errors="coerce")
    pm = registry.drop_duplicates("fighter_id", keep="first").set_index("fighter_id")
    for s in ("a", "b"):
        hyp[f"height_{s}"] = hyp[f"fighter_{s}_id"].map(pm["height_in"])
        hyp[f"reach_{s}"] = hyp[f"fighter_{s}_id"].map(pm["reach_in"])
        hyp[f"dob_{s}"] = hyp[f"fighter_{s}_id"].map(pm["dob"])
        hyp[f"stance_{s}"] = hyp[f"fighter_{s}_id"].map(pm["STANCE"])

    feats, _ = build_features_v3(pd.concat([fights, hyp], ignore_index=True))
    identity = ["date", "fighter_a_id", "fighter_b_id"]
    new = feats.merge(hyp[identity].assign(_u=1), on=identity, how="inner")
    new = new[new["date"] >= hyp["date"].min()]

    m, _ = load_matched_cached(build_features_v3, "v3", bout_cols=[])
    # A stable deployment seed keeps unchanged predictions reproducible across
    # repeated card snapshots; training-data changes still change the fitted model.
    ensemble = fit_ensemble(m, n_models=BOOTSTRAP_MODELS,
                            seed=event_seed(MODEL_VERSION, "upcoming"))
    cols = MODEL_FEATURES

    out = []
    for _, r in up.iterrows():
        row = new[(new["fighter_a_id"] == r["fighter_a_id"])
                  & (new["fighter_b_id"] == r["fighter_b_id"])]
        oa = float(str(r["odds_a"]).replace("+", ""))
        ob = float(str(r["odds_b"]).replace("+", ""))
        p_line, pa, pb = market_probability(
            oa, ob, r.get("market_prob_a", np.nan)
        )
        if not len(row):
            continue
        ll = slogit(np.clip(p_line, 0.02, 0.98))
        fv = [row.iloc[0][c] for c in FOCUS + ["ko_recent"]]
        X = pd.DataFrame([[ll, abs(ll)] + fv], columns=cols)
        p, se = predict_probabilities(ensemble, X)
        p, se = float(p[0]), float(se[0])
        execution_a = _optional_number(r.get("best_odds_a"), oa)
        execution_b = _optional_number(r.get("best_odds_b"), ob)
        execution_pa = american_to_prob(execution_a)
        execution_pb = american_to_prob(execution_b)
        ea, eb = p - execution_pa, (1 - p) - execution_pb
        pick_a = ea >= eb
        net = max(ea, eb) - se
        pick_probability = p if pick_a else 1.0 - p
        ladder = execution_ladder(pick_probability, se)
        books = _optional_number(r.get("market_books"))
        books = int(books) if books is not None else None
        spread = _optional_number(r.get("market_spread"))
        age_minutes = quote_age_minutes(r.get("fetched_at"))
        source = str(r.get("odds_source", "manual_or_unknown"))
        enough_books = books is None or books >= MIN_MARKET_BOOKS
        fresh_quote = (age_minutes is None or age_minutes <= MAX_ODDS_AGE_MINUTES
                       or not source.startswith("the-odds-api"))
        quality_ok = enough_books and fresh_quote
        execution_price = execution_a if pick_a else execution_b
        execution_book = r.get("best_book_a" if pick_a else "best_book_b", "")
        execution_book = _clean_meta(execution_book, "consensus")
        consensus_price = oa if pick_a else ob
        consensus_opp_price = ob if pick_a else oa
        market_pick = p_line if pick_a else 1.0 - p_line
        execution_pick = execution_pa if pick_a else execution_pb
        out.append({
            "_p_a": p, "_row_idx": int(row.index[0]), "_pick_a": bool(pick_a),
            "_net_raw": net, "_quality_ok": quality_ok,
            "ladder": ladder,
            "pick": r["display_a"] if pick_a else r["display_b"],
            "opp": r["display_b"] if pick_a else r["display_a"],
            "price": f"{int(execution_price):+d}",
            "execution_price": f"{int(execution_price):+d}",
            "execution_book": execution_book,
            "execution_implied": round(execution_pick * 100, 1),
            "consensus_price": f"{int(consensus_price):+d}",
            "consensus_opp_price": f"{int(consensus_opp_price):+d}",
            "market": round(market_pick * 100, 1),
            "model": round(pick_probability * 100, 1),
            "edge": round(max(ea, eb) * 100, 1),
            "se": round(se * 100, 1),
            "net": round(net * 100, 1),
            "qualified": bool(net >= EDGE_RULE and quality_ok),
            "bet": False,
            "stake": 0,
            "date": str(pd.Timestamp(r["date"]).date()),
            "scheduled_start": (str(r.get("commence_time"))
                                if pd.notna(r.get("commence_time", np.nan))
                                and str(r.get("commence_time", "")).strip()
                                else ""),
            "odds_source": source,
            "odds_fetched_at": r.get("fetched_at", ""),
            "market_books": books,
            "market_spread": round(spread * 100, 1) if spread is not None else None,
            "quote_age_minutes": round(age_minutes, 1) if age_minutes is not None else None,
            "market_warning": bool(spread is not None
                                   and spread > MARKET_DISAGREEMENT_WARNING),
            "eligibility_reason": ("eligible" if net >= EDGE_RULE and quality_ok
                                   else "price stale" if not fresh_quote
                                   else "fewer than 3 paired books" if not enough_books
                                   else "below edge rule"),
            "meta": f"{_clean_meta(r.get('weightclass'))} | {r['date']}",
        })
    if out:
        allocation_net = np.array([
            item["_net_raw"] if item["_quality_ok"] else -np.inf
            for item in out
        ])
        stakes = allocate_stakes(
            allocation_net,
            groups=np.array([item["date"] for item in out]),
            group_cap=EVENT_DAY_STAKE_CAP,
        )
        for item, stake in zip(out, stakes):
            item["stake"] = int(stake)
            item["bet"] = bool(stake > 0)
            if item["qualified"] and not item["bet"]:
                item["eligibility_reason"] = "event-day exposure cap"
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
                pick_is_a = o["_pick_a"]
                o["props"] = {("pick_" if (k[0] == "a") == pick_is_a
                               else "opp_") + k[2:]: v
                              for k, v in pr.items()}
    except Exception as exc:
        print("props skipped:", exc)
    for o in out:
        o.pop("_p_a", None); o.pop("_row_idx", None); o.pop("_pick_a", None)
        o.pop("_net_raw", None); o.pop("_quality_ok", None)
    return out


def recent_results(days=120):
    """Exact production walk-forward picks at recent completed events."""
    m, _ = load_matched_cached(build_features_v3, "v3", bout_cols=[])
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    window = m[m["date"] >= start]
    scored_events = []
    for date in sorted(window["date"].unique()):
        train = m[m["date"] < date]
        test = window[window["date"] == date].copy()
        if len(train) < 2000:
            continue
        models = fit_ensemble(train, n_models=BOOTSTRAP_MODELS,
                              seed=event_seed(date))
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




def _legacy_american_from_prob(p):
    """Fair American odds for an implied probability."""
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _legacy_sizing_ladder(lr, cols, feat_row, imp_now, vig, se_sub=0.0):
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
                ths[tier] = _legacy_american_from_prob(grid[b])
                break
    return ths

# --------------------------------------------------------------- site
def build_site(upcoming, recent, summary, freshness=None):
    with open("site_template.html") as f:
        tpl = f.read()
    stamp = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    freshness = freshness or {}
    freshness_banner = (
        f'<div class="freshness {freshness.get("status", "check")}">'
        f'Results through <b>{freshness.get("results_through", "unknown")}</b> | '
        f'{freshness.get("message", "freshness not checked")}</div>'
    )
    html = (tpl.replace("__UPCOMING__", json.dumps(upcoming))
               .replace("__RECENT__", json.dumps(recent))
               .replace("__SUMMARY__", json.dumps(summary))
               .replace("__MAX_ODDS_AGE__", str(MAX_ODDS_AGE_MINUTES))
               .replace("__FRESHNESS_BANNER__", freshness_banner)
               .replace("__STAMP__", stamp))
    import os
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(html)
    print(f"docs/index.html written "
          f"({len(upcoming)} upcoming, {len(recent)} recent)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-paper-trades", action="store_true",
                    help="lock one official qualifying paper wager per fight")
    args = ap.parse_args()

    up = pd.read_csv("odds_upcoming.csv")
    from paper_ledger import (assert_pre_event, lock_paper_trades,
                              record_prediction_snapshots)
    if len(up):
        # Validate timing before expensive feature/model work.  Exact API
        # commence times are preferred; date-only rows must be future-dated.
        assert_pre_event(up.to_dict("records"))
        upcoming = predict_upcoming(up)
    else:
        print("odds_upcoming.csv contains no fights; building an empty card")
        upcoming = []
    recent, summary = recent_results()
    from freshness import assess_freshness
    freshness = assess_freshness(pd.read_csv("fights_v2.csv"))
    with open("data_freshness.json", "w", encoding="utf-8") as output:
        json.dump(freshness, output, indent=2)
    build_site(upcoming, recent, summary, freshness)

    from model_manifest import sha256, write_manifest
    write_manifest()
    provenance = {"model_version": MODEL_VERSION,
                  "manifest_hash": sha256("model_manifest.json")}
    added = record_prediction_snapshots(upcoming, provenance=provenance)
    print(f"prediction snapshots: appended {added}")
    if args.lock_paper_trades:
        locked = lock_paper_trades(upcoming, provenance=provenance)
        print(f"official paper trades: locked {locked}")


if __name__ == "__main__":
    main()
