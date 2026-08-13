"""Predict the upcoming card and generate the site (docs/index.html).

Reads odds_upcoming.csv (from fetch_odds.py or edited by hand), computes
point-in-time features for each matchup, trains the production model on
all matched history, and writes a self-contained HTML page with:
  - the upcoming card: model probability vs market, edge, bet flags
  - a rolling results ledger: how model picks fared at recent events

Usage: python predict_card.py [--lock-paper-trades]
"""

import argparse
import html
import json
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import logit as slogit

from adapter import parse_height, parse_reach
from backtest import american_to_prob
from cards import card_ufc_experience, event_groups
from data_quality import assert_clean, identity_warnings
from features_v3 import build_features_v3
from identity import (assign_fighter_identities, canonical_name,
                      fighter_registry)
from pipeline import load_matched_cached
from config import (BOOTSTRAP_MODELS, EDGE_RULE, EVENT_DAY_STAKE_CAP, FOCUS,
                    MARKET_DISAGREEMENT_WARNING, MAX_EXECUTION_DEVIATION,
                    MAX_ODDS_AGE_MINUTES, MIN_MARKET_BOOKS, MODEL_VERSION,
                    RESEARCH_TWO_UNIT_RULE)
from production import (MODEL_FEATURES, allocate_stakes, event_pnl, event_seed,
                        fit_ensemble, predict_probabilities, score_bets)
import method_model as MM

PROMOTION_CHOICES = ("ufc", "dwcs", "all")


def _promotion_value(row):
    """The promotion a row actually claims, from provider metadata only.

    `capture_promotion` is deliberately not consulted: it records which
    capture was requested, not what the event is, and reading it back here is
    how a `--promotion ufc` run came to assert UFC over a card of debutants.
    """
    text = " ".join(str(row.get(column, "")) for column in (
        "promotion", "event_title",
    )).lower()
    if "contender series" in text or "dwcs" in text:
        return "dwcs"
    if "ufc" in text or "ultimate fighting championship" in text:
        return "ufc"
    return ""


def ufc_experience(upcoming, fights):
    """Share of each card's fighters holding a prior UFC bout.

    The provider names no promotion, so this is the measurable stand-in: a
    UFC card is fighters who have fought in the UFC. Cards are grouped by
    start time because the feed's event_id is per bout, not per card - all 44
    rows on the 2026-08-10 board carried 44 distinct event_ids.
    """
    veterans = set()
    for column in ("fighter_a", "fighter_b"):
        veterans |= set(fights[column].dropna().map(canonical_name))
    pairs = [(canonical_name(row.get("fighter_a", "")) in veterans,
              canonical_name(row.get("fighter_b", "")) in veterans)
             for _, row in upcoming.iterrows()]
    groups = event_groups(upcoming["commence_time"].tolist()
                          if "commence_time" in upcoming
                          else [""] * len(upcoming))
    return card_ufc_experience(groups, pairs)


# A card on which nobody has ever fought in the UFC is not a UFC card. The
# threshold is loose because a real UFC card occasionally features a debutant
# or two; it is the all-debutant card this is meant to catch.
NO_UFC_HISTORY = 0.10


def filter_upcoming_promotion(upcoming, promotion, fights=None):
    """Filter an upcoming card by promotion, measuring when metadata is absent.

    The feed labels every event "MMA", so the provider label is empty in
    practice and a metadata-only filter could never select a Contender Series
    card. When `fights` is supplied the roster is measured instead.
    """
    if promotion == "all" or not len(upcoming):
        return upcoming.copy()
    mask = upcoming.apply(_promotion_value, axis=1)
    if fights is not None and (mask == "").any():
        share = pd.Series(ufc_experience(upcoming, fights), index=upcoming.index)
        mask = mask.where(mask != "",
                          share.map(lambda s: "" if s > NO_UFC_HISTORY
                                    else "no-ufc-history"))
    if promotion == "ufc":
        # Measured-non-UFC cards are kept here on purpose. Labelling them
        # correctly is one decision; dropping them from the snapshot record
        # is another, and this filter is not the place to make the second one
        # quietly. They are still gated from trading by the unresolved
        # identity rule, and now carry an honest label instead of "UFC".
        return upcoming[mask.isin(["ufc", "", "no-ufc-history"])].copy()
    if promotion == "dwcs":
        return upcoming[mask.isin(["dwcs", "no-ufc-history"])].copy()
    return upcoming[mask == promotion].copy()


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


def leader_gap_for_pick(leader_prob_a, follower_prob_a, pick_a):
    """Orient recorded leader/follower probabilities onto the picked side.

    Returns ``(leader, follower, gap_points)``, where the gap reads as "the
    market-setting books make this pick that much more likely than the books
    that follow them". Flipping the pick flips the sign. Research provenance
    only; nothing downstream reads it back into the model.
    """
    leader, follower = leader_prob_a, follower_prob_a
    if not pick_a:
        leader = None if leader is None else 1.0 - leader
        follower = None if follower is None else 1.0 - follower
    gap = (None if leader is None or follower is None
           else round((leader - follower) * 100, 2))
    return leader, follower, gap


_TITLE_NOISE = re.compile(r"\bufc\b|\binterim\b|\btitle\b|\btournament\b", re.I)


def _base_division(value):
    """Strip title/interim wording down to the plain division."""
    text = _TITLE_NOISE.sub(" ", str(value or ""))
    return " ".join(text.split()).strip()


def infer_weightclass(upcoming, fights):
    """Fill each fight's division from the two fighters' most recent bouts.

    The odds feed carries no weight class - the column was written as a
    hardcoded "" for every fight ever captured - while every training row has
    one. That skew is expensive twice over. It zeroes the division flags at
    serve time, which moved held-out distance probabilities by +14 points at
    heavyweight and -15 at women's flyweight and cost more accuracy than the
    rounds model had over a division lookup table in the first place. And it
    breaks identity: two fighters named Jean Silva are separated only by
    division, so with none supplied the bout resolves to neither and is gated
    out of trading entirely.

    Both corners usually agree. Where they do not, the more recent bout wins,
    on the grounds that the fighter who competed last is the better guide to
    where this one is being made. On the 2026-08-10 board 15 of 38 disagreed
    and 14 of those were cosmetic - the divisions differed but mapped to the
    same model flags - leaving one fight where the choice mattered at all.

    Returns (values, sources); the source column keeps the inference legible
    rather than letting it pass as something the feed supplied.
    """
    seen = fights.dropna(subset=["weightclass"]).sort_values("date")
    latest, when = {}, {}
    for row in seen.itertuples():
        for name in (row.fighter_a, row.fighter_b):
            key = canonical_name(name)
            latest[key] = row.weightclass
            when[key] = row.date

    values, sources = [], []
    for _, row in upcoming.iterrows():
        # `NaN or ""` returns the NaN, because a nan float is truthy, and
        # str() of it is the word "nan" - which is not falsy either. An
        # explicit null check is the only safe read of a missing cell.
        raw = row.get("weightclass", "")
        supplied = "" if raw is None or pd.isna(raw) else str(raw).strip()
        if supplied:
            values.append(supplied)
            sources.append("provider")
            continue
        a, b = (canonical_name(row.get("fighter_a", "")),
                canonical_name(row.get("fighter_b", "")))
        known = [(latest[k], when[k]) for k in (a, b) if k in latest]
        if not known:
            values.append("")
            sources.append("")
        elif len(known) == 1:
            values.append(_base_division(known[0][0]))
            sources.append("history-one-corner")
        elif _base_division(known[0][0]) == _base_division(known[1][0]):
            values.append(_base_division(known[0][0]))
            sources.append("history-agree")
        else:
            values.append(_base_division(max(known, key=lambda k: k[1])[0]))
            sources.append("history-recent")
    return values, sources


def quote_quality(books, age_minutes, source, market_pick, execution_pick,
                  identity_resolved=True):
    """Return ``(ok, reason)`` for a live quote, before the edge rule applies.

    ``market_pick`` is the de-vigged paired-book consensus for the picked side
    and ``execution_pick`` is the raw implied probability of the best captured
    price. The executable price carries vig, so it normally implies slightly
    more probability than the consensus. A price implying materially less is a
    broken quote at one book rather than a line-shopping gain.
    """
    # A fighter with no UFCStats identity gets neutral career features, which
    # is the model saying "I know nothing" - and a fight priced on that is a
    # fight priced as though both corners were debutants. 29% of scored
    # snapshots involve one. None has ever become a trade, but only because
    # non-UFC bouts usually draw fewer than three books, which is incidental
    # rather than a control. This makes it a control.
    if not identity_resolved:
        return False, "unresolved fighter identity"
    stale = (age_minutes is not None and age_minutes > MAX_ODDS_AGE_MINUTES
             and str(source).startswith("the-odds-api"))
    if stale:
        return False, "price stale"
    if books is not None and books < MIN_MARKET_BOOKS:
        return False, "fewer than 3 paired books"
    if (market_pick - execution_pick) > MAX_EXECUTION_DEVIATION:
        return False, "book price outlier"
    return True, ""


def _flag_flat_cards(out, fights):
    """Mark whole cards whose totals carry no information beyond the prior.

    Checked per card rather than per fight, because the signature is a card
    of prices that agree with each other and with the base rate. A warning,
    not a gate: on a short card five real fights look this flat 11% of the
    time, so blocking on it would throw away genuine prices.
    """
    import rounds_model as RM
    base = float(fights["method"].astype(str).str.contains("DEC", case=False).mean())
    groups = event_groups([str(o.get("scheduled_start") or o.get("date", ""))
                           for o in out])
    for group in set(groups):
        members = [o for o, g in zip(out, groups) if g == group and o.get("rounds")]
        if not RM.flat_card([o["rounds"]["distance_pct"] / 100.0
                             for o in members], base):
            continue
        for o in members:
            o["rounds"]["uninformative_card"] = True
        print(f"  WARNING: totals for {len(members)} fights on {members[0]['date']} "
              f"sit at the base rate with no spread; the model has no read on "
              f"this card")


def _assert_aligned(label, priced, hyp):
    """Fail loudly if a prop list came back bound to the wrong fights.

    Both prop models sort internally and used to return date order, which
    lined up with the caller's fights only because the odds feed happens to
    write them date-sorted. A hand-edited CSV was one step from pricing a
    main event as a prelim, and the length check could not see it. Names are
    compared because they are the thing that must match.
    """
    for position, row in enumerate(priced):
        if not row:
            continue
        expected = (hyp.at[position, "fighter_a"], hyp.at[position, "fighter_b"])
        got = (row.get("fighter_a"), row.get("fighter_b"))
        if got != expected:
            raise ValueError(
                f"{label} misaligned at row {position}: "
                f"expected {expected}, got {got}")


def predict_upcoming(up):
    fights = pd.read_csv("fights_v2.csv", parse_dates=["date"])
    assert_clean(fights, up)
    for warning in identity_warnings(fights):
        print(f"  WARNING: {warning}")
    up = up.copy()
    up["display_a"] = up["fighter_a"]
    up["display_b"] = up["fighter_b"]
    # Must precede identity resolution: that is what disambiguates two
    # fighters sharing a name, and it needs the division to do it.
    up["weightclass"], up["weightclass_source"] = infer_weightclass(up, fights)
    filled = sum(1 for s in up["weightclass_source"] if s.startswith("history"))
    if filled:
        print(f"  weightclass inferred from fighter history for {filled} "
              f"of {len(up)} fights")
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
        execution_price = execution_a if pick_a else execution_b
        execution_book = r.get("best_book_a" if pick_a else "best_book_b", "")
        execution_book = _clean_meta(execution_book, "consensus")
        consensus_price = oa if pick_a else ob
        consensus_opp_price = ob if pick_a else oa
        market_pick = p_line if pick_a else 1.0 - p_line
        execution_pick = execution_pa if pick_a else execution_pb
        resolved_a = not str(r.get("fighter_a_id", "")).startswith("unresolved:")
        resolved_b = not str(r.get("fighter_b_id", "")).startswith("unresolved:")
        quality_ok, quality_reason = quote_quality(
            books, age_minutes, source, market_pick, execution_pick,
            identity_resolved=resolved_a and resolved_b,
        )
        neutral_identity = not (resolved_a and resolved_b)
        leader_prob, follower_prob, leader_gap = leader_gap_for_pick(
            _optional_number(r.get("leader_prob_a")),
            _optional_number(r.get("follower_prob_a")),
            pick_a,
        )
        out.append({
            "_p_a": p, "_row_idx": int(row.index[0]), "_pick_a": bool(pick_a),
            "_net_raw": net, "_quality_ok": quality_ok,
            "_neutral_identity": neutral_identity,
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
            "leader_prob": round(leader_prob * 100, 1) if leader_prob is not None else None,
            "leader_books": _optional_number(r.get("leader_books")),
            "follower_prob": round(follower_prob * 100, 1) if follower_prob is not None else None,
            "follower_books": _optional_number(r.get("follower_books")),
            "leader_gap": leader_gap,
            "market_spread": round(spread * 100, 1) if spread is not None else None,
            "quote_age_minutes": round(age_minutes, 1) if age_minutes is not None else None,
            "market_warning": bool(spread is not None
                                   and spread > MARKET_DISAGREEMENT_WARNING),
            "eligibility_reason": ("eligible" if net >= EDGE_RULE and quality_ok
                                   else quality_reason or "below edge rule"),
            "meta": " | ".join(part for part in (
                _clean_meta(r.get("promotion"), ""),
                _clean_meta(r.get("event_title"), ""),
                _clean_meta(r.get("weightclass"), ""),
                str(r["date"]),
            ) if part),
        })
    if out:
        for item, group in zip(out, event_groups(
                [item.get("scheduled_start") or item.get("date") for item in out])):
            item["event_group"] = group
        allocation_net = np.array([
            item["_net_raw"] if item["_quality_ok"] else -np.inf
            for item in out
        ])
        stakes = allocate_stakes(
            allocation_net,
            # Group by event, not by UTC date. A card starting 00:20 UTC and
            # one starting 21:10 UTC are different events on the same UTC day,
            # and grouping them shared one 2-unit cap between two shows.
            groups=np.array([item["event_group"] for item in out]),
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
            _assert_aligned("method props", props, hyp)
            for o, pr in zip(out, props):
                pick_is_a = o["_pick_a"]
                o["props"] = {("pick_" if (k[0] == "a") == pick_is_a
                               else "opp_") + k[2:]: v
                              for k, v in pr.items()
                              if k not in ("fighter_a", "fighter_b")}
    except Exception as exc:
        print("props skipped:", exc)
    # totals and distance (fair prices only; see rounds_model for why these
    # are not tradable). Trained inline rather than from a pickle: the fit
    # costs about three seconds, and no job refreshes a committed one, so a
    # stored model would quietly serve a stale fit against fresh careers.
    try:
        import rounds_model as RM
        allf = pd.concat([fights, hyp], ignore_index=True)
        allf["date"] = pd.to_datetime(allf["date"])
        priced = RM.card_prices(RM.train(fights), allf)
        if len(priced) == len(out):
            _assert_aligned("rounds prices", priced, hyp)
            for o, rounds in zip(out, priced):
                if rounds:
                    # Two ways to have nothing to say: no resolved identity,
                    # or a resolved fighter who has simply never fought. The
                    # second is the one that bit - a Contender Series fighter
                    # resolves fine and still has zero career features.
                    rounds["baseline_only"] = bool(
                        o["_neutral_identity"] or rounds["no_career_data"])
                    rounds.pop("fighter_a", None)
                    rounds.pop("fighter_b", None)
                    o["rounds"] = rounds
            _flag_flat_cards(out, fights)
        else:
            print(f"rounds skipped: priced {len(priced)} of {len(out)} fights")
    except Exception as exc:
        print("rounds skipped:", exc)
    for o in out:
        o.pop("_p_a", None); o.pop("_row_idx", None); o.pop("_pick_a", None)
        o.pop("_net_raw", None); o.pop("_quality_ok", None)
        o.pop("_neutral_identity", None)
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
def build_site(upcoming, recent, summary, freshness=None, card_context=None):
    with open("site_template.html") as f:
        tpl = f.read()
    stamp = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    freshness = freshness or {}
    status = html.escape(str(freshness.get("status", "check")), quote=True)
    results_through = html.escape(str(freshness.get("results_through", "unknown")))
    message = html.escape(str(freshness.get("message", "freshness not checked")))
    freshness_banner = (
        f'<div class="freshness {status}">'
        f'Results through <b>{results_through}</b> | '
        f'{message}</div>'
    )
    card_context = card_context or {
        "eyebrow": "Fight Ledger | walk-forward model v3",
        "headline": "Model<br>Card Read",
        "subcopy": ("Every upcoming fight, priced by a model trained on "
                    "<b>8,600+ UFC fights</b> and benchmarked against closing "
                    "lines since 2019. Consensus prices inform the prediction; "
                    "the best captured sportsbook price determines whether a "
                    "fight clears <b>4 net points</b>."),
    }
    def safe_json(value):
        return json.dumps(value).replace("</", "<\\/")
    page_html = (tpl.replace("__UPCOMING__", safe_json(upcoming))
                    .replace("__RECENT__", safe_json(recent))
                    .replace("__SUMMARY__", safe_json(summary))
                    .replace("__MAX_ODDS_AGE__", str(MAX_ODDS_AGE_MINUTES))
                    .replace("__FRESHNESS_BANNER__", freshness_banner)
                    .replace("__EYEBROW__", card_context["eyebrow"])
                    .replace("__HEADLINE__", card_context["headline"])
                    .replace("__SUBCOPY__", card_context["subcopy"])
                    .replace("__STAMP__", stamp))
    import os
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(page_html)
    print(f"docs/index.html written "
          f"({len(upcoming)} upcoming, {len(recent)} recent)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-paper-trades", action="store_true",
                    help="lock one official qualifying paper wager per fight")
    ap.add_argument(
        "--promotion", choices=PROMOTION_CHOICES, default="ufc",
        help="score ufc, dwcs, or all tagged rows from odds_upcoming.csv",
    )
    args = ap.parse_args()

    up = pd.read_csv("odds_upcoming.csv")
    up = filter_upcoming_promotion(
        up, args.promotion, pd.read_csv("fights_v2.csv", usecols=[
            "fighter_a", "fighter_b"]))
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
    contexts = {
        "ufc": {
            "eyebrow": "Fight Ledger | walk-forward model v3",
            "headline": "Model<br>Card Read",
            "subcopy": ("Every upcoming fight, priced by a model trained on "
                        "<b>8,600+ UFC fights</b> and benchmarked against "
                        "closing lines since 2019. Consensus prices inform "
                        "the prediction; the best captured sportsbook price "
                        "determines whether a fight clears <b>4 net points</b>."),
        },
        "dwcs": {
            "eyebrow": "Fight Ledger | DWCS model read | walk-forward model v3",
            "headline": "DWCS<br>Card Read",
            "subcopy": ("Dana White's Contender Series fights, priced through "
                        "the UFC-trained moneyline model. Many DWCS fighters "
                        "lack UFCStats identities, so neutral-history reads "
                        "are expected and paper-trade quality gates remain "
                        "strict."),
        },
        "all": {
            "eyebrow": "Fight Ledger | MMA model read | walk-forward model v3",
            "headline": "MMA<br>Card Read",
            "subcopy": ("Tagged upcoming MMA fights, priced through the "
                        "UFC-trained moneyline model. Non-UFC fighters often "
                        "use neutral career features, so this is exploratory "
                        "unless identity and market quality gates clear."),
        },
    }
    build_site(upcoming, recent, summary, freshness, contexts[args.promotion])

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
