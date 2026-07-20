"""Fail-closed checks for production input tables."""

import pandas as pd

from backtest import american_to_prob, norm_name


def _pairs(frame, a="fighter_a", b="fighter_b"):
    return [frozenset((norm_name(x), norm_name(y)))
            for x, y in zip(frame[a], frame[b])]


def ambiguous_names(fights):
    """Return normalized names represented by multiple raw spellings."""
    names = {}
    for side in ("fighter_a", "fighter_b"):
        for value in fights[side].dropna().astype(str):
            names.setdefault(norm_name(value), set()).add(value.strip())
    return {key: sorted(values) for key, values in names.items() if len(values) > 1}


def identity_warnings(fights):
    id_columns = {"fighter_a_id", "fighter_b_id", "fighter_a_url",
                  "fighter_b_url"}
    if not id_columns.intersection(fights.columns):
        return ["stable fighter IDs/URLs are absent; name matching remains a risk"]
    return []


def audit_fights(fights):
    errors = []
    required = {"date", "fighter_a", "fighter_b", "winner"}
    errors.extend(f"fights missing column: {c}" for c in sorted(required - set(fights)))
    if errors:
        return errors
    dates = pd.to_datetime(fights["date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"fights contain {int(dates.isna().sum())} invalid dates")
    pairs = _pairs(fights)
    duplicate = pd.DataFrame({"date": dates, "pair": pairs}).duplicated().sum()
    if duplicate:
        errors.append(f"fights contain {int(duplicate)} duplicate date/pair rows")
    bad_winners = ~fights["winner"].isin(["A", "B", "draw"])
    if bad_winners.any():
        errors.append(f"fights contain {int(bad_winners.sum())} invalid winner values")
    same = [norm_name(a) == norm_name(b)
            for a, b in zip(fights["fighter_a"], fights["fighter_b"])]
    if any(same):
        errors.append(f"fights contain {sum(same)} self-match rows")
    return errors


def audit_upcoming(upcoming):
    errors = []
    required = {"date", "fighter_a", "fighter_b", "odds_a", "odds_b"}
    errors.extend(f"upcoming missing column: {c}" for c in sorted(required - set(upcoming)))
    if errors:
        return errors
    dates = pd.to_datetime(upcoming["date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"upcoming contains {int(dates.isna().sum())} invalid dates")
    try:
        oa = pd.to_numeric(upcoming["odds_a"].astype(str).str.replace("+", "", regex=False))
        ob = pd.to_numeric(upcoming["odds_b"].astype(str).str.replace("+", "", regex=False))
        if ((oa == 0) | (ob == 0) | oa.isna() | ob.isna()).any():
            errors.append("upcoming contains missing or zero American odds")
        if not ((american_to_prob(oa) > 0).all()
                and (american_to_prob(ob) > 0).all()):
            errors.append("upcoming contains invalid American odds")
    except (TypeError, ValueError):
        errors.append("upcoming odds are not numeric American odds")
    pairs = _pairs(upcoming)
    duplicate = pd.DataFrame({"date": dates, "pair": pairs}).duplicated().sum()
    if duplicate:
        errors.append(f"upcoming contains {int(duplicate)} duplicate date/pair rows")
    same = [norm_name(a) == norm_name(b)
            for a, b in zip(upcoming["fighter_a"], upcoming["fighter_b"])]
    if any(same):
        errors.append(f"upcoming contains {sum(same)} self-match rows")
    return errors


def assert_clean(fights, upcoming=None):
    errors = audit_fights(fights)
    if upcoming is not None:
        errors.extend(audit_upcoming(upcoming))
    if errors:
        raise ValueError("Input quality gate failed:\n- " + "\n- ".join(errors))
