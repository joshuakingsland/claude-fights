"""Stable fighter identities shared by ingestion and feature builders."""

import re
import unicodedata

import pandas as pd


def norm_name(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(value.lower().replace(".", "").replace("-", " ").split())


DIVISION_LIMITS = (
    ("light heavyweight", 205.0),
    ("heavyweight", 265.0),
    ("middleweight", 185.0),
    ("welterweight", 170.0),
    ("lightweight", 155.0),
    ("featherweight", 145.0),
    ("bantamweight", 135.0),
    ("flyweight", 125.0),
    ("strawweight", 115.0),
)

# Source-specific spelling corrections are explicit so name matching never
# expands into a fuzzy, potentially cross-fighter join.
NAME_ALIASES = {
    "ian garry": "ian machado garry",
    "paulo henrique costa": "paulo costa",
    "rafael cerquiera": "rafael cerqueira",
    "ramazonbek temirov": "ramazan temirov",
    "stephen erceg": "steve erceg",
}


def fighter_id_from_url(value):
    match = re.search(r"/fighter-details/([0-9a-f]+)", str(value), re.I)
    return match.group(1).lower() if match else ""


def division_limit(value):
    text = str(value).lower()
    for label, limit in DIVISION_LIMITS:
        if label in text:
            return limit
    return None


def fighter_registry(tott, details=None):
    """Normalize the UFCStats fighter table and validate stable URLs."""
    registry = tott.copy()
    for column in registry.columns:
        if pd.api.types.is_string_dtype(registry[column]):
            registry[column] = registry[column].str.strip()
    required = {"FIGHTER", "URL"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"fighter registry is missing columns: {sorted(missing)}")
    if details is not None:
        detail_rows = details.copy()
        for column in detail_rows.columns:
            if pd.api.types.is_string_dtype(detail_rows[column]):
                detail_rows[column] = detail_rows[column].fillna("").str.strip()
        detail_required = {"FIRST", "LAST", "URL"}
        detail_missing = detail_required - set(detail_rows.columns)
        if detail_missing:
            raise ValueError(
                f"fighter details are missing columns: {sorted(detail_missing)}"
            )
        detail_rows["FIGHTER"] = (
            detail_rows["FIRST"] + " " + detail_rows["LAST"]
        ).str.strip()
        for column in registry.columns:
            if column not in detail_rows:
                detail_rows[column] = pd.NA
        registry = pd.concat(
            [registry, detail_rows[registry.columns]], ignore_index=True
        )

    registry["fighter_id"] = registry["URL"].map(fighter_id_from_url)
    registry["name_key"] = registry["FIGHTER"].map(norm_name)
    weight = registry.get("WEIGHT", pd.Series("", index=registry.index))
    registry["registered_weight"] = pd.to_numeric(
        weight.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    if registry["fighter_id"].eq("").any():
        raise ValueError("fighter registry contains a missing or invalid UFCStats URL")
    registry = registry.drop_duplicates(["name_key", "fighter_id"], keep="first")
    url_counts = registry.groupby("fighter_id")["URL"].nunique()
    if (url_counts > 1).any():
        raise ValueError("fighter registry maps one fighter ID to multiple URLs")
    return registry.reset_index(drop=True)


def resolve_fighter(name, weightclass, registry, strict=True):
    """Resolve a display name to one UFCStats fighter using division context."""
    name_key = norm_name(name)
    name_key = NAME_ALIASES.get(name_key, name_key)
    candidates = registry[registry["name_key"] == name_key]
    if len(candidates) == 1:
        return candidates.iloc[0]
    if len(candidates) > 1:
        limit = division_limit(weightclass)
        known = candidates.dropna(subset=["registered_weight"]).copy()
        if limit is not None and len(known):
            known["distance"] = (known["registered_weight"] - limit).abs()
            best = known[known["distance"] == known["distance"].min()]
            if len(best) == 1:
                return best.iloc[0]
        if limit is None and len(known) == 1:
            return known.iloc[0]
    if strict:
        detail = "not found" if not len(candidates) else "ambiguous"
        raise ValueError(
            f"fighter identity {detail}: {name!r} in {weightclass!r}"
        )
    return None


def assign_fighter_identities(frame, registry, strict=True):
    """Attach fighter IDs and URLs to both sides of a bout table."""
    out = frame.copy()
    for side in ("a", "b"):
        ids, urls = [], []
        for name, weightclass in zip(out[f"fighter_{side}"], out["weightclass"]):
            fighter = resolve_fighter(name, weightclass, registry, strict=strict)
            if fighter is None:
                ids.append(f"unresolved:{norm_name(name)}")
                urls.append("")
            else:
                ids.append(fighter["fighter_id"])
                urls.append(fighter["URL"])
        out[f"fighter_{side}_id"] = ids
        out[f"fighter_{side}_url"] = urls
    return out


def fighter_keys(frame, side):
    """Return stable feature keys, falling back only for legacy fixtures."""
    id_column = f"fighter_{side}_id"
    if id_column in frame:
        ids = frame[id_column].fillna("").astype(str).str.strip()
        fallback = frame[f"fighter_{side}"].map(lambda value: f"name:{norm_name(value)}")
        return ids.where(ids.ne(""), fallback)
    return frame[f"fighter_{side}"].map(lambda value: f"name:{norm_name(value)}")
