"""Stable fighter identities shared by ingestion and feature builders."""

import re
import unicodedata

import pandas as pd


def norm_name(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower().replace(".", "").replace("-", " ")
    # Sources disagree on apostrophes for the same fighter ("L'udovit Klein"
    # against "Ludovit Klein"), and dropping them merges no distinct names.
    value = value.replace("'", "").replace("’", "")
    return " ".join(value.split())


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
    # UFCStats lists him as "Nicholas Galanti" in the fighter tables and
    # spells the bout "Nick Galanti" in the results table - one source, one
    # fighter URL (8988705468d04ef7), two spellings. Shortened first names do
    # not normalise, so this halted every result refresh from 2026-10-19.
    "nick galanti": "nicholas galanti",
    "paulo henrique costa": "paulo costa",
    "rafael cerquiera": "rafael cerqueira",
    "ramazonbek temirov": "ramazan temirov",
    "stephen erceg": "steve erceg",
}


def canonical_name(value):
    """Return the explicit cross-source alias used for deterministic joins."""
    name_key = norm_name(value)
    return NAME_ALIASES.get(name_key, name_key)


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
    name_key = canonical_name(name)
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


def alias_candidates(name, registry, limit=3):
    """Registry names sharing this one's surname, as alias targets.

    The remedy for an unresolved fighter is always a `NAME_ALIASES` entry, and
    the work is in knowing what to map it to. UFCStats spells the same man two
    ways across its own tables - "Nick Galanti" in the results, "Nicholas
    Galanti" in the fighter list - so the counterpart almost always shares the
    surname. Offered as candidates, never applied: picking one automatically
    would be the fuzzy cross-fighter join `NAME_ALIASES` exists to avoid.
    """
    parts = norm_name(name).split()
    if not parts or "name_key" not in registry:
        return []
    surname = parts[-1]
    keys = registry["name_key"].astype(str)
    hits = keys[keys.str.split().str[-1].eq(surname) & keys.ne(canonical_name(name))]
    return sorted(set(hits))[:limit]


def assign_fighter_identities(frame, registry, strict=True):
    """Attach fighter IDs and URLs to both sides of a bout table.

    Every unresolvable name in the table is collected before raising. Failing
    on the first one meant fixing a single alias per run, and each run is a
    day apart: one name ('Nick Galanti') stalled the result refresh from
    2026-10-19 to the 29th, and finding a second behind it would have cost the
    same again. The error also names the remedy, because it is always the same
    remedy.
    """
    out = frame.copy()
    unresolved = {}
    for side in ("a", "b"):
        ids, urls = [], []
        for name, weightclass in zip(out[f"fighter_{side}"], out["weightclass"]):
            try:
                fighter = resolve_fighter(name, weightclass, registry, strict=strict)
            except ValueError as failure:
                unresolved.setdefault(
                    (str(name), str(weightclass)),
                    ("ambiguous" if "ambiguous" in str(failure) else "not found",
                     alias_candidates(name, registry)),
                )
                fighter = None
            if fighter is None:
                ids.append(f"unresolved:{norm_name(name)}")
                urls.append("")
            else:
                ids.append(fighter["fighter_id"])
                urls.append(fighter["URL"])
        out[f"fighter_{side}_id"] = ids
        out[f"fighter_{side}_url"] = urls
    if unresolved and strict:
        lines = []
        for (name, weightclass), (detail, candidates) in sorted(unresolved.items()):
            line = f"  {name!r} in {weightclass!r} ({detail})"
            if candidates:
                suggestion = ", ".join(
                    f'"{canonical_name(name)}": "{candidate}"'
                    for candidate in candidates
                )
                line += f" - same surname in the registry: {suggestion}"
            lines.append(line)
        noun = "identity" if len(unresolved) == 1 else "identities"
        raise ValueError(
            f"{len(unresolved)} fighter {noun} could not be resolved; add a "
            f"NAME_ALIASES entry in identity.py for each:\n" + "\n".join(lines)
        )
    return out


def fighter_keys(frame, side):
    """Return stable feature keys, falling back only for legacy fixtures."""
    id_column = f"fighter_{side}_id"
    if id_column in frame:
        ids = frame[id_column].fillna("").astype(str).str.strip()
        fallback = frame[f"fighter_{side}"].map(lambda value: f"name:{norm_name(value)}")
        return ids.where(ids.ne(""), fallback)
    return frame[f"fighter_{side}"].map(lambda value: f"name:{norm_name(value)}")
