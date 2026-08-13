"""Fight records for fighters the UFCStats data has never seen.

29% of scored snapshots involve at least one fighter with no career history in
`fights_v2.csv`. Those fighters are not unknown quantities in reality - they
have long records in PFL, Bellator, ONE, KSW, and regional promotions - they
are only unknown to this repository, which is built entirely on UFCStats.
Until that gap is filled the model prices them on neutral history, which is
the same as pricing them as debutants.

This module collects those records from Sherdog.

On sources. Sherdog's robots.txt is `User-agent: * / Allow: /` with no
restrictions. Tapology's names ClaudeBot explicitly and disallows it, so
Tapology is not touched here and should not be added later; if a second source
is wanted, check its robots.txt first and pick one that permits access.

Collection is deliberately slow and cached. One request per fighter for the
search, one for the record, a pause between each, and raw HTML written to disk
so a re-run costs nothing. Records do not change often, so re-fetching a
fighter is almost always waste.
"""

import argparse
import csv
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

SEARCH = "https://www.sherdog.com/stats/fightfinder?SearchTxt={query}"
BASE = "https://www.sherdog.com"
AGENT = "claude-fights-research/1.0 (personal fight-model research)"
CACHE = Path("raw/sherdog")

BOUT_FIELDS = [
    "fighter_slug", "fighter_name", "order", "result", "opponent",
    "promotion", "event", "event_date", "method", "round", "time",
]

# Sherdog writes the event cell as "Promotion - Event Title Mon / DD / YYYY".
DATE_IN_EVENT = re.compile(r"([A-Z][a-z]{2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})\s*$")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _strip(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def fetch(url, pause=1.5, timeout=30, attempts=3):
    """GET with caching, a pause, and backoff.

    The cache is keyed by URL, so re-running the collector after adding new
    fighters re-fetches only the new ones.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", url)[-120:]
    path = CACHE / f"{key}.html"
    if path.exists() and path.stat().st_size > 500:
        return path.read_text(encoding="utf-8", errors="ignore")
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="ignore")
            path.write_text(body, encoding="utf-8")
            time.sleep(pause)
            return body
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last = error
            time.sleep(2 ** attempt)
    raise last


def find_fighter(name, pause=1.5):
    """Return the Sherdog path for a name, or None when it is ambiguous.

    A single unambiguous hit is required. Returning the first of several
    matches is how one fighter's record ends up attached to another's name,
    which is exactly the failure the UFCStats id work was done to prevent.
    """
    html = fetch(SEARCH.format(query=urllib.parse.quote_plus(name)), pause=pause)
    table = re.search(r'<table[^>]*fightfinder.*?</table>', html, re.S | re.I)
    if not table:
        return None
    links = re.findall(r'href="(/fighter/[^"]+)"', table.group(0))
    unique = list(dict.fromkeys(links))
    if len(unique) != 1:
        return None
    return unique[0]


def parse_record(html, slug):
    """Return one row per bout on a fighter page."""
    name = ""
    heading = re.search(r'<span[^>]*class="fn"[^>]*>(.*?)</span>', html, re.S)
    if heading:
        name = _strip(heading.group(1))
    rows = []
    for table in re.findall(r"<table[^>]*>.*?</table>", html, re.S):
        entries = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)
        if len(entries) < 2:
            continue
        header = _strip(entries[0])
        if "Result" not in header or "Event" not in header:
            continue
        for order, entry in enumerate(entries[1:]):
            cells = [_strip(c) for c in
                     re.findall(r"<td[^>]*>(.*?)</td>", entry, re.S)]
            if len(cells) < 6:
                continue
            result, opponent, event, method, rnd, clock = cells[:6]
            promotion, title, date = _split_event(event)
            rows.append({
                "fighter_slug": slug,
                "fighter_name": name,
                "order": order,
                "result": result.lower(),
                "opponent": opponent,
                "promotion": promotion,
                "event": title,
                "event_date": date,
                # The referee is appended to the method cell; keep the method.
                "method": method.split(")")[0] + ")" if ")" in method else method,
                "round": rnd,
                "time": clock,
            })
    return rows


def _split_event(event):
    """Pull promotion, title, and date out of one event cell."""
    date = ""
    match = DATE_IN_EVENT.search(event)
    if match:
        month, day, year = match.groups()
        if month in MONTHS:
            date = f"{year}-{MONTHS[month]:02d}-{int(day):02d}"
        event = event[:match.start()].strip()
    if " - " in event:
        promotion, title = event.split(" - ", 1)
    else:
        promotion, title = event, ""
    return canonical_promotion(promotion), title.strip(), date


# Sherdog does not always separate promotion from event. Sometimes the cell is
# "Bellator - Bellator 293", sometimes just "Bellator 293", which would make
# every numbered card its own promotion and shatter the rating graph into
# hundreds of one-event fragments.
EVENT_SUFFIX = re.compile(
    r"\s+(?:\d+|[IVXLC]+|"
    r"(?:FC|MMA|Fighting|Championship\w*)?\s*\d+)\s*$", re.I)


def canonical_promotion(raw):
    """Collapse a promotion string to a stable organisation name."""
    name = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not name:
        return ""
    # Strip a trailing card number, repeatedly: "Bellator 293" -> "Bellator".
    for _ in range(3):
        stripped = EVENT_SUFFIX.sub("", name).strip(" :-")
        if stripped == name or not stripped:
            break
        name = stripped
    # Drop a trailing season or year marker: "PFL 2024 Season" -> "PFL".
    name = re.sub(r"\s+(?:19|20)\d{2}(?:\s+Season)?\s*$", "", name).strip()
    return name


def collect(names, out_path="data/sherdog_bouts.csv", pause=1.5, limit=None,
            verbose=True):
    out_path = Path(out_path)
    done, rows = set(), []
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                done.add(row["fighter_name"].lower())
                rows.append(row)
        if verbose:
            print(f"{len(done)} fighters already collected")
    pending = [n for n in names if n.lower() not in done]
    if limit:
        pending = pending[:limit]
    if verbose:
        print(f"{len(pending)} to fetch")

    unresolved = []
    for name in pending:
        try:
            path = find_fighter(name, pause=pause)
        except Exception as error:  # noqa: BLE001 - reported, run continues
            unresolved.append((name, repr(error)[:60]))
            continue
        if path is None:
            # Ambiguous or absent. Recorded, never guessed at.
            unresolved.append((name, "no unambiguous match"))
            continue
        try:
            record = parse_record(fetch(BASE + path, pause=pause), path)
        except Exception as error:  # noqa: BLE001
            unresolved.append((name, repr(error)[:60]))
            continue
        if not record:
            unresolved.append((name, "no bouts parsed"))
            continue
        for row in record:
            row["fighter_name"] = row["fighter_name"] or name
        rows.extend(record)
        done.add(name.lower())
        if verbose:
            print(f"  {name:28s} {len(record):3d} bouts  {path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOUT_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if verbose:
        print(f"wrote {len(rows)} bouts for {len(done)} fighters to {out_path}")
        if unresolved:
            print(f"unresolved ({len(unresolved)}):")
            for name, why in unresolved[:15]:
                print(f"  {name:28s} {why}")
    return rows, unresolved


def unknown_fighters(snapshots_path="prediction_snapshots.csv",
                     fights_path="fights_v2.csv"):
    """Names appearing in scored snapshots with no UFCStats career."""
    import pandas as pd

    from identity import norm_name

    fights = pd.read_csv(fights_path)
    known = set(fights["fighter_a"].map(norm_name)) | set(
        fights["fighter_b"].map(norm_name))
    snapshots = pd.read_csv(snapshots_path)
    seen = {}
    for column in ("pick", "opp"):
        for name in snapshots[column].dropna():
            seen.setdefault(norm_name(name), name)
    return [display for key, display in sorted(seen.items())
            if key not in known]


def opponent_names(bouts_path="data/sherdog_bouts.csv"):
    """Opponents named in collected records whose own record is not yet held.

    This is the second hop, and it is what makes strength of schedule
    computable. A 20-0 run means nothing until you know whether those twenty
    opponents were themselves 3-105 or were future PFL fighters. Without it a
    promotion can only be ranked by its label; with it, by who actually fought
    there.
    """
    import pandas as pd

    from identity import norm_name

    bouts = pd.read_csv(bouts_path)
    have = {norm_name(n) for n in bouts["fighter_name"].dropna()}
    seen = {}
    for name in bouts["opponent"].dropna():
        key = norm_name(name)
        if key and key not in have:
            seen.setdefault(key, name)
    return [display for _, display in sorted(seen.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sherdog_bouts.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--names", default="",
                        help="comma-separated; defaults to unknown fighters")
    parser.add_argument("--opponents", action="store_true",
                        help="second hop: collect opponents of collected fighters")
    args = parser.parse_args()
    if args.opponents:
        names = opponent_names(args.out)
        print(f"{len(names)} opponents without a record of their own")
    else:
        names = ([n.strip() for n in args.names.split(",") if n.strip()]
                 or unknown_fighters())
        print(f"{len(names)} fighters with no UFCStats history")
    collect(names, args.out, pause=args.pause, limit=args.limit)


if __name__ == "__main__":
    main()
