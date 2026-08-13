"""The sealed holdout, enforced in code rather than by intention.

PREREGISTRATION.md commits every card from 2025-09-01 onward to a holdout that
is opened exactly once, after a decision has been reached on the earlier data.
A rule like that kept only in a document is a rule that gets broken by
accident: some future analysis loads the whole archive because loading the
whole archive is the obvious thing to do, nobody notices, and the holdout is
quietly spent.

So every loader goes through here. `development()` is the default and drops
sealed rows. Reading the holdout requires calling `unseal()` with an explicit
reason, which writes an audit line to sealed_access.log - if that file has more
than one entry, the holdout is no longer a holdout and the final check is void.

This does not stop a determined person from reading the CSV directly. It stops
the accident, which is the realistic failure.
"""

import datetime
from pathlib import Path

import pandas as pd

SEAL_DATE = pd.Timestamp("2025-09-01")
ACCESS_LOG = Path("sealed_access.log")


def _dates(frame, column):
    if column not in frame.columns:
        raise ValueError(
            f"cannot apply the seal: no {column!r} column to judge dates by. "
            "Refusing to return rows rather than guessing."
        )
    return pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_localize(None)


def is_sealed(frame, column="event_date"):
    """Boolean mask of rows inside the holdout."""
    return _dates(frame, column) >= SEAL_DATE


def development(frame, column="event_date"):
    """Everything outside the holdout. The default for all analysis.

    Rows with an unreadable date are dropped rather than kept: an undated row
    could be from either side, and letting it through would leak on the side
    that matters.
    """
    dates = _dates(frame, column)
    return frame[(dates < SEAL_DATE) & dates.notna()].copy()


def unseal(frame, reason, column="event_date"):
    """The holdout. Every call is logged; more than one voids the check.

    `reason` is required and recorded. It exists so the log says what the
    holdout was spent on, not merely that it was spent.
    """
    if not reason or not str(reason).strip():
        raise ValueError("unsealing requires a stated reason; it goes in the log")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    previous = ACCESS_LOG.read_text(encoding="utf-8").strip().splitlines() \
        if ACCESS_LOG.exists() else []
    with ACCESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{reason.strip()}\n")
    if previous:
        print(f"  WARNING: the holdout has been opened {len(previous)} time(s) "
              f"before. Per PREREGISTRATION.md it is opened once; any further "
              f"result from it is not a clean out-of-sample check.")
    dates = _dates(frame, column)
    return frame[(dates >= SEAL_DATE) & dates.notna()].copy()


def report(frame, column="event_date"):
    """How the archive splits, safe to call without touching holdout rows."""
    dates = _dates(frame, column)
    sealed = int((dates >= SEAL_DATE).sum())
    return {
        "rows_total": int(len(frame)),
        "rows_development": int(((dates < SEAL_DATE) & dates.notna()).sum()),
        "rows_sealed": sealed,
        "rows_undated": int(dates.isna().sum()),
        "seal_date": str(SEAL_DATE.date()),
        "holdout_opened": (len(ACCESS_LOG.read_text(encoding="utf-8").strip().splitlines())
                           if ACCESS_LOG.exists() else 0),
    }
