"""Wait out transient odds API failures instead of dying on them.

The API key is shared with another repository, so every request here competes
with whatever that one is doing. A rate limit is not an error about our
request; it is a statement about timing, and the right response is to wait.

Two callers need this and they need it for different reasons. The historical
backfill fires tens of thousands of requests over hours, so a collision that
kills the job wastes a lot of work. The live capture runs on a schedule every
few hours, so a collision means a failed workflow and a stale card. Backfilling
densely makes the second more likely, which is why both got fixed at once.

Permanent failures are never retried. A bad key, a malformed request, or a date
outside the plan will fail identically on every attempt, and retrying only
delays a message someone needs to read.
"""

import time
import urllib.error

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 120.0


def backoff_seconds(attempt, retry_after=None, cap=BACKOFF_CAP_SECONDS):
    """Exponential backoff, deferring to the server's own Retry-After."""
    if retry_after:
        try:
            return min(float(retry_after), cap)
        except (TypeError, ValueError):
            pass
    return min(BASE_BACKOFF_SECONDS * (2 ** attempt), cap)


def with_retry(call, describe="odds API", attempts=MAX_ATTEMPTS,
               sleep=time.sleep, log=print):
    """Run `call`, retrying transient HTTP and network failures.

    `call` takes no arguments and either returns or raises. Failures are
    re-raised once the attempts are spent rather than swallowed: a caller that
    records results treats what it is handed as real, so returning a placeholder
    would launder a rate limit into data.
    """
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last = RuntimeError(f"{describe} HTTP {exc.code}: {detail[:500]}")
            if exc.code not in RETRY_STATUSES:
                raise last from exc
            wait = backoff_seconds(
                attempt, exc.headers.get("Retry-After") if exc.headers else None)
        except urllib.error.URLError as exc:
            # A dropped connection is the same class of problem as a 503.
            last = RuntimeError(f"{describe} unreachable: {exc.reason}")
            wait = backoff_seconds(attempt)
        if attempt == attempts - 1:
            break
        log(f"  transient {describe} failure ({last}); retrying in {wait:.0f}s "
            f"[attempt {attempt + 1}/{attempts}]")
        sleep(wait)
    raise last
