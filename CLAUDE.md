# claude-fights

A UFC moneyline model, a forward-test ledger, and a published card at `docs/`.
Scheduled workflows refresh results, capture odds, price the card, and commit
the outputs.

## The most important thing in this repository

**The model does not have an edge, and nothing here should imply it does.**

Twenty-four pre-registered hypotheses were tested and are recorded in
`PREREGISTRATION.md` with their results. The model's own edge rule failed its
primary test (H1). Its forward ledger is small and has not established an
edge; use the current corrected `paper_validation.json` rather than a stale
profitability claim. The one effect that survived is **line shopping** - taking
the best of ~16 book prices instead of one, worth about 3 points - and that is
mechanical rather than predictive.

So: the card, the notifications, and any new surface must read as information
rather than as a recommendation. `notify_card.py` states the ledger's real
record in every notice for exactly this reason. Do not remove that.

## Rules that are not style preferences

- **Never make a check pass by weakening it.** Do not widen a grace window,
  lower a threshold, delete an assertion, swallow an exception, or skip a test
  to turn a run green. These guards protect a betting ledger and a published
  page. A red workflow is cheap; a silently wrong ledger is not.
- **`PREREGISTRATION.md` is append-only.** It records what was predicted before
  each test was run, which is the only thing making those results meaningful.
  Add addenda; never edit or delete a recorded prediction or result.
- **The sealed holdout is spent.** `sealed.py` enforced a holdout that was
  opened once, on 2026-08-23, and `sealed_access.log` records it. No result
  from this archive is out-of-sample any more. Do not describe one as if it is.
- **Data and audit files are not code.** Do not edit `raw/`, `fights_v2.csv`,
  `paper_trades.csv`, `paper_settlements.csv`, `prediction_snapshots.csv`,
  `h22_forward_log.csv`, `notified_trades.csv`, `unmatchable_bookings.csv`, or
  `sealed_access.log` by hand. The workflows own them.

## Where failures come from

Most scheduled failures have been `freshness.py`, and all of those from one
root cause: **the guard cannot tell "the result is late" from "no result is
ever coming"**, and only the first is fixed by waiting. Variants seen so far:

| symptom | cause | fix |
|---|---|---|
| whole card missing | third-party results scrape publishes days late | grace window |
| one fighter, several opponents | booking replaced, or the same man spelled two ways | `known_superseded` |
| an old Contender Series bout appears | a fighter's later UFC debut made it retroactively "trackable" | scope judged as of the fight date |
| a real fight is "unmatchable" with its result on record | the odds feed revised `commence_time`, so the stale date is a phantom | `known_rescheduled` |
| a bout no card ever held sits waiting | it was quoted but never happened | `known_absent_from_card` |

The fix is a new classification in `assess_freshness`, not a longer grace
window. Isolated stragglers past the window are written off automatically into
`unmatchable_bookings.csv`; more than five at once is refused, because that
means something systemic broke.

**A write-off is the fallback, not the answer.** The quarantine log reached
fifteen entries, of which fourteen were explainable and three were real UFC
bouts - Pantoja vs Van among them - written off on a date the feed had
abandoned while their results sat in `fights_v2.csv`. The lid never tripped
because they arrived three to five at a time, so the pipeline read `current`
while discarding genuine fights. `--prune-quarantine` now releases rows the
classifications can account for, and `known_quarantine_resolved` reports a
write-off whose result later turned up. Prefer explaining a booking to
excusing it.

### The other source: one source, two spellings

UFCStats spells the same fighter differently across its own tables. The
fighter list had "Nicholas Galanti"; the results table wrote the bout as
"Nick Galanti". Shortened first names do not normalise, so `resolve_fighter`
found nothing and `update_data.py` aborted on every scheduled run from
2026-10-19 to the 29th.

The guard is right to refuse - `audit_fights` rejects a bout with a blank
fighter URL, so a bout with an unidentified fighter must never reach
`fights_v2.csv`, and `strict=False` is not a way out of this. **The remedy is
always a `NAME_ALIASES` entry in `identity.py`**, verified against the fighter
URL before adding it: two spellings of one man share a URL, two men do not.
`alias_candidates` suggests same-surname targets and never applies one,
because that would be the fuzzy cross-fighter join the explicit table exists
to avoid.

What made this expensive was not the name:

- **It aborted step one of fourteen.** Odds capture, the card refresh and
  settlement all got skipped, and only `update.yml` settles, so five paper
  trades sat unsettled for eighteen days. The step is now
  `continue-on-error` with the alarm at the end, exactly as the freshness
  step already was - safe because `update_data.py` stages its download and
  promotes it only after the audit passes, so a failure leaves the previous
  results intact and settlement matches fewer results rather than wrong ones.
- **It reported one name at a time.** Each run is a day apart, so a second
  unresolved name behind the first would have cost another ten days.
  `assign_fighter_identities` now collects every one before raising.
- **Nothing told anyone.** `CLAUDE_CODE_OAUTH_TOKEN` was empty, so the
  autofix workflow died inside the action on every attempt. It now checks
  that secret first and says so.

`known_absent_from_card` has its own lid (`MAX_ABSENT_PER_CARD`) and requires
that bookings for that night actually matched results, not merely that results
exist. Both conditions are there for one reason: if `canonical_name` ever stops
matching, the results are still present and nothing matches them, and without
those guards every bout would be excused as "not on the card" - the guard would
go green on the worst failure it can have. Do not relax either.

**Reproducing a freshness failure needs `python update_data.py` first.** The
committed `fights_v2.csv` often predates the problem, so the check passes
locally until you refresh results the way CI does.

## Testing

```bash
python3.12 -m venv .venv                 # 3.12 specifically - see below
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests    # what CI runs
```

**The venv must be Python 3.12.** `requirements.txt` pins `scipy==1.18.0`,
which requires >=3.12, and CI pins 3.12.13. On a container whose default
`python3` is 3.11 the install fails with "Could not find a version that
satisfies the requirement scipy==1.18.0" and a long list of versions that
looks like a dependency problem. It is an interpreter problem. Do not relax
the pin to match a local interpreter.

Tests are the specification here, and several have caught real bugs in fixes
that looked right. When a test fails after a change, work out which of the two
is wrong before editing either - more than once the fixture was the thing at
fault, and more than once it was not.

## Layout

| file | what it does |
|---|---|
| `predict_card.py` | prices the upcoming card, writes `docs/index.html` |
| `freshness.py` | judges whether results are current; source of most failures |
| `paper_ledger.py` | the forward-test ledger; settlement, CLV |
| `shopping.py` | best price and what it saves - the one validated effect |
| `market_residual.py`, `sharp_fade.py`, `fair_line.py`, `devig.py` | H9-H24 research |
| `forward_test.py` | H22's forward log, the only route left to an answer |
| `notify_email.py`, `notify_card.py` | Resend / webhook / SMTP notifications |
