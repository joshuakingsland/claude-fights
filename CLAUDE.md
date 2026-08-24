# claude-fights

A UFC moneyline model, a forward-test ledger, and a published card at `docs/`.
Scheduled workflows refresh results, capture odds, price the card, and commit
the outputs.

## The most important thing in this repository

**The model does not have an edge, and nothing here should imply it does.**

Twenty-four pre-registered hypotheses were tested and are recorded in
`PREREGISTRATION.md` with their results. The model's own edge rule failed its
primary test (H1) and its forward ledger is negative on both money and
closing-line value. The one effect that survived is **line shopping** - taking
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

Almost every scheduled failure has been `freshness.py`, and all from one root
cause: **the guard cannot tell "the result is late" from "no result is ever
coming"**, and only the first is fixed by waiting. Variants seen so far:

| symptom | cause | fix |
|---|---|---|
| whole card missing | third-party results scrape publishes days late | grace window |
| one fighter, several opponents | booking replaced, or the same man spelled two ways | `known_superseded` |
| an old Contender Series bout appears | a fighter's later UFC debut made it retroactively "trackable" | scope judged as of the fight date |

If a fourth variant appears, the fix is usually a new classification in
`assess_freshness`, not a longer grace window. Isolated stragglers past the
window are written off automatically into `unmatchable_bookings.csv`; more than
five at once is refused, because that means something systemic broke.

**Reproducing a freshness failure needs `python update_data.py` first.** The
committed `fights_v2.csv` often predates the problem, so the check passes
locally until you refresh results the way CI does.

## Testing

```bash
python -m unittest discover -s tests    # what CI runs
```

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
