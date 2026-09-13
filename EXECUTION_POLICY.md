# Five-minute paper execution observations

Policy `quote-persistence-5m-v1` begins with locks dated 2026-09-14 00:00 UTC.
It observes the existing strategy; it does not alter selection, stake size, or
the model. Earlier locks are historical diagnostics, separated in the JSON.

For each canonical first-touch paper wager:

1. Verify the requested price against the archived quote captured at
   `odds_fetched_at`, with the same bookmaker, unordered fighter pair, and
   scheduled start. That capture must precede the lock by at most 15 minutes.
   Record the provider event ID and bookmaker key from this source.
2. Request one additional market capture five minutes after new locks. Multiple
   locks in the same run share one capture, scheduled from the latest lock.
   Accept observations only between five and seven minutes after each lock.
3. Use the first event snapshot in that window. Never select a later snapshot
   because its price is better. Use the original bookmaker and event ID; handle
   reversed corners. The quote must belong to a priced region, precede both
   original and current start times, and have a book update no more than
   15 minutes old and no later than the capture timestamp.
4. Append one final observation per trade after its window closes. Keep missing
   books, missing windows, invalid prices, and stale quotes explicit. Rerunning
   the process never rewrites an existing check. Duplicate rescheduled locks
   use the canonical paper ledger view.

`execution_checks.csv` is an observation ledger owned by the script. It does
not record accepted bets. A displayed quote does not establish account access,
stake limits, acceptance, or a fill. Missing evidence is not a rejected bet.
Raw paper trades, settlements, and prediction snapshots remain unchanged.

`execution_report.json` and `EXECUTION_REPORT.md` show forward cumulative and
trailing seven-day lock cohorts, coverage and missing-status counts, price
persistence, slippage, and results for the settled observed subset. Positive
slippage is deterioration: delayed implied probability minus requested implied
probability, in percentage points. ROI reprices that same subset at its delayed
quotes, including worse quotes; it is a diagnostic, not a new executable policy
or a whole-portfolio return. The original P&L on that subset is also shown.

Two closing comparisons are labeled separately: the existing entry-consensus
to close-consensus movement, and closing fair probability minus the delayed
price's break-even probability. Both report coverage. Neither proves an edge.
There is no automatic model promotion or staking change.

Capture runs immediately after new locks in the normal market/update jobs.
It adds up to seven minutes and one extra market collection only when recent
new locks exist. It uses the existing API credentials and regions. Observation
fetches append book quotes without replacing card inputs or their manifest.
A failed research capture is visible in Actions but does not discard official
locks; subsequent reports retain the observation gap.

A Monday 18:30 UTC workflow refreshes the report without an API call, writes a
dated copy under `reports/execution/`, and displays it in the Actions summary.
Existing result-settlement jobs supply outcomes. No messages or wagers are sent.

```bash
python -m unittest discover -s tests
python execution_observations.py            # archived observations and report
python execution_observations.py --capture  # live follow-up for recent locks
```
