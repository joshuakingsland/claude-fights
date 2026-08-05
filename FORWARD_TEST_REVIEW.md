# Forward-test review, 2026-07-20 to 2026-08-05

The first sustained run of the six-hour market snapshot produced 1,731 model
snapshots over 96 distinct events and 5,260 paired sportsbook quotes. That is
enough forward data to check things the historical audits could not. This note
records what it shows. Production probabilities are unchanged.

## What the ledger actually captured

| | |
| --- | ---: |
| Observation window | 16 days |
| Model snapshots recorded | 1,731 |
| Distinct fights that were eligible at some point | 5 |
| Fights locked as official paper trades | 2 |
| Capture rate | 40% |
| Settled | 2 (1 win, 1 loss, -0.49 units) |

Official trades are locked only on the Wednesday run. Eligibility is sampled
every six hours, so a signal that qualifies for three days but sits below the
edge rule at the single Wednesday timestamp never becomes a trade.

Both misses were exactly this. Ravena Oliveira vs Juliana Miller held 4.0-4.3
net edge points across ten snapshots on August 2-4 and read 3.5-3.8 on the
August 5 lock run. Diyar Nurgozhay vs Bruno Lopes ran 4.2-4.4 on August 2-3 and
3.3 on the lock run. Neither was rejected on quality; both were sampled at the
wrong moment.

At two trades per 16 days the ledger reaches the 200-bet live gate in roughly
4.4 years. The forward test is the binding constraint on the whole project, and
its throughput is currently set by a scheduling artifact rather than by the
model or the edge rule.

## The one-book price outlier

On 2026-08-01T03:49Z the model marked L'udovit Klein vs Tofiq Musayev eligible
at **26.5 net edge points**, roughly triple any other signal ever produced.

The paired-book consensus for Klein was -265 in the snapshot before and -265 in
the snapshot after. In that one capture a single book posted **+126** on the
same side. The system took it as the best executable price, computed an implied
44.2% against a consensus of 69.7%, and passed it through the edge rule.

`market_spread` recorded the anomaly correctly at 28.0 probability points,
5.6x the `MARKET_DISAGREEMENT_WARNING` threshold, but that field only set a
display flag. It was not part of `quality_ok`, so nothing blocked the signal.
The only reason it did not become an official trade is that it appeared on a
Saturday and locks happen on Wednesday.

Context for how far outside normal this was: across all 1,641 priced snapshots,
consensus-minus-execution deviation runs from -5.3 to +3.0 probability points.
Klein sat at +25.5. Real line shopping is worth one to two points.

`predict_card.quote_quality` now rejects a quote whose executable price implies
more than `MAX_EXECUTION_DEVIATION` (8 points) less probability than the
paired-book consensus, with eligibility reason `book price outlier`. Replayed
over the full snapshot history the guard fires once, on this row, and changes
nothing else.

## Edge buckets are not monotonic

From `staking_validation.json`, ROI by net-edge bucket:

| Bucket | Production bets | Production ROI | Entry bets | Entry ROI |
| --- | ---: | ---: | ---: | ---: |
| 4-6 pts | 192 | +6.1% | 107 | +2.2% |
| 6-8 pts | 103 | -2.4% | 44 | -4.9% |
| 8-10 pts | 50 | -12.3% | 29 | -20.5% |
| 10+ pts | 71 | +25.8% | 23 | +51.4% |

Edge does not order returns. The middle buckets lose money and the extreme
bucket carries the entire result on 23 entry bets. That pattern is what a
mixture of genuine edge and mispriced inputs looks like, and the Klein row is a
live example of how a signal lands in the 10+ bucket without being real.

The research 2-unit-at-10-points tier is sized off that bucket. Until the 10+
bucket is shown to survive a bad-quote filter, its ROI should be read as
unverified rather than as the best tier. The active flat 1-unit policy is
unaffected.

## Cross-book dispersion, previously deferred

`MODEL_IMPROVEMENT_AUDIT.md` deferred the sportsbook-dispersion candidate for
insufficient sample. Forward capture now gives a first measurement, though it
is still thin: 1,611 fight-snapshots, of which 613 carry three or more paired
books, covering 40 distinct fights.

Median cross-book standard deviation is 0.69 probability points and the median
best-to-worst range is 1.91 points. Both are well inside the 4-point edge rule,
which is the useful conclusion for now: normal book disagreement cannot
manufacture a qualifying signal on its own. A single broken quote can, which is
what the guard above addresses. The candidate stays deferred; 40 fights is not
a basis for promotion.

## Book coverage limits the card, not the edge rule

Of 1,731 snapshot rows, 998 (58%) were rejected for fewer than three paired
books, far more than the 551 rejected on edge. 876 of 1,611 fight-snapshots
carried exactly one book.

Most of that is horizon rather than a data problem. 637 rejected rows come from
events more than 90 days out, including cards in July and August 2027 that the
odds endpoint returns alongside the live ones. Those rows will never be
tradable and are diluting every rate computed over the snapshot file. Coverage
inside 7 days is good: mean 7.6 books per fight.

## Cadence change

Locking moved from the Wednesday run to the first scheduled run that sees a
qualifying signal, under staking policy
`paper-flat-1u-first-touch-cap2-v2`. Stake sizes and the event-day cap are
unchanged; only the timing of the sample moved.

Because locks now spread across runs, `lock_paper_trades` enforces the 2-unit
event-day cap against wagers already on the ledger rather than only within one
card scoring, and takes candidates strongest-first so an early marginal signal
cannot take the last slot from a stronger one later the same day.

The two wagers locked under the old cadence keep
`paper-flat-1u-day-cap2-v1`. `paper_validation.json` reports each policy
separately as well as pooled, so the change does not silently blend two
different rules into one ROI figure.

## Not changed here

- Model, features, edge rule, and stake sizing are untouched.
- Snapshot horizon filtering is unchanged; the far-future rows are an audit
  record, and dropping them silently would edit that record.
- The research 2-unit-at-10-points tier stays research-only and unsized.
