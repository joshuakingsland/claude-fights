# Archived pre-integrity artifacts

`paper_trades_legacy_mixed_predictions.csv` is the original file in which every
model display row—including passes and potentially repeated or retroactive
runs—was stored under a trade-oriented filename. It is preserved unchanged for
transparency but is excluded from all verified forward-test statistics.

`odds_upcoming_2026-07-18_stale.csv` is the past-dated card that exposed the
retroactive-recording failure mode. It is preserved as a regression fixture and
must not be copied back into the live input without updating exact future
`commence_time` values.
