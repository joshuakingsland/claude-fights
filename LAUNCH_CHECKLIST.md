# Launch checklist

## Public dashboard

1. Rotate the API key that was previously shared and store only the replacement
   in the repository Actions secret named `ODDS_API_KEY`.
2. Upload the GitHub-ready package. Keep the local-research package private.
3. In repository Actions settings, allow workflows to read and write contents.
4. In Pages settings, deploy the `main` branch from `/docs`.
5. Run the `Test` workflow and require it to pass before merging changes.
6. Run `Update Fight Ledger` manually once. Confirm that it fetches current
   odds, generates `docs/index.html`, and commits without a push conflict.
7. Confirm `data_freshness.json` is `current` and the dashboard shows the same
   `results_through` date.
8. Leave prop discovery at its default zero-request cap until API usage is
   reviewed. Confirm the T-30 workflow has not duplicated event IDs.
9. Open the published page and verify its UTC update stamp, fighter names,
   consensus prices, best book/price, start times, and `paper_only` language
   before sharing the URL.
10. Review API quota before enabling `Snapshot MMA Market`; its six-hour
    schedule adds roughly 124 current-odds calls in a 31-day month.
11. Confirm `staking_validation.json` says the active and 2-unit candidate
    policies are `paper_only`.

## Operating rules

- Do not upload ignored historical odds CSVs or API response artifacts.
- Do not bypass `--require-key` in the scheduled workflow.
- Do not edit generated ledger rows after they are committed.
- Review failed or empty-card workflow runs before rerunning them.
- Upgrade pinned dependencies only with fresh production and entry-price audits.
- Keep all selections paper-only until the promotion gates pass. Publishing
  the dashboard is not approval for real-money operation.
- Keep active allocation flat at 1 unit with the 2-unit event-day cap. Do not
  treat the displayed 10-point 2-unit threshold as an approved stake tier.
