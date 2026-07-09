# Fight Ledger — self-updating model site

What this is: the trained UFC prediction model + a pipeline that refreshes
fight data after each event, prices the next card against real odds, and
publishes a dashboard to a public URL. Total cost: $0. No server.

## One-time setup (~15 minutes)

1. **Create a GitHub repository** (free account is fine). Upload this
   entire folder — including the hidden `.github/` directory.

2. **Enable the website**: repo Settings → Pages → Source: "Deploy from a
   branch" → Branch: `main`, folder `/docs` → Save.
   Your dashboard will live at `https://<your-username>.github.io/<repo>/`.

3. **Odds feed (pick one)**
   - Automatic: get a free API key at the-odds-api.com (500 requests/mo;
     this uses ~8/mo). Repo Settings → Secrets and variables → Actions →
     New repository secret → name `ODDS_API_KEY`, paste the key.
   - Manual: skip the key. Edit `odds_upcoming.csv` in the GitHub web UI
     with lines from your book whenever you want a card priced, then run
     the workflow (step 4).

4. **First run**: repo Actions tab → "Update Fight Ledger" →
   "Run workflow". ~10 minutes later your site is live.

From then on it runs itself every Sunday and Monday: pulls the latest
results (the upstream stats repo re-scrapes after every event), retrains,
prices the next card, updates the page and the rolling results ledger.

## Running locally instead

    pip install -r requirements.txt
    python update_data.py      # after each event
    python fetch_odds.py       # or edit odds_upcoming.csv by hand
    python predict_card.py     # writes docs/index.html — open it

## Things to know

- **Results lag**: the upstream data source usually updates within a day
  or two of an event. The Sunday AND Monday schedule covers this.
- **New odds dataset**: model training uses historical closing odds
  through Mar 2026 (raw/ufc-master.csv). The model still prices new
  fights fine — the training window just stays fixed until that source
  updates. Rolling results after Mar 2026 use your captured odds.
- **Debutants** and fighters off multi-year layoffs are outside the
  model's reliable range; the page's methodology note covers this.
- The model's verified edge is +6.9% ROI with a 90% CI of +0.9%…+13.1%,
  and it had a flat year (2024). Flat stakes or quarter-Kelly, always.
