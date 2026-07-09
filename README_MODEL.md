# UFC Fight Prediction — Starter Pipeline

A leakage-safe skeleton for predicting fight outcomes: point-in-time
feature engineering, Elo ratings, time-based evaluation, and calibrated
probability metrics.

## Files

- `elo.py` — chronological Elo; records each fighter's rating *before* the fight.
- `features.py` — the core. Explodes fights to (fighter, fight) rows, computes
  expanding career stats with `.shift(1)` (the leakage firewall), merges back,
  and builds A−B differentials. Also `symmetrize()` for corner-swap augmentation.
- `synthetic.py` — fake data with hidden skill, for smoke tests.
- `train.py` — time-split train/val/test, Elo-only baseline, logistic
  regression, gradient boosting, symmetry-averaged predictions, calibration
  table, permutation importances.

## Run

```bash
python train.py --synthetic        # smoke test, no data needed
python train.py --data fights.csv  # real data
```

## Data schema for fights.csv

One row per fight, chronological order not required (it gets sorted):

| column | meaning |
|---|---|
| date | fight date |
| fighter_a, fighter_b | names/ids |
| winner | 'A', 'B', or 'draw' |
| method | 'KO/TKO', 'SUB', 'DEC', ... |
| dob_a/b, reach_a/b, height_a/b | physical attributes |
| fight_time_min | duration in minutes |
| sig_str_landed_a/b, sig_str_absorbed_a/b | per-fight striking totals |
| td_landed_a/b, td_attempted_a/b | per-fight takedown counts |

Missing stat columns are skipped automatically. Kaggle UFC datasets map onto
this with light renaming; write a small adapter that outputs this schema.

**Important:** the career-stat builder only sees fights in this file. If your
CSV starts in 2010, a veteran's first row looks like a debut. Either include
full histories or add pre-UFC record columns.

## Interpreting results

- Log loss is the primary metric; 0.693 = coin flip, ~0.60 ≈ Vegas closing odds.
- On **synthetic** data the numbers (acc ~0.73) are better than you should
  expect on real fights (~0.62–0.66) — real MMA is noisier.
- If real-data accuracy exceeds ~70%, suspect leakage before celebrating.

## Real data (included path)

`adapter.py` converts the raw UFCStats scrape from
github.com/Greco1899/scrape_ufc_stats (the source behind most Kaggle UFC
datasets) into this pipeline's schema:

```bash
mkdir -p raw && cd raw
for f in ufc_event_details.csv ufc_fight_results.csv \
         ufc_fight_stats.csv ufc_fighter_tott.csv; do
  curl -sLO "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/$f"
done
cd .. && python adapter.py --raw-dir raw --out fights.csv
python train.py --data fights.csv
```

Results on 8,547 real fights (1994–2026), test = last ~15% (Nov 2023 → May 2026, n=1,283):

| model | log loss | accuracy |
|---|---|---|
| Elo formula only | 0.683 | 0.550 |
| Logistic regression | **0.648** | **0.624** |
| Gradient boosting | 0.655 | 0.611 |
| GBM symmetry-averaged | 0.654 | 0.610 |

Top features by permutation importance: age_diff, career striking rates,
elo_diff. Note UFCStats lists the winner first in most bouts (A wins ~64%
of raw rows) — the differential features + symmetrization make the model
corner-blind, but never add corner identity as a feature.

## Suggested next steps

1. Adapter script: Kaggle CSV → this schema.
2. Add features: opponent-quality-adjusted stats, weight-class, short-notice
   flags, damage absorbed recently, southpaw/orthodox matchup.
3. Rolling-origin evaluation (retrain each year, test the next) for a more
   honest picture than one fixed split.
4. Probability calibration (`CalibratedClassifierCV`, isotonic) if the
   calibration table drifts on real data.

## Final model (v3, locked 2026-07)

`final_model.py` — walk-forward ridge logistic on
`[logit(line), |logit(line)|, 9 FOCUS differentials, ko_recent]`, refit
before every event, flat 1u bets where model prob > raw implied + 0.04.

Full-period walk-forward, 2019 → Mar 2026 (3,218 fights, 302 events):
log loss 0.6018 vs closing line 0.6039; accuracy 68.2% vs 67.5%;
841 bets, +58.4u, ROI +6.9% (CI90 [+0.9%, +13.1%]), positive 6 of 8
years, worst year −0.2%; quarter-Kelly ROI +9.3%.

Development discipline: features/config selected on 2019 → Jan 2025 only;
2025+ used as confirmation (model log loss 0.5736 vs line 0.5775 there).
Research trail: research.py … research4.py, final_test.py.

Known limits: single-book closing prices, no line movement, no
injury/camp/short-notice news; edge is regime-dependent (2024 ≈ flat).

## Pre-UFC records experiment (negative result, documented)

Hypothesis: the model's biggest edges cluster on fights involving
debutants/low-experience fighters, where its confidence is partly a
"veteran prior" rather than information. Adding full pro records
(regional + amateur-era fights) should convert that prior into signal.

Build: career W-L-D snapshot for 2,990 fighters (ufc.com athlete pages,
`ext/` + `pro_records.csv`), reconciled against point-in-time UFC records.
Key bug found and fixed: the snapshot dates to ~Apr 2025, so non-UFC
records must be derived against UFC records *through the snapshot date*,
not all-time (otherwise every active fighter goes negative). Final
point-in-time coverage: 92-95% of matched fights through 2025
(`features_v4.py`).

Result: no improvement — overall dev log loss -0.0011 vs -0.0017 for the
base model; low-experience-segment log loss and ROI both marginally worse.
Conclusion: the market already prices records fully (they are the most
public fact about any fighter); its advantage on debutants comes from
tape and camp information, not W-L data. Feature excluded from the final
model.

Surprise finding: the low-experience segment is the base model's BEST
segment — ROI +12.4% (CI90 [+2.6%, +22.3%], n=330 dev bets) vs +7.5%
overall. The disciplined experience/Elo prior appears to beat the
market's hype-driven pricing of unproven fighters even without record
data.

## Staking rule v2 (locked 2026-07)

Bet-selection meta-layer experiment: a learned selector (logistic on
bet-level features) failed dev walk-forward and the 2025+ confirmation —
it rediscovered priced information (favorites win more). But the edge-ROI
relationship it exposed is strongly monotonic (2-4pt edges: -1.8%; 4-8pt:
+3.1%; >8pt: +14.8%), so the production rule is now TIERED FLAT STAKES:
1u on 4-8pt edges, 2u above 8pts. Full period 2019-2026: 845 bets, 1,119u
staked, +98.9u, ROI +8.8% (CI90 [+2.1%, +15.6%]), positive every full
year including 2024. Confirmation window 2025+: +7.3% vs +4.0% flat.

## Judge-scorecard experiment (negative result, documented)

Parsed 1,027 clean OCR'd scorecards (2020-2024) into a point-in-time
"decision lean" feature (career avg judge margin per round). Dev window
(2023 -> Jan 2025): log loss neutral, betting ROI point estimate lower
than base. Not adopted, despite looking better on the 2025+ confirm
window — dev decides, and adopting confirm-flattered features is
test-set mining. Likely cause: judge margins restate striking
differentials the model already has, and mmadecisions data is public.
Artifacts: judge_lean.csv, dec_lean_feature.csv, raw/scorecards.csv.

## Odds snapshot logger

fetch_odds.py now appends every fetch to odds_log.csv (timestamped),
and the workflow runs Wed + Sun + Mon. Over months this accumulates a
self-owned early-vs-closing line dataset — the raw material for the
line-movement model and, once prop capture is added, prop validation.

## Staking rule v3 (locked 2026-07): net-edge tiers

Interaction-feature arc: all five style-matchup cross-terms (power x chin,
control x control-vulnerability, sub threat x sub losses, pressure x fade,
fade x five-round) failed dev screening — no adoptions. Combined with the
earlier GBM failure: no recoverable interaction structure at this sample
size against closing lines.

Uncertainty-aware staking: per-fight predictive SE from a 30-model
bootstrap ensemble; production rule is now NET edge (edge minus SE):
1u above 4pts net, 2u above 8pts net. Dev +13.7% vs +9.1% tiered;
confirm 2025+ +10.1% vs +7.3%; full period 443 bets, 569u staked,
+73.8u, ROI +13.0% (CI90 [+4.2%, +22.3%]). Half the volume of the gross
rule at materially higher per-unit return; gross tiers remain the
documented volume alternative.

## Regional Elo experiment (negative result + leakage case study)

Built Elo over 73,504 regional MMA fights (Tapology, 2020-2022; 66,899
fighters) as a debutant quality prior. First pass used a static
end-of-2022 snapshot: DEV showed logloss -0.0394 and ROI +27.1%
(CI90 [+22.2,+31.9]) — impossibly good, and correctly diagnosed as
leakage (snapshot contains future results, including 1,655 UFC fights
being predicted). Rebuilt point-in-time (binary-search Elo-as-of-date):
the effect vanished entirely — logloss -0.0012 vs base -0.0018, low-exp
segment WORSE (0.6013 vs 0.5988). Not adopted. Likely causes: 3-year
window means ratings are dominated by UFC results we already model;
disconnected regional scenes all start at 1500 so Elo cannot rank scene
quality; name-collision noise. Artifacts: tap_elo.csv, ext/tapology-data.
Keep the leaked-vs-honest pair as the canonical example of how future
information manufactures fake edges.

## Research ledger (complete)

ADOPTED (each dev-selected, confirm-consistent):
  1. Residual modeling vs the closing line (+line, |line| recalibration)
  2. ko_recent — fighting within 365d of being KO'd
  3. Tiered stakes by edge magnitude (1u/2u)
  4. Net-edge staking: edge minus bootstrap SE (production rule,
     full period ROI +13.0%, CI90 [+4.2%,+22.3%])
  5. Frame symmetrization (correctness fix, deployment-critical)

REJECTED (all documented above with artifacts):
  pre-UFC records · judge-margin decision lean · 5 style-matchup
  interactions · fade x five-round · recency-weighted training ·
  GBM residuals · learned bet-selection meta-model · regional Elo

PATTERN: public facts are priced; computed combinations sometimes
are not; anything that looks spectacular is leakage until proven
otherwise.

OPEN (acquisition-bound): scale BFO open/close harvest; weigh-in and
short-notice data; prop-odds capture (odds_log.csv accumulating).

## Pre-deployment audit (2026-07)

Ten-point audit before live betting. PASSED: odds math fair-EV
roundtrips; swap-order symmetry on the deployed model (0/15
asymmetries); card-vs-ladder consistency; method props sum to win
probability (max err 0.001); point-in-time safety by flip-winner test
(own-row features invariant, later fights change); walk-forward strict
date exclusion; no duplicate (date,pair) rows; documented P&L exactly
replicated. FOUND & FIXED: (1) fuzzy name matcher assigned debutant
John Garza the career of retired Pablo Garza — fuzzy fallback removed,
unknown fighters stay unknown; (2) fetch_odds could ingest Draw
outcomes from 3-way books — now guarded. FLAGGED: 4 active homonym
pairs (bruno silva, jean silva, mike davis, victor valenzuela) have
blended career features; predict_card now warns when they appear on a
card; proper fix is URL-based fighter IDs (future refactor).
