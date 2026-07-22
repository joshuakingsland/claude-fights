"""Stable configuration shared by production and research entry points."""

# Production code imports this module, never a research script with its own
# experimental dependencies.
FOCUS = [
    "age_diff", "c_apm_diff", "c_ctrld_pm_diff", "c_won_diff",
    "reach_diff", "c_tdd_diff", "c_ko_loss_n_diff", "elo_slow_diff",
    "r3_lpm_diff",
]

# A qualifying production bet needs this much edge after uncertainty.
EDGE_RULE = 0.04

# Production remains a paper-only, flat-stake policy until the forward ledger
# clears its launch gates. The larger tier is tracked as research, not used to
# allocate official paper trades.
PRODUCTION_MAX_STAKE = 1
EVENT_DAY_STAKE_CAP = 2
RESEARCH_TWO_UNIT_RULE = 0.10
STAKING_POLICY_VERSION = "paper-flat-1u-day-cap2-v1"

# Live quote quality controls. A generated page can age after deployment, so
# the browser also re-checks MAX_ODDS_AGE_MINUTES before presenting a signal.
MIN_MARKET_BOOKS = 3
MAX_ODDS_AGE_MINUTES = 360
MARKET_DISAGREEMENT_WARNING = 0.05

MODEL_VERSION = "production-v3"
BOOTSTRAP_MODELS = 30
ODDS_CONSENSUS_VERSION = "paired-book-devig-v1"
