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
# v2 locks a qualifying signal on the first scheduled run that sees it, instead
# of only on a fixed weekday. Stake sizing and the event-day cap are unchanged;
# the version bump keeps rows locked under the old cadence separable.
STAKING_POLICY_VERSION = "paper-flat-1u-first-touch-cap2-v2"

# Live quote quality controls. A generated page can age after deployment, so
# the browser also re-checks MAX_ODDS_AGE_MINUTES before presenting a signal.
MIN_MARKET_BOOKS = 3
MAX_ODDS_AGE_MINUTES = 360
MARKET_DISAGREEMENT_WARNING = 0.05

# A single sportsbook can publish a stale or mis-mapped price that is far more
# generous than the paired-book consensus. Line shopping is worth about one to
# two probability points; anything past this gap is a bad quote, not an edge,
# so the signal is rejected instead of being priced off that book.
MAX_EXECUTION_DEVIATION = 0.08

MODEL_VERSION = "production-v3"
BOOTSTRAP_MODELS = 30
ODDS_CONSENSUS_VERSION = "paired-book-devig-v1"

# Regions requested from the odds API, each as its own request. The API bills
# one credit per region per market, so every added region multiplies the
# per-request quota cost.
ODDS_REGIONS = ("us", "eu")

# Regions whose books actually feed the model consensus and the executable
# price. A region can be captured for research without being priced, so
# widening ODDS_REGIONS never silently moves the model's most important
# feature, and never credits an edge to a price the ledger cannot execute.
# `eu` is captured to measure what Pinnacle does to the consensus before any
# decision to price it.
PRICED_ODDS_REGIONS = ("us",)

# Books that set the MMA market rather than follow it. On the 2022 per-book
# archive every other book converged toward BetOnline's entry price while it
# did not converge back; LowVig behaved the same way, and Pinnacle is the
# reference sharp market the `eu` region exists to capture.
#
# This drives a research column only. Leader and follower probabilities are
# recorded next to each snapshot so the leader-versus-follower gap can be
# tested on forward data later. Nothing here reaches the model, the consensus,
# the edge rule, or the executable price.
LEADER_BOOK_KEYS = ("pinnacle", "betonlineag", "lowvig")
