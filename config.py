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

MODEL_VERSION = "production-v2"
BOOTSTRAP_MODELS = 30
