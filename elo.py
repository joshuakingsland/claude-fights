"""Chronological Elo ratings for fighters.

Key design point: for each fight we record each fighter's rating *before*
the fight, then update. This guarantees the feature is point-in-time safe.
"""

import pandas as pd


def compute_elo(
    fights: pd.DataFrame,
    k: float = 32.0,
    base: float = 1500.0,
    finish_bonus: float = 8.0,
) -> pd.DataFrame:
    """Add pre-fight Elo columns to a chronologically sortable fight table.

    Parameters
    ----------
    fights : DataFrame with columns
        date        : fight date (sortable)
        fighter_a   : name/id of fighter A
        fighter_b   : name/id of fighter B
        winner      : 'A', 'B', or 'draw'
        method      : optional, contains 'KO', 'SUB', 'DEC', ... (used for
                      a small K bump on finishes; ignored if absent)
    k : base K-factor.
    base : starting rating for unseen fighters.
    finish_bonus : extra K applied when the fight ends in a finish.

    Returns
    -------
    Copy of `fights` sorted by date with new columns:
        elo_a_pre, elo_b_pre : ratings going *into* the fight
        elo_diff             : elo_a_pre - elo_b_pre
    """
    df = fights.sort_values("date", kind="stable").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    elo_a_pre, elo_b_pre = [], []
    has_method = "method" in df.columns

    for row in df.itertuples(index=False):
        ra = ratings.get(row.fighter_a, base)
        rb = ratings.get(row.fighter_b, base)
        elo_a_pre.append(ra)
        elo_b_pre.append(rb)

        # Expected score for A
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

        if row.winner == "A":
            sa = 1.0
        elif row.winner == "B":
            sa = 0.0
        else:  # draw / no contest
            sa = 0.5

        k_eff = k
        if has_method and isinstance(row.method, str):
            m = row.method.upper()
            if "KO" in m or "SUB" in m or "TKO" in m:
                k_eff += finish_bonus

        delta = k_eff * (sa - ea)
        ratings[row.fighter_a] = ra + delta
        ratings[row.fighter_b] = rb - delta

    df["elo_a_pre"] = elo_a_pre
    df["elo_b_pre"] = elo_b_pre
    df["elo_diff"] = df["elo_a_pre"] - df["elo_b_pre"]
    return df
