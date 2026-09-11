"""Historical state for query rows that must never update a fighter's career."""

import pandas as pd


def observed_bouts(fights):
    event = fights.get('event', pd.Series('', index=fights.index))
    return event.fillna('').ne('UPCOMING')


def prior_stat(values, observed, how):
    """Aggregate completed rows only, then look up state before every row.

    Indexing by the count of prior observations keeps query rows out of rolling
    windows and preserves missing historical measurements without filling them.
    """
    observed = observed.reindex(values.index).astype(bool)
    history = values[observed].reset_index(drop=True)
    if how == 'mean':
        state = history.expanding().mean()
    elif how == 'sum':
        state = history.expanding().sum()
    elif how == 'last3':
        state = history.rolling(3, min_periods=1).mean()
    elif how == 'last':
        state = history
    else:
        raise ValueError(how)
    positions = observed.astype(int).cumsum().shift(1, fill_value=0) - 1
    result = state.reindex(positions.to_numpy()).reset_index(drop=True)
    result.index = values.index
    return result
