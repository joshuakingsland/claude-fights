"""Card structure inferred from scheduled start times.

The odds feed gives a commence time per fight but says nothing about which
card a fight belongs to or how many rounds it is scheduled for. Both matter,
and neither can be read off a calendar date: 2026-08-08 carried two separate
cards, and `five_rounds` was hardcoded to 0 for every fight ever fetched.

Grouping uses a gap in start times rather than the UTC date, because a card
running past midnight UTC is still one card and two cards on one UTC date are
still two.

Round count is inferred from position: the last fight on a UFC card is the
main event and has been scheduled for five rounds since 2012. That inference
is only made when start times inside a group actually separate. Far-future
cards arrive with every fight sharing one placeholder time, and guessing a
main event from an arbitrary tie would be worse than admitting the card order
is unknown. Title fights below the main event are also five rounds and are
not detectable here, so this is a floor on the five-round count, not a
complete answer.
"""

import pandas as pd

EVENT_GAP_HOURS = 8.0


def _moment(value):
    if not value:
        return None
    try:
        return pd.Timestamp(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def event_groups(starts):
    """Cluster scheduled start times into cards."""
    order = sorted(range(len(starts)), key=lambda i: (starts[i] or ""))
    groups = [0] * len(starts)
    label, previous = 0, None
    for position in order:
        moment = _moment(starts[position])
        if moment is not None and previous is not None:
            if (moment - previous).total_seconds() / 3600.0 > EVENT_GAP_HOURS:
                label += 1
        groups[position] = label
        if moment is not None:
            previous = moment
    return [f"event-{g}" for g in groups]


def card_ufc_experience(groups, experienced):
    """Per row, the share of that card's fighters with a prior UFC bout.

    The odds feed labels every event "MMA" and names no promotion, so what
    kind of card this is cannot be read off the provider. It can be measured.
    A UFC card is made of fighters who have fought in the UFC; a Contender
    Series or regional card is made of fighters who have not.

    On 2026-08-10 that separated cleanly: the five-bout 2026-08-12 card scored
    0.00 with not one of its ten fighters holding a UFC bout, and every other
    card on the board scored 0.96 to 1.00.

    `experienced` is one boolean per fight per corner, as (a, b) pairs, so the
    caller owns the definition of "has fought in the UFC" and this stays a
    pure count.
    """
    totals, seen = {}, {}
    for group, (a, b) in zip(groups, experienced):
        totals[group] = totals.get(group, 0) + int(bool(a)) + int(bool(b))
        seen[group] = seen.get(group, 0) + 2
    return [totals[group] / seen[group] if seen.get(group) else 0.0
            for group in groups]


def infer_five_rounds(starts):
    """1 for each group's main event, 0 elsewhere, 0 when order is unknown.

    A group whose start times are all identical carries no ordering, so every
    fight in it stays 0 rather than one being picked arbitrarily.
    """
    groups = event_groups(starts)
    moments = [_moment(value) for value in starts]
    flags = [0] * len(starts)
    for group in set(groups):
        members = [i for i, g in enumerate(groups) if g == group]
        known = [i for i in members if moments[i] is not None]
        if len(known) < 2:
            continue
        latest = max(moments[i] for i in known)
        if latest == min(moments[i] for i in known):
            continue  # one placeholder time for the whole card; order unknown
        winners = [i for i in known if moments[i] == latest]
        if len(winners) == 1:
            flags[winners[0]] = 1
    return flags
