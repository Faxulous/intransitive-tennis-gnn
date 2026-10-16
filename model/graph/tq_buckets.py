"""Tournament tier weights. Maps tourney_level to one of four tiers:
    GS (Grand Slam), Finals (Tour Finals/Masters Cup), 1000, 500.
GS is pinned at 1.0 in the model; the other three are tuned. Unknown levels
fall back to '500' (the lowest tier present in the filtered dataset).
"""
TIER_MAP = {
    'Grand Slam':         'GS',
    'Masters Cup':        'Finals',
    'Tour Championships': 'Finals',
    'Masters 1000':       '1000',
    'WTA1000':            '1000',
    'ATP500':             '500',
    'WTA500':             '500',
}

TIERS = ['GS', 'Finals', '1000', '500']
FALLBACK_TIER = '500'  # for unseen tourney_level (e.g., ATP250 if it ever appears)


def get_tier(level: str) -> str:
    """Returns tier for a tourney_level; falls back to '500' for unseen."""
    return TIER_MAP.get(level, FALLBACK_TIER)
