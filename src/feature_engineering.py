"""Build the modeling dataset from raw games + team stats.

Pipeline:
  1. add_rolling      - per-team trailing N-game averages that EXCLUDE the
                        current game (the core anti-leakage step).
  2. build_dataset    - merge rolling team form onto each game, add context
                        features, build home-minus-away differentials.
  3. chronological_split - time-ordered train/test split (no shuffling).
"""

import pandas as pd

ROLL_STATS = ["pass_yards", "rush_yards", "turnovers",
              "points_scored", "points_allowed"]


def add_rolling(df, stat_cols, window=5):
    """Trailing mean of each stat over the previous `window` games per team.

    The `.shift(1)` is what guarantees no leakage: each row sees only games
    that happened BEFORE it. `min_periods=1` gives an expanding window for
    early-season games (so we keep rows instead of dropping them to NaN).
    """
    df = df.sort_values(["team", "season", "week"]).copy()
    for col in stat_cols:
        df[f"{col}_roll{window}"] = (
            df.groupby("team")[col]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return df
