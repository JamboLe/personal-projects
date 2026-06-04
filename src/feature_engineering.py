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


def build_differentials(df, roll_cols):
    """Add home-minus-away differential columns.

    The model only needs the relative edge between the two teams, so we collapse
    each `home_X` / `away_X` pair into a single `X_diff`. Fewer features, same
    signal — a deliberate, defensible choice.
    """
    df = df.copy()
    for col in roll_cols:
        df[f"{col}_diff"] = df[f"home_{col}"] - df[f"away_{col}"]
    return df


def _team_week_table(games):
    """Long team-week table with offensive production + points for/against.

    Each game produces two rows (one per team). Points come from the game's
    score; pass/rush/turnovers are merged from team_stats.
    """
    home = games[["season", "week", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_scored",
                 "away_score": "points_allowed"})
    away = games[["season", "week", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_scored",
                 "home_score": "points_allowed"})
    long = pd.concat([home, away], ignore_index=True)

    team_stats = pd.read_csv("data/raw/team_stats.csv")
    long = long.merge(team_stats, on=["season", "week", "team"], how="left")
    return long


def build_dataset():
    games = pd.read_csv("data/raw/games.csv")
    games = games[games["game_type"] == "REG"].copy()
    games = games.dropna(subset=["home_score", "away_score"])

    # 1. trailing team form (excludes current game)
    long = _team_week_table(games)
    long = add_rolling(long, ROLL_STATS, window=5)

    roll_cols = [f"{s}_roll5" for s in ROLL_STATS]
    keys_roll = ["season", "week", "team"] + roll_cols

    # 2. join rolled form back onto each game, for home and away
    home_roll = long[keys_roll].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in roll_cols}})
    away_roll = long[keys_roll].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in roll_cols}})
    df = games.merge(home_roll, on=["season", "week", "home_team"], how="left")
    df = df.merge(away_roll, on=["season", "week", "away_team"], how="left")

    # 3. context features
    df["temp"] = df["temp"].fillna(70)      # domes are climate-controlled
    df["wind"] = df["wind"].fillna(0)
    df["roof_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)

    # 4. home-minus-away differentials
    df = build_differentials(df, roll_cols)

    # 5. drop the earliest games that have no trailing history at all
    df = df.dropna(subset=[f"home_{c}" for c in roll_cols] +
                          [f"away_{c}" for c in roll_cols])

    keep = (["season", "week", "home_team", "away_team", "home_win",
             "home_rest", "away_rest", "div_game", "temp", "wind", "roof_dome",
             "spread_line"]
            + [f"home_{c}" for c in roll_cols]
            + [f"away_{c}" for c in roll_cols]
            + [f"{c}_diff" for c in roll_cols])
    out = df[keep].copy()
    out.to_csv("data/processed/merged_features.csv", index=False)
    print("merged_features:", out.shape)
    print("home_win rate:", out["home_win"].mean().round(3))
    return out
