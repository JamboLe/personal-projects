"""Pull team-week and player-week stats from nflreadpy.

Discovered API notes (verified 2026-06-04):
- The handoff doc's `load_stats()` does not exist as a function; it's a submodule.
- Real functions: `load_team_stats()` (team-week level) and
  `load_player_stats()` (player-week level). Both default to summary_level='week'.
- `load_team_stats` has no single 'turnovers' column, so we derive offensive
  turnovers = interceptions thrown + fumbles lost (sack/rushing/receiving).
- Points scored/allowed are NOT in team_stats; we get those from games.csv later.
"""

import nflreadpy as nfl
import pandas as pd

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def pull_team_stats():
    df = nfl.load_team_stats(seasons=SEASONS).to_pandas()

    # offensive turnovers = interceptions thrown + all fumbles lost
    df["turnovers"] = (
        df["passing_interceptions"].fillna(0)
        + df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )

    keep = ["season", "week", "team", "passing_yards", "rushing_yards", "turnovers"]
    team_week = df[keep].rename(
        columns={"passing_yards": "pass_yards", "rushing_yards": "rush_yards"}
    )
    team_week.to_csv("data/raw/team_stats.csv", index=False)
    print("team_stats:", team_week.shape)
    return team_week


def pull_player_stats():
    df = nfl.load_player_stats(seasons=SEASONS).to_pandas()
    keep = ["season", "week", "team", "player_display_name", "position",
            "passing_yards", "rushing_yards", "receiving_yards"]
    players = df[keep]
    players.to_csv("data/raw/player_stats.csv", index=False)
    print("player_stats:", players.shape)
    return players


if __name__ == "__main__":
    pull_team_stats()
    pull_player_stats()
