import pandas as pd

from src.feature_engineering import (
    add_rolling, build_differentials, chronological_split)


def test_rolling_excludes_current_game():
    # one team, 3 games, scoring 10, 20, 30
    df = pd.DataFrame({
        "team": ["A", "A", "A"],
        "season": [2024, 2024, 2024],
        "week": [1, 2, 3],
        "points_scored": [10, 20, 30],
    })
    out = add_rolling(df, ["points_scored"], window=5).sort_values("week")
    # game 1: no prior games -> NaN (no history)
    # game 2: only game 1 (10) -> 10
    # game 3: games 1,2 (10,20) -> 15  (MUST NOT include the 30 from game 3)
    vals = out["points_scored_roll5"].tolist()
    assert vals[1] == 10
    assert vals[2] == 15


def test_differential_is_home_minus_away():
    row = pd.DataFrame({
        "home_pass_yards_roll5": [250],
        "away_pass_yards_roll5": [200],
    })
    out = build_differentials(row, ["pass_yards_roll5"])
    assert out["pass_yards_roll5_diff"].iloc[0] == 50


def test_split_has_no_season_overlap():
    df = pd.DataFrame({"season": [2019, 2023, 2024, 2025], "x": [1, 2, 3, 4]})
    train, test = chronological_split(df, train_max=2023)
    assert set(train["season"]) == {2019, 2023}
    assert set(test["season"]) == {2024, 2025}
    assert len(set(train["season"]) & set(test["season"])) == 0
