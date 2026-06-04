import pandas as pd

from src.feature_engineering import add_rolling


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
