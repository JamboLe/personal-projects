"""Prediction engine used by the web app.

Loads the trained XGBoost (nospread, the honest model), assembles a feature row
for a hypothetical matchup from each team's most recent rolling form, and returns
the win probability plus the top SHAP factors and display stats/players.
"""

import json
import pickle

import pandas as pd
import shap

VARIANT = "nospread"   # app uses the honest model by default

_model = pickle.load(open(f"models/xgboost_{VARIANT}.pkl", "rb"))
_features = json.load(open(f"models/feature_list_{VARIANT}.json"))
_data = pd.read_csv("data/processed/merged_features.csv")
_players = pd.read_csv("data/raw/player_stats.csv")
_explainer = shap.TreeExplainer(_model)

ROLL_STATS = ["pass_yards", "rush_yards", "turnovers",
              "points_scored", "points_allowed"]

# friendly labels for SHAP factors shown in the UI
LABELS = {
    "spread_line": "Vegas line",
    "home_rest": "Home rest days", "away_rest": "Away rest days",
    "div_game": "Divisional game", "week": "Week of season",
    "temp": "Temperature", "wind": "Wind", "roof_dome": "Dome",
    "pass_yards_roll5_diff": "Passing-yards edge",
    "rush_yards_roll5_diff": "Rushing-yards edge",
    "turnovers_roll5_diff": "Turnover edge",
    "points_scored_roll5_diff": "Scoring-offense edge",
    "points_allowed_roll5_diff": "Scoring-defense edge",
}

TEAMS = sorted(set(_data["home_team"]) | set(_data["away_team"]))


def _latest_team_form(team):
    """Most recent trailing 5-game form for a team (as of its last game)."""
    rows = _data[(_data["home_team"] == team) | (_data["away_team"] == team)]
    if rows.empty:
        raise ValueError(f"Unknown team: {team}")
    last = rows.sort_values(["season", "week"]).iloc[-1]
    side = "home" if last["home_team"] == team else "away"
    return {s: float(last[f"{side}_{s}_roll5"]) for s in ROLL_STATS}


def _build_feature_row(home_team, away_team):
    home = _latest_team_form(home_team)
    away = _latest_team_form(away_team)

    # neutral pre-game context for a hypothetical matchup
    row = {
        "home_rest": 7, "away_rest": 7, "div_game": 0,
        "week": int(_data["week"].max()), "temp": 70, "wind": 0, "roof_dome": 0,
    }
    for s in ROLL_STATS:
        row[f"{s}_roll5_diff"] = home[s] - away[s]
    return pd.DataFrame([row])[_features]


def _team_stat_card(team):
    f = _latest_team_form(team)
    return {
        "Passing yds/g": round(f["pass_yards"], 1),
        "Rushing yds/g": round(f["rush_yards"], 1),
        "Turnovers/g": round(f["turnovers"], 2),
        "Points scored/g": round(f["points_scored"], 1),
        "Points allowed/g": round(f["points_allowed"], 1),
    }


def _top_players(team):
    """Top 3 players for a team by their primary yardage in the latest season."""
    df = _players[_players["team"] == team]
    if df.empty:
        return []
    latest_season = df["season"].max()
    df = df[df["season"] == latest_season].copy()
    df["primary_yards"] = df[["passing_yards", "rushing_yards",
                              "receiving_yards"]].max(axis=1)
    totals = (df.groupby(["player_display_name", "position"], as_index=False)
                .agg(passing=("passing_yards", "sum"),
                     rushing=("rushing_yards", "sum"),
                     receiving=("receiving_yards", "sum")))
    totals["key_yards"] = totals[["passing", "rushing", "receiving"]].max(axis=1)
    totals["key_stat"] = totals[["passing", "rushing", "receiving"]].idxmax(axis=1)
    top = totals.sort_values("key_yards", ascending=False).head(3)
    return [{"name": r["player_display_name"], "position": r["position"],
             "stat": f"{int(r['key_yards'])} {r['key_stat']} yds ({latest_season})"}
            for _, r in top.iterrows()]


def predict_game(home_team, away_team):
    feat = _build_feature_row(home_team, away_team)
    prob = float(_model.predict_proba(feat)[0, 1])

    shap_vals = _explainer.shap_values(feat)[0]
    top = sorted(zip(_features, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "home_team": home_team, "away_team": away_team,
        "home_win_prob": prob, "away_win_prob": 1 - prob,
        "shap_factors": [{"feature": LABELS.get(f, f),
                          "impact": round(float(v), 3),
                          "direction": "home" if v > 0 else "away"}
                         for f, v in top],
        "home_stats": _team_stat_card(home_team),
        "away_stats": _team_stat_card(away_team),
        "home_players": _top_players(home_team),
        "away_players": _top_players(away_team),
    }
