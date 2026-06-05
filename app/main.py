"""FastAPI backend for the NFL win predictor.

Routes:
  GET  /         -> renders the page with the team dropdowns
  POST /predict  -> returns the prediction JSON for a matchup
"""

import nflreadpy as nfl
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.predict import TEAMS, predict_game

app = FastAPI(title="NFL Win Predictor")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# team metadata: logo URL and primary color keyed by abbr
_team_df = nfl.load_teams().to_pandas()
TEAM_META = {
    row["team_abbr"]: {
        "logo": row["team_logo_espn"],
        "color": row["team_color"],
        "color2": row["team_color2"],
    }
    for _, row in _team_df.iterrows()
    if row["team_abbr"] in TEAMS
}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"teams": TEAMS, "team_meta": TEAM_META})


@app.get("/model-stats")
def model_stats():
    """Static model info for the 'show why' panel — pre-computed, no recalculation."""
    return {
        "accuracy": [
            {"label": "Always pick home (baseline)", "value": 53.5, "color": "#9ca3af"},
            {"label": "Logistic Regression",          "value": 63.6, "color": "#6366f1"},
            {"label": "Random Forest",                "value": 60.8, "color": "#8b5cf6"},
            {"label": "XGBoost (this model)",         "value": 61.6, "color": "#1f6feb"},
        ],
        "shap_features": [
            {"label": "Scoring offense edge",  "value": 0.408},
            {"label": "Scoring defense edge",  "value": 0.276},
            {"label": "Rushing yards edge",    "value": 0.191},
            {"label": "Passing yards edge",    "value": 0.163},
            {"label": "Turnover edge",         "value": 0.105},
            {"label": "Temperature",           "value": 0.097},
        ],
        "trained_on": "1,295 regular-season games (2019–2023)",
        "tested_on":  "544 regular-season games (2024–2025)",
    }


@app.post("/predict")
def predict(home_team: str = Form(...), away_team: str = Form(...)):
    if home_team == away_team:
        return JSONResponse({"error": "Pick two different teams."}, status_code=400)
    result = predict_game(home_team, away_team)
    result["home_meta"] = TEAM_META.get(home_team, {})
    result["away_meta"] = TEAM_META.get(away_team, {})
    return result
