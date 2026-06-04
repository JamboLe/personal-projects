"""FastAPI backend for the NFL win predictor.

Routes:
  GET  /         -> renders the page with the team dropdowns
  POST /predict  -> returns the prediction JSON for a matchup
"""

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.predict import TEAMS, predict_game

app = FastAPI(title="NFL Win Predictor")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"teams": TEAMS})


@app.post("/predict")
def predict(home_team: str = Form(...), away_team: str = Form(...)):
    if home_team == away_team:
        return JSONResponse({"error": "Pick two different teams."}, status_code=400)
    return predict_game(home_team, away_team)
