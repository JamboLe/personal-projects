# NFL Win Predictor — Design Spec

**Date:** 2026-06-04
**Status:** Approved, ready for implementation plan
**Owner:** Jamie

---

## Goal

Build an interview-ready machine learning project that predicts the winner of
an NFL regular-season game from pre-game information. The deliverable is two
things at once:

1. A documented modeling project (notebooks + clean scripts) that shows the
   full data-science process and can be explained in a job interview.
2. A working web app (FastAPI + plain HTML/CSS/JS) where you pick two teams and
   see a predicted win probability, each team's recent stats, top players, and
   the factors driving the prediction.

The model is the focus. The UI exists to demo the model and should look
intentional and clean — like a sports-analytics tool, not an AI-generated page.

---

## Primary use case (the demo)

1. Open the web app.
2. Select a home team and an away team from two dropdowns.
3. Click "Predict."
4. See:
   - A win-probability bar (e.g. "Chiefs 67% — Ravens 33%").
   - Side-by-side team stat cards (rolling 5-game averages).
   - Top 3 players per team by position, from real Pro Football Reference stats.
   - The top 3 features that drove this specific prediction (from SHAP).

---

## Architecture

```
NFLwinloss/
├── data/
│   ├── raw/               games.csv (DONE), team_stats.csv, player_stats.csv
│   └── processed/         merged_features.csv, train.csv, test.csv
├── models/                xgboost_model.pkl, logistic_model.pkl,
│                          rf_model.pkl, feature_list.json, scaler.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb     (started)
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
├── src/
│   ├── data_collection.py            (DONE)
│   ├── feature_engineering.py        builds rolling features + merged dataset
│   ├── train.py                      trains + evaluates all three models
│   └── predict.py                    loads model, returns prediction + SHAP
├── app/
│   ├── main.py                       FastAPI app
│   ├── templates/index.html
│   └── static/{styles.css, app.js}
└── docs/
    ├── models_explained.md
    ├── features_explained.md
    └── ml_concepts.md
```

**Tech choices**
- **Modeling:** pandas, scikit-learn (LogisticRegression, RandomForest),
  xgboost, shap.
- **Process style (Option A):** explore in notebooks, then extract the final
  logic into clean `src/*.py` modules that both the training pipeline and the
  app import. Notebooks tell the story; scripts run in production.
- **Backend:** FastAPI (modern Python API — good to talk about in an interview).
- **Frontend:** server-rendered HTML template + one CSS file + one vanilla JS
  file that calls the API. No React, no framework.

---

## Data pipeline

Three stages after the existing `data_collection.py`:

1. **Pull team + player stats.** Use `nflreadpy`:
   - `load_stats()` for weekly team-level performance (passing/rushing yards,
     turnovers, points). Confirm in exploration whether it returns team-level or
     player-level rows and aggregate to team-week if needed.
   - `load_pfr_advstats()` (or `load_player_stats`) for the top-player cards in
     the UI (QB passing yards, RB rushing yards, WR receiving yards).
2. **Merge.** Join team stats onto each game row so every game has both home and
   away team stats *as of that week*. Join key: team + season + week.
3. **Rolling averages.** For each team, compute the mean of each stat over their
   **previous 5 games** — explicitly excluding the current game. This is the
   single most important step for avoiding data leakage and for capturing
   current form rather than season-long averages.

Output: `data/processed/merged_features.csv`, then a chronological
train/test split written to `train.csv` (2019–2023) and `test.csv` (2024–2025).

---

## Feature selection (decided)

These are the features the model will use, with the reasoning for each so every
choice is defensible in an interview. They split into two groups.

### Group 1 — Pre-game context (from the schedule, known before kickoff)

| Feature | Type | Why it's in |
|---|---|---|
| `spread_line` | numeric | The Vegas point spread. The market aggregates enormous amounts of information; it is consistently the strongest single predictor. Including it is realistic — it's available before kickoff. **Used only in the "with-spread" model variant — see Models section.** |
| `home_rest` / `away_rest` | numeric | Days of rest since each team's last game. Fatigue and short weeks (Thursday games) measurably affect performance. |
| `div_game` | binary | Divisional matchups tend to be closer and less predictable; teams know each other well. |
| `week` | numeric | Captures season progression — early-season noise vs. late-season form and playoff motivation. |
| `temp` / `wind` | numeric | Cold and especially wind suppress the passing game. Missing values (domes) are imputed (see below). |
| `roof_dome` | binary | Dome vs. open-air, one-hot encoded. Controlled conditions favor passing offenses. |

### Group 2 — Team form (rolling 5-game averages, computed per team-week)

For **both** home and away team, the difference or the raw rolling values of:

| Feature | Why it's in |
|---|---|
| `pass_yards_roll5` | Recent passing production — core measure of offensive strength. |
| `rush_yards_roll5` | Recent rushing production — complements passing, signals ball control. |
| `turnovers_roll5` | Giveaways. Turnover margin is one of the most outcome-correlated stats in football. |
| `points_scored_roll5` | Direct measure of recent offensive output. |
| `points_allowed_roll5` | Direct measure of recent defensive strength. |

We will model these as **home-minus-away differentials** (e.g.
`pass_yards_diff = home_pass_yards_roll5 - away_pass_yards_roll5`) where it makes
sense, because the model only needs the relative edge between the two teams, and
differentials reduce the feature count while keeping all the signal. This is a
deliberate choice we can explain.

### Features explicitly EXCLUDED (leakage)

`home_score`, `away_score`, `result`, `total`, `overtime` — all only known
*after* the game. Including any of them would be data leakage: the model would
look near-perfect in testing and be useless in production. This exclusion is a
talking point, not an oversight.

### Features excluded for simplicity (YAGNI)

`surface`, `referee`, raw QB/coach names, moneylines. They add encoding
complexity for marginal signal beyond what `spread_line` already captures. We
note them as possible future work rather than building them now.

### Missing-value handling

- `temp` / `wind` for dome games → imputed with a neutral indoor baseline
  (e.g. temp = 70°F, wind = 0), with a note that domes are climate-controlled.
- Early-season games with fewer than 5 prior games → use the average of
  available games (expanding window) rather than dropping the row, so we keep
  data. This choice is documented.

---

## Models

Three models, trained in increasing order of sophistication. The point of
training all three is to show a baseline → improvement narrative.

1. **Logistic Regression** — the sanity-check baseline. A linear model: if a
   well-tuned XGBoost can't beat a straight line through the features, something
   is wrong with the features or the split. Needs feature scaling (a `scaler.pkl`
   is saved for the app).
2. **Random Forest** — bagged decision trees, majority vote. Captures
   non-linear interactions, robust to outliers, gives feature importance for
   free.
3. **XGBoost** — gradient-boosted trees built sequentially, each correcting the
   prior trees' errors. Typically the strongest on tabular data; this is the
   model the app uses.

### Two model variants (the spread experiment)

Each model is trained **twice** — once excluding `spread_line` and once
including it — to separate "did my own feature engineering work?" from "did the
model just copy the betting market?"

- **Variant A — "no-spread" (the honest model).** Uses only our engineered
  features (rolling team form, rest, weather, divisional, week). This is the
  true measure of whether *our* features have predictive power. **This is the
  model the app uses by default**, so the demo reflects our own work, not Vegas's.
- **Variant B — "with-spread" (the benchmark).** Adds `spread_line`. Shows how
  much the market's prediction boosts accuracy and how close Variant A gets to
  the market on its own.

The reasoning, and why this matters, is captured in `features_explained.md`:
including the spread is **not data leakage** (the line is set before kickoff, so
it's available at prediction time), but it *is* a crutch — the spread is itself
the output of a prediction market that already absorbed every stat we use, so a
model leaning on it is partly copying rather than predicting. NFL betting markets
alone pick winners ~66–67% of the time, so a with-spread model near that number
may have learned little of its own. We therefore treat Variant A as the real
accomplishment and Variant B as a market benchmark.

**Split:** strictly chronological — train on 2019–2023, test on 2024–2025.
Never shuffle. This prevents temporal leakage.

**Evaluation:** accuracy, ROC-AUC, and a confusion matrix for each model and
each variant, all compared against the **53.5% naive baseline** (always pick the
home team) and against the **~66% market benchmark**. Targets: Variant A
~60–63% (proves our features work); Variant B ~64–67% (approaches the market).

**Interpretability:** SHAP values on the final XGBoost model. A SHAP summary plot
ranks features by overall impact; per-prediction SHAP values power the "top 3
factors" shown in the UI.

---

## Interview-prep documentation

Three plain-English markdown files in `docs/`, each ending with sample
interview Q&As:

- **`models_explained.md`** — how Logistic Regression, Random Forest, and
  XGBoost each work (in plain language), why we use all three, the
  strengths/weaknesses of each, and why XGBoost is the production model.
- **`features_explained.md`** — every feature and why it was chosen, what data
  leakage is and exactly how we prevented it, why rolling 5-game averages beat
  raw or season-long stats, why we used differentials, how we handled missing
  values, and the full **spread-line argument** (leakage vs. crutch, the
  two-variant experiment, and how to answer "isn't using the Vegas line
  cheating?" in an interview).
- **`ml_concepts.md`** — supervised learning, the target variable,
  train/test split and why chronological, the naive baseline, accuracy vs.
  ROC-AUC, overfitting/underfitting, feature scaling, and SHAP — each in plain
  English with a sample interview question and a strong answer.

These docs are a first-class deliverable, not an afterthought. They are written
so Jamie can read them and confidently explain every decision unprompted.

---

## The web app

**Backend (`app/main.py`)** — FastAPI with two routes:
- `GET /` → renders `index.html` with the list of teams for the dropdowns.
- `POST /predict` → takes `home_team` + `away_team`, builds the feature row from
  the latest available rolling stats, calls `predict.py`, returns JSON:
  win probability, both teams' stat cards, top-3 players each, and top-3 SHAP
  factors.

**`src/predict.py`** — loads `xgboost_model.pkl` + `scaler.pkl` +
`feature_list.json`, assembles the feature vector for a given matchup from the
processed data, returns the probability and the SHAP contributions. Imported by
both the app and any notebook.

**Frontend** — one server-rendered page:
- Two dropdowns (home, away) + Predict button.
- A win-probability bar with both team names and percentages.
- Two stat cards side by side (rolling pass yds, rush yds, turnovers, points
  for/against), in each team's colors.
- Top-3 player list per team (name, position, key stat) from PFR data, labeled
  honestly as recent PFR stats — **not** "PFF ranked."
- A short "What drove this prediction" list from SHAP (e.g. "Vegas line +11%,
  home rest +6%, away passing form −5%").

**Design direction:** clean two-column layout, real team colors, restrained use
of shadows/gradients, a readable system or sports-style font. It should look
like a focused analytics tool that someone put deliberate effort into — and
specifically should not read as a generic AI-generated UI.

---

## Out of scope (YAGNI)

- Live/auto-updating data feeds — the app reads from the processed CSV.
- User accounts, databases, deployment/hosting.
- Predicting exact scores or point totals (classification only).
- Playoff games (regular season only, `game_type == 'REG'`).
- Player-level modeling — players appear in the UI for context, not as model
  inputs.

---

## Success criteria

1. The no-spread XGBoost (Variant A) beats the 53.5% baseline (target ~60–63%),
   proving our engineered features have real predictive power; the with-spread
   variant (B) is also reported as a market benchmark (~64–67%).
2. No data leakage: chronological split, rolling features exclude the current
   game, post-game columns excluded.
3. The app runs locally and returns a sensible prediction + stats + SHAP factors
   for any valid matchup.
4. The three `docs/*.md` files let Jamie explain every modeling decision
   confidently in an interview.
