# NFL Win Predictor — Resume Summary

A short, copy-paste-ready summary of the project for your resume, LinkedIn, and
portfolio. Pick the version that fits the space you have.

---

## One-line version (LinkedIn headline / tight resume)

> Built an end-to-end NFL game-outcome predictor (Python, scikit-learn, XGBoost,
> SHAP, FastAPI) that beats the 53.5% home-field baseline by ~10 points, with a
> web app for live matchup predictions.

---

## Resume bullet points (pick 3–4)

- Built an **end-to-end machine-learning pipeline** in Python to predict NFL
  regular-season winners from 1,839 games (2019–2025), pulling and merging
  schedule and team-stat data via the nflverse API.
- Engineered leakage-free features — **trailing 5-game form differentials**, rest,
  weather, and venue — enforcing a strict chronological train/test split and a
  unit-tested rolling-average shift to prevent data leakage.
- Trained and compared **Logistic Regression, Random Forest, and XGBoost**,
  reaching **63.6% accuracy / 0.68 ROC-AUC** versus a 53.5% naive baseline.
- Designed a **two-variant experiment** (with/without the Vegas spread) to
  separate genuine model skill from echoing the betting market, demonstrating
  awareness of feature provenance and "predicting vs. copying."
- Made the model interpretable with **SHAP**, surfacing the top per-game drivers,
  and shipped a **FastAPI + JavaScript web app** for interactive predictions with
  team stat cards and top-player breakdowns.

---

## Short paragraph (portfolio / "Projects" section)

> **NFL Win Predictor** — An end-to-end ML project that predicts NFL game winners
> from pre-game data. I pulled seven seasons of games via the nflverse API,
> engineered leakage-free rolling-form features, and trained Logistic Regression,
> Random Forest, and XGBoost models with a strict chronological split, reaching
> ~64% accuracy against a 53.5% baseline. I used SHAP for interpretability and
> built a FastAPI web app where you pick two teams and get a win probability, team
> stat comparison, and the factors driving the prediction. A deliberate
> with/without-Vegas-spread experiment distinguishes real predictive signal from
> simply mirroring the betting market.
> *Tech: Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, JavaScript.*

---

## Skills demonstrated (for the skills section / talking points)

- **ML:** supervised classification, model comparison, train/test methodology,
  overfitting diagnosis, ROC-AUC, model interpretability (SHAP)
- **Data engineering:** API ingestion, joins/merges, feature engineering,
  leakage prevention, rolling-window features
- **Software:** Python packaging, unit testing (pytest), REST API (FastAPI),
  front-end (HTML/CSS/JS), Git version control
- **Judgment:** baseline-driven evaluation, feature-provenance awareness, YAGNI
  scoping

---

### Notes for honesty (important)

- Quote **63.6%** as the headline accuracy — that's the **no-spread** model, your
  own work. If you mention the ~67.5% number, clarify it's the *with-spread*
  benchmark, because an interviewer who knows sports modeling will ask.
- "~10 points over baseline" refers to the with-spread model (67.5 − 53.5). For
  the no-spread model it's ~10 points too (63.6 − 53.5) — either way the claim
  holds, but be ready to say which model you mean.
