"""Generate real ML visualizations from the trained models for the web app.

Outputs (styled dark to match the app's 'how it works' panel):
  app/static/viz/roc_curves.png       - ROC curves for all 3 models (no-spread)
  app/static/viz/confusion_matrix.png - XGBoost confusion matrix on the test set
  app/static/viz/shap_beeswarm.png    - SHAP beeswarm summary for XGBoost

These are genuine model outputs, not hand-drawn charts.
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             confusion_matrix, roc_auc_score, roc_curve)

OUT = "app/static/viz"

# dark theme so plots blend into the panel
DARK_BG = "#0f1923"
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.edgecolor": "#334155",
    "font.size": 12,
})


def roc_curves():
    test = pd.read_csv("data/processed/test.csv")
    y = test["home_win"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"Logistic Regression": "#6366f1",
              "Random Forest": "#8b5cf6",
              "XGBoost": "#1f6feb"}

    # load saved xgboost + retrain lr/rf inline for clean probability curves
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train = pd.read_csv("data/processed/train.csv")
    feats = json.load(open("models/feature_list_nospread.json"))
    Xtr, ytr = train[feats], train["home_win"]
    Xte = test[feats]

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
    xgbc = pickle.load(open("models/xgboost_nospread.pkl", "rb"))

    probs = {
        "Logistic Regression": lr.predict_proba(scaler.transform(Xte))[:, 1],
        "Random Forest": rf.predict_proba(Xte)[:, 1],
        "XGBoost": xgbc.predict_proba(Xte)[:, 1],
    }
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=colors[name], lw=2.2, label=f"{name} (AUC {auc:.2f})")

    ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1.3, label="Random guess (AUC 0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — how well each model ranks games", color="#e2e8f0", pad=12)
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/roc_curves.png", dpi=130)
    plt.close(fig)
    print("saved roc_curves.png")


def confusion():
    test = pd.read_csv("data/processed/test.csv")
    feats = json.load(open("models/feature_list_nospread.json"))
    model = pickle.load(open("models/xgboost_nospread.pkl", "rb"))
    y = test["home_win"]
    pred = model.predict(test[feats])
    cm = confusion_matrix(y, pred)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Away win", "Home win"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 15})
    ax.set_title("XGBoost predictions vs. actual\n(2024–2025 test games)", color="#e2e8f0", pad=12)
    ax.set_xlabel("Predicted", color="#e2e8f0")
    ax.set_ylabel("Actual", color="#e2e8f0")
    # recolor tick labels
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color("#cbd5e1")
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=130)
    plt.close(fig)
    print("saved confusion_matrix.png")


def beeswarm():
    test = pd.read_csv("data/processed/test.csv")
    feats = json.load(open("models/feature_list_nospread.json"))
    model = pickle.load(open("models/xgboost_nospread.pkl", "rb"))

    # friendlier feature names
    rename = {
        "points_scored_roll5_diff": "Scoring offense edge",
        "points_allowed_roll5_diff": "Scoring defense edge",
        "rush_yards_roll5_diff": "Rushing yards edge",
        "pass_yards_roll5_diff": "Passing yards edge",
        "turnovers_roll5_diff": "Turnover edge",
        "home_rest": "Home rest", "away_rest": "Away rest",
        "div_game": "Divisional game", "week": "Week", "temp": "Temperature",
        "wind": "Wind", "roof_dome": "Dome",
    }
    X = test[feats].rename(columns=rename)
    explainer = shap.TreeExplainer(model)
    sv = explainer(test[feats])
    sv.feature_names = [rename.get(f, f) for f in feats]

    plt.figure(figsize=(7, 5))
    shap.plots.beeswarm(sv, show=False, color_bar=True)
    fig = plt.gcf()
    fig.patch.set_facecolor(DARK_BG)
    ax = plt.gca()
    ax.set_facecolor(DARK_BG)
    ax.set_title("SHAP — every dot is one game", color="#e2e8f0", pad=12)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color("#cbd5e1")
    ax.xaxis.label.set_color("#e2e8f0")
    fig.tight_layout()
    fig.savefig(f"{OUT}/shap_beeswarm.png", dpi=130, facecolor=DARK_BG)
    plt.close(fig)
    print("saved shap_beeswarm.png")


if __name__ == "__main__":
    roc_curves()
    confusion()
    beeswarm()
