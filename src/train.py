"""Train and evaluate three models in two feature variants.

Models (increasing sophistication, so we can tell a baseline -> improvement story):
  1. Logistic Regression - linear sanity check (needs scaled features).
  2. Random Forest        - bagged trees, non-linear, free feature importance.
  3. XGBoost              - boosted trees, usually strongest on tabular data.

Variants:
  - nospread : only our engineered features (the HONEST model the app uses).
  - spread   : adds the Vegas line (a market BENCHMARK; see features_explained.md).

Split is chronological (2019-2023 train, 2024-2025 test) -> no temporal leakage.
"""

import json
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

CONTEXT = ["home_rest", "away_rest", "div_game", "week", "temp", "wind", "roof_dome"]
DIFFS = ["pass_yards_roll5_diff", "rush_yards_roll5_diff", "turnovers_roll5_diff",
         "points_scored_roll5_diff", "points_allowed_roll5_diff"]
FEATURES_NOSPREAD = CONTEXT + DIFFS
FEATURES_SPREAD = FEATURES_NOSPREAD + ["spread_line"]

BASELINE = 0.535   # always pick the home team
MARKET = 0.66      # rough NFL betting-market accuracy, for context


def _evaluate(name, model, Xte, yte):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    cm = confusion_matrix(yte, pred).tolist()
    print(f"{name:28s} acc={acc:.3f}  auc={auc:.3f}  (baseline {BASELINE})")
    return {"name": name, "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4), "confusion_matrix": cm}


def train_variant(train, test, features, tag):
    Xtr, ytr = train[features], train["home_win"]
    Xte, yte = test[features], test["home_win"]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), columns=features)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=features)

    metrics = []
    lr = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
    metrics.append(_evaluate(f"LogReg [{tag}]", lr, Xte_s, yte))

    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
    metrics.append(_evaluate(f"RandomForest [{tag}]", rf, Xte, yte))

    xgbc = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             eval_metric="logloss", random_state=42).fit(Xtr, ytr)
    metrics.append(_evaluate(f"XGBoost [{tag}]", xgbc, Xte, yte))

    pickle.dump(xgbc, open(f"models/xgboost_{tag}.pkl", "wb"))
    pickle.dump(scaler, open(f"models/scaler_{tag}.pkl", "wb"))
    json.dump(features, open(f"models/feature_list_{tag}.json", "w"))
    return metrics


def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    print(f"\n=== nospread (honest model) — {len(FEATURES_NOSPREAD)} features ===")
    m1 = train_variant(train, test, FEATURES_NOSPREAD, "nospread")
    print(f"\n=== spread (market benchmark) — {len(FEATURES_SPREAD)} features ===")
    m2 = train_variant(train, test, FEATURES_SPREAD, "spread")

    all_metrics = {"baseline": BASELINE, "market_benchmark": MARKET,
                   "results": m1 + m2}
    json.dump(all_metrics, open("models/metrics.json", "w"), indent=2)
    print("\nsaved models/metrics.json")


if __name__ == "__main__":
    main()
