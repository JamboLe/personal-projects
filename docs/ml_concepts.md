# ML Concepts — Explained for the Interview

Plain-English explanations of every machine-learning concept this project
touches. Each section ends with a likely interview question and a strong answer
you can say out loud.

---

## 1. Supervised learning

**What it is:** Learning from labeled examples. We have thousands of past games
where we already know who won. We show the model the pre-game information (the
**features**) paired with the known outcome (the **label**), and it learns the
relationship. Then we apply that learned relationship to new games.

The two ingredients that make something "supervised": you have **inputs** and you
have **known correct answers** to learn from. (Unsupervised learning has inputs
but no answers — e.g. clustering customers into groups nobody labeled.)

> **Q: "Why is this a supervised learning problem?"**
> "Because every training example is labeled. Each past game has a known outcome,
> so the model can learn the relationship between pre-game stats and who won, then
> apply it to future games. If we had no outcomes to learn from, it would be
> unsupervised."

---

## 2. Classification vs. regression

**Classification** predicts a category. **Regression** predicts a number. We
predict *who wins* (home or away) — two categories — so this is **binary
classification**. If we'd predicted the exact final score, that would be
regression.

We chose classification because the winner is what the app needs, and because a
classifier naturally outputs a **probability** (e.g. 67% home win), which is more
useful and honest than a single guessed score.

> **Q: "Why classification instead of predicting the score?"**
> "The product needs the winner and a confidence level, not an exact score.
> Classification gives me a calibrated probability directly, and it's an easier,
> better-posed problem than regressing a noisy final score."

---

## 3. The target variable

The **target** (or label) is what the model predicts. Ours is `home_win`: `1` if
the home team won, `0` if it lost. Every supervised problem needs a clearly
defined target — it's the thing you measure everything else against.

We built it with one line: `(home_score > away_score)` converted to 1/0.

> **Q: "What's your target variable and why?"**
> "`home_win`, a binary 0/1. I framed it from the home team's perspective so the
> model output maps cleanly to 'home win probability,' which is exactly what the
> UI shows."

---

## 4. Features

**Features** are the inputs — the columns the model uses to make its prediction.
Ours fall into two groups:

- **Pre-game context:** rest days, divisional game, week, temperature, wind,
  dome, (optionally) the Vegas spread.
- **Recent team form:** each team's trailing 5-game averages for passing yards,
  rushing yards, turnovers, points scored, and points allowed — expressed as
  **home-minus-away differences**.

The golden rule for every feature: *would I have known this before kickoff?* If
not, it can't be a feature (see leakage below).

---

## 5. Train/test split — and why we split by TIME

You never evaluate a model on the same data it learned from — it would just
memorize. So we hold out a **test set** the model never sees during training.

The crucial twist for time-based data: we split **chronologically**. Train on
2019–2023, test on 2024–2025. We never shuffle. This mimics reality — you always
predict the future from the past, never the reverse.

- Train: 1,295 games (2019–2023)
- Test: 544 games (2024–2025)

> **Q: "How did you split your data?"**
> "Chronologically — trained on 2019 through 2023 and tested on 2024–2025, with no
> shuffling. For time-series-like data, a random split would let future games leak
> into training and inflate the score. Splitting by time tests the model the way
> it'll actually be used."

---

## 6. Data leakage

**Leakage** is when information that wouldn't be available at prediction time
sneaks into training. The model looks brilliant in testing and then fails in the
real world. It's the single most common way to fool yourself in ML.

Three ways we prevented it:
1. **Excluded post-game columns** — `home_score`, `away_score`, `result`,
   `total`, `overtime` are never features.
2. **Chronological split** — no future seasons in training.
3. **`.shift(1)` on rolling averages** — each game's "recent form" uses only
   games played *before* it, never the current game itself.

> **Q: "What is data leakage and how did you prevent it?"**
> "Leakage is future or outcome information bleeding into training, which makes a
> model look more accurate than it really is. I prevented it three ways: I dropped
> all post-game columns, I split by time instead of shuffling, and my rolling
> averages are shifted by one game so a team's form only ever reflects prior
> games."

---

## 7. The naive baseline

Before trusting any model, you need to know what "no skill" looks like. In the
NFL the home team wins about **53.5%** of the time. So always guessing "home"
scores 53.5%. **Any model that can't beat 53.5% is worthless.** The baseline is
the bar.

> **Q: "How did you know your model was actually good?"**
> "I set a naive baseline first — the home team wins 53.5% of the time, so that's
> the floor. My models all cleared it; the best hit about 64% without any betting
> data, which is a real improvement over just picking the home team."

---

## 8. Accuracy vs. ROC-AUC

- **Accuracy:** the fraction of games you call correctly. Intuitive, but it only
  looks at the final yes/no, not the confidence.
- **ROC-AUC:** measures how well the model *ranks* games by risk — the
  probability that a randomly chosen actual-home-win is scored higher than a
  randomly chosen home-loss. 0.5 is coin-flip, 1.0 is perfect. It rewards
  well-ordered probabilities, not just the threshold call.

We report both. Our best model: ~0.64 accuracy, ~0.68 AUC (no-spread).

> **Q: "Why look at AUC and not just accuracy?"**
> "Accuracy only judges the final 50% cutoff. AUC judges the whole probability
> ranking, so it tells me whether the model's confidence is meaningful. A model
> can have decent accuracy but poorly calibrated probabilities — AUC catches that."

---

## 9. Overfitting and underfitting

- **Overfitting:** the model memorizes training noise and fails on new data —
  great train score, poor test score. Complex models (deep trees, big forests)
  are prone to it on small datasets.
- **Underfitting:** the model is too simple to capture the real pattern — poor on
  both train and test.

We saw this directly: our **Random Forest and XGBoost slightly overfit** the
~1,300-game training set, while the simpler **Logistic Regression generalized
best**. That's a textbook reminder that more complexity isn't automatically
better.

> **Q: "Did you run into overfitting?"**
> "Yes, mildly. My tree ensembles scored higher on training but a bit lower on the
> held-out seasons than my logistic regression. With only ~1,300 training games,
> the simpler linear model actually generalized best — which is exactly why I
> always train a simple baseline alongside the complex models."

---

## 10. Feature scaling

Some models care about the *scale* of features. Logistic Regression does:
passing yards (hundreds) and turnovers (0–3) live on wildly different scales, and
without scaling the big-numbered feature dominates. We use `StandardScaler` to
put every feature on a comparable scale (mean 0, standard deviation 1) for the
logistic model. Tree models (Random Forest, XGBoost) don't need scaling because
they split on thresholds, not distances — but scaling doesn't hurt them.

> **Q: "Why scale features for logistic regression but not the trees?"**
> "Logistic regression weighs features by magnitude, so unscaled features on
> different ranges bias it. Standardizing fixes that. Trees split on thresholds
> rather than absolute magnitudes, so they're scale-invariant and don't need it."

---

## 11. SHAP — explaining predictions

A model that can't explain itself is hard to trust. **SHAP** (SHapley Additive
exPlanations) breaks any single prediction into the contribution of each feature:
"the rushing-yards edge pushed this +8% toward the home team, the turnover edge
pulled it −5%," and so on. Averaged across all games, SHAP also **ranks features
by overall importance**.

In our model the top drivers were the recent **scoring-defense edge**, then the
**rushing** and **passing** yardage edges, then **turnover margin** — all
features we engineered ourselves.

> **Q: "How did you make the model interpretable?"**
> "I used SHAP values on the XGBoost model. For any single game it tells me how
> much each feature pushed the prediction toward home or away, and the summary plot
> ranks features by average impact. That's how I confirmed my engineered
> form-differentials, not noise, were driving the predictions — and it powers the
> 'what drove this' panel in the app."

---

## 12. Why three models?

We trained Logistic Regression, Random Forest, and XGBoost on purpose:

- **Logistic Regression** is the sanity check. If a complex model can't beat a
  straight line, the features or the data are the problem, not the algorithm.
- **Random Forest** captures non-linear interactions and gives feature
  importance.
- **XGBoost** is usually the strongest on tabular data and is the model the app
  uses (its tree structure also makes SHAP fast and exact).

> **Q: "Why train three models instead of just the best one?"**
> "To get a baseline-to-improvement story and to avoid fooling myself. The
> logistic regression sets the floor; the ensembles test whether added complexity
> actually helps. On this dataset it barely did — which is itself a useful,
> honest finding."
