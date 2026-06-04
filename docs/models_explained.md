# Models Explained — for the Interview

How each model works in plain English, why we chose it, its strengths and
weaknesses, and our actual results. Read this and you can confidently defend
every modeling choice.

---

## The results table (test set: 2024–2025)

| Model | Variant | Accuracy | ROC-AUC |
|---|---|---|---|
| Logistic Regression | no-spread | **0.636** | 0.683 |
| Random Forest | no-spread | 0.608 | 0.659 |
| XGBoost | no-spread | 0.607 | 0.655 |
| Logistic Regression | with-spread | **0.675** | 0.726 |
| Random Forest | with-spread | 0.658 | 0.725 |
| XGBoost | with-spread | 0.651 | 0.706 |

- **Naive baseline** (always pick home): 0.535
- **Market benchmark** (Vegas alone): ~0.66

**Every model beats the baseline.** The best no-spread model (Logistic
Regression, 0.636) is the headline: it shows our *own* engineered features have
real predictive power, with no betting data.

---

## 1. Logistic Regression — the baseline

**How it works (plain English):** It draws the best straight-line boundary
through the feature space that separates home wins from home losses. For each
feature it learns a weight (positive pushes toward a home win, negative away),
adds them up, and squashes the total through an S-curve (the logistic/sigmoid
function) to produce a probability between 0 and 1.

**Why we use it:** It's the **sanity check**. It's simple, fast, and
interpretable. If a sophisticated model can't beat a straight line, something is
wrong with your features or your data split — so you always want this number to
compare against.

**Strengths:** Simple, fast, naturally outputs calibrated probabilities, hard to
overfit, coefficients are readable.
**Weaknesses:** Assumes a roughly linear relationship; can't capture complex
interactions on its own.

**Requires feature scaling** — it weighs features by magnitude, so we standardize
inputs first (see `ml_concepts.md` §10).

**Our result:** It *won*. On a ~1,300-game dataset with mostly linear signal, the
simplest model generalized best (0.636 no-spread). Great story: complexity isn't
automatically better.

> **Q: "Why start with logistic regression if XGBoost is usually better?"**
> "It's my sanity check and my floor. If a boosted model can't beat a straight
> line through the same features, the problem is my features or my split, not the
> algorithm. On this dataset the logistic regression actually generalized best,
> which validated keeping a simple baseline."

---

## 2. Random Forest — bagged trees

**How it works (plain English):** A single decision tree asks a series of
yes/no questions ("is the rushing-yards edge > 15?") and follows the branches to
a prediction. One tree overfits badly. A **random forest** builds *hundreds* of
trees, each on a random subset of the data and a random subset of features, then
takes a **majority vote**. The randomness de-correlates the trees so their errors
cancel out — this is called **bagging** (bootstrap aggregating).

**Why we use it:** It captures **non-linear** relationships and interactions a
straight line can't, it's robust to outliers, and it gives **feature importance**
essentially for free.

**Strengths:** Handles non-linearity, robust, little tuning needed, resistant to
overfitting compared to a single tree.
**Weaknesses:** Larger/slower, less interpretable than one tree, can still overfit
small noisy datasets.

**Our result:** 0.608 no-spread — beat the baseline but slightly *below* the
logistic regression, a sign of mild overfitting on our small dataset.

> **Q: "How does a random forest avoid the overfitting of a single tree?"**
> "It trains many trees, each on a random sample of rows and features, and
> averages their votes. Because the trees are de-correlated, their individual
> errors tend to cancel, so the ensemble generalizes far better than any single
> deep tree."

---

## 3. XGBoost — gradient boosting (the production model)

**How it works (plain English):** Also a forest of trees, but built very
differently. Instead of many independent trees voting, XGBoost builds trees
**sequentially**, and each new tree focuses on correcting the **mistakes** the
previous trees made. It's like a study group where each member specializes in the
questions the group keeps getting wrong. This is called **boosting**, and
"gradient" refers to using the error gradient to decide what to fix next.

**Why we use it:** It's typically the **strongest performer on tabular data**,
and its tree structure lets **SHAP** compute exact, fast explanations — which we
need for the app's "what drove this prediction" panel. So even though it didn't
top the accuracy table here, it's the model the app loads.

**Strengths:** Usually best-in-class on tabular data, captures complex patterns,
strong with proper tuning, integrates cleanly with SHAP.
**Weaknesses:** More hyperparameters to tune, can overfit small datasets if not
regularized, less interpretable without tools like SHAP.

**Key settings we used:** `n_estimators=300` (number of trees), `max_depth=4`
(shallow trees to limit overfitting), `learning_rate=0.05` (small steps so each
tree only nudges the prediction).

**Our result:** 0.607 no-spread. Competitive, beats the baseline, and its SHAP
support made it the right choice for the explainable app even though logistic
regression edged it on raw accuracy.

> **Q: "Why is the app using XGBoost if logistic regression scored higher?"**
> "Two reasons. First, the accuracy gap is small and XGBoost gives well-calibrated
> probabilities. Second, XGBoost's tree structure lets SHAP produce exact,
> instant per-prediction explanations, which is what powers the app's
> interpretability panel. I documented the logistic-regression win honestly, but
> for an explainable product XGBoost is the better engine."

> **Q: "What's the difference between bagging and boosting?"**
> "Bagging — like random forest — builds many independent trees in parallel and
> averages them to reduce variance. Boosting — like XGBoost — builds trees
> sequentially, each one correcting the previous trees' errors, to reduce bias.
> Bagging fights overfitting; boosting chases accuracy."

---

## How they relate

```
Logistic Regression   →  straight line, the floor / sanity check
Random Forest         →  many independent trees, majority vote (bagging)
XGBoost               →  sequential trees correcting errors (boosting)
```

The progression is the whole point: a baseline to measure against, then two ways
of adding non-linear power. Training all three — rather than just picking one —
is what let us discover that the simple model generalized best, which is a more
honest and interesting result than "I used XGBoost and got 65%."
