# Features Explained — for the Interview

Every feature, why it's there, how we prevented leakage, and the full argument
about the Vegas spread. This is the doc to read before any question about *data*.

---

## The feature set

### Group 1 — Pre-game context (from the schedule)

| Feature | Why it's in |
|---|---|
| `home_rest` / `away_rest` | Days of rest since each team's last game. Short weeks (Thursday games) and bye-week rest measurably affect performance. |
| `div_game` | Divisional matchups are closer and less predictable — familiar opponents. |
| `week` | Season progression; early-season noise vs. late-season form and stakes. |
| `temp` / `wind` | Cold and especially wind suppress the passing game. |
| `roof_dome` | Dome/closed roof = controlled conditions that favor passing offenses. |
| `spread_line` | The Vegas line. **Only in the with-spread variant** — see the big section below. |

### Group 2 — Recent team form (rolling 5-game, as home-minus-away differences)

| Feature | Why it's in |
|---|---|
| `pass_yards_roll5_diff` | Recent passing production edge — core offensive strength. |
| `rush_yards_roll5_diff` | Recent rushing edge — ball control, complements passing. |
| `turnovers_roll5_diff` | Turnover-margin edge — one of the most outcome-correlated stats in football. |
| `points_scored_roll5_diff` | Recent scoring-offense edge. |
| `points_allowed_roll5_diff` | Recent scoring-defense edge (the model's #1 driver). |

---

## Why rolling 5-game averages (not raw or season-long stats)

A single game is noisy — a blowout or a fluke can mislead. A **season-long
average** is too slow — it can't tell that a team caught fire in November. A
**trailing 5-game average** is the sweet spot: it captures *current form* while
smoothing out one-game noise.

> **Q: "Why rolling averages instead of season averages?"**
> "Season averages react too slowly to capture a team's current form, and single
> games are too noisy. A trailing 5-game window balances both — it reflects how a
> team is playing right now while smoothing out one-game flukes."

---

## Why home-minus-away differentials

For each rolling stat we have a home value and an away value. Instead of feeding
both, we feed the **difference** (`home − away`). The model only needs the
*relative edge* between the two teams, so the difference carries the same signal
with half the columns. Fewer features means less overfitting and a simpler model
to explain.

> **Q: "Why did you use differences instead of both teams' raw stats?"**
> "The outcome depends on the matchup — the relative edge — not the absolute
> numbers. Collapsing each home/away pair into a single difference keeps all the
> signal, halves the feature count, and reduces overfitting risk. It's a cleaner
> representation of the actual question."

---

## How we prevented data leakage (the most important part)

Leakage is the cardinal sin: future or outcome information sneaking into
training, making the model look great in testing and fail in reality. Three
defenses:

1. **Excluded all post-game columns.** `home_score`, `away_score`, `result`,
   `total`, `overtime` are *never* features — they're only known after the game.
   We verified the final feature table contains none of them.
2. **Chronological split.** Train 2019–2023, test 2024–2025, never shuffled — so
   no future season informs training.
3. **`.shift(1)` on every rolling average.** Each game's "recent form" is built
   from games played *strictly before* it. Without the shift, a team's 5-game
   average would include the very game we're predicting — classic leakage. We
   wrote a unit test (`test_rolling_excludes_current_game`) that fails if the
   current game ever sneaks into its own rolling window.

> **Q: "Walk me through how you avoided data leakage."**
> "Three things. I dropped every column only known after kickoff. I split by time
> instead of shuffling so the model never trains on the future. And my rolling
> averages are shifted by one game, so a team's form only ever reflects prior
> games — I even have a unit test that fails if the current game leaks into its
> own average."

---

## Missing-value handling

- **Dome games have no temp/wind.** We impute `temp = 70°F` and `wind = 0`,
  reflecting that domes are climate-controlled, and add a `roof_dome` flag so the
  model knows it's indoors.
- **Early-season games have fewer than 5 prior games.** Instead of dropping them,
  the rolling average uses however many games exist so far (an expanding window,
  `min_periods=1`). The very first games of 2019, which have *zero* prior history,
  are dropped — that's the only data we lose (1,960 → 1,839 games).

> **Q: "How did you handle missing values?"**
> "Context-dependent. Dome games legitimately have no weather, so I imputed a
> neutral indoor baseline and flagged them as domes. For early-season games with
> fewer than five prior games, I used an expanding window rather than dropping
> data, and only dropped the season-opening games that had no history at all."

---

## The Vegas spread: is using it cheating?

This is the most interesting question in the project, and the answer shows
maturity. Short version: **it's not leakage, but it is a crutch — so we built the
model both ways.**

### Is it data leakage? No.

Leakage means using information you wouldn't have at prediction time. The spread
is set *days before* kickoff and is frozen when the game starts. In real life
it's sitting right there before the game — so using it doesn't break the "would I
know this before kickoff?" rule. It is **not** leakage.

### Is it a crutch? Yes — and that's the real critique.

The spread is itself the **output of a prediction market**. Thousands of sharp
bettors and oddsmakers already digested every stat we use (and more) to produce
it. So handing the spread to our model is letting it **copy a finished answer**
rather than reason from raw inputs. A model leaning on the spread can look good
while having learned almost nothing of its own. NFL betting markets alone pick
winners about **66%** of the time — so a with-spread model near that number may
just be echoing Vegas.

### What we did about it: two variants

- **No-spread (the honest model):** only our engineered features. This is the
  true measure of *our* work. Best result: **0.636** (Logistic Regression).
- **With-spread (the benchmark):** adds the spread. Best result: **0.675** — about
  4 points higher, pulling us toward the ~0.66 market line.

The ~4-point lift is real, but we attribute it to the model absorbing the
market's prediction, not to new skill of our own. That's why **the app uses the
no-spread model** and we report the with-spread number as a benchmark.

> **Q: "Isn't using the Vegas line basically cheating?"**
> "It's a fair question. It's not *leakage* — the line is set before kickoff, so
> it's genuinely available at prediction time. But it is a *crutch*: the spread is
> the output of a betting market that already absorbed every stat I'm using, so a
> model leaning on it is partly copying rather than predicting. That's why I
> trained two versions. Without the spread my model hit about 64% on my own
> features, which is the result I'm actually proud of. Adding the spread pushed it
> to about 68% — but I treat that as a market benchmark, not as my model getting
> smarter. The app uses the no-spread model on purpose."

That answer demonstrates you understand the difference between **predicting** and
**copying** — a level most candidates never reach.

---

## Features we deliberately left out (YAGNI)

`surface`, `referee`, raw QB/coach names, and moneylines. They add encoding
complexity for marginal signal beyond what we already capture. Noting them as
*possible future work* — rather than cramming them in — is itself good practice
to mention.

> **Q: "What would you add with more time?"**
> "Quarterback-level adjustments — converting QB names into a rolling passer
> rating or EPA — and injury data for key starters. I left them out to keep the
> first version focused and leakage-free, but they're the natural next features."
