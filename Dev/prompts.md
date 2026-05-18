# Prompt Log — Retail Sales / Walmart Revenue Analysis

This file logs the prompts sent to the LLM assistant during the analysis, along with notes on
what worked, where push-back was required, and where human judgment corrected the model's reasoning.

---

## Phase 1 Prompts — Data Audit & EDA

### Prompt 1 — Initial Task Setup

```
You are a Senior Data Scientist with 10+ years of experience in time-series
forecasting, retail analytics, macroeconomic analysis, and financial forecasting.
Your task is to perform a complete exploratory data analysis on the following
datasets: retail_sales_fred.csv and walmart_revenue.csv.

Business Question: "Can retail sales act as a leading indicator for Walmart
quarterly revenue?"

Begin with PHASE 1 — DATA AUDIT. Validate schemas, datatypes, missing values,
duplicate records, date continuity, frequency consistency, outliers, monotonicity,
unit consistency, structural breaks, and regime shifts. For every issue discovered,
explain its downstream impact, severity, possible handling strategies, and
recommended action.
```

**What worked:** Correctly identified unit mismatch (RSXFS in millions, Walmart in absolute USD),
flagged the Walmart fiscal calendar (Feb-Jan FY), and proposed fiscal-quarter-correct aggregation
over naive QE resampling.

**Where I had to push back:** The assistant initially computed a 220% YoY retail growth in 2011-Q1
without flagging it as suspicious. This would have propagated a spurious artifact into every
downstream correlation. Human judgment was required to investigate.

---

### Prompt 2 — Debugging the Spurious 220% YoY

```
The retail YoY for January 2011 shows 220% growth. Before accepting this or
excluding it silently, explain why this might be occurring, what the correct
handling is, and what the downstream impact would be if we retained it.
```

**What worked:** Correctly diagnosed that FY-Q4 2010 (ending Jan 31, 2010) contained only
January 2010 retail data (Nov/Dec 2009 outside dataset). The inflated YoY was a boundary artifact.

**Where the LLM failed first:** The initial code used `dropna()` which incidentally excluded the
corrupted row, making pre-COVID r appear as 0.855 — correct number, wrong reason. The fix must
be explicit (n_months == 3 filter), not accidental.

**Manual correction:** Rewrote quarterly aggregation to count n_months and explicitly drop
incomplete quarters before computing YoY.

---

### Prompt 3 — Cross-Correlation and Lead-Lag Analysis

```
Compute the cross-correlation function (CCF) between retail YoY and Walmart YoY.
Run it on the pre-COVID subsample only. Explain what the CCF pattern means for
the leading-indicator hypothesis. Be specific about whether the CCF supports a
clean lead from retail → Walmart or a symmetric co-movement pattern.
```

**What worked:** Correctly identified pre-COVID subsample requirement. CCF showed the
bi-directional relationship clearly.

**Where the LLM required clarification:** Initial full-sample CCF showed near-zero correlations
and was reported as "weak signal." This framing was wrong — COVID destroyed the signal. Required
reframing as "regime-sensitivity problem, not weak signal."

**Key finding:** CCF showed significant correlations at BOTH positive and negative lags —
Walmart also "leads" retail. This symmetric structure means both respond to a common macro
driver (GDP, employment). Initial LLM output glossed over this. Push-back required.

---

### Prompt 4 — Granger Causality Testing

```
Run a Granger causality test to formally test whether lagged retail YoY adds
predictive content for Walmart YoY beyond Walmart's own lags. Run on both the
full sample and the pre-COVID subsample. Discuss the caveats clearly: small n,
structural breaks, and the distinction between Granger causality and economic
causality.
```

**What worked:** Correctly ran on both subsets. Pre-COVID lag-1 significant (p=0.010);
full-sample not significant. Key contrast was surfaced.

**What needed correction:** Initial output included verbose Granger output inline.
Cleaned up using verbose=False and extracting only F-statistics and p-values.

---

### Prompt 5 — Rolling Correlation

```
Compute rolling 8-quarter Pearson correlations between retail YoY (contemporaneous
and lag+1) and Walmart YoY. Plot both on the same axis and annotate the COVID
regime. Interpret the chart in terms of signal stability.
```

**What worked:** Rolling correlation chart was the single most informative visualization.
Pre-COVID stable ~0.75, COVID inversion to −0.71 — visually unambiguous regime break.

**No push-back needed** on this prompt.

---

### Prompt 6 — EDA Summary

```
Synthesize all EDA findings into a structured summary covering: key observations,
predictive signals, structural risks, data limitations, useful forecasting
directions, approaches likely to fail, and evidence that would increase confidence.
Ground every statement in specific numbers from the data.
```

**What worked:** Well-structured summary. Key findings correctly cited with numbers.

**One overstatement corrected:** "retail sales is a leading indicator" → "retail sales
was a leading indicator pre-COVID; the relationship is currently unvalidated."

---

## Phase 2 Prompts — Feature Engineering

### Prompt 7 — Publication Lag Analysis

```
PHASE 2 — FEATURE ENGINEERING STRATEGY

You are a Senior Data Scientist with 10+ years of experience in:
feature engineering, time-series forecasting, econometrics,
macroeconomic modeling, retail analytics.

Before engineering any features, document explicitly:
which FRED monthly retail observations are available before Walmart reports each
of its four fiscal quarters? Account for FRED's monthly release schedule
(~14th of the following month) and Walmart's typical reporting calendar
(2-4 weeks after fiscal quarter end).
```

**What worked:** Correctly built full schedule. Identified that last month of every quarter
is available 3–5 days before WMT reports (borderline).

**What needed correction:** Initial framing said "current-quarter data is NOT available."
This is too conservative — advance estimate IS published before WMT reports.
Corrected to: available but carries ±0.5% revision risk.

---

### Prompt 8 — Transformation Selection

```
Evaluate all six candidate transforms for the retail series: level, log-level,
first difference, YoY%, QoQ%, log-YoY. For each: test stationarity (ADF + KPSS),
explain economic meaning, and specify which is appropriate for use in a regression
predicting Walmart YoY revenue growth.
```

**What worked:** Correctly identified I(1) issue for levels and log-levels.
Recommended YoY% as primary. QoQ correctly flagged as stationary but noisy.

**No push-back needed.**

---

### Prompt 9 — Aggregation Method

```
Compare three aggregation approaches for converting monthly retail to Walmart
fiscal quarters: sum, mean, and last-month. Show whether they produce different
YoY growth rates and which is more predictive.
```

**What worked:** Correctly derived that sum and mean are mathematically identical for
YoY ratios (constant 1/3 cancels). Last-month is noisier and slightly less predictive.

**LLM initially glossed over the algebraic proof that sum=mean.** Required explicit verification.

---

### Prompt 10 — Feature Group Engineering (12 Groups)

```
Engineer all 12 feature groups: YoY retail growth, QoQ retail growth,
aggregation, lagged features, rolling averages, rolling volatility,
seasonal adjustments, z-score normalization, revenue momentum,
interaction terms, regime indicators, and COVID indicator.
For each group: (1) document three approaches, (2) state tradeoffs,
(3) specify leakage status, (4) explain economic intuition,
(5) explain statistical intuition, (6) recommend approach.
```

**What worked:** All 12 groups constructed. The expanding-window z-score was correctly
identified as requiring fit-on-train-only.

**One failure caught:** The assistant included the rolling 8Q mean in the "recommended"
set because full-sample r=0.39. This is spurious — pre-COVID r=0.07. The full-sample
correlation is driven by shared nominal trend (both series trend upward 2010–2019),
not consumer demand. Required explicit pre/post-COVID comparison to expose.

**Key failure type:** LLM optimized for full-sample correlation without regime awareness.
Human oversight required at every feature-relevance step.

---

### Prompt 11 — Incremental R² Analysis

```
Compare three nested OLS models on the pre-COVID dataset:
A: AR(1) + seasonal dummies
B: A + retail_lag1
C: B + retail_lag2
Report R², adj-R², AIC, and run a partial F-test for the incremental
contribution of retail_lag1 controlling for the AR and seasonal baseline.
```

**What worked:** Partial F-test result (F≈6.8, p≈0.014) is the key quantitative evidence
that retail adds real incremental information. Correctly set up.

**Correction required:** Initial summary text stated Delta R² of ~0.16.
Actual model output is ~0.09. The 0.16 was from an earlier version with a different
data subset. Always verify summary statistics against actual model output.

---

### Prompt 12 — VIF Analysis

```
Run VIF analysis for the minimal, extended, and regime-aware feature sets.
Flag any VIF > 5. Explain what moderate collinearity (VIF ~3.5-4) means
for model stability and coefficient interpretation.
```

**What worked:** VIF values correctly computed. VIF ~3.6-4.0 for f_wmt_ar1 and
f_retail_lag1 — acceptable. Explanation of widened CIs was appropriate.

**No push-back needed.**

---

## Phase 3 Prompts — Baseline, Forecasting Models, Evaluation

### Prompt 13 — Walk-Forward Validation Framework

```
PHASE 3 — BASELINE STRATEGY

You are a Senior Data Scientist with 10+ years of experience in forecasting,
financial modeling, time-series evaluation, econometrics, retail demand forecasting.

Implement a strictly proper walk-forward (expanding window) cross-validation
framework. Explain why random k-fold is invalid for time series. Define all 6
evaluation metrics: MAE, RMSE, MAPE, sMAPE, Directional Accuracy, Forecast Bias.
For each metric, explain business interpretation, mathematical formula, and
when to prefer it over others.
```

**What worked:** Framework implemented correctly. Metrics correctly defined with formulas.
Expanding vs rolling window tradeoff correctly articulated.

**One subtlety flagged:** MAPE is unstable when actuals near zero (which can happen if
a quarter has near-0% YoY growth). sMAPE is more stable — both should be reported.

---

### Prompt 14 — Baseline Models

```
Implement at minimum three baseline models:
1. Seasonal naive (same quarter last year)
2. Historical growth mean (expanding)
3. Revenue autoregressive naive (prior observation)

For each: explain intuition, explain why it matters, explain strengths/weaknesses.
Explicitly state: any model that cannot beat ALL baselines provides no business value.
```

**What worked:** All three baselines implemented and compared correctly.

**Key diagnostic surfaced:** Seasonal naive is the hardest naive to beat because Walmart's
YoY growth is highly persistent quarter-over-quarter. The LLM correctly identified this.

---

### Prompt 15 — Linear and Regularised Models

```
Evaluate models incrementally:
1. OLS-AR: AR(1) + seasonal only (no retail signal) — benchmark
2. OLS-MIN: OLS + retail_lag1 — test the business hypothesis
3. OLS-EXT: OLS + extended feature set
4. Ridge Regression (L2) with GCV alpha selection
5. Lasso Regression (L1) with TimeSeriesSplit CV

For every model: explain why it is appropriate, its assumptions,
tradeoffs, interpretability, robustness. Show coefficient table for Ridge.
Show feature selection behavior for Lasso.
```

**What worked:** Model hierarchy correctly implemented. OLS-AR vs OLS-MIN comparison
is the cleanest test of the business hypothesis. Ridge coefficient table and Lasso
zeroing behavior both implemented.

**One issue with Lasso:** Initial implementation used standard 5-fold KFold for alpha
selection — INVALID for time series. Corrected to TimeSeriesSplit(n_splits=3).

**Human correction:** LLM's initial Lasso used `LassoCV(cv=5)` without specifying that
the CV must respect temporal ordering. Time-series-aware CV (TimeSeriesSplit) is required.

---

### Prompt 16 — Time Series Models (ARIMA + SARIMAX)

```
Implement ARIMA(1,0,1) on the stationary YoY series and SARIMAX(1,0,1)
with f_retail_lag1 as exogenous regressor.

Explain: why d=0 (already stationary), why no seasonal differencing
(YoY already removes seasonality), the exact meaning of ΔRMSE(ARIMA→SARIMAX)
as a walk-forward quantification of the retail signal's value, and how this
relates to the Granger test from Phase 1 EDA.
```

**What worked:** ARIMA and SARIMAX correctly implemented. ΔRMSE(ARIMA→SARIMAX) is the
cleanest OOS complement to the Granger test — the LLM correctly framed this.

**Technical issue caught:** In statsmodels 0.14.6, `res.forecast(steps=1)` returns
a numpy array (not pandas Series). `.iloc[0]` raises AttributeError. Fixed by using
`np.asarray(res.forecast(steps=1))[0]`.

**dtype issue fixed:** pandas 3.x nullable Float64 type conflicts with statsmodels'
`np.isfinite()` checks. Required explicit `.astype(float)` casts on all feature arrays
before passing to statsmodels.

---

### Prompt 17 — Tree-Based Models (Conditional)

```
Tree-based models should only be included if justified by the data.
Evaluate whether non-linear effects are present. If included, implement
Random Forest and XGBoost with conservative hyperparameters appropriate
for n=56 observations. Explicitly state why these are supplementary only.
```

**What worked:** LLM correctly argued for conditional inclusion. Depth limits and
regularisation constraints were appropriately set.

**Key framing enforced:** "If a simple OLS or Ridge achieves similar RMSE, the simpler
model wins on interpretability grounds (Occam's Razor applied to finance)."

---

### Prompt 18 — Robustness Checks

```
Run five robustness checks:
RC1: Pre-COVID OOS performance
RC2: COVID window performance
RC3: Post-COVID recovery performance
RC4: Lag sensitivity (what if we use lag-2 instead of lag-1?)
RC5: Training window sensitivity (INITIAL_TRAIN = 16, 20, 24 quarters)

For each: explain what it reveals about model stability.
Discuss structural breaks, instability, model degradation.
```

**What worked:** All five checks implemented. COVID/Pre-COVID ratio correctly
identified as the key stress-test metric.

**Key finding:** All models show RMSE_COVID / RMSE_pre ≈ 2–3x. This is the
most important robustness finding — no model is immune to structural breaks.

---

### Prompt 19 — Business Conclusion (7 Questions)

```
Answer directly and explicitly:
1. Does retail sales improve forecasting?
2. By how much?
3. Is the improvement meaningful?
4. Is it stable?
5. What are the caveats?
6. Would this be trusted in production?
7. What evidence would change our conclusion?

Explicitly distinguish: predictive usefulness vs causal claims.
The distinction matters for investment and operational decisions.
```

**What worked:** All 7 questions answered directly with quantitative grounding.
Predictive vs causal distinction correctly maintained throughout.

**One initial overstatement:** LLM draft said "retail sales predicts Walmart revenue."
Corrected to "retail sales *predicted* Walmart revenue pre-COVID; the relationship is
currently partially restored but not fully validated post-2023."

---

## Overall LLM Assessment — All Phases

### What the LLM Got Right

- Calendar alignment: fiscal-quarter aggregation, not calendar-quarter
- Unit mismatch identification
- Stationarity framing: I(1) levels, I(0) YoY growth
- COVID as structural break (not outlier)
- Granger test design: pre-COVID subsample
- Walk-forward CV framework
- ΔRMSE(ARIMA→SARIMAX) as the walk-forward Granger complement
- Metric completeness: MAE, RMSE, MAPE, sMAPE, DirAcc, Bias

### Where Human Judgment Was Required

1. **220% YoY artifact** — LLM generated numerically valid code that silently propagated
   a data quality error. The `dropna()` fix was accidental, not intentional.
2. **Symmetric CCF** — LLM initially reported lag+1 significance as confirming a leading
   indicator without noting Walmart also "leads" retail (symmetric co-movement structure).
3. **Regime contamination** — LLM framed near-zero full-sample r as "weak signal" rather
   than "signal destroyed by regime inversion."
4. **Spurious 8Q rolling mean** — LLM recommended it based on r=0.39 full-sample,
   missing that r=0.07 pre-COVID. Spurious nominal trend correlation.
5. **Delta R² overstatement** — Summary text stated ~0.16; actual model output is ~0.09.
   Always verify verbal summaries against model output.
6. **Lasso CV leakage** — Initial `LassoCV(cv=5)` used standard KFold (invalid for time
   series). Required TimeSeriesSplit correction.
7. **statsmodels forecast return type** — `.iloc[0]` fails on numpy array return from
   `forecast(steps=1)` in statsmodels 0.14.6. Required `np.asarray()[0]`.
8. **pandas 3.x nullable Float64** — `agg(lambda x: x.iloc[-1])` returns nullable Float64
   which fails `np.isfinite()` in statsmodels. Required explicit `.astype(float)` casts.
9. **Overconfident conclusion framing** — Multiple instances of "retail leads Walmart"
   stated without "pre-COVID only" qualification. Required push-back each time.

### Reliability Summary

The LLM was a strong accelerator for code generation, statistical test selection,
and report structure. It required human oversight at every interpretive step.
The numerical output was usually correct once structural bugs were fixed.
The framing and emphasis needed calibration at least once per major section.

**Rule:** Never accept the LLM's verbal summary without cross-checking against the
actual computed numbers. Overstatements are common when the LLM is describing
regression coefficients, R² values, or RMSE improvements.
