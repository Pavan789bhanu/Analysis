# Prompt log — Walmart × FRED take-home

This is the chronological log of prompts I sent to my LLM coding assistant during this exercise. Each prompt is reproduced verbatim (lightly edited for clarity) and tagged with the **artifact it produced**, so a reviewer can re-run any single prompt against the supplied data and reproduce that specific output.

The reflective note (≤200 words) sits at the top.

---

## Reflective note

The assistant was strong at the mechanical layers — data loading, statsmodels boilerplate, walk-forward CV scaffolding, matplotlib styling — and weak in two places that mattered. **First**, on the FRED-to-Walmart join: it initially aggregated FRED to calendar quarters, which is wrong because Walmart's fiscal Q4 spans Nov–Jan and straddles calendar years. I had it switch to a fiscal-quarter mapping (`Feb–Apr=Q1`, `May–Jul=Q2`, `Aug–Oct=Q3`, `Nov–Jan=Q4`). **Second**, on look-ahead: its first feature builder used FRED months up to the target-quarter-end, ignoring publication lag. I tightened the cutoff to `target_date − 1 month` so features mirror what an analyst could actually have on the forecast date.

How I checked it: I ran `seasonal_decompose` and saw FRED's seasonal_strength was 0.024 — the series is seasonally adjusted, which the assistant had not flagged. That changed the comparison to YoY-on-YoY. I also asked for a pre-/post-2020 regime split that the assistant initially offered only as a single-window number; the regime split is what surfaced the real falsifiable claim ("FRED worked pre-pandemic, stopped working after").

I trusted the assistant for code, not for the framing.

---

## Prompt → artifact mapping

| #  | Prompt theme | Primary artifact |
|----|--------------|------------------|
| 1  | Data-quality + temporal EDA | `analysis.ipynb` §2-3, `INSIGHTS.md` §1 |
| 2  | Distribution, decomposition, stationarity, ACF | `analysis.ipynb` (EDA appendix), `INSIGHTS.md` §1 |
| 3  | Leakage-safe feature engineering + fiscal-quarter merge | `analysis.py` §3-4, `analysis.ipynb` §3, `INSIGHTS.md` §2 |
| 4  | Forecast protocol (target, horizon, info set, CV) | `analysis.py` §1, `analysis.ipynb` §3, `INSIGHTS.md` §3 |
| 5  | Implement the 7-model menu | `analysis.py` §6, `analysis.ipynb` §4 |
| 6  | Rolling-origin CV + metrics + paired comparisons | `analysis.py` §7-9, `analysis.ipynb` §5-8 |
| 7  | Pre-/post-2020 regime split | `analysis.py` §10, `analysis.ipynb` §9, fig4 |
| 8  | Cost/latency benchmark + production tradeoff matrix | `analysis.py` §11, `analysis.ipynb` §11, `tradeoff_matrix.csv` |
| 9  | Memo-ready figures (MAPE bars + forecast overlay) | `figures/fig1`, `figures/fig2` |
| 10 | Stakeholder explainability figures | `figures/fig3-fig6` |
| 11 | One-page PM memo | `memo.md` |
| 12 | Long-form professional insights | `INSIGHTS.md` |
| 13 | Notebook generation | `analysis.ipynb` |

---

## The prompts

### Prompt 1 — Data-quality + temporal EDA  →  `analysis.ipynb` §2-3, `INSIGHTS.md` §1

> *"I have two CSVs: `walmart_revenue.csv` (quarterly Walmart revenue) and `retail_sales_fred.csv` (monthly FRED retail sales, series RSXFS). Act as a senior data scientist. Perform a complete data-quality audit and temporal-structure audit on both files: shape, dtypes, missing, duplicates, value ranges, cadence between observations, and the unique months observed. Walmart reports on a fiscal year ending January — map every Walmart date to its fiscal quarter (Feb-Apr=Q1, May-Jul=Q2, Aug-Oct=Q3, Nov-Jan=Q4). Document every observation in OBSERVATIONS blocks."*

### Prompt 2 — Distribution, decomposition, stationarity, ACF  →  `INSIGHTS.md` §1

> *"On the same two files, compute: distribution summary (skew, kurtosis) per series; additive seasonal decomposition with period=4 for Walmart and period=12 for FRED; Hyndman trend_strength and seasonal_strength; ADF and KPSS tests on raw, first-differenced, seasonally differenced, and log-seasonally-differenced versions; ACF/PACF on the appropriately-differenced series. Surface anything surprising in the results — in particular, sanity-check whether the FRED file is seasonally adjusted by looking at its seasonal_strength."*

### Prompt 3 — Leakage-safe feature engineering + fiscal-quarter merge  →  `analysis.py` §3-4, `INSIGHTS.md` §2

> *"Build a leakage-safe feature engineering function. For each Walmart fiscal-quarter end date `d` (the target), produce: Walmart lags 1-4 (`wm_lag1..4`); Walmart YoY of the last reported quarter (`wm_yoy_lag1` = y_{t-1}/y_{t-5} - 1); fiscal-quarter one-hot encodings (`is_Q1..Q4`); and four FRED features that use only data through `(d - 1 month)` — `fred_last`, `fred_last_yoy`, `fred_3m_mean`, `fred_6m_yoy`. The 1-month FRED hold-back is critical: RSXFS publishes with a ~2-week lag, so a real-time forecaster wouldn't have the latest in-quarter month. For the merge from monthly FRED to quarterly Walmart, sum the 3 calendar months ending at each Walmart quarter-end and flag the first row (incomplete FRED coverage) as NaN."*

### Prompt 4 — Forecast protocol design  →  `analysis.py` §1, `INSIGHTS.md` §3

> *"Codify the forecast protocol explicitly: target = Walmart quarterly revenue (level, USD); horizon = 1 fiscal quarter ahead; forecast origin = end of the prior fiscal quarter (the day Walmart's prior 10-Q became public); information set = Walmart actuals through y_{t-1} only (we don't know the current quarter yet) plus FRED data ≤ target-quarter-end − 1 month; CV scheme = rolling-origin / walk-forward expanding window with `MIN_TRAIN_QUARTERS = 24` (~6 fiscal years of training before OOS begins). Document why each choice is honest."*

### Prompt 5 — Implement the 7-model menu  →  `analysis.py` §6

> *"Implement seven forecasters as functions `(train_df, test_row) → ŷ`. Every model re-fits on each CV step. The menu: (1) seasonal_naive = y_{t-4}; (2) seasonal_naive_drift = y_{t-4} × (1 + trailing-4q-avg YoY); (3) sarima_walmart_only = SARIMAX(1,1,0)(0,1,1,4) on Walmart history only; (4) ols_walmart_only = OLS on `[wm_lag4, wm_yoy_lag1, is_Q1..Q3]`; (5) ols_walmart_plus_fred = same OLS plus `[fred_last_yoy, fred_6m_yoy, fred_3m_mean]`; (6) ridge_walmart_plus_fred = standardised Ridge(α=1) on the same features; (7) gbr_walmart_plus_fred = GradientBoostingRegressor on Walmart lags + FRED features. Tag (5)/(6)/(7) as 'uses_FRED'."*

### Prompt 6 — Rolling-origin CV + metrics + paired comparison  →  `analysis.py` §7-9

> *"Run rolling-origin CV across all seven models. At each step record y, ŷ, date, and fit+predict wall-clock time. Report MAPE, RMSE, sMAPE, bias, and average fit+predict ms per quarter. Add two paired-comparison tables: every model vs `seasonal_naive` (mean Δ|%err| in percentage points, std, and share of OOS quarters where the model beats baseline); and every FRED-augmented model vs `sarima_walmart_only` — this is the apples-to-apples test of whether FRED adds value over the best Walmart-only competitor."*

### Prompt 7 — Pre-/post-2020 regime split  →  `analysis.py` §10, fig4

> *"Split the OOS window on `2020-03-01` and compute MAPE separately for pre-COVID (n=12) and post-COVID (n=24) sub-samples for every model. This is to test whether the FRED–Walmart linear relationship is stable. Report the per-regime MAPE table and flag the models whose regime ranking *flips* — those carry the falsifiable claim."*

### Prompt 8 — Cost/latency benchmark + production tradeoff matrix  →  `analysis.py` §11, `tradeoff_matrix.csv`

> *"Use the wall-clock times collected during CV to build a production tradeoff matrix with columns: model, MAPE_%, bias_USD_bn, avg_fit_pred_ms, latency_vs_cheapest_x (anchor to the cheapest *non-trivial* model so we don't divide by zero on the naive baseline), interpretability {trivial, high, medium, low}, deps {none, sklearn, statsmodels}, uses_FRED, main_failure_mode. Write the recommendation from the matrix as a senior data scientist would explain it to a product manager — what to ship, what to use only for diagnostics, what to avoid."*

### Prompt 9 — Memo-ready figures  →  `figures/fig1`, `figures/fig2`

> *"Produce exactly two clean memo-ready figures: (a) horizontal bar chart of OOS MAPE by model, colour-coded by FRED usage, with values labelled at the bar tips; (b) time-series of actuals vs the strongest Walmart-only model (SARIMA), the strongest FRED-augmented model (Ridge), and the seasonal-naive baseline, with a red dotted line at March 2020 marking the structural break."*

### Prompt 10 — Stakeholder explainability figures  →  `figures/fig3-fig6`

> *"Add four stakeholder-facing figures designed for a non-technical reader (portfolio manager). Each must carry a one-line 'business takeaway' annotation baked into the canvas. (a) Cost-accuracy frontier: scatter of latency (log x) vs MAPE, each model labelled, the Pareto frontier drawn as a dashed line. (b) Pre-vs-post-2020 grouped bar chart highlighting that the FRED-augmented models won pre-2020 and lost post-2020. (c) Feature-impact bar chart using Ridge standardised coefficients translated to '% of mean revenue per +1σ' and labelled in business English (e.g. 'Walmart same quarter, prior year' instead of `wm_lag4`). Exclude the trending-level FRED feature to avoid collinearity-driven misinterpretation. (d) A matplotlib-only decision flowchart that answers 'which forecaster do we ship?' in one page."*

### Prompt 11 — One-page PM memo  →  `memo.md`

> *"Write a one-page memo for a portfolio manager who took stats in college but hasn't done much since. Structure: short answer (1 paragraph); regime-split evidence table; falsifiable claim sentence; one figure embedded (the regime comparison); 'what we worry about' bullets; 'what would change our minds' bullets; production recommendation. Plain English; no model formulas in the body."*

### Prompt 12 — Long-form professional insights  →  `INSIGHTS.md`

> *"Write a long-form analyst notes document covering every observation and insight from EDA, feature engineering, and modeling. Organised by phase: §1 EDA, §2 Feature engineering, §3 Modeling protocol, §4 Baseline & univariate results, §5 FRED-augmented results, §6 Regime split, §7 Production tradeoffs, §8 Open questions. Voice: senior data scientist reviewing the work for a peer, not a PM. Cite specific numbers from the analysis files."*

### Prompt 13 — Notebook generation  →  `analysis.ipynb`

> *"Generate `analysis.ipynb` as a thin presentation layer over `analysis.py`. Each markdown cell narrates one prompt's worth of work; each code cell calls into the library and renders the output. The notebook must execute top-to-bottom without errors and embed all six figures inline."*

---

## What the assistant got subtly wrong (and how I caught it)

1. **Calendar vs fiscal quarter join.** First-pass code used `.resample("QE")` on FRED and merged on date. That assumes Walmart's quarter-ends are calendar quarter-ends, which they aren't (Walmart Q4 spans Nov–Jan). Caught by inspecting Walmart's unique observed months: `[1, 4, 7, 10]`.

2. **No FRED publication lag.** First-pass features used FRED data up to the target-quarter end. In production, that data isn't available on the day you'd be forecasting. Caught by tracing the data lineage and asking "could a forecaster on Oct 31 actually have the Oct retail print?" (No — RSXFS October prints mid-November.)

3. **Seasonality of FRED.** The assistant did not initially notice that RSXFS in this file is seasonally adjusted. Caught by reading the additive decomposition output: `seasonal_strength = 0.024` is incompatible with a NSA retail series. This re-framed every cross-series feature to YoY-on-YoY.

4. **In-sample feature impact for the explainability chart.** The first version showed a +6.50 standardised coefficient on FRED's *level* feature (collinear with Walmart's growing trend). That visually contradicted the OOS finding that FRED-augmented models don't beat Walmart-only. Fixed by switching the explainability chart to Ridge coefficients on growth-rate features only.

5. **Divide-by-zero in the latency ratio.** The first tradeoff matrix reported `latency_vs_cheapest_x = 11,230,000` because `seasonal_naive` had been rounded to 0.00 ms. Fixed by anchoring the ratio against the cheapest *non-trivial* model.

## What the assistant got right

- Walk-forward CV scaffolding with re-fit on every step.
- Sensible default SARIMAX orders for a quarterly series.
- Honest paired-comparison reporting (mean Δ|%err| + share of quarters beaten) instead of a full Diebold-Mariano test that would be statistically dicey at n=36.
- A model menu that explicitly includes both naive baselines, a univariate strong competitor (SARIMA), and three FRED-augmented variants spanning OLS → Ridge → GBR.
- Production-perspective tradeoff matrix that explicitly names the failure mode of each model.
