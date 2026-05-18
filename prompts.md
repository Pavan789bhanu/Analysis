# Prompt Log — YipitData Take-Home Exercise

## Overview Note (< 200 words)

I used Cursor's chat as a senior pair-programmer rather than a code generator. Across seven prompts I (a) asked it to *decompose* the problem into a falsifiable plan before any code was written, (b) generated boilerplate I would otherwise type by hand (EDGAR JSON parsing, ADF/KPSS scaffolding, statsmodels OLS / RidgeCV walk-forward loops), and (c) used it as a sanity-checker on subtle pitfalls (look-ahead bias, R² vs MAPE confusion, mean vs median for the seasonal-naive growth scaler). The LLM was directly useful where the work was mechanical and well-trodden — pulling SEC XBRL, ADF call signatures, the structure of an expanding-window walk-forward loop. It was wrong or naïve in three specific places that I caught and corrected: it initially proposed *shuffled* k-fold on a time series; it used the mean (not median) of historical YoY growth for the seasonal-naive scaler, which is heavily distorted by 2020 outliers; and it reported in-sample R² as if it were the predictive metric. I verified every output: I cross-checked the CCF peak lag against the raw `np.corrcoef` values, audited the Walmart fiscal-quarter alignment by month-counting (Nov-Dec-Jan → fiscal Q1), and compared per-fold predictions against actuals by hand for the first three OOS quarters.

---

## Prompt 1 — Problem Decomposition & Research Design

**Prompt sent:**
> You are a senior data scientist. I am working on a take-home exercise for a quant research firm. The research question is: does the FRED monthly retail-sales index (series RSXFS) serve as a statistically reliable leading indicator of Walmart's quarterly revenue, and does it outperform a seasonal-naive baseline? Before writing any code, help me decompose this into a rigorous analytical plan. The plan must: (1) state a testable null hypothesis, (2) identify look-ahead bias as a primary validity threat and state how to prevent it, (3) specify walk-forward expanding-window validation as the evaluation protocol — explicitly ruling out shuffled k-fold, (4) define the seasonal-naive benchmark we must beat before calling the FRED signal useful, and (5) list the EDA steps required to establish stationarity, leading-indicator lag structure, and structural breaks before touching a model.

**LLM response summary:**
> The LLM produced a structured 5-section analysis plan covering hypothesis framing, data alignment risks, EDA sequence (ADF → CCF → Chow test), baseline definition, and walk-forward evaluation logic. It correctly named look-ahead bias and proposed `shift(1)` on the target. It incorrectly suggested using R² as the primary evaluation metric without qualifying that this must be out-of-sample.
>
> → **See `analysis.ipynb` — Section 1 (Problem Framing)** for the accepted null hypothesis and evaluation framework. The R² caveat was corrected: Section 5 reports walk-forward MAPE and RMSE as the primary metrics; in-sample R² from `model.summary()` is presented in Section 5.3 with an explicit markdown warning that it is **not** a predictive metric.

**What I accepted / rejected / modified:**
> Accepted: the 5-part plan structure, null hypothesis wording, and the `shift(1)` suggestion for look-ahead prevention.
> Rejected: using R² as the headline metric. Modified the plan to require walk-forward MAPE as the primary criterion, with R² relegated to a diagnostic in the full-sample OLS summary.
> → Corresponding notebook location: **`analysis.ipynb` → Section 1, Section 5.3, Section 5.5**

---

## Prompt 2 — Data Ingestion & Alignment

**Prompt sent:**
> Write a Python data ingestion module for the following two series: (1) FRED series RSXFS — monthly U.S. retail sales index from 2010-01-01 to present, pulled via the FRED REST API or CSV fallback at `data/retail_sales_fred.csv`; (2) Walmart quarterly revenue from SEC EDGAR XBRL, CIK 0000104169. The EDGAR pull must handle the ASC 606 concept switch: use `us-gaap:Revenues` for fiscal years through 2018 and `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` from fiscal 2019 onward. Q4 is not filed as a quarterly fact — derive it as `FY_total - (Q1 + Q2 + Q3)` on matching fiscal-year boundaries. After ingestion, align both series to a common quarterly index. Handle the Walmart fiscal calendar (quarters ending Jan/Apr/Jul/Oct) vs. FRED calendar quarters explicitly. Add data quality assertions: no nulls after 2010-Q1, FRED values are numeric (coerce `.` to NaN and log count), date range covers at least 2010–2023.

**LLM response summary:**
> The LLM produced a working dual-source ingestion block with CSV fallback (Path A) and EDGAR concept-switch logic (Path B). It included the Q4 = FY − (Q1+Q2+Q3) derivation and the dual-concept pull. Its first attempt at fiscal-calendar alignment naïvely mapped Walmart's Jan-31 fiscal-quarter-end to calendar Q1 — which would associate Walmart's Nov/Dec/Jan revenue with FRED's Jan/Feb/Mar retail data and crush every correlation. I rewrote the alignment to use a trailing-3-month FRED average ending at each Walmart fiscal-quarter-end month (so fiscal Q1 ↔ FRED mean(Nov, Dec, Jan)). The LLM also did not add the explicit data-quality assertions — those were added manually.
>
> → **See `analysis.ipynb` → Section 2 (Data Ingestion)** for the accepted pipeline and the `trailing_3m_mean` alignment function. Data quality assertion block starts with `assert retail_raw['date'].is_monotonic_increasing`.

**What I accepted / rejected / modified:**
> Accepted: EDGAR dual-concept pull, Q4 derivation logic, CSV fallback pattern.
> Rewrote: the fiscal-calendar alignment from a period-merge to an explicit `trailing_3m_mean` helper that aggregates the three FRED months ending at each Walmart fiscal-quarter-end month.
> Added manually: `assert` blocks for null checks, NaN coercion logging, date-range validation, monotonicity check.
> → **`analysis.ipynb` → Section 2** (`trailing_3m_mean` function and the assertion block immediately after data load).

---

## Prompt 3 — EDA: Stationarity, CCF, and Structural Breaks

**Prompt sent:**
> Using the merged DataFrame from the ingestion step, produce the following EDA outputs in a Jupyter notebook. Each figure must have a descriptive title, labeled axes with units, and be followed by a one-sentence markdown caption. Required outputs: (1) Dual-axis time-series plot of raw retail index and Walmart revenue (billions USD), 2010–present, with COVID-19 onset (March 2020) and reopening (Q3 2021) annotated. (2) YoY growth rate chart for both series on the same axes, with NBER recession shading for 2020 Q1–Q2. (3) ADF and KPSS tests on both raw and YoY-differenced series — output as a clean summary table with columns: series, test, statistic, p-value, stationary (Y/N). (4) Cross-correlation function (CCF) between retail YoY and revenue YoY, lags -4 to +4 quarters, with 95% confidence bands — annotate the peak lag in a markdown cell. (5) Structural break test (Chow test or statsmodels breaktest) for a candidate break at 2020 Q1. (6) Histogram + QQ plot for both YoY series with skewness and kurtosis printed.

**LLM response summary:**
> The LLM generated all 6 EDA blocks. The CCF helper used `statsmodels.tsa.stattools.ccf` which returns one-sided correlations; I replaced it with an explicit `manual_ccf` for two-sided lags so the lead vs lag direction is unambiguous. The stationarity table was produced correctly. The structural break test used `statsmodels.stats.diagnostic.breaks_cusumolsresid` rather than a Chow test — this was retained because the small-sample (≈ 60 quarters) Chow F-test is unreliable, and the supplemental pre-2020 vs post-2020 OLS-coefficient comparison is more interpretable.
>
> → **See `analysis.ipynb` → Section 3 (EDA)**, subsections 3.1 through 3.6.

**What I accepted / rejected / modified:**
> Accepted: all 6 EDA figures, stationarity table structure, CUSUM break test, distribution diagnostics.
> Replaced: `statsmodels.tsa.stattools.ccf` (one-sided) → `manual_ccf` helper that returns correlations for lags −4 through +4 with unambiguous sign-direction.
> Retained: CUSUM over Chow — documented the reasoning in the markdown cell preceding Section 3.5.
> → **`analysis.ipynb` → Section 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

---

## Prompt 4 — Feature Engineering with Look-Ahead Safeguards

**Prompt sent:**
> Build the feature engineering pipeline. Enforce the following rules explicitly in code comments: (1) Aggregate monthly FRED retail data to quarterly using `.resample('QE').mean()` — not `.sum()` — because RSXFS is an index level, not a flow variable; document this distinction in a markdown cell. (2) Apply `shift(1)` to the revenue target to reflect the reporting lag: Walmart Q3 revenue is publicly available only in mid-Q4, so the model must predict revenue for quarter Q using features available through Q-1 only. (3) Construct these features: `retail_yoy` (YoY growth of quarterly FRED), `retail_lag1` (retail_yoy lagged 1Q), `retail_lag2` (lagged 2Q — drop if VIF > 10), `revenue_yoy_lag1` (lagged target as AR term), `quarter_dummy` (Q1–Q4 categorical dummies), `covid_flag` (binary, 1 for 2020 Q1–Q2). (4) Compute VIF for all features and print a table. Drop any feature with VIF > 10. (5) Define the walk-forward split: training window starts 2010 Q1, initial cutoff 2018 Q4, expand by 1 quarter per step, forecast horizon 1 quarter ahead, no shuffling.

**LLM response summary:**
> The LLM implemented all five requirements. The `.mean()`-vs-`.sum()` justification was correctly written as a markdown cell. The look-ahead enforcement uses `revenue_yoy.shift(1)` for the autoregressive feature; future-quarter features are never constructed. The VIF table reported all primary features below 10. The walk-forward split returns a list of `(train_idx, test_idx)` tuples and includes an explicit minimum-training-window assertion (`MIN_TRAIN_QUARTERS = 32`).
>
> → **See `analysis.ipynb` → Section 4 (Feature Engineering)**, subsections 4.1 (resampling rationale), 4.2 (look-ahead enforcement), 4.3 (feature construction), 4.4 (VIF table), 4.5 (walk-forward split definition with assertion).

**What I accepted / rejected / modified:**
> Accepted: all five components.
> Added manually: assertion that the minimum training window contains ≥ 32 quarters before the first forecast (`assert len(folds_primary) >= 10`).
> → **`analysis.ipynb` → Section 4.1 through 4.5**

---

## Prompt 5 — Walk-Forward Modeling & Baseline Comparison

**Prompt sent:**
> Implement four models using the walk-forward split defined in Section 4. For each model, iterate over the split folds: fit on the training window, predict one step ahead, collect the prediction. After all folds, compute aggregate walk-forward MAPE and RMSE. Models: (1) Seasonal-naive baseline: ŷ(Q) = y(Q-4) × (1 + median historical YoY growth). This is the mandatory benchmark — every other model is evaluated relative to it. (2) AR(1): OLS with `revenue_yoy_lag1` as the only feature, via statsmodels. (3) OLS with FRED signal: features are `retail_lag1`, `revenue_yoy_lag1`, `quarter_dummy` — statsmodels OLS. After the walk-forward loop, fit a final full-sample OLS and print `model.summary()`. (4) Ridge with extended features: `retail_lag1`, `retail_lag2`, `revenue_yoy_lag1`, `quarter_dummy`, `covid_flag` — sklearn RidgeCV with alphas [0.01, 0.1, 1, 10, 100]. Produce a model comparison table with columns: Model, Walk-Forward MAPE, Walk-Forward RMSE, MAPE vs. Naive Baseline. Highlight the best MAPE row in green using pandas Styler.

**LLM response summary:**
> The LLM implemented all four models with walk-forward loops. Two issues required correction. First, the LLM's seasonal-naive formula used the **mean** of all historical YoY growth — I changed it to the **median** for robustness against the 2020 COVID outlier (documented in the markdown cell preceding Section 5.1). Second, the LLM initially reported MAPE on YoY-percentage values, which is unstable when YoY is near zero (it produced absurd 100%+ MAPE figures). I switched the headline to **revenue-dollar MAPE** — the conventional metric for revenue forecasting — and kept YoY MAE in percentage points as a divide-by-near-zero-resistant secondary diagnostic. The full-sample OLS summary is printed with an in-sample-warning markdown cell. The pandas Styler highlight-min on the leaderboard correctly marks AR(1) as the best out-of-sample model.
>
> → **See `analysis.ipynb` → Section 5 (Modeling)**, subsections 5.1 (seasonal-naive with median), 5.2 (AR(1)), 5.3 (OLS + FRED with in-sample R² caveat), 5.4 (Ridge), 5.5 (leaderboard with pandas Styler).

**What I accepted / rejected / modified:**
> Accepted: all four model structures, walk-forward loop logic, RidgeCV alpha grid, Styler highlight-min.
> Modified: seasonal-naive growth scaler `mean` → `median` (Section 5.1 markdown).
> Modified: headline metric scale — replaced YoY-%-MAPE with revenue-$ MAPE (Section 5.5 metric helper `revenue_metrics`); reported YoY MAE as the supplementary divide-safe metric.
> → **`analysis.ipynb` → Section 5.1 through 5.5**

---

## Prompt 6 — Diagnostics, Residual Analysis & Subsample Stability

**Prompt sent:**
> For the best-performing OLS model from Section 5, run the following diagnostics: (1) Residuals vs. fitted values scatter plot. (2) ACF and PACF of residuals, lags 1–8. (3) Ljung-Box test on residuals — report p-value and state whether autocorrelation is present (fail threshold: p < 0.05). (4) Interpret the Durbin-Watson statistic from `model.summary()` — flag if outside [1.5, 2.5]. (5) Subsample stability: re-fit the OLS model on pre-2020 data only and compare coefficients and walk-forward MAPE to the full-sample model in a side-by-side table. State explicitly whether the FRED signal's coefficient sign and magnitude are stable across regimes. (6) Quantify the effect size: state the out-of-sample MAPE improvement of the best model vs. the seasonal-naive baseline in absolute percentage points and relative percentage terms.

**LLM response summary:**
> The LLM produced all six diagnostic outputs. The Durbin-Watson value was inside [1.5, 2.5] (≈ 2.08 → essentially no first-order autocorrelation). The Ljung-Box p-value at lag 4 is slightly below 0.05 — a partial flag, surfaced in the markdown caption as a known limitation rather than papered over. The subsample-stability comparison surfaced a critical finding the LLM did not flag on its own: only **two** OOS quarters lie strictly pre-2020, so the headline OOS evaluation is dominated by post-2020 data. I added a regime-MAPE breakdown (pre / COVID / post-2020) to make this explicit, and recorded the per-regime n in `DASHBOARD['regimes']`. The effect-size statement was rewritten to honestly conclude that the best model is **AR(1)** — *not* OLS + FRED — and to state this in the headline verdict.
>
> → **See `analysis.ipynb` → Section 6 (Diagnostics)**, subsections 6.1 (residuals plot), 6.2 (ACF/PACF), 6.3 (Ljung-Box + Durbin-Watson), 6.4 (subsample stability + regime MAPE breakdown), 6.5 (effect-size verdict).

**What I accepted / rejected / modified:**
> Accepted: residuals plot, ACF/PACF, Ljung-Box + DW reporting, full-sample-vs-pre-2020 coefficient table.
> Added: explicit regime-MAPE breakdown (pre-2020 / COVID-2020 / post-2020) with per-regime n, OLS-MAPE, naive-MAPE, and Δpp; this surfaces the small-pre-2020 OOS sample issue that the LLM glossed over.
> Rewrote: the effect-size verdict to honestly identify AR(1) as the best out-of-sample model — the FRED signal does not improve over the AR(1) baseline on the available OOS window.
> → **`analysis.ipynb` → Section 6.1 through 6.5**

---

## Prompt 7 — memo.md Drafting

**Prompt sent:**
> Write a one-page executive memo for a portfolio manager audience. The reader took statistics in college but is not a practitioner. The memo must follow this exact structure: (1) The Question — state the research question in plain English, define "leading indicator" and "naive baseline" in one sentence each, state the evaluation method briefly. (2) The Answer — lead with the verdict and quantified effect size in the first sentence (MAPE improvement over naive in percentage points). Do not bury the finding. (3) How We Tested It — 2–3 sentences on walk-forward validation protocol. (4) What to Worry About — 4–5 specific caveats: reporting lag, COVID structural break, FRED data revisions, fiscal/calendar quarter mismatch, sample size. (5) What Would Change Our Minds — 3 falsifiable conditions for when confidence in the signal would increase. No jargon without a one-phrase definition. One page maximum.

**LLM response summary:**
> The LLM produced a clean, correctly structured memo. The verdict sentence led with the quantified MAPE improvement, but the LLM defaulted to framing OLS + FRED as the winning model — I rewrote the verdict to reflect the actual finding (AR(1) is the best OOS model; the FRED signal does **not** beat AR(1)). The caveats section covered all five required items. The "What Would Change Our Minds" section produced three appropriately falsifiable conditions. Minor terminology fix: the LLM used "heteroskedasticity" without defining it — replaced with "uneven forecast errors across time periods."
>
> → **See `memo.md`** — the final accepted document with the corrected verdict and the plain-English terminology fix applied directly.

**What I accepted / rejected / modified:**
> Accepted: full structure, the five caveats, three falsifiable conditions, plain-English ground rules.
> Rewrote: the verdict sentence to match the actual model-comparison finding (AR(1) wins; FRED signal adds nothing material).
> Replaced: one undefined technical term (`heteroskedasticity` → "uneven forecast errors across time periods").
> → **`memo.md` — final file**
