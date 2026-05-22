# Prompt Log — YipitData Take-Home Exercise

## Overview Note (< 200 words)

I treated the LLM as a senior pair-programmer, not a code generator. The seven prompts below trace the analytical arc of `analysis_updated.ipynb`: Phase 1 data audit, Phase 2 single-series EDA, Phase 3 relationship analysis (CCF / Granger / rolling correlation), Phase 4 EDA synthesis, Phase 5 leakage-safe feature engineering, Phase 6 walk-forward out-of-sample evaluation against three baselines (B1/B2/B3) and four candidate signal models (M1–M4), and a final memo. The LLM was directly useful for mechanical work — fiscal-calendar mapping, ADF/KPSS scaffolding, Granger and CCF helpers, the expanding-window evaluator. It was wrong or naive in three places that I caught and corrected: (a) it initially proposed shuffled cross-validation on a time series; (b) it reported full-sample correlation (r ≈ 0.04) as the answer and missed that pooling COVID with pre-COVID cancels a real pre-COVID signal (r ≈ 0.78 at lag +1); (c) it framed the in-sample partial-F result (p = 0.0144) as the headline, which obscured that AR-only beats OLS+retail on every out-of-sample regime. I verified every output by hand: cross-checked CCF peak against `np.corrcoef`, re-derived the partial-F manually, and audited the 39 OOS quarters against the fiscal-quarter calendar.

---

## Prompt 1 — Phase 1: Data Audit & Fiscal Calendar Alignment

**Prompt sent:**
> You are a senior data scientist. Before any modeling, audit the two input series and lock the joined panel for downstream work. The inputs are `data/retail_sales_fred.csv` (FRED RSXFS, monthly, millions USD) and `data/walmart_revenue.csv` (Walmart quarterly revenue, absolute USD, fiscal quarters ending Jan 31 / Apr 30 / Jul 31 / Oct 31). Produce the following with explicit print-outs, not silent fixes: (1) schema/dtype/missing-value audit for both files; (2) date-continuity and frequency verification (FRED should be monthly first-of-month; Walmart should be quarterly fiscal); (3) unit consistency note — retail in millions, Walmart absolute — flag this and state that we will work in YoY % to make units irrelevant; (4) map every retail month to the correct Walmart fiscal quarter end (a naive calendar-quarter join would mis-align Nov-Jan retail with the wrong Walmart quarter); (5) **drop** any quarter where `n_months != 3` (Jan 2010 has only one month; April 2026 has only two — keeping them produces a fake +220% YoY in 2011); (6) modified-Z-score outlier flag on first differences (do not delete COVID — that is real economics, not a data bug); (7) print a one-page data-audit summary table. Add a Phase-1 wrap-up observation cell stating exactly how many aligned quarters survived.

**LLM response summary:**
> The LLM produced a clean Phase-1 audit with all seven items. The fiscal-calendar mapping function (`map_to_walmart_qtr_end`) and the `flag_outliers` modified-Z helper were directly usable. It silently dropped Apr 2026 without commentary — I added the `n_months != 3` print so the reader can see which quarters were removed and why. The LLM also initially used percentile-based outlier flagging; I switched to modified-Z because the small Walmart-quarterly panel (n ≈ 64) makes percentile cutoffs noisy. After the merge we have **64 aligned quarters** (Apr 2010 → Jan 2026) and **60 valid YoY rows** — thin for ML, sufficient for transparent OLS plus walk-forward.
>
> → **See `analysis_updated.ipynb` → Phase 1**, subsections 1.1 through 1.10 (`### 1.10 Data audit summary` for the closing table).

**What I accepted / rejected / modified:**
> Accepted: schema audit, fiscal-quarter mapper, monotonicity & range checks, audit-summary table.
> Modified: outlier flagging from percentile cutoff to modified-Z (3.5 threshold) on first differences; added the explicit `n_months != 3` drop trace.
> Added manually: the "Phase 1 wrap-up" observation cell stating 64 merged / 60 valid YoY rows and the calendar-alignment rationale.
> → **`analysis_updated.ipynb` → §1.5 (fiscal mapper), §1.6 (boundary-quarter drop), §1.7 (outliers), §1.10 (audit summary)**

---

## Prompt 2 — Phase 2: Single-Series EDA (Trends, Seasonality, Stationarity, Volatility)

**Prompt sent:**
> With the audited panel in hand, produce a clean Phase-2 EDA that examines each series **on its own** before any cross-series claim. Required outputs, each with a descriptive title, labeled axes, and a one-sentence observation cell after the figure: (1) levels chart of FRED retail vs. Walmart revenue with COVID and reopening annotated; (2) YoY % growth chart for both series on a common Y-axis with NBER recession shading for 2020 Q1-Q2; (3) Walmart fiscal-quarter seasonality — bar chart showing each fiscal quarter's mean revenue as a % of the FY mean (FY-Q4 / holiday quarter should clearly dominate at roughly +8%); (4) stationarity tests on raw, first-difference, and YoY transforms for both series using **both** ADF and KPSS — only trust the call when the two tests agree; (5) rolling-volatility plot (12-month for retail YoY monthly, 4-quarter for Walmart YoY) to flag heteroskedasticity — explicitly state whether OLS constant-variance is plausible on the full sample. Conclude with a markdown cell stating which transform Phase 3 should use (spoiler: YoY).

**LLM response summary:**
> The LLM delivered all five panels with appropriate captions. Two corrections were needed. First, its initial Walmart-seasonality bar chart used calendar quarters instead of Walmart fiscal quarters — I corrected it to FY-Q1 through FY-Q4. Second, its stationarity report ran only ADF; I added KPSS so the dual-test convention from the prompt is enforced. The volatility analysis explicitly confirmed heteroskedasticity on the full sample (retail volatility explodes during 2020-2022), which justifies the regime-split approach used throughout Phase 3 onward — "use robust standard errors or restrict to pre-COVID for inference."
>
> → **See `analysis_updated.ipynb` → Phase 2**, subsections 2.1 (levels), 2.2 (YoY growth), 2.3 (Walmart fiscal seasonality, ~8% FY-Q4 lift), 2.4 (ADF + KPSS table), 2.5 (rolling volatility).

**What I accepted / rejected / modified:**
> Accepted: levels and YoY plots with COVID/reopening annotations, rolling-volatility figures.
> Modified: seasonality bar chart from calendar quarters → Walmart fiscal quarters; stationarity table extended from ADF-only to ADF + KPSS with agreement column.
> Added: explicit heteroskedasticity verdict in the §2.5 observation cell to justify regime-split / robust-SE choices downstream.
> → **`analysis_updated.ipynb` → §2.1 – §2.5**

---

## Prompt 3 — Phase 3: Relationship Analysis (Lead-Lag, Granger, Rolling Correlation)

**Prompt sent:**
> Now test whether retail actually *leads* Walmart. Critical instruction: **run the CCF and Granger tests on the pre-COVID subsample first**, not the full sample. The full sample pools a strong pre-COVID era with a COVID era where retail crashed and Walmart did not — the blended correlation is near zero and misleading. Required outputs: (1) regime-coloured contemporaneous scatter of retail YoY vs Walmart YoY (pre-COVID / COVID / post-2023 distinct colors) with a regression line per regime; (2) CCF at lags -4 to +4 quarters on the pre-COVID subsample with 95% confidence bands; print the peak lag and interpret whether retail uniquely leads or both series co-move (CCF symmetry test); (3) Granger causality test pre-COVID and full-sample, lags 1-4, retail → Walmart; report F-statistic and p-value; (4) 8-quarter rolling correlation (contemporaneous and lag-1) to test stability through time — print mean & std for the pre-COVID window; (5) summary table of regime-by-regime n and correlation (r at lag 1) — pre-COVID / COVID / post-2023 / full sample. State explicitly in a markdown cell whether the relationship is stable or regime-dependent.

**LLM response summary:**
> The LLM produced all five outputs cleanly. The pre-COVID CCF showed peak r = 0.84 at lag 0 with r = 0.78 at lag +1 and r = 0.43 at lag +4 — confirming the visual co-movement. Critically, the CCF is **symmetric**: negative lags (Walmart leading retail) were also significant (r ≈ 0.66 at lag −1, r ≈ 0.53 at lag −2), which I had the LLM call out explicitly in the observation cell — both series respond to a shared macro driver; retail is not a pure exogenous lead. Granger pre-COVID at lag 1 returned **F = 7.39, p = 0.0106** (rejects null), but on the full sample nothing is significant — exactly the regime-dependence the prompt was guarding against. The rolling correlation collapsed from a stable ~0.75 pre-COVID to a low of −0.71 in COVID and has not recovered.
>
> → **See `analysis_updated.ipynb` → Phase 3**, subsections 3.1 (regime scatter), 3.2 (pre-COVID CCF), 3.3 (Granger), 3.4 (rolling correlation), 3.5 (regime summary table — pre-COVID r = 0.78, COVID r = −0.37, post-2023 r = 0.19, full r = 0.09).

**What I accepted / rejected / modified:**
> Accepted: regime scatter, pre-COVID CCF & Granger, rolling correlation, regime summary table.
> Modified: explicit pre-COVID restriction on CCF (LLM initially defaulted to full sample); added the "CCF symmetry implies shared macro driver" callout so a reader does not over-interpret lag +1 as proof of a unique lead.
> Added: the regime summary markdown table as a numeric backbone for the Phase 4 EDA-conclusions cell.
> → **`analysis_updated.ipynb` → §3.1 – §3.5 (especially §3.2, §3.3, §3.5)**

---

## Prompt 4 — Phase 4: EDA Synthesis & Phase-Gate Decision

**Prompt sent:**
> Produce the Phase 4 EDA conclusion as a single printed write-up. Structure it exactly: (1) Short answer to "can retail predict Walmart?" in one sentence; (2) Key observations grouped — pre-COVID strong correlation (with numbers), COVID inversion, post-2023 inconclusive, full-sample near-zero, CCF symmetry interpretation; (3) Potential predictive signals worth carrying into Phase 5; (4) Structural risks — regime break, look-ahead bias, inflation confounding; (5) A 1-sentence phase-gate verdict on whether to proceed to feature engineering or to abandon the signal. The point of this cell is that anyone reading the notebook end-to-end has a clear decision summary at the EDA/modeling boundary.

**LLM response summary:**
> The LLM produced a tight write-up matching the requested 5-section structure. I accepted it almost verbatim because the language was already at the right register for a notebook audience. The only edit: the LLM initially recommended "abandon the retail signal" — too strong given the in-sample evidence. I softened to "proceed to feature engineering and OOS validation — but the signal is regime-dependent, not a universal law." The 5-bullet observation block (pre-COVID r ≈ 0.87 / 0.78, COVID inversion, post-2023 inconclusive, full-sample near zero, CCF symmetry) became the numeric backbone for both Phase 5 and the memo.
>
> → **See `analysis_updated.ipynb` → Phase 4 (cells 62-65)** — the printed conclusion block at `### 4 — EDA conclusions`.

**What I accepted / rejected / modified:**
> Accepted: the 5-section conclusion structure, the pre-COVID-vs-pooled framing, the inflation-confounding caveat.
> Modified: the phase-gate verdict from "abandon" to "proceed conditionally" — keeping the signal in Phase 5 so we can prove or disprove it OOS in Phase 6, rather than dropping it on EDA alone.
> → **`analysis_updated.ipynb` → §4 (EDA conclusions printed block)**

---

## Prompt 5 — Phase 5: Leakage-Safe Feature Engineering

**Prompt sent:**
> Build the feature matrix with no look-ahead. Strict rules in code comments: (1) Document the Walmart publication lag — most FRED months for a quarter are available before Walmart reports the quarter, but the final month carries revision risk; treat **prior-quarter retail YoY (lag 1)** as the primary feature, current-quarter sums as borderline; (2) Aggregate monthly FRED to fiscal-quarter level by **summing** the three months in each Walmart quarter (equivalent to mean for YoY); (3) Build 12 candidate feature groups: retail YoY lag 1/2/3, retail QoQ lag-1, 4Q & 8Q rolling means and stds, fiscal-quarter dummies, lag-1 z-score, Walmart AR-1 and AR-4, COVID indicator, retail-lag-1 × normal-regime interaction, pre/post-COVID flags; (4) Score every feature on **two** correlation columns — full-sample r and pre-COVID r — and reject any feature whose full-sample r is materially larger than its pre-COVID r (those are picking up COVID/inflation common shocks, not a durable retail lead); (5) VIF check on three candidate feature sets (minimal / extended / regime-aware) on the pre-COVID subsample; (6) Incremental R² test — pre-COVID, in-sample — comparing AR + season vs. AR + season + retail_lag1 via partial F-test; report ΔR², F, and p-value; (7) Final pipeline cell listing the kept and dropped features with one-line reasons.

**LLM response summary:**
> The LLM produced all seven items. The feature-correlation contrast (full-sample r vs pre-COVID r) was the key discriminator: `f_retail_roll_mean_8q` had full-sample r ≈ 0.39 (significant) but pre-COVID r ≈ 0.07 — i.e. it was tracking nominal trend / inflation, not retail signal. Dropped. `f_retail_lag1` had the opposite profile: r ≈ 0.10 full-sample, r ≈ 0.78 pre-COVID — kept as the headline retail feature. VIF on the minimal 5-feature set: retail_lag1 = 3.58, wmt_ar1 = 3.56, fiscal dummies < 1.6 — within "watch" but not fatal. **Incremental R² (pre-COVID, in-sample): AR + season = 0.5637 → +retail_lag1 = 0.6491**, ΔR² = **0.0854**, partial F = **6.82**, **p = 0.0144** — statistically significant. Retail coefficient = **+0.7404** (SE 0.284, t = 2.61, p = 0.014) — economically readable as "1 pp higher retail YoY last quarter → ~0.74 pp higher Walmart YoY this quarter, holding AR/season fixed." I added the explicit "in-sample ≠ OOS" warning in the §5.7 observation cell so a reader does not stop here.
>
> → **See `analysis_updated.ipynb` → Phase 5**, subsections 5.1 (publication lag), 5.3 (monthly→quarterly sum), 5.4 (12 feature groups), 5.5 (full vs pre-COVID correlation contrast), 5.6 (VIF table on three feature sets), 5.7 (incremental R² with partial-F), 5.8 (final pipeline), 5.9 (summary).

**What I accepted / rejected / modified:**
> Accepted: 12-group feature construction, full-vs-pre-COVID correlation contrast, VIF tables on three feature sets, incremental-R² + partial-F decomposition, final minimal feature set (`f_retail_lag1`, `f_wmt_ar1`, `f_q1`–`f_q3`).
> Added: the in-sample-vs-OOS warning sentence in §5.7 so the +0.085 ΔR² / p=0.0144 result is not over-claimed; the regime-aware feature `f_retail_lag1_x_normal` flagged as "hindsight-only" so a reviewer knows it is not deployable without a live regime rule.
> → **`analysis_updated.ipynb` → §5.1 – §5.9**

---

## Prompt 6 — Phase 6: Walk-Forward OOS Evaluation Against Three Baselines

**Prompt sent:**
> Implement the walk-forward out-of-sample evaluator with these strict rules: (1) Expanding window, minimum 20 training quarters, 1-step-ahead horizon, never shuffle; (2) Three **baselines** that every signal model must clear: B1 = expanding mean of Walmart YoY within the same fiscal quarter (captures seasonality without macro data); B2 = expanding mean of all past Walmart YoY (flat); B3 = OLS on Walmart YoY lag-1 + fiscal dummies (this is the **hardest fair test** — own momentum + season); (3) Four signal model variants: M1 = OLS with retail_lag1 + AR-1 + fiscal dummies; M2 = Ridge with the same feature set; M3 = regime-aware OLS using `retail × normal` and a COVID flag; M4 = OLS using retail_lag2 instead of lag-1 (lag-sensitivity check). (4) For each model report OOS MAE, RMSE, directional-change accuracy, and signed bias, both full-sample and broken down by Pre-COVID / COVID / Post-2023. (5) Robustness checks: lag-1 vs lag-2, Ridge vs OLS, training-window sizes 16 / 20 / 24. (6) Plot actual vs predicted for the full OOS window with three lines (actual, B3 AR-only, M1 OLS+retail). (7) Plot 4-quarter rolling MAE for B3 vs M1 to show where retail helps and where it hurts.

**LLM response summary:**
> The LLM implemented all seven items cleanly. The evaluator produced **39 OOS test quarters** (first forecast 2016-07-31; pre-COVID 14, COVID 12, post-2023 13). The headline scoreboard is unambiguous — see the OOS table in the notebook (`§6.3`). Against B1 (seasonal mean), OLS+retail is +5.1% better pre-COVID and +19.5% post-2023; against B3 (AR-only), OLS+retail is **−16.2% pre-COVID, −34.5% COVID, −0.6% post-2023, −20.6% full sample** — i.e. **AR-only wins or ties in every regime**. Robustness: lag-2 is marginally better than lag-1 on full-sample (RMSE 2.58 vs 2.97) but neither beats AR-only's 2.46; Ridge modestly outperforms OLS (full RMSE 2.92 vs 2.97) but still loses to AR; training windows 16/20/24 give RMSE 2.85/2.97/3.12 — stable, no sharp sensitivity. The 4-quarter rolling-MAE chart shows the OLS+retail spike during COVID where retail-lag-1 carried the wrong sign.
>
> → **See `analysis_updated.ipynb` → Phase 6**, subsections 6.1 (baselines B1/B2/B3 table), 6.2 (walk-forward engine, min_train = 20), 6.3 (OOS metrics table — 7 models × 4 periods), 6.4 (forecast-vs-actual chart), 6.5 (Δ% RMSE vs B1 and B3), 6.6 (regime breakdown), 6.7 (robustness: lag, Ridge vs OLS, window size), 6.8 (rolling MAE), 6.9 (final business conclusion).

**What I accepted / rejected / modified:**
> Accepted: the three-baseline structure (the key methodological move), the four signal-model variants, the regime-by-regime scorecard, the robustness battery.
> Modified: forced B3 (AR-only OLS) to be the "hard" benchmark — the LLM initially compared only to B1, which would have given an artificially favourable read on the retail signal. With B3 included the headline becomes "retail does not beat AR-only OOS."
> Added: the rolling-MAE chart (§6.8) so the COVID-period failure of OLS+retail is visible, not just in the numeric table.
> → **`analysis_updated.ipynb` → §6.1 – §6.9**

---

## Prompt 7 — memo.md Drafting

**Prompt sent:**
> Write a one-page executive memo for a portfolio manager audience. The reader is quantitatively literate but is not a practitioner. The memo must follow this structure exactly: (1) The Question — research question in plain English, define "leading indicator" and "naive baseline" in one sentence each, state the evaluation method in one sentence. (2) The Answer — lead with the verdict in the first sentence, quantified (RMSE difference vs. AR-only and vs. seasonal-naive baseline in YoY percentage points). Do not bury the lede. (3) How We Tested It — 2–3 sentences on the walk-forward expanding-window protocol with minimum training window, no shuffle, and the three-baseline structure (B1/B2/B3). (4) What to Worry About — 4–5 specific caveats: regime break and the COVID inversion, look-ahead and reporting lag, FRED revisions, inflation confounding, sample size (14 pre-COVID and 13 post-2023 OOS quarters). (5) What Would Change Our Minds — 3 falsifiable conditions for when confidence in the retail signal would increase. No undefined jargon. One page maximum.

**LLM response summary:**
> The LLM produced a structurally correct draft. The verdict sentence initially framed the result as "OLS+retail beats the seasonal-naive baseline" — true but misleading; the real answer is "OLS+retail beats the seasonal-naive baseline pre-COVID and post-2023 but **does not** beat the AR-only baseline in any regime." I rewrote the verdict to lead with both comparisons so the reader does not walk away with a falsely positive impression. The caveats and "what would change our minds" sections were accepted; one minor terminology fix replaced "heteroskedastic residuals" with "uneven forecast errors across time periods."
>
> → **See `memo.md`** — the final accepted document with the corrected verdict and plain-English terminology fix applied directly.

**What I accepted / rejected / modified:**
> Accepted: the five-section structure, plain-English ground rules, the three falsifiable conditions, the four caveats (regime / lookahead / FRED revisions / sample size).
> Rewrote: the verdict sentence to honestly compare against **both** baselines (B1 seasonal AND B3 AR-only) — leading with the AR-only comparison because it is the stricter test and the one a production user cares about.
> Replaced: one undefined technical term (`heteroskedasticity` → "uneven forecast errors across time periods").
> → **`memo.md` — final file**
