# Memo: FRED Retail Sales as a Leading Indicator for Walmart Quarterly Revenue

**To:** Portfolio Management
**From:** YipitData Take-Home Candidate
**Date:** May 22, 2026
**Re:** Does the FRED RSXFS retail-sales index predict Walmart quarterly revenue better than a fair baseline?

---

## The Question

Can the FRED U.S. Retail Sales index (series `RSXFS`, published monthly) help us anticipate Walmart's quarterly revenue *before* Walmart reports? Two definitions a portfolio manager should hold in mind. A **leading indicator** is a series whose moves tend to *precede* moves in the target — if retail growth accelerates this quarter, does Walmart's revenue growth tend to accelerate next quarter? A **naive baseline** is the bar a model must clear before we call it useful; we tested two: (B1) the expanding mean of Walmart's revenue YoY within the same fiscal quarter (pure seasonality, no macro data), and (B3) an OLS regression using Walmart's own previous-quarter YoY growth plus fiscal-quarter dummies (own-momentum + season — the stricter benchmark). We evaluated every model with **walk-forward expanding-window forecasting**, one quarter ahead, retraining each step, from 2016 Q3 through 2026 Q1 — 39 strictly out-of-sample quarters.

---

## The Answer

**The retail signal beats the seasonal-naive baseline pre-COVID and post-2023, but it does *not* beat a simple Walmart-own-momentum model in any regime.** Concretely: over the full 39-quarter out-of-sample window, the AR-only OLS model (Walmart YoY lag-1 + fiscal dummies) delivers RMSE **2.46 pp**, vs. **2.97 pp** for OLS-with-retail and **2.85 pp** for the seasonal-mean baseline — the AR baseline is **0.51 pp better** than the retail-augmented model (a 20.6 % worse RMSE for the retail model). Pre-COVID (n = 14) the retail model improves on the seasonal mean (RMSE 1.65 vs 1.74) but still loses to AR-only (1.42). During COVID (n = 12) the retail signal *actively misleads* — RMSE 4.24 vs 3.15 for AR-only — because broad retail crashed in April 2020 while Walmart's essential-goods mix kept revenue growing. Post-2023 (n = 13) OLS-with-retail and AR-only are essentially tied (RMSE 2.64 vs 2.62). The in-sample evidence pre-COVID is suggestive (retail coefficient +0.74, p = 0.014; ΔR² of +0.085, partial-F p = 0.014; pre-COVID lag-1 correlation r = 0.78; Granger F = 7.39, p = 0.011) but **does not survive a strict out-of-sample test against Walmart's own quarterly momentum**. The honest production recommendation today is **AR-only + fiscal-quarter seasonality**; treat retail as a regime indicator to monitor, not a default forecast input.

---

## How We Tested It

We used a strict **walk-forward expanding-window** protocol: for every quarter starting at 2016 Q3 we trained each model on every prior quarter (minimum 20 training quarters required), produced a one-step-ahead forecast, then expanded the training window by one quarter and repeated — never shuffling, never using future data. We tested **three baselines** (B1 fiscal-quarter expanding mean, B2 flat historical mean, B3 OLS on Walmart YoY lag-1 + fiscal dummies) and **four signal models** (M1 OLS+retail, M2 Ridge+retail, M3 regime-aware OLS with a `retail × normal` interaction, M4 OLS with retail lag-2). Results are reported as RMSE and MAE in YoY percentage points, broken down by regime (Pre-COVID / COVID / Post-2023) so the headline number does not silently average across a regime where the signal inverted.

---

## What to Worry About

* **Regime break and signal inversion (2020).** RSXFS covers *all* retail; Walmart is *essential-goods* heavy. During the April 2020 stay-at-home shock, broad retail fell ~17 % while Walmart's revenue held up. The rolling 8-quarter correlation between the two YoY series went from a stable +0.75 pre-COVID to a low of **−0.71** in COVID and has not recovered (current value ≈ 0 post-2023). Any model using retail as a feature without a regime detector inherits this inversion risk.
* **Reporting / publication lag.** Walmart files the 10-Q about 5–6 weeks after fiscal-quarter end. The final FRED retail month of a quarter is typically released *before* Walmart reports — but FRED publishes a first-print "advance estimate" that is subsequently revised by ~±0.5 %. We used the latest-vintage FRED values in this back-test; live performance with first-print vintages would be modestly worse.
* **Inflation confounding.** Post-2021 both series are nominal. A naive correlation may be picking up shared CPI exposure rather than real consumer demand. We dropped long-rolling-mean retail features (8-quarter mean had full-sample r ≈ 0.39 but pre-COVID r ≈ 0.07 — clear nominal-trend contamination, not a real-demand signal).
* **Sample size.** 39 out-of-sample forecasts is enough for a directional verdict but not enough for tight confidence intervals on small effects. The pre-COVID OOS window is only 14 quarters and the post-2023 window is only 13 — the regime where the signal *might* be re-stabilising has the fewest observations. We also see uneven forecast errors across time periods (residuals from the COVID era are visibly larger), so conventional OLS standard errors should be interpreted with mild caution.
* **Fiscal vs. calendar quarter mismatch.** Walmart's fiscal quarters end Jan 31 / Apr 30 / Jul 31 / Oct 31 — one month before the standard calendar quarter ends. We map every monthly FRED observation to the correct Walmart fiscal quarter before aggregating; a naive calendar-quarter merge would mis-align two-thirds of the data and crush every correlation downstream.

---

## What Would Change Our Minds

1. **Post-COVID re-stabilisation.** If on data through 2027-2028 the walk-forward RMSE of OLS+retail falls **below** AR-only by at least 0.2 pp over a window of ≥ 24 post-2023 OOS quarters, and the rolling 8-quarter correlation has been ≥ 0.5 for at least 8 consecutive quarters, we would re-promote retail to a production feature.
2. **Robust to publication lag and revisions.** If we re-run the same back-test using only first-print FRED vintages (no revisions) and OLS+retail still beats AR-only on full-sample RMSE, the signal is operationally viable rather than only an artifact of clean revised data.
3. **Sub-sector replacement.** Walmart's revenue mix maps poorly to aggregate retail. If a more Walmart-aligned FRED sub-series — e.g. RSGCSS (general merchandise), RSDBS (department stores), or RSGMS (general-merchandise stores) — shows pre-COVID lag-1 correlation ≥ 0.85 *and* maintains lag-1 correlation ≥ 0.5 post-2023 over ≥ 12 quarters, that sub-series replaces RSXFS and the test is re-run end-to-end.
