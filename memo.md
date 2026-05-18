# Memo: FRED Retail Sales as a Leading Indicator for Walmart Revenue

**To:** Portfolio Management
**From:** YipitData Take-Home Candidate
**Date:** May 18, 2026
**Re:** Can FRED retail-sales data predict Walmart quarterly revenue?

---

## The Question

Does the FRED U.S. Retail Sales index (series `RSXFS`, monthly) tell us anything *ahead of time* about Walmart's quarterly revenue that we couldn't already get from a simple, mechanical year-ago benchmark? A **leading indicator** is a series whose moves *precede* moves in the target — if FRED retail growth turns up this quarter, does Walmart revenue growth tend to turn up next quarter? A **naive baseline** ("seasonal-naive") sets next quarter's prediction equal to the same quarter one year ago, scaled by the typical historical growth rate — it requires no analyst and no model. We evaluated the question with a **walk-forward out-of-sample test** from 2019 onward: train on every quarter through Q-1, predict Q, expand the training window by one quarter, repeat.

---

## The Answer

**The FRED retail-sales signal does not reliably outperform a simple autoregressive model out-of-sample.** Our best out-of-sample model — a plain **AR(1) on Walmart's own previous quarterly YoY growth** — achieves a walk-forward revenue MAPE of **~2.0 %**, versus **~2.8 %** for the seasonal-naive year-ago benchmark (a ~0.8 percentage-point improvement, or ~29 % relative reduction in forecast error). Adding the FRED `retail_lag1` feature on top of AR(1) — either in OLS or Ridge form — slightly *worsens* out-of-sample MAPE (OLS+FRED ≈ 2.51 %; Ridge ≈ 2.44 %). The FRED retail coefficient in the full-sample fit is positive (≈ +0.04) but statistically indistinguishable from zero (p ≈ 0.52). The leading-indicator story does not survive a strict walk-forward test on this data.

---

## How We Tested It

We trained the four models on data from 2010 Q1 through each successive quarter ending at or after 2018 Q4, predicted exactly one quarter ahead, expanded the training window by one quarter, and repeated for **27 out-of-sample folds**. No future data was used at any point: features for forecasting quarter *Q* come strictly from data available through *Q-1* (which we enforce via `shift(1)` on the autoregressive target and a documented Walmart reporting-lag assumption). The four models tested were: seasonal-naive, AR(1), OLS with the FRED signal plus AR and quarter dummies, and Ridge with an extended feature set including a second FRED lag and a COVID indicator. Walk-forward MAPE on revenue dollars (the standard revenue-forecasting metric) is the headline; YoY MAE in percentage points is the supplementary diagnostic.

---

## What to Worry About

* **Reporting lag.** Walmart's fiscal Q3 results are not public until mid-Q4 (the 10-Q is filed about 5-6 weeks after quarter end). Any prediction made at quarter end must use data through Q-1 only — the `shift(1)` in our pipeline enforces this. A model that "uses" Q-end retail data on the same day Q's revenue is predicted would be cheating.
* **COVID structural break.** The 2020 shock fundamentally distorted both series in opposite directions (FRED spiked on stay-at-home demand; Walmart revenue lagged on inventory and supply-chain constraints). Out of 27 OOS quarters in this study, only **2 lie strictly pre-2020** — so the headline comparison is dominated by post-COVID quarters where the historical relationship has not yet restabilized. Stability post-2022 is unconfirmed in this window.
* **FRED data revisions.** RSXFS is revised backward over multiple subsequent vintages. The walk-forward test here uses the latest-vintage values; in live trading the analyst would see *less accurate* real-time first-prints — so the back-tested error is a lower bound on real-time error.
* **Fiscal vs. calendar quarter mismatch.** Walmart's fiscal quarters end Jan / Apr / Jul / Oct — one month before each calendar-quarter end. We align by aggregating the three FRED months that match each Walmart fiscal-quarter window (e.g. fiscal Q1, ending Jan 31, is aligned to mean(FRED Nov, Dec, Jan)). A reader who instead used the same-labeled calendar quarter would be off by one month and would crush the correlation.
* **Sample size.** ~27 out-of-sample forecasts is enough for a directional verdict but is not enough for tight confidence intervals on small effects. We also flagged uneven forecast errors across time periods (residuals from the COVID years are visibly larger than non-COVID), which means the conventional OLS standard errors should be interpreted with mild caution.

---

## What Would Change Our Minds

1. **Post-COVID re-stabilisation.** If on data through 2027-2028 the walk-forward MAPE of OLS + FRED falls below AR(1) by at least 0.5 pp and the `retail_lag1` coefficient becomes statistically significant at p < 0.05, we would conclude the relationship has restabilized and revisit the signal.
2. **Robust to publication lag.** If we incorporate an explicit FRED-revision lag (use the first-vintage values, not the latest revisions) and the OLS + FRED model still beats AR(1) and seasonal-naive on walk-forward MAPE, the signal is operationally viable rather than only an artifact of revised data.
3. **Sub-sector replacement.** If a Walmart-aligned FRED sub-series — e.g. RSGCSS (general merchandise), RSDBS (department stores), or RSXFSN (non-store retail) — shows a cross-correlation at lag 1 above 0.6 (vs. ≈ 0.10 for aggregate RSXFS in our window), that sub-series replaces RSXFS and the test is re-run.
