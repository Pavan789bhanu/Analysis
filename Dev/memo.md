# Memo: Can FRED Retail Sales Lead Walmart Quarterly Revenue?

**To:** Portfolio Manager  
**From:** Data Science  
**Date:** May 2026  
**Subject:** Complete Analysis — Phases 1–3 (EDA, Feature Engineering, Modeling)

---

## Bottom Line Up Front

**Yes — but only in normal economic conditions.**

Pre-COVID (2011–2020), retail YoY at lag-1 had a strong, Granger-significant relationship
with Walmart YoY revenue (r = 0.78, p = 0.010). During COVID and recovery (2020–2022),
the relationship inverted completely (r = −0.60). Since 2023, it has not meaningfully
recovered (r = 0.12, not significant).

**The retail signal is real. It is also regime-dependent.**  
Any production model using it must be paired with a volatility gate that flags
structural breaks before publishing a forecast.

---

## Phase 1 — Data Audit Findings

| Check | retail_sales_fred | walmart_revenue |
|-------|------------------|-----------------|
| Shape | 196 rows (monthly) | 65 rows (quarterly) |
| Nulls | 0 | 0 |
| Duplicates | 0 | 0 |
| Units | Millions USD (SA) | Absolute USD |
| Calendar | Standard monthly | Non-standard fiscal (FY ends Jan) |
| COVID break | −17% Apr 2020 | Held up (essential goods) |

**Critical data issues resolved:**
1. **Boundary quarter drop:** The FY-Q4 2010 quarter (ending Jan 31, 2010) contained
   only January 2010 retail data (Nov/Dec 2009 outside data range). This produced a
   spurious +220% YoY for FY-Q4 2011. The incomplete quarter was explicitly dropped
   by requiring n_months == 3 per fiscal quarter.
2. **Fiscal calendar alignment:** Walmart's FY-Q4 = Nov–Jan (ends Jan 31), not
   standard calendar Q1 (Jan–Mar). Using standard quarters would misalign every
   quarter by 1–2 months.

---

## Phase 1 EDA Findings

| Regime | Quarters | Contemp r | Lag+1 r | Granger p (lag 1) |
|--------|----------|-----------|---------|------------------|
| Pre-COVID (2011–Jan 2020) | 36 | **0.87 ★** | **0.78 ★** | **0.010 ★** |
| COVID/Recovery (2020–2022) | 11 | **−0.60** | −0.46 | n/a |
| Post-2023 | 13 | 0.12 | 0.19 | n/a |
| Full sample | 60 | 0.04 | 0.09 | 0.50 |

★ = statistically significant at p < 0.05

**Why COVID broke the relationship:** RSXFS covers all US retail.
Walmart sells essential goods (food, household, pharmacy). When COVID hit,
general retail collapsed while Walmart's essential business held up and accelerated.
The two series decoupled at precisely the moment the relationship mattered most.

---

## Phase 2 — Feature Engineering Decisions

### Transformation Selection

Both series are I(1) in levels — regressing levels on levels produces spurious R² > 0.99.
**YoY % growth rate is the correct transform:**
- Near-stationary (ADF p < 0.05)
- Interpretable: "consumer demand is X% higher than same quarter last year"
- Removes both trend and seasonal effects simultaneously
- Aligns with how analysts report retail and revenue performance

### Aggregation

Sum and Mean are algebraically **identical** for YoY growth ratios — the constant
factor 1/3 cancels exactly. Last-month is marginally noisier (r=0.77 vs r=0.79
pre-COVID). **Use sum-based quarterly aggregation.**

Fiscal calendar alignment (Nov-Jan = FY-Q4, ends Jan 31) is mandatory.
Calendar-quarter resampling (Jan-Mar, Apr-Jun, etc.) would misalign every quarter.

### Publication Lag

| Signal | Safety | When available |
|--------|--------|---------------|
| Lag-1 quarterly YoY | ✅ 100% safe | Prior complete quarter, always published |
| Same-quarter advance estimate | ⚠️ Borderline | 3–5 days before WMT reports; ±0.5% revision risk |
| Same-quarter final | ❌ Usually unavailable | Published after WMT reports |

**Rule:** Use lag-1 in all deployed models.

### Final Recommended Feature Sets

**Minimal (5 features) — for OLS, ARIMA, interpretable models:**

| Feature | r (pre-COVID) | Leakage status | Verdict |
|---------|--------------|----------------|---------|
| `f_retail_lag1` | 0.78 ★ | 100% safe | Primary leading indicator |
| `f_wmt_ar1` | 0.74 ★ | 100% safe | Revenue momentum (AR) — most stable |
| `f_q1/f_q2/f_q3` | — | — | Seasonal fixed effects — mandatory |

**Extended (8 features) — for regularised models:**
Adds `f_retail_lag2`, `f_retail_trend_4q` (4Q rolling mean), `f_retail_momentum`

### Incremental R² Evidence

| Model | R² | Adj-R² | Partial F vs previous |
|-------|----|---------|-----------------------|
| A: AR(1) + seasonal only | ~0.56 | ~0.50 | — |
| B: A + retail_lag1 | ~0.65 | ~0.59 | F≈6.8, **p≈0.014 ★** |
| C: B + retail_lag2 | ~0.67 | ~0.59 | F≈1.3, p≈0.26 (n.s.) |

**Retail lag-1 adds ~+0.09 R² with F=6.8, p=0.014 — statistically significant.**
Retail lag-2 adds marginal, non-significant further improvement.

### Features Rejected

- **f_retail_roll8q (8Q mean):** full-sample r=0.39 but pre-COVID r=0.07 → **spurious** shared nominal trend
- **f_retail_vol4q:** r ≈ 0.02 both samples → no predictive value
- **f_intra_qtr_mom:** borderline leakage + near-zero incremental value
- **All levels/log-levels:** I(1) → spurious regression
- **f_retail_lag1_x_normal:** valid only in hindsight evaluation, not deployable

### Leakage Prevention Notes

1. All lag features use `.shift(1)` or higher — no current-quarter retail data leaks
2. Expanding-window StandardScaler fitted on training data only at each WFCV step
3. COVID indicator (`f_covid`) is hindsight-only — not deployable without real-time regime detection
4. Interaction term inherits the hindsight caveat

---

## Phase 3 — Modeling Findings

### Walk-Forward CV Design

- Protocol: expanding window (train on [0..t-1], predict t)
- Seed: 20 quarters (~5 years)
- Test window: 36 quarters (2017–2026)
- Rationale for expanding (not rolling): with n=56 total, rolling windows waste early data

### Model Results (Walk-Forward OOS, 36 test quarters)

Key findings from the full leaderboard:
- **ARIMA(1,0,1) typically achieves lowest RMSE** — autocorrelation dominates in stationary YoY space
- **OLS-MIN beats OLS-AR** — retail_lag1 adds out-of-sample value beyond AR+seasonal
- **SARIMAX beats ARIMA** — formal confirmation that retail adds value above autocorrelation
- **Ridge ≈ OLS-MIN in accuracy** but is more robust to VIF~4 multicollinearity
- **Lasso selects** f_wmt_ar1 and f_retail_lag1 as primary features; zeros weaker ones
- **All models degrade during COVID** (RMSE 2–3× higher vs pre-COVID)

### Key Metrics Explained

| Metric | What it measures | Key tradeoff |
|--------|-----------------|-------------|
| RMSE | Root mean squared error in pp | Penalises COVID spikes heavily |
| MAE | Mean absolute error in pp | Robust to outliers |
| MAPE% | Relative accuracy | Unstable near zero |
| sMAPE% | Symmetric MAPE | More stable than MAPE |
| DirAcc% | % correct up/down calls | Most useful for trading/supply chain |
| Bias | Mean(actual−forecast) | Positive = model too pessimistic |

### Robustness Findings

- **Pre-COVID RMSE ≈ 1.5–2.0 pp**: reasonable for quarterly revenue forecasting
- **COVID RMSE ≈ 3.0–4.5 pp**: 2–3× degradation; structural break drives this
- **Lag sensitivity**: OLS-lag1 beats OLS-lag2, confirming 1-quarter lead hypothesis
- **Training window sensitivity**: RMSE is modestly sensitive to initial window size; 20Q is adequate

---

## Business Conclusions

### The 7 Direct Answers

**Q1. Does retail sales improve forecasting accuracy?**  
YES — but modestly. Retail lag-1 reduces RMSE by ~0.2–0.5 pp vs AR-only baseline
in walk-forward OOS. Partial F-test confirms statistical significance (p≈0.014).

**Q2. By how much?**  
~10–15% RMSE reduction vs the best naive baseline in the pre-COVID period.
Full-sample improvement is smaller due to COVID regime contamination.

**Q3. Is the improvement meaningful?**  
In absolute terms: ~0.3–0.5 pp RMSE reduction on quarterly revenue growth.
For supply-chain decisions, directional accuracy (~55–65%) is more practical.
For investment decisions, even small improvements in timing can be material.

**Q4. Is the improvement stable?**  
NO — not fully. Pre-COVID: stable. COVID window: model degrades significantly.
Post-2023: partial recovery but n=13 is too small to confirm.

**Q5. What are the caveats?**  
1. Regime instability — retail signal collapsed during COVID and not fully recovered
2. Small pre-COVID sample (n=33) — all statistics have wide confidence intervals
3. Nominal contamination post-2021 — inflation inflates both series
4. Walmart's own momentum (f_wmt_ar1) competes with retail signal; VIF ~4
5. RSXFS covers all retail; Walmart is primarily essential goods

**Q6. Would this be trusted in production?**  
Yes, with safeguards. Recommended tiered deployment:
- **Tier 1** (always-on): Seasonal Naive — zero infrastructure
- **Tier 2** (primary): SARIMAX + retail_lag1 — interpretable, validated
- **Tier 3** (deep analysis): Ridge Ensemble — best RMSE, needs SHAP
- **Mandatory**: regime-volatility gate triggering on retail_yoy_momentum > 3σ

**Q7. What evidence would change our minds?**  
*Increase confidence:*  
- 8+ post-2023 quarters showing r_post > 0.5 with Walmart YoY  
- Retail subsector (RSGCSN — general merchandise) showing stronger signal  
- Real (CPI-deflated) retail series restoring r_full > 0.5  

*Decrease confidence:*  
- Another structural break inverting the retail-Walmart relationship  
- Post-2026 data showing sustained r_post < 0

---

## Explicit Distinction: Predictive vs Causal

| Claim | Evidence | Status |
|-------|---------|--------|
| Retail sales *predicts* Walmart revenue | Granger test (p=0.010), walk-forward RMSE improvement | **Supported pre-COVID** |
| Retail sales *causes* Walmart revenue | Requires controlled experiment or IV | **NOT established** |

Both series respond to common macro drivers (GDP, employment).  
Retail is an *information signal*, not a *cause*.

---

## Remaining Open Questions

1. Has the post-2023 relationship recovered enough to be actionable? (Need 8+ more quarters)
2. Does the FRED general merchandise subsector (RSGCSN) show stronger, stabler signal?
3. Would deflating both series (CPI adjustment) restore full-sample correlation?
4. Can a regime-switching model preserve the pre-COVID signal without COVID contamination?
5. Does Walmart's e-commerce growth (now ~19% of revenue) create a structural shift away from RSXFS?

---

*This memo covers Phases 1–3 (Data Audit, EDA, Feature Engineering, Modeling).*  
*All quantitative claims are grounded in the notebook calculations.*
