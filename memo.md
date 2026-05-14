# Memo: Can FRED Retail Sales Lead Walmart Quarterly Revenue?

**To:** Portfolio Manager  
**From:** Data Science  
**Date:** May 2026  
**Subject:** EDA Findings — Retail Sales as a Leading Indicator for Walmart Revenue

---

## The Question

We examined whether the monthly U.S. retail sales index from FRED (RSXFS) can predict Walmart's quarterly revenue before Walmart reports it. A useful leading indicator would move before Walmart's revenue does, giving us a forward-looking signal.

---

## Bottom Line Up Front

**Yes — but only in normal economic conditions, and the COVID regime broke it.**

Before COVID, retail YoY growth at a one-quarter lag had a strong, statistically significant relationship with Walmart YoY revenue growth (r = 0.78, Granger F = 7.61, p = 0.010). During COVID and recovery (2020–2022), the relationship **inverted** completely (r = –0.60). Since 2023, it has not recovered (r = 0.12, not significant).

Any model built on the full history will detect essentially no signal (r = 0.04). The signal only exists within the pre-COVID macro regime.

---

## What the Data Shows

| Regime | Quarters | Contemp r | Lag+1 r | Granger p (lag 1) |
|---|---|---|---|---|
| Pre-COVID (2011–Jan 2020) | 36 | **0.87** ★ | **0.78** ★ | **0.010** ★ |
| COVID/Recovery (2020–2022) | 11 | **–0.60** | –0.46 | n/a |
| Post-2023 | 13 | 0.12 | 0.19 | n/a |
| Full sample | 60 | 0.04 | 0.09 | 0.50 |

★ = statistically significant at p < 0.05

**Pre-COVID rolling correlation** was consistently high (mean = 0.75, std = 0.12) — a genuinely stable signal.  
**COVID rolling correlation** dropped to –0.71 — the signal didn't just weaken; it reversed sign.

---

## Why the Relationship Broke During COVID

RSXFS covers **all** U.S. retail. Walmart is primarily **essential** goods — food, household supplies, pharmacy. When COVID hit in spring 2020, general retail (restaurants, apparel, electronics) collapsed, pulling RSXFS down 17% in two months. Walmart's essential-goods business held up and then accelerated. The two series decoupled at precisely the moment the forecasting relationship mattered most.

---

## Key Risks and Caveats

**1. Regime instability is the primary risk.** A model calibrated on pre-COVID data will fail in the next economic stress event if it follows a similar essential-goods bifurcation.

**2. Both series share a macro driver.** The correlation is partly because both respond to the same underlying GDP and employment cycle — retail does not uniquely lead Walmart. Walmart's own lagged revenue carries nearly as much predictive information.

**3. Post-2021 inflation confounding.** After 2021, both series are nominal. A common inflation factor may inflate the apparent correlation between them even when real consumer volume diverges.

**4. Look-ahead bias is manageable but real.** Walmart reports 2–4 weeks after quarter-end. Most FRED retail data for the preceding quarter is released before Walmart reports, but the final monthly estimate carries revision risk of ±0.5%. In a real-time model, use only the estimates available as-of the prediction date.

**5. Broad index vs. subsector.** RSXFS measures all retail. Walmart's revenue mix (food/general merchandise) is better matched to a subsector index. This data was not available in the provided dataset.

---

## Open Questions

- Has the post-2023 relationship recovered enough to be useful? (Need 8+ more quarters to test.)
- Does the general-merchandise FRED subsector (RSGCSN) show a stronger, more stable lag structure?
- Would deflating both series (CPI adjustment) restore the post-2020 relationship?
- Can a regime-switching model (with a COVID indicator) preserve the pre-COVID signal without contamination?

---

---

## Phase 2 — Feature Engineering Decisions

### Transformation Choice

Both series are I(1) in levels — regressing levels on levels produces spurious results. **YoY % growth rate is the correct transform**: near-stationary, interpretable ("demand accelerating vs. last year"), and removes both trend and seasonal effects.

### Aggregation

Monthly retail sums to quarterly using Walmart's fiscal-calendar boundaries (Nov-Jan, Feb-Apr, May-Jul, Aug-Oct). Sum and mean are mathematically identical for YoY growth rate analysis. The **first month of the dataset (Jan 2010) was an incomplete quarter** — using it would create a spurious +220% YoY in 2011. Explicitly dropped.

### Publication Lag

The last month of every Walmart fiscal quarter is published by FRED approximately 3-4 days before Walmart reports earnings. All months are technically available, but the advance estimate carries ±0.5% revision risk. For a genuine **forecasting** application, only the **prior quarter's retail data (lag-1)** is entirely safe. For nowcasting on the day of earnings, the current-quarter sum is usable with documented caveats.

### Recommended Feature Set

**Five features for the minimal OLS model (pre-COVID):**

| Feature | Type | r (pre-COVID) | Notes |
|---|---|---|---|
| `f_retail_lag1` | Retail YoY lag-1 | 0.78 ★ | Primary signal; 100% leakage-safe |
| `f_wmt_ar1` | Walmart YoY lag-1 | 0.74 ★ | AR component; most stable predictor |
| `f_q1/f_q2/f_q3` | Seasonal dummies | n/a | Essential: FY-Q4 is ~8% above FY mean |

### Model Performance (Pre-COVID In-Sample)

| Model | R² | Adj-R² | AIC |
|---|---|---|---|
| AR(1) + seasonal only | 0.564 | 0.504 | 134.8 |
| **+ retail_lag1** | **0.649** | **0.587** | **129.4** |

Adding `retail_lag1` increases R² by **+0.085**. Partial F-test: **F=6.82, p=0.014** — statistically significant incremental contribution. Coefficient: **0.74** (p=0.014), meaning a 1 pp higher retail YoY last quarter predicts 0.74 pp higher Walmart YoY this quarter, after controlling for AR and seasonality.

### Features Rejected

- **Rolling 8Q mean**: full-sample r=0.39 (significant) but pre-COVID r=0.07 — captures nominal trend/inflation, not consumer signal. **Spurious.**
- **Rolling volatility (4Q std)**: r≈0 in both samples.
- **QoQ growth lag-1**: r_pre=0.31 (marginal), high noise.
- **Intra-quarter momentum**: r≈0, no value.
- **Levels/log-levels**: I(1) — spurious regression.

### Leakage Prevention Notes

1. All lag features use `.shift(1)` or higher — no current-quarter retail data leaks into the feature.
2. The regime indicator (`f_covid`) is flagged as hindsight-only — not deployable without a real-time regime-detection rule.
3. The interaction term (`retail_lag1 × non-COVID`) inherits the hindsight caveat.
4. Expanding-window z-score normalization uses only past data at each point in time.

### Remaining Concerns After Phase 2

- All model evidence is in-sample (pre-COVID). OOS performance on post-COVID data is unknown and is the key risk.
- n=33 usable pre-COVID observations with 5 parameters gives adequate but not comfortable degrees of freedom.
- f_wmt_ar1 and f_retail_lag1 are correlated (VIF ~3.6), widening individual coefficient confidence intervals even when joint fit is reasonable.
- Post-COVID regime (2023+, n=13): signal has not recovered (r=0.12). The model's real-world deployability is currently unvalidated.

---

## What This Means for Forecasting (Next Phase)

The evidence supports building an **OLS model** on **pre-COVID data** using **retail YoY at 1-quarter lag** with **fiscal-quarter fixed effects**, evaluated on a proper **expanding-window out-of-sample test** (no shuffled k-fold). The model must be compared against a **seasonal-naive baseline** (same quarter last year). If it beats that baseline by more than 10% on MAPE in the out-of-sample window, the signal is practically useful. If not, the honest answer is that seasonality already explains most of what can be forecasted, and the retail leading indicator is not adding value at current data availability.

---

*This memo covers Phase 1 (Data Audit) through Phase 4 (EDA Summary). Modeling results will follow.*
