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

---

## Phase 3 — Forecasting Results

### Methodology

**Expanding-window (walk-forward) validation.** Train on all available history through quarter *t*, forecast quarter *t+1*, repeat. Minimum 20 training quarters before first forecast. No shuffled splits. All features use only data observable before Walmart reports (lag-1 retail is 100% safe per Phase 2 publication-lag audit).

**39 out-of-sample test quarters** (Jul 2016 – Jan 2026): 14 pre-COVID, 12 COVID/recovery, 13 post-2023.

### Baselines

| Baseline | Description | Role |
|---|---|---|
| B1: Seasonal Naive | Expanding mean of same fiscal quarter's YoY | Naive benchmark |
| B2: Historical Mean | Expanding overall mean of Walmart YoY | Flat benchmark |
| B3: AR-Only OLS | AR(1) + seasonal dummies; uses Walmart own history | Hard benchmark |

### Model Performance (Out-of-Sample RMSE in percentage points)

| Model | Full | Pre-COVID | COVID | Post-2023 |
|---|---|---|---|---|
| Seasonal Naive [B1] | 2.85 | 1.74 | 3.35 | 3.28 |
| AR-Only OLS [B3] | **2.46** | **1.42** | **3.15** | **2.62** |
| OLS Signal [M1] | 2.97 | 1.65 | 4.24 | 2.64 |
| Ridge Signal [M2] | 2.92 | 1.59 | 4.16 | 2.63 |

Bold = best in each period.

### The Answer to the Business Question

**Does retail sales predict Walmart revenue better than a naive baseline?**

- **vs. Seasonal Naive (B1):** Yes in pre-COVID (+5% RMSE improvement) and post-2023 (+20%). No during COVID (35% worse). Even-ish full-sample.
- **vs. AR-Only (B3, the honest baseline):** No — consistently worse in every period. Full-sample: OLS Signal RMSE=2.97 vs AR-Only RMSE=2.46 (-21%). Pre-COVID: 1.65 vs 1.42 (-16%).

**The retail signal does not beat Walmart's own momentum once AR(1)+seasonal is included as the baseline.** The AR-only model — using only Walmart's own history and seasonal effects — outperforms every retail-augmented model in every period.

### Why Retail Fails to Beat AR-Only

1. **COVID inversion:** Retail crashed 17% in Apr 2020; Walmart held up (essential goods). Retail lag-1 transmitted the wrong signal during COVID, causing OLS Signal RMSE to spike to 4.24 vs AR-Only's 3.15.
2. **Shared macro driver:** Both retail and Walmart respond to the same GDP/employment cycle. Once Walmart's own AR captures that shared signal, retail adds noise, not information.
3. **Directional accuracy:** AR-Only correctly predicts acceleration/deceleration more often than OLS Signal in every regime.

### Caveats and Worries

1. **COVID instability is the primary risk.** Any model using retail as an input must handle the event that retail crashes but Walmart does not — or vice versa.
2. **Small sample.** 14 pre-COVID and 13 post-2023 OOS quarters. Directional conclusions are robust but individual RMSE comparisons have wide confidence bands.
3. **Post-2023 is tentatively favorable.** OLS Signal and AR-Only are nearly tied (2.64 vs 2.62) post-2023. With 8 more quarters of clean data, this could justify revisiting the retail signal.
4. **Inflation confound post-2021.** Both series carry nominal effects. A common inflation factor may inflate apparent correlation without signaling real demand.

### Recommendation

**Use AR-Only (Walmart AR1 + seasonal dummies) as the production baseline.** It outperforms every retail-augmented model and requires no external data.

**Keep monitoring the retail signal** as a supplementary indicator: track rolling correlation between retail YoY lag-1 and Walmart YoY. If it exceeds 0.5 sustained over 8 quarters post-2023, re-test whether adding retail materially improves the AR model.

**Do not add retail to a production model** until it demonstrably beats the AR-Only baseline on a further 2+ years of post-COVID OOS data.

### What Evidence Would Change Our Conclusion

1. Post-2023 rolling correlation recovering to 0.5+ sustained over 8 quarters.
2. OOS RMSE improvement >10% over AR-Only on the 2023-2026 window (need ~6 more quarters).
3. General-merchandise FRED subsector (RSGCSN) showing tighter, more stable lag relationship.
4. Regime-switching model that automatically suppresses retail during stress events.

---

*Analysis complete. Phases 1-4 (EDA), Phase 2 (Feature Engineering), and Phase 3 (Modeling and Evaluation) are all documented in analysis.ipynb.*
