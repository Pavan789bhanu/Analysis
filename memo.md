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

## What This Means for Forecasting (Next Phase)

The evidence supports building an **OLS model** on **pre-COVID data** using **retail YoY at 1-quarter lag** with **fiscal-quarter fixed effects**, evaluated on a proper **expanding-window out-of-sample test** (no shuffled k-fold). The model must be compared against a **seasonal-naive baseline** (same quarter last year). If it beats that baseline by more than 10% on MAPE in the out-of-sample window, the signal is practically useful. If not, the honest answer is that seasonality already explains most of what can be forecasted, and the retail leading indicator is not adding value at current data availability.

---

*This memo covers Phase 1 (Data Audit) through Phase 4 (EDA Summary). Modeling results will follow.*
