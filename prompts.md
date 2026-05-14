# Prompt Log — Retail Sales / Walmart Revenue EDA

This file logs the prompts sent to the LLM assistant during the analysis, along with notes on what worked, where push-back was required, and where human judgment corrected the model's reasoning.

---

## Prompts Used

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

**What worked:** The assistant correctly identified unit mismatch between RSXFS (millions USD) and Walmart revenue (absolute USD), correctly flagged the Walmart fiscal calendar (Feb-Jan FY), and proposed fiscal-quarter-correct aggregation over naive QE resampling.

**Where I had to push back:** The assistant initially computed a 220% YoY retail growth in 2011-Q1 without flagging it as suspicious. It would have propagated a spurious, artifact-driven data quality issue into every downstream correlation calculation without raising any alarm. Human judgment was required to pause and investigate.

---

### Prompt 2 — Debugging the Spurious 220% YoY

```
The retail YoY for January 2011 shows 220% growth. Before accepting this or 
excluding it silently, explain why this might be occurring, what the correct 
handling is, and what the downstream impact would be if we retained it.
```

**What worked:** The assistant correctly diagnosed that the January 2010 Walmart fiscal quarter (FY-Q4) only captured one month of retail data (January 2010), because November and December 2009 fall outside the dataset's start date. The inflated YoY was entirely a boundary artifact.

**Where the LLM failed first:** The initial code used `merged.dropna(subset=['retail_yoy', 'walmart_yoy', 'retail_yoy_lag1', 'retail_yoy_lag2'])`, which incidentally excluded the January 2011 corrupted row (because lag-2 was NaN for it). This made the pre-COVID pearsonr appear to be r=0.855 — but this was an accidental fix, not an intentional one. The code was giving correct-looking numbers for the wrong reason. The correct fix is to explicitly drop the incomplete quarter and document why.

**Manual correction:** Rewrote the quarterly aggregation to count `n_months` per fiscal quarter and explicitly drop any quarter with fewer than 3 months before computing YoY.

---

### Prompt 3 — Cross-Correlation and Lead-Lag Analysis

```
Compute the cross-correlation function (CCF) between retail YoY and Walmart YoY.
Run it on the pre-COVID subsample only. Explain what the CCF pattern means for 
the leading-indicator hypothesis. Be specific about whether the CCF supports a 
clean lead from retail → Walmart or a symmetric co-movement pattern.
```

**What worked:** The assistant correctly identified that the CCF should be run on the pre-COVID subsample rather than the full sample, because the COVID regime inverts the relationship. This matches domain knowledge.

**Where the LLM required clarification:** The initial CCF on the full sample showed near-zero correlations at all lags (the COVID cluster cancels the pre-COVID signal). The assistant initially reported this as evidence that "the signal is weak" — without noting that this conclusion was driven entirely by regime contamination. Human judgment was required to frame this as a regime-sensitivity problem, not a weak-signal problem.

**Key finding surfaced:** The CCF on the pre-COVID subsample showed significant correlations at **both positive and negative lags** — meaning Walmart also "leads" retail by 1–2 quarters. This symmetric structure means both series respond to a common macro driver (GDP, employment). This is a more nuanced finding than "retail leads Walmart" and the LLM initially glossed over it. Push-back required.

---

### Prompt 4 — Granger Causality Testing

```
Run a Granger causality test to formally test whether lagged retail YoY adds 
predictive content for Walmart YoY beyond Walmart's own lags. Run on both the 
full sample and the pre-COVID subsample. Discuss the caveats clearly: small n, 
structural breaks, and the distinction between Granger causality and economic 
causality.
```

**What worked:** The assistant correctly ran Granger on both subsets and surfaced the key contrast: pre-COVID lag-1 is significant (p=0.010), full-sample is not.

**Where the LLM needed correction:** The initial Granger output included a warning about `verbose=False` deprecation. The assistant initially included `verbose=True` output inline in the notebook, which was noisy. Cleaned up by using `verbose=False` and extracting only the F-statistics and p-values.

---

### Prompt 5 — Rolling Correlation

```
Compute rolling 8-quarter Pearson correlations between retail YoY (contemporaneous 
and lag+1) and Walmart YoY. Plot both on the same axis and annotate the COVID 
regime. Interpret the chart in terms of signal stability — is this a reliable 
signal or a fragile one?
```

**What worked:** Rolling correlation chart was the most informative single visualization. The drop from ~0.75 (pre-COVID stable) to –0.71 (COVID inversion) made the regime-instability story visually unambiguous.

**No push-back needed** on this prompt.

---

### Prompt 6 — EDA Summary

```
Synthesize all EDA findings into a structured summary covering: key observations, 
predictive signals, structural risks, data limitations, useful forecasting 
directions, approaches likely to fail, and evidence that would increase confidence 
in the signal. Ground every statement in specific numbers from the data.
```

**What worked:** The assistant produced a well-structured summary. All key findings (regime break, Granger significance, rolling stability) were correctly cited with specific numbers.

**One overstatement corrected:** The assistant initially wrote "retail sales is a leading indicator for Walmart revenue" without qualification. This was corrected to "retail sales was a leading indicator pre-COVID; the relationship is currently unvalidated." This distinction matters for practical use.

---

## Overall LLM Assessment

**Got right:**
- Calendar alignment: Correctly identified fiscal-quarter aggregation as the correct approach over naive calendar-quarter resampling.
- Unit mismatch: Flagged immediately.
- Stationarity framing: Correctly warned against level-on-level regression.
- COVID as regime break (not outlier): Correctly framed the COVID period as a structural break requiring separate analysis.
- Granger test design: Correctly chose to run on pre-COVID subsample.

**Where human judgment was required:**
1. **Spotting the 220% YoY artifact** — LLM was generating numerically valid code that produced a silent data quality error. Without the explicit audit step, this would have flowed through to every downstream correlation.
2. **Interpreting the symmetric CCF** — LLM initially reported lag+1 significance as confirming a leading indicator without noting the symmetric structure that qualifies the interpretation.
3. **Regime contamination of full-sample statistics** — LLM initially framed near-zero full-sample r as "weak signal" rather than "signal destroyed by regime inversion."
4. **Overconfident conclusion framing** — Required push-back to add "pre-COVID only" qualifier to the leading-indicator conclusion.

**Reliability summary:** The LLM was a strong accelerator for code generation, test selection, and report structure. It required human oversight at every interpretive step — the numerical output was usually correct, but the framing and emphasis needed calibration.
