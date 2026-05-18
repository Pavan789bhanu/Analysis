# Prompt log — Walmart × FRED take-home

I used Claude as my coding assistant for this exercise. Below: the prompts I sent, in order, and a reflective note on where the assistant helped, where it had to be corrected, and how I checked its output.

---

## Reflective note (≤ 200 words)

The assistant was strong at the mechanical layers — data loading, statsmodels boilerplate, walk-forward CV scaffolding, and matplotlib styling — and weak in two places that mattered.

**Where I had to push back.** First, on the FRED-to-Walmart join: it initially aggregated FRED to calendar quarters. That is wrong because Walmart's fiscal Q4 spans Nov-Jan, straddling calendar years. I had it switch to a fiscal-quarter mapping (`Feb-Apr=Q1`, `May-Jul=Q2`, `Aug-Oct=Q3`, `Nov-Jan=Q4`). Second, on look-ahead: its first feature builder used FRED months up to the target-quarter end, ignoring publication lag. I tightened the cutoff to `target_date – 1 month` to mirror what an analyst could actually have on the forecast date.

**How I checked it.** I ran `seasonal_decompose` and saw FRED's seasonal_strength was 0.024 — i.e. it is seasonally adjusted, which the assistant had not flagged. That changed my whole comparison strategy to YoY-on-YoY. I also ran the rolling-origin CV separately for pre- and post-2020 — the assistant offered a single full-window number; the regime split is what surfaced the real falsifiable claim ("FRED worked pre-pandemic, stopped working after").

I trusted the assistant for code, not for the framing.

---

## Prompts (chronological)

1. **Exploratory pass.** *"I have attached two CSV files (Walmart quarterly revenue and FRED monthly retail sales). Please act as a senior data scientist with 10+ years of experience. Do a complete end-to-end EDA — data quality, temporal structure, distributions, decomposition, stationarity, autocorrelation. Then feature engineering (lags, rolling, growth) and a merge strategy. Document every observation. Produce a single, well-commented .py file."*

2. **Pivot to modelling.** *"Based on the prior EDA, the customer's question is: does FRED predict Walmart's quarterly revenue better than a naive baseline? Build a baseline first, then try a few candidate models. Compare them on accuracy, training cost, and inference latency. Make the recommendation as a senior data scientist would — think production economics, not just MAPE."*

3. **Confirming protocol.** I picked from a clarification widget: pure exploratory analysis as a deliverable for round 1; for the modelling round I asked for the full bundle (modelling .py + memo + prompt log) and asked the assistant to benchmark accuracy and latency on the real data rather than ship a qualitative tradeoff table.

4. **Format follow-up.** *"I want both a `.py` source and a runnable `analysis.ipynb`. The notebook should run top-to-bottom without errors. The memo audience is a portfolio manager."*

5. **Correcting the join (paraphrased from the conversation).** I prompted the assistant to map every Walmart date to its fiscal quarter before computing share-of-retail, and to NaN-out the first row whose FRED 3-month window was incomplete. The first version inflated Walmart's share to 37% on that row.

6. **Correcting the leakage policy.** I asked the assistant to use only FRED data through `target_date – 1 month`, because RSXFS publishes ~2 weeks after the reference month ends and the safer convention for a real-time forecaster is a one-month hold-back.

7. **Regime split.** I asked the assistant to add a separate pre-2020 vs post-2020 MAPE table after I noticed the post-pandemic forecasts looked noisier than the early ones in the actuals-vs-forecasts plot.

8. **Tradeoff matrix repair.** The first version of the production tradeoff table reported a `latency_vs_cheapest_x` of 11 million because the seasonal-naive baseline had been rounded to 0.00 ms. I asked the assistant to anchor the ratio against the cheapest *non-trivial* model so the column is interpretable.

## Where the assistant got it subtly wrong

- **Calendar vs fiscal quarter join.** First-pass code resampled FRED with `.resample("QE")` and merged on date. That assumes Walmart's quarter-ends are calendar quarter-ends, which they aren't.
- **No FRED publication lag.** First-pass features used FRED data up to the target-quarter end. In production, that data isn't available on the day you'd be forecasting.
- **Seasonality of FRED.** The assistant did not initially notice that RSXFS in this file is seasonally adjusted. I caught it by reading the decomposition output, then re-framed every cross-series feature in YoY-on-YoY terms.

## What it got right

- Clean walk-forward CV scaffolding, with re-fit on every step.
- Sensible default SARIMAX orders for a quarterly series.
- Honest paired-comparison reporting (mean Δ|%err|, share of quarters where each model wins) rather than a full DM test that would be statistically dicey at n = 36.
- Useful default model set: two naive baselines, one univariate, three FRED-augmented covering OLS / Ridge / GBR.
- Production-perspective tradeoff matrix that explicitly names the failure mode of each model.
