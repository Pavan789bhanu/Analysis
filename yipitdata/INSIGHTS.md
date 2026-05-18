# Walmart × FRED — Analyst notes

A long-form companion to `memo.md`. The memo is the one-pager for a portfolio manager; this document is the depth document for a peer reviewer who wants to see every observation and the reasoning behind every modelling choice.

Numbers in this document are quoted directly from `analysis.py`, `leaderboard.csv`, and `tradeoff_matrix.csv`. They can be reproduced by running `python analysis.py`.

---

## 1. EDA — what the two series actually look like

### 1.1 Data quality

Both files are pristine. No imputation required.

| Series | Rows | Date range | Nulls | Dup dates | Non-positive values |
|---|---|---|---|---|---|
| Walmart quarterly revenue | 65 | 2010-01-31 → 2026-01-31 | 0 | 0 | 0 |
| FRED RSXFS monthly retail | 195 | 2010-01-01 → 2026-03-01 | 0 | 0 | 0 |

### 1.2 Temporal structure — the fiscal-year trap

Walmart's observed month-ends are exclusively `{1, 4, 7, 10}`. This is *not* a calendar quarterly cadence; it's the fiscal calendar of a company whose fiscal year ends 31 January. Concretely:

- Fiscal Q1 = Feb–Apr (period ends April)
- Fiscal Q2 = May–Jul (period ends July)
- Fiscal Q3 = Aug–Oct (period ends October)
- Fiscal Q4 = Nov–Jan (period ends January, **straddling calendar years**)

Any join from monthly FRED to Walmart that uses calendar quarter boundaries will misalign every Walmart Q4 reading. We use a custom `to_fq` helper that maps each Walmart date to its fiscal quarter and fiscal year.

The cadence between Walmart observations is mean 91.3 days (min 89, max 92) — textbook quarterly with no missing quarters. FRED is mean 30.4 days between observations with no gaps. Both files are temporally clean.

### 1.3 Distribution and summary statistics

| Series | Mean | Median | Std | Min | Max | Skew | Kurt |
|---|---|---|---|---|---|---|---|
| Walmart (USD bn) | 133.5 | 128.0 | 21.6 | 99.8 | 185.5 | +0.77 | −0.23 |
| FRED (USD bn) | 452.5 | 423.2 | 103.0 | 302.3 | 651.8 | +0.48 | −1.20 |

Both are mildly right-skewed and broadly distributed — typical of growing nominal time series. Walmart's max is the most recent FY26-Q4 print; FRED's max is the March 2026 release. Log-transforming both before any linear modelling is the right move.

### 1.4 Trend and seasonality (Hyndman strengths)

| Series | Trend strength | Seasonal strength | Conclusion |
|---|---|---|---|
| Walmart | 0.993 | **0.924** | Strong trend AND strong seasonality — NSA |
| FRED RSXFS | 0.993 | **0.024** | Strong trend, essentially no seasonality — **SA** |

**This is the single most important EDA finding.** A retail series with seasonal strength of 0.024 cannot be "raw retail sales" — it must be seasonally adjusted. The FRED file we were handed is the SA version (most likely RSAFS / RSXFS-SA), not the NSA series. Three consequences:

1. We must **not** re-seasonally-adjust FRED.
2. Comparing FRED level to Walmart level — and computing Walmart's share of US retail — is biased: Walmart's holiday-quarter print is the Nov-Dec-Jan spike; FRED's December print has already had that seasonality removed.
3. The clean comparison is **YoY-on-YoY**, which removes seasonality from both sides structurally.

### 1.5 Walmart fiscal-quarter seasonal profile

| Fiscal quarter | Avg revenue (USD bn) | Index vs annual mean |
|---|---|---|
| Q1 (Feb–Apr) | 126.7 | 94.9 |
| Q2 (May–Jul) | 132.6 | 99.3 |
| Q3 (Aug–Oct) | 131.6 | 98.6 |
| Q4 (Nov–Jan) | 142.5 | 106.8 |

Q4 is ~9% above the annual mean every fiscal year. The pattern is highly repeatable — see seasonal strength of 0.924 above.

### 1.6 Stationarity (ADF and KPSS)

ADF null = unit root (non-stationary). KPSS null = stationary. We require both to agree before declaring stationarity.

| Series | Raw | 1st diff | Seasonal diff | log + seasonal diff |
|---|---|---|---|---|
| Walmart | non-stat | non-stat | non-stat | borderline (ADF p=0.14, KPSS p=0.09) |
| FRED | non-stat | **stationary** (ADF p=0.003, KPSS p=0.10) | non-stat | non-stat |

Walmart needs `log + seasonal-diff(4)` plus possibly an extra first-diff to model on a stationary footing. FRED becomes stationary after a single first-diff — entirely consistent with it being SA.

### 1.7 Autocorrelation

After log + seasonal-diff(4), Walmart's ACF decays from ~+0.6 at lag 1 to ~+0.15 by lag 4; PACF spikes at lag 1 (+0.61). An AR(1) on YoY log-revenue is a defensible univariate baseline. FRED after log + seasonal-diff(12) is similar — ACF persistent at lags 1-3, PACF short memory.

### 1.8 Cross-correlation between the two series

Aligned on Walmart's quarterly grid, we computed corr(Walmart-YoY-log-growth, FRED-YoY-log-growth) at lags ±4 quarters:

| Lag (quarters) | Interpretation | Correlation | n |
|---|---|---|---|
| −4 | Walmart leads | +0.194 | 57 |
| −2 | Walmart leads | +0.094 | 59 |
| 0 | contemporaneous | **+0.072** | 61 |
| +2 | FRED leads | +0.214 | 59 |
| +4 | FRED leads | **+0.255** | 57 |

Two things to notice. First, the *contemporaneous* correlation is essentially zero. Whatever shared signal exists is at horizons of a full year or more, not within a quarter. Second, every correlation is below 0.30 — these are not closely-coupled series. The "leading indicator" hypothesis is weak from the start, before we even fit a single model.

---

## 2. Feature engineering — what's available, and what isn't

### 2.1 Walmart features

- `wm_lag{1..4}` — historic revenue. Always available at forecast origin.
- `wm_yoy_lag1 = y_{t-1}/y_{t-5} − 1` — YoY growth of the most recently reported quarter. Always available.
- `is_Q1..Q4` — fiscal-quarter one-hot encoding. Always available.

### 2.2 FRED features — and the publication-lag discipline

The exam brief specifically warns about look-ahead bias. FRED RSXFS publishes around the 15th of the following month — i.e. the September 2025 print becomes public ~Oct 15, 2025. A real-time forecaster standing on Oct 31, 2025 would have FRED through September, **not** October.

Our discipline: for every target date `d`, FRED features use only months with `month_end ≤ d − 1 calendar month`. The first day of `(d − 1 month)` is the cutoff. This is intentionally conservative; shortening the cutoff to `d − 14 days` could flatter FRED's apparent performance but wouldn't survive in production.

The resulting features:

- `fred_last` — level of the latest available month
- `fred_last_yoy` — YoY % change of that month
- `fred_3m_mean` — mean of the latest 3 available months
- `fred_6m_yoy` — mean YoY across the latest 6 available months

We deliberately use **growth-rate** features (YoY) as the primary FRED inputs. The level features are confounded with Walmart's own growing nominal trend — putting both `wm_lag4` and `fred_3m_mean` in the same regression splits trend variance arbitrarily between them.

### 2.3 Merge strategy — monthly FRED to fiscal-quarter Walmart

For each Walmart fiscal-quarter end `d`, sum the three calendar months ending at `d` (this is `fred_quarter_total`). Flag rows whose 3-month window has incomplete FRED coverage (only one — the very first row, 2010-01-31, which lacks Nov-Dec 2009).

This aggregation is **not** the same as `df.resample("QE").sum()`. Calendar QE bins end on March 31 / June 30 / September 30 / December 31; Walmart's quarter-ends are April 30 / July 31 / October 31 / January 31. A calendar QE join would misalign every observation by a month.

### 2.4 Walmart's share of US retail

With the corrected merge, Walmart's share of US retail (Walmart revenue ÷ FRED 3-month total) sits in the ~8.8–10.0% band across recent quarters. Fiscal Q4 (Nov–Jan) is consistently ~50–100 bps higher than the rest, which reflects (a) Walmart's grocery-heavy holiday share gain and (b) the SA-vs-NSA artefact noted in §1.4. Don't read the Q4 bump as pure share gain.

---

## 3. Modelling protocol

### 3.1 Target, horizon, info set

- **Target** `y_t`: Walmart fiscal-quarter revenue for the quarter ending at `date`. We forecast the *level* (USD), not growth, because the exam's customer asks about predicting revenue.
- **Horizon**: 1 fiscal quarter ahead.
- **Forecast origin**: the day Walmart's prior 10-Q became public. Walmart files 10-Qs about three weeks after fiscal-quarter close. Conceptually, the model wakes up the day filings hit EDGAR with `y_{t-1}` newly known and is asked to predict `y_t`.
- **Information set at origin**: Walmart actuals through `y_{t-1}` (i.e. lags ≥ 1), FRED data through `target-quarter-end − 1 month`, plus all calendar/fiscal indicators.

### 3.2 Cross-validation

Rolling-origin walk-forward expanding window:

```
for i in range(MIN_TRAIN_QUARTERS, N):
    train_df = data.iloc[:i]      # everything before quarter i
    test_row = data.iloc[i]       # exactly one held-out quarter
    yhat = model.fit(train_df).predict(test_row)
```

`MIN_TRAIN_QUARTERS = 24` gives ~6 fiscal years of history before any OOS prediction is recorded. The final OOS window is **36 quarters** spanning **2017-04-30 → 2026-01-31** (i.e. FY18-Q1 → FY26-Q4).

Every model re-fits on every step. This is more expensive than re-fitting periodically but it's the honest setup — a production system would do the same.

### 3.3 Metrics

We report four metrics on the same OOS predictions per model:

- **MAPE** — primary, easy to communicate to a PM.
- **RMSE** — penalises large errors, surfaces tail behaviour.
- **sMAPE** — symmetric variant, robust to sign asymmetry. Useful as a sanity check on MAPE.
- **Bias** — mean of (ŷ − y). Tells us whether the model systematically under- or over-shoots.

---

## 4. Baseline & univariate models

### 4.1 seasonal_naive (lag-4)

`ŷ_t = y_{t-4}` — the simplest possible quarterly forecaster. Zero parameters.

- **OOS MAPE 4.08%**, RMSE $7.24 bn, bias **−$6.13 bn**.

The huge negative bias is the giveaway: Walmart is a growing nominal series, so "predict last year's same quarter" systematically under-shoots. This is the baseline the exam brief warned would already be quite good — and on a naive view it isn't bad, but the next baseline takes a chunk out of it for almost no extra work.

### 4.2 seasonal_naive + drift

`ŷ_t = y_{t-4} × (1 + trailing-4q-avg YoY)` — one degree of freedom (the drift), still no ML.

- **OOS MAPE 2.03%**, RMSE $3.94 bn, bias **−$0.13 bn**.

This baseline is the one every more-complex model should be measured against. It uses no FRED, no statsmodels, ~0.16 ms/forecast, and reduces error by half over the lag-4 baseline.

### 4.3 sarima_walmart_only — SARIMAX(1,1,0)(0,1,1,4)

The classical Walmart-univariate forecaster. The (1,1,0)(0,1,1,4) order was chosen from the ACF/PACF in §1.7: one regular AR term, one regular differencing, one seasonal MA term at the quarterly period.

- **OOS MAPE 1.73%**, RMSE $3.79 bn, bias **−$0.06 bn**.

This is the best model on the full OOS window. It costs 11.6 ms per fit+predict (~70× the drift baseline) and adds a statsmodels dependency, but it produces the most accurate Walmart-only forecast.

---

## 5. FRED-augmented models

### 5.1 OLS Walmart + FRED

Same regression specification as `ols_walmart_only` but with FRED columns appended.

- Walmart-only OLS: 2.10% MAPE
- Walmart + FRED OLS: 2.05% MAPE

A 0.05pp absolute improvement on 36 OOS quarters is within noise. Adding FRED to the OLS does not visibly help.

### 5.2 Ridge Walmart + FRED

Same feature set, with L2 regularisation (α=1) on standardised features.

- **OOS MAPE 2.01%** — the best FRED-augmented model on the full window.

Ridge edges OLS by 0.04pp. Still inferior to SARIMA (1.73%).

### 5.3 Gradient Boosting Regressor

`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`. The tree-ensemble probe.

- **OOS MAPE 3.28%**, RMSE $6.18 bn.

GBR underperforms every non-trivial model. This is the classic small-sample failure mode: with ~30 training rows expanding to ~60 by the end of the CV, a 200-tree boosted ensemble overfits the in-sample idiosyncrasies. This is a useful negative result — it confirms that "more complex model" isn't automatically "more accurate" on a dataset this small.

### 5.4 Paired comparison vs the best Walmart-only model (SARIMA)

| FRED model | Mean Δ\|%err\| vs SARIMA (pp) | Share of quarters beats SARIMA |
|---|---|---|
| ols_walmart_plus_fred | **−0.32** | 44% |
| ridge_walmart_plus_fred | **−0.28** | 42% |
| gbr_walmart_plus_fred | **−1.55** | 28% |

All FRED-augmented models have **negative** mean uplift vs SARIMA on the full OOS window — they lose to the Walmart-only competitor on average and they win in fewer than half of OOS quarters.

---

## 6. Regime split — the most useful finding

Splitting the OOS window on 2020-03-01 reveals a different story:

| Model | MAPE pre-2020 (n=12) | MAPE post-2020 (n=24) |
|---|---|---|
| seasonal_naive | 2.41% | 4.92% |
| seasonal_naive + drift | 1.56% | 2.26% |
| sarima_walmart_only | 0.95% | **2.12%** |
| ols_walmart_only | 1.63% | 2.33% |
| **ols_walmart_plus_fred** | **0.87%** | 2.63% |
| **ridge_walmart_plus_fred** | 0.92% | 2.55% |
| gbr_walmart_plus_fred | 3.44% | 3.19% |

**Pre-pandemic**: OLS Walmart + FRED was the **best** model (0.87% MAPE), beating SARIMA (0.95%) and the naive-drift baseline (1.56%) by meaningful margins. The leading-indicator story held.

**Post-pandemic**: every FRED-augmented model is **worse** than SARIMA, by 0.43–1.07 pp. The relationship decoupled — likely because the stimulus / inflation / channel-mix shocks of 2020–2023 hit US-aggregate retail and Walmart on different timelines.

**The falsifiable claim that goes in the memo**: adding FRED RSXFS does not improve out-of-sample Walmart quarterly revenue forecasts over a Walmart-only SARIMA on 2017-Q1 → 2026-Q4, and is strictly worse on the 2020-Q1 → 2026-Q4 sub-sample. We would revisit if (a) 4+ more clean quarters of post-stimulus data showed FRED-augmented models recovering, or (b) a Walmart-specific retail sub-series replaced the aggregate.

---

## 7. Production tradeoffs

### 7.1 Cost / latency benchmark

Wall-clock fit+predict times per OOS step, single-machine, Python 3.10:

| Model | avg fit+predict (ms) | ratio to cheapest non-trivial |
|---|---|---|
| seasonal_naive | ~0 (essentially a lookup) | — |
| seasonal_naive_drift | 0.16 | 1.0× |
| ols_walmart_plus_fred | 0.55 | 3.4× |
| ridge_walmart_plus_fred | 0.61 | 3.8× |
| ols_walmart_only | 0.79 | 4.9× |
| sarima_walmart_only | 11.59 | 72.4× |
| gbr_walmart_plus_fred | 35.44 | 221.5× |

### 7.2 Production recommendation

For most use-cases, **ship `seasonal_naive + drift`**:

- 2.03% MAPE (within 0.30 pp of SARIMA, within 0.02 pp of the best FRED model).
- ~0.16 ms / forecast — sub-millisecond at any practical batch size.
- No statsmodels, no sklearn, no FRED data dependency.
- Interpretability: trivial. A PM can compute it on a napkin.

**Use SARIMA as the system of record only if 0.30 pp of MAPE is worth:**

- A statsmodels dependency (occasional fragility, particularly around regime breaks).
- ~70× the latency (still well within any realistic budget).
- Slightly less interpretability ("the SARIMA said so" is harder to defend than "lag-4 plus 3.5% drift").

**Do not ship any FRED-augmented model on this evidence:**

- No measurable accuracy improvement post-2020.
- Adds a FRED ingestion + monitoring + publication-lag-checking pipeline.
- Introduces new failure modes (FRED release delays, RSXFS methodology changes, NSA-vs-SA confusion).

**Avoid GBR.** Strictly dominated on this dataset — worse MAPE *and* 200× the latency. Tree-ensemble forecasters on n≈30 quarters are an antipattern.

---

## 8. Open questions

1. **Does the post-2020 decoupling recover?** With 4–8 more clean quarters of post-stimulus data, a fresh OOS test might show FRED-augmented models re-beating SARIMA. Worth re-running this analysis annually.
2. **Would a regime-conditional model rehabilitate FRED?** We tested only unconditional models. A model that *learns* when to weight FRED (e.g. a state-space model with regime indicators, or a simple ensemble that switches by regime detector) might extract the pre-2020 signal without the post-2020 noise.
3. **Sub-sector decomposition.** RSXFS is whole-economy retail. A Walmart-specific category mix (grocery + general merchandise, ideally excluding food services) might carry idiosyncratic signal. FRED publishes several sub-series (e.g. RSGCS for grocery, RSGMSN for general merchandise). Worth testing.
4. **Higher-frequency Walmart proxies.** If the goal is in-quarter nowcasting rather than next-quarter forecasting, alternative Walmart proxies (foot-traffic, card-spend, web-scraped pricing) would let FRED's monthly cadence matter more.
5. **Robustness to the publication-lag convention.** Sensitivity analysis: how do the FRED-augmented MAPEs change as we vary the FRED hold-back from 0 to 2 months? If a 0-month hold-back flatters FRED meaningfully, that's evidence the "signal" is partly look-ahead and the apparent gap to SARIMA is even larger than reported.
