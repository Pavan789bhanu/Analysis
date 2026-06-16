# Walmart Revenue Forecasting Analysis

> **Can FRED U.S. Retail Sales (RSXFS) predict Walmart quarterly revenue better than a fair baseline?**

A rigorous time-series forecasting study evaluating whether the FRED retail sales index can anticipate Walmart's quarterly revenue growth — using walk-forward out-of-sample validation across 39 quarters (2016 Q3 – 2026 Q1).

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-4C72B0?style=flat-square)](https://www.statsmodels.org/)

---

## Overview

This project tests whether the **FRED RSXFS** (Advance Monthly Sales for Retail and Food Services) index can serve as a **leading indicator** for Walmart's quarterly revenue — evaluated against strict baselines using leakage-safe, walk-forward expanding-window forecasting.

### Key Finding

**Walmart's own quarterly momentum (AR-only OLS) beats every retail-augmented model out-of-sample.**

| Model | Full-Sample OOS RMSE |
|-------|---------------------|
| **AR-only** (Walmart YoY lag-1 + fiscal dummies) | **2.46 pp** |
| Seasonal-mean baseline (B1) | 2.85 pp |
| OLS + retail signal (M1) | 2.97 pp |

The retail signal shows promise pre-COVID (in-sample r = 0.78 at lag +1) but **inverts during COVID** (rolling correlation drops to −0.71) and does not survive strict out-of-sample testing against Walmart's own momentum.

**Production recommendation:** AR-only + fiscal-quarter seasonality. Monitor retail as a regime indicator, not a default forecast input.

---

## Repository Structure

```
Analysis/
├── analysis_updated.ipynb   # Full analysis pipeline (Phases 1–6)
├── analysis_old.ipynb       # Archived earlier version
├── dashboard.html           # Interactive results dashboard
├── memo.md                  # Executive summary for portfolio management
├── prompts.md               # LLM prompt log (methodology transparency)
├── requirements.txt         # Python dependencies
└── data/
    ├── retail_sales_fred.csv    # FRED RSXFS monthly retail sales
    └── walmart_revenue.csv        # Walmart quarterly revenue (fiscal)
```

---

## Methodology

### Data

| Series | Source | Frequency | Period |
|--------|--------|-----------|--------|
| FRED RSXFS | `data/retail_sales_fred.csv` | Monthly | Jan 2010 – Apr 2026 |
| Walmart Revenue | `data/walmart_revenue.csv` | Quarterly (fiscal) | Jan 2010 – Jan 2026 |

Walmart fiscal quarters end **Jan 31 / Apr 30 / Jul 31 / Oct 31** — all FRED monthly observations are mapped to the correct fiscal quarter before aggregation.

### Models Tested

**Baselines:**
- **B1** — Fiscal-quarter expanding mean (seasonality only)
- **B2** — Flat historical mean
- **B3** — OLS on Walmart YoY lag-1 + fiscal-quarter dummies (AR-only)

**Signal Models:**
- **M1** — OLS + retail YoY feature
- **M2** — Ridge + retail
- **M3** — Regime-aware OLS (`retail × normal` interaction)
- **M4** — OLS with retail lag-2

### Evaluation Protocol

- **Walk-forward expanding window** — one quarter ahead, retraining each step
- **39 strictly out-of-sample quarters** (2016 Q3 – 2026 Q1)
- Minimum 20 training quarters required per forecast
- Results broken down by regime: Pre-COVID / COVID / Post-2023

---

## Results by Regime

| Regime | AR-only RMSE | OLS+Retail RMSE | Verdict |
|--------|-------------|-----------------|---------|
| Pre-COVID (n=14) | **1.42 pp** | 1.65 pp | AR-only wins |
| COVID (n=12) | **3.15 pp** | 4.24 pp | Retail actively misleads |
| Post-2023 (n=13) | **2.62 pp** | 2.64 pp | Essentially tied |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/Pavan789bhanu/Analysis.git
cd Analysis
pip install -r requirements.txt
```

### Run the Analysis

```bash
jupyter notebook analysis_updated.ipynb
```

### View the Dashboard

Open `dashboard.html` in any browser for an interactive summary of KPIs, model comparison charts, and regime breakdowns.

---

## Key Risks & Caveats

- **Regime break (2020):** Broad retail crashed ~17% in April 2020 while Walmart's essential-goods mix kept revenue growing — the signal inverted.
- **Fiscal vs. calendar quarter mismatch:** Naive calendar-quarter merging mis-aligns two-thirds of the data.
- **Inflation confounding:** Post-2021 both series are nominal; long-rolling-mean features show nominal-trend contamination.
- **Sample size:** 39 OOS forecasts is sufficient for directional verdicts but not tight confidence intervals.

See [`memo.md`](memo.md) for the full executive summary and [`prompts.md`](prompts.md) for the complete LLM-assisted methodology log.

---

## Author

**Pavan Kumar Malasani** — AI/ML Engineer

[GitHub](https://github.com/Pavan789bhanu) · [Portfolio](https://pavan-portfolio.com) · [LinkedIn](https://www.linkedin.com/in/pavan789bhanu/)
