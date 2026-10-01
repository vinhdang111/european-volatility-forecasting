# European Equity Volatility Forecasting

**Can machine learning and deep learning forecast stock market volatility better than classical econometric models?**

This project benchmarks volatility forecasting models, from simple baselines and GARCH to gradient boosting and transformer-based neural networks, on **50 large European stocks and 9 European equity indices** (2000 to today), and evaluates their practical value for risk management through **Value-at-Risk backtesting**.

> 🚧 **Work in progress.** The project is built step by step; see [Project Status](#project-status) below.

---

## Why volatility?

Stock prices are close to unpredictable, but **volatility (how much prices move) is not**. It shows well-documented patterns:

- **Volatility clustering:** turbulent days tend to follow turbulent days
- **Leverage effect:** falling prices increase volatility more than rising prices do
- **Mean reversion:** after a crisis, volatility gradually returns to its long-run level

Volatility forecasts are used every day in finance:

| Use case | Who uses it |
|---|---|
| Value-at-Risk and trading limits | Bank risk management, regulators (Basel) |
| Volatility targeting and risk parity | Asset managers, funds |
| Option pricing | Derivatives traders |
| Hedging decisions | Corporate treasury |
| Margin requirements | Exchanges and clearing houses |

## Research questions

1. Do ML and deep learning models beat classical models (GARCH, HAR) at forecasting volatility at 1-day, 1-week and 1-month horizons?
2. Are the differences **statistically significant**, and do they hold in both calm and crisis periods?
3. Which information matters most (past volatility, price ranges, volume, VIX)?
4. Does a single model trained on all European stocks generalise to markets it has never seen?
5. Do better forecasts translate into better **Value-at-Risk** estimates?

## Data

- **Source:** Yahoo Finance via `yfinance`: daily Open, High, Low, Close, Adjusted Close and Volume
- **Period:** January 2000 to present, covering the dot-com crash, the 2008 financial crisis, the euro debt crisis (2011), Brexit (2016), COVID-19 (2020), the energy and rate shock (2022) and the Credit Suisse collapse (2023)
- **Universe:** 60 series, defined in [`data/universe.csv`](data/universe.csv)

| Type | Count | Content |
|---|---|---|
| Stocks | 50 | Large, liquid European companies across 13 countries and 11 sectors |
| Indices | 9 | Euro Stoxx 50, CAC 40, DAX, FTSE 100, SMI, AEX, IBEX 35, FTSE MIB, OMX Stockholm 30 |
| External | 1 | VIX (US market-fear index) |

**Stocks by country**

| Country | # | Tickers |
|---|---|---|
| France | 8 | MC.PA, TTE.PA, SAN.PA, BNP.PA, AIR.PA, OR.PA, SU.PA, AI.PA |
| Germany | 8 | SAP.DE, SIE.DE, ALV.DE, DTE.DE, BAS.DE, MBG.DE, DBK.DE, BAYN.DE |
| United Kingdom | 7 | SHEL.L, AZN.L, HSBA.L, BP.L, ULVR.L, RIO.L, VOD.L |
| Switzerland | 6 | NESN.SW, NOVN.SW, RO.SW, UBSG.SW, ZURN.SW, ABBN.SW |
| Netherlands | 4 | ASML.AS, INGA.AS, PHIA.AS, AD.AS |
| Spain | 4 | SAN.MC, IBE.MC, ITX.MC, TEF.MC |
| Italy | 3 | ENEL.MI, ENI.MI, ISP.MI |
| Sweden | 3 | VOLV-B.ST, ERIC-B.ST, ATCO-A.ST |
| Denmark | 2 | NOVO-B.CO, MAERSK-B.CO |
| Finland | 2 | NOKIA.HE, NDA-FI.HE |
| Luxembourg | 1 | MT.AS (ArcelorMittal) |
| Norway | 1 | EQNR.OL |
| Belgium | 1 | ABI.BR |

**Stocks by sector:** Banks 8 · Health Care 7 · Industrials 7 · Energy 5 · Consumer Staples 5 · Materials 4 · Technology 4 · Consumer Discretionary 3 · Telecom 3 · Insurance 2 · Utilities 2

### Download pipeline

[`src/data/fetch_prices.py`](src/data/fetch_prices.py) downloads every series and saves:

- `data/raw/prices.parquet`: all prices in long format (`date, ticker, open, high, low, close, adj_close, volume`)
- `data/raw/download_report.csv`: per-ticker status, first and last date, years of history, missing values, zero-volume days and suspicious daily jumps (> 50%)

Design choices:

- **Raw data is kept raw.** Prices are stored unadjusted and unfilled, so every cleaning decision is made explicitly and documented in Step 2.
- **Each series keeps its own trading calendar.** European exchanges have different holidays; no artificial rows are created.
- **Robust downloading.** Each ticker is retried up to 3 times, and a missing ticker does not stop the pipeline.
- **Raw data is not committed** to the repository (see `.gitignore`); anyone can rebuild it with one command.

Known limitations to address in Step 2:

- **Survivorship bias:** the universe contains today's large companies, so firms that disappeared (e.g. Credit Suisse) are excluded.
- **Shorter histories:** some companies were listed, merged or restructured after 2000 (e.g. UBS Group, Shell's unified share line, Mercedes-Benz, Ahold Delhaize).
- **London prices** are quoted in pence and occasionally switch units on Yahoo, creating artificial 100x jumps.
- **VIX** closes after European markets, so it must be lagged by one day to avoid look-ahead bias.

## Methodology (planned)

| Level | Models |
|---|---|
| Baselines | Naive, historical mean, EWMA (RiskMetrics) |
| Econometrics | GARCH(1,1), GJR-GARCH, HAR-RV |
| Machine learning | LightGBM (per-stock and pooled) |
| Deep learning | LSTM / GRU, PatchTST or TFT |

**Evaluation:** walk-forward backtesting (no look-ahead), MSE and QLIKE loss, Diebold-Mariano tests, regime analysis, ablation studies, and VaR backtesting with the Kupiec test.

## Tech stack

- **Language:** Python
- **Data:** pandas, NumPy, yfinance, Parquet
- **Econometrics:** arch, statsmodels
- **Machine learning:** scikit-learn, LightGBM, Optuna, SHAP
- **Deep learning:** PyTorch, neuralforecast
- **Experiment tracking and testing:** MLflow, pytest
- **Visualisation and app:** matplotlib, plotly, Streamlit

## Repository structure

```
european-volatility-forecasting/
├── data/
│   ├── universe.csv    # list of tickers with country, sector, currency
│   ├── raw/            # downloaded price data (not committed)
│   └── processed/      # cleaned datasets
├── notebooks/          # exploration, analysis and results
├── src/
│   ├── data/           # download and cleaning
│   ├── features/       # volatility estimators, targets, features
│   ├── models/         # baselines, econometric, ML and DL models
│   └── evaluation/     # walk-forward backtest, loss functions, tests
├── tests/              # unit tests (incl. data-leakage checks)
├── reports/figures/    # charts used in this README
├── app/                # Streamlit dashboard
├── requirements.txt
└── requirements-dl.txt # deep learning dependencies
```

## Getting started

```bash
git clone https://github.com/vinhdang111/european-volatility-forecasting.git
cd european-volatility-forecasting

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # makes `src` importable from notebooks

python -m src.data.fetch_prices  # download all price data (~1-2 minutes)
```

Deep learning models (Step 8) need extra packages: `pip install -r requirements-dl.txt` (a GPU, e.g. Google Colab, is recommended).

## Project status

| Step | Description | Status |
|---|---|---|
| 0 | Project setup | ✅ Done |
| 1 | Data collection | ✅ Done |
| 2 | Data cleaning and quality checks | ⏳ Next |
| 3 | Exploratory analysis: stylised facts of volatility | ⬜ |
| 4 | Volatility targets and feature engineering | ⬜ |
| 5 | Evaluation framework and baselines | ⬜ |
| 6 | Econometric models (GARCH, GJR-GARCH, HAR) | ⬜ |
| 7 | Machine learning (LightGBM, SHAP) | ⬜ |
| 8 | Deep learning (LSTM, transformer) | ⬜ |
| 9 | Model comparison and statistical analysis | ⬜ |
| 10 | Application: Value-at-Risk backtesting | ⬜ |
| 11 | Interactive dashboard | ⬜ |
| 12 | Final report and documentation | ⬜ |

## Author

**Thanh Vinh Dang**: Accounting and Audit background, moving into data-driven finance.

[LinkedIn](https://www.linkedin.com/in/thanhvinhdang2001) · thanhvinhdang.work@gmail.com
