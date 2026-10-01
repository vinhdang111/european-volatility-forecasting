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

### Storage: PostgreSQL

All data is stored in a local **PostgreSQL** database (`volatility_db`). The schema is defined in [`sql/schema.sql`](sql/schema.sql):

```
universe ──┬──< prices_raw          (ticker, date)  raw daily OHLCV from Yahoo
           ├──< prices_clean        (ticker, date)  cleaned prices (Step 2)
           ├──< cleaning_log                        every correction made (Step 2)
           └──< download_log >──── download_runs    audit trail of every download
```

| Table / view | Grain | Content |
|---|---|---|
| `universe` | 1 row per ticker | Name, asset type, country, exchange, sector, currency (loaded from `data/universe.csv`) |
| `prices_raw` | 1 row per ticker and trading day | Open, high, low, close, adjusted close, volume, download timestamp |
| `download_runs` | 1 row per run | Date range requested, start/end time, tickers and rows downloaded |
| `download_log` | 1 row per ticker per run | Status, first/last date, years of history, missing values, zero-volume days, suspicious jumps (> 50%) |
| `latest_download_report` *(view)* | 1 row per ticker | Quality report of the most recent run |
| `prices_clean` | 1 row per ticker and trading day | Cleaned prices, with sanity constraints (Step 2) |
| `cleaning_log` | 1 row per change | Ticker, date, rule, action and detail of every correction (Step 2) |
| `cleaning_summary`, `data_coverage` *(views)* | 1 row per rule / ticker | Summary of the cleaning; rows kept and dropped per ticker |

### Download pipeline

1. [`src/data/setup_db.py`](src/data/setup_db.py) creates the database and tables and loads the universe.
2. [`src/data/fetch_prices.py`](src/data/fetch_prices.py) downloads every series from Yahoo Finance and writes it to `prices_raw`, logging each ticker in `download_log`.

Design choices:

- **Raw data is kept raw.** Prices are stored unadjusted and unfilled, and `prices_raw` deliberately has no sanity constraints (e.g. `high >= low`): wrong values must be stored so they can be detected, logged and fixed in Step 2.
- **Idempotent loads.** Prices are written with an *upsert* (`INSERT ... ON CONFLICT DO UPDATE`) on the primary key `(ticker, date)`, so re-running the download updates rows instead of duplicating them.
- **Referential integrity.** Every price and log row must reference a ticker in `universe` (foreign keys).
- **Auditability.** Every run and every ticker's outcome is recorded, so data issues can be traced back to a specific download.
- **Each series keeps its own trading calendar.** European exchanges have different holidays; no artificial rows are created.
- **Robust downloading.** Each ticker is retried up to 3 times, and a missing ticker does not stop the pipeline.
- **Credentials stay local.** Connection settings live in a `.env` file that is never committed (template: [`.env.example`](.env.example)).

## Data quality

Raw Yahoo Finance data contains several kinds of errors. [`src/data/clean_prices.py`](src/data/clean_prices.py) rebuilds `prices_clean` from `prices_raw` with seven documented rules, and records **every change** (ticker, date, rule, action, detail) in the `cleaning_log` table. The full investigation is in [`notebooks/01_data_quality.ipynb`](notebooks/01_data_quality.ipynb).

![Rows affected by each cleaning rule](reports/figures/02_cleaning_rules.png)

| Issue found | Example | Rule | Action |
|---|---|---|---|
| Holidays and missing days filled with an old price (zero volume) | Christmas, 1 May, Good Friday | `stale_bar` | Row dropped |
| Frozen early history | Shell before the July 2005 unification: no trade on >90% of days in 2000 | `before_valid_from` | Rows dropped (`valid_from` in `universe.csv`) |
| Isolated bad prints that revert the next day | Novo Nordisk: 33 prices 2-5x away from their neighbours | `price_spike` | Row dropped |
| Open or close outside the [low, high] range | Mostly before 2005 | `range_repair` | Range widened |
| Close-only bars (open = high = low = close) | FTSE MIB until 2003, Vodafone until 2002 | `close_only_bar` | Open/high/low set to NULL |
| Negative adjusted close | AB InBev 2000-2008, Inditex 2001-2002 | `invalid_adj_close` | Set to NULL |
| Session still trading when downloaded | Data downloaded during market hours | `incomplete_session` | Row dropped |

Only **1.1% of rows are dropped**. Genuine extreme moves are kept, e.g. ABB's -62% on 22 Oct 2002 (asbestos crisis) or Rio Tinto's -37% on 25 Nov 2008. The spike filter only removes prices that immediately revert.

Why it matters: a handful of bad prints is enough to make volatility estimates meaningless.

![Novo Nordisk: raw vs clean prices and volatility](reports/figures/02_novo_spikes.png)

Design decisions:

- **Returns will be computed from `close`**, which Yahoo adjusts for stock splits. `adj_close` is unreliable (negative values, 25-40% one-day jumps in the adjustment factor) and is kept for reference only.
- **Database constraints** on `prices_clean` (`close > 0`, `high >= low`, open and close within the range…) guarantee that no inconsistent row can be stored.
- **Unit tests** ([`tests/test_clean_prices.py`](tests/test_clean_prices.py)) check each rule on synthetic data, including that genuine crashes and VIX spikes are *not* removed.

Known limitations:

- **Survivorship bias:** the universe contains today's large companies, so firms that disappeared (e.g. Credit Suisse) are excluded.
- **Shorter histories:** some series start after 2000 (e.g. Ahold Delhaize, ArcelorMittal, Euro Stoxx 50, OMX Stockholm 30 on Yahoo).
- **Coarse price ticks in the early 2000s** produce more unchanged-price days (e.g. Equinor), which slightly lowers measured volatility in that period.
- **VIX** closes after European markets, so it must be lagged by one day when used as a feature (Step 4).

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
- **Data:** pandas, NumPy, yfinance
- **Database:** PostgreSQL, SQLAlchemy, psycopg2
- **Econometrics:** arch, statsmodels
- **Machine learning:** scikit-learn, LightGBM, Optuna, SHAP
- **Deep learning:** PyTorch, neuralforecast
- **Experiment tracking and testing:** MLflow, pytest
- **Visualisation and app:** matplotlib, plotly, Streamlit

## Repository structure

```
european-volatility-forecasting/
├── data/
│   └── universe.csv    # list of tickers with country, sector, currency
├── sql/
│   └── schema.sql      # PostgreSQL tables, keys, constraints and views
├── notebooks/          # analysis notebooks (01_data_quality, ...)
├── src/
│   ├── db.py           # database connection (reads .env)
│   ├── data/           # database setup, download and cleaning
│   ├── features/       # volatility estimators, targets, features
│   ├── models/         # baselines, econometric, ML and DL models
│   └── evaluation/     # walk-forward backtest, loss functions, tests
├── tests/              # unit tests (incl. data-leakage checks)
├── reports/figures/    # charts used in this README
├── app/                # Streamlit dashboard
├── .env.example        # template for database credentials
├── requirements.txt
└── requirements-dl.txt # deep learning dependencies
```

## Getting started

**Prerequisites:** Python 3.10+ and [PostgreSQL](https://www.postgresql.org/download/) running locally.

```bash
git clone https://github.com/vinhdang111/european-volatility-forecasting.git
cd european-volatility-forecasting

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # makes `src` importable from notebooks
```

Copy `.env.example` to `.env` and enter your PostgreSQL password, then:

```bash
python -m src.data.setup_db      # create database, tables and load the universe
python -m src.data.fetch_prices  # download all price data into PostgreSQL (~2-3 minutes)
python -m src.data.clean_prices  # clean the prices (prices_clean + cleaning_log)
python -m pytest                 # run the unit tests
```

Deep learning models (Step 8) need extra packages: `pip install -r requirements-dl.txt` (a GPU, e.g. Google Colab, is recommended).

## Project status

| Step | Description | Status |
|---|---|---|
| 0 | Project setup | ✅ Done |
| 1 | Data collection | ✅ Done |
| 2 | Data cleaning and quality checks | ✅ Done |
| 3 | Exploratory analysis: stylised facts of volatility | ⏳ Next |
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

**Thanh Vinh Dang**

[LinkedIn](https://www.linkedin.com/in/thanhvinhdang2001) · thanhvinhdang.work@gmail.com
