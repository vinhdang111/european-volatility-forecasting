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
           ├──< features            (ticker, date)  daily variance, features, targets (Step 4)
           ├──< forecasts           (horizon, ticker, date)  out-of-sample forecasts, one column per model (Step 5)
           ├──< model_scores >───── models          losses per model, horizon, ticker and test year (Step 5)
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
| `features` | 1 row per stock / index and trading day | Daily variance, 18 features and 3 forecast targets (Step 4) |
| `model_dataset` *(view)* | 1 row per stock / index and trading day | `features` joined with sector, country and asset type, plus volatilities in % |
| `forecasts` | 1 row per horizon, ticker and forecast date | Out-of-sample variance forecasts, one column per model (Step 5) |
| `forecasts_long`, `forecast_vs_actual` *(views)* | 1 row per model, horizon, ticker and date | The same forecasts in long format; forecast and realised volatility in % side by side (for Power BI) |
| `models` | 1 row per model | Model name, family and description |
| `model_scores` | 1 row per model, horizon, ticker and test year | QLIKE, log MSE and MAE, computed in SQL on the common sample ([`sql/score_models.sql`](sql/score_models.sql)) |
| `model_scores_overall` *(view)* | 1 row per model and horizon | Overall losses and rank of each model |

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

Raw Yahoo Finance data contains several kinds of errors. [`src/data/clean_prices.py`](src/data/clean_prices.py) rebuilds `prices_clean` from `prices_raw` with nine documented rules, and records **every change** (ticker, date, rule, action, detail) in the `cleaning_log` table. The full investigation is in [`notebooks/01_data_quality.ipynb`](notebooks/01_data_quality.ipynb).

![Rows affected by each cleaning rule](reports/figures/02_cleaning_rules.png)

| Issue found | Example | Rule | Action |
|---|---|---|---|
| Holidays and missing days filled with an old price (zero volume) | Christmas, 1 May, Good Friday | `stale_bar` | Row dropped |
| Frozen early history | Shell before the July 2005 unification: no trade on >90% of days in 2000 | `before_valid_from` | Rows dropped (`valid_from` in `universe.csv`) |
| Isolated bad prints that revert the next day | Novo Nordisk: 33 prices 2-5x away from their neighbours | `price_spike` | Row dropped |
| Intraday prices on a different basis than the close | Vodafone before mid-2007: range-based volatility 10 to 100 times too high | `unreliable_range` | Open/high/low set to NULL (`range_valid_from` in `universe.csv`) |
| Impossible high or low | Shell and BP in 2019: low quoted in pounds instead of pence; bad prints 15%+ beyond both open and close | `implausible_range` | Open/high/low set to NULL |
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
- **VSTOXX**, the European equivalent of the VIX, would be the closer match for European stocks, but it is not available on Yahoo Finance and has no free, automatable source, so the project uses the VIX as its implied-volatility feature.

## Key findings from the exploratory analysis

[`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb) checks the *stylised facts* that volatility models rely on, using 331,000 daily stock returns (2000 to September 2026). Each finding drives a design choice in the following steps.

**Volatility moves between calm regimes and short, violent crises.** The volatility of the median stock ranges from 12% to 101% (3 November 2008). It averaged 40% during the financial crisis and 51% during COVID-19, against 22% outside the shaded periods.

![21-day volatility of 50 European stocks](reports/figures/03_volatility_regimes.png)

**Volatility is forecastable, returns are not.** Daily returns have no memory (autocorrelation ≈ 0), but their size does: the autocorrelation of absolute returns is 0.23 at one day and still 0.09 after 100 days.

![Autocorrelation of returns and absolute returns](reports/figures/03_volatility_clustering.png)

**Falls raise volatility more than rises.** After a drop of more than 4%, volatility over the next five days averages 54%, against 48% after a rise of the same size.

![Leverage effect](reports/figures/03_leverage_effect.png)

| Finding | Evidence | Consequence for the project |
|---|---|---|
| Fat tails | Moves beyond 5 standard deviations are about 4,700 times more frequent than under a normal distribution | Robust loss function (QLIKE); VaR backtesting |
| Clustering and long memory | Absolute returns stay autocorrelated for 100+ days | HAR (day / week / month) as the benchmark to beat |
| Regimes | Median-stock volatility between 12% and 101% | Results reported separately for calm and crisis periods |
| Leverage effect | Falls are followed by more volatility than rises | Negative-return features; GJR-GARCH |
| Skewness | Skewness 3.1 for volatility, 0.5 for its logarithm | Models forecast log volatility |
| Co-movement | Average correlation of 0.87 between index volatilities; lagged VIX as informative as an index's own past volatility | One pooled model for all series; VIX feature tested by ablation |
| Sector levels | From 23% (consumer staples) to 43% (technology) | Per-series scaling or sector features |

## Volatility measure, targets and features

Details and checks in [`notebooks/03_features.ipynb`](notebooks/03_features.ipynb); code in [`src/features/`](src/features/).

**Measure.** Volatility on a single day cannot be observed. The squared daily return is unbiased but extremely noisy, so the project measures daily variance from the full open/high/low/close information:

> `rv` = (overnight return)² + Garman-Klass variance of the trading session

with a fallback to the squared close-to-close return when the intraday range is missing (0.9% of rows). The range-based measure has the same average level as the close-to-close one (33% against 32% annualised for stocks) but is far less noisy: its day-to-day autocorrelation is 0.58, against 0.15 for squared returns.

![Two measures of daily volatility during the COVID-19 crash](reports/figures/04_volatility_estimators.png)

**Targets.** For each horizon of 1, 5 and 22 trading days: the logarithm of the average daily variance over the following *h* days.

**Features.** 18 variables, each motivated by a finding of the exploratory analysis:

| Group | Features | Motivation |
|---|---|---|
| Own volatility | Day, week, month, quarter, long-run level | Clustering and long memory (the first three form the HAR model) |
| Returns | Daily, weekly, monthly return; negative return; share of variance from down days | Leverage effect |
| Instability, activity | Volatility of volatility; volume relative to its monthly average | Unstable or unusually active markets |
| Market | Volatility of the median European stock (day, week, month) | Co-movement across markets |
| Implied volatility | VIX level and 5-day change, lagged one US session | Forward-looking expectations |
| Calendar | Day of the week | Seasonality |

The final dataset has **371,839 complete rows** (59 series, December 2000 to August 2026).

**No look-ahead.** A row dated *t* contains only what is known at the close of day *t*; targets use days *t+1* onwards. This is enforced by unit tests ([`tests/test_features.py`](tests/test_features.py)): the features are rebuilt on data truncated at a date *T* and must be identical to the full-sample features up to *T*, and changing today's prices must leave today's targets unchanged. The VIX is taken from the last US session strictly before *t*, because the US market closes after Europe.

## Evaluation framework and baselines

Every model of the project is evaluated by the same code, on the same dates and with the same loss functions. Details in [`notebooks/04_baselines.ipynb`](notebooks/04_baselines.ipynb); code in [`src/evaluation/`](src/evaluation/) and [`src/models/`](src/models/).

**Walk-forward backtest.** Each calendar year from 2006 is forecast by a model that has only seen earlier data (expanding window, 21 test years). The last *h* days before each test year are removed from training, because their targets overlap the test period (embargo).

![Walk-forward evaluation](reports/figures/05_walk_forward.png)

**Loss functions.**

| Loss | Role | Why |
|---|---|---|
| QLIKE | Primary | Standard in the volatility literature: robust to the noise in the variance measure, and under-predicting risk costs more than over-predicting it |
| MSE of log variance | Secondary | Symmetric; the quantity the machine-learning models are trained on |
| MAE in annualised volatility points | Interpretation | "The forecast is off by 7 volatility points on average" |

Models are compared on a **common sample** (dates on which every model has a forecast), and the scores are computed in SQL from the stored forecasts, so they can be reproduced without Python.

**Baselines.** Five rules with no estimated parameters: the last observed value (naive), the average of the last month, the historical mean, an EWMA of the range-based variance (λ = 0.94) and RiskMetrics (the same EWMA applied to squared returns).

Out-of-sample QLIKE, 2006 to 2026, about 307,000 forecasts per model and horizon (lower is better):

| Baseline | 1 day | 5 days | 22 days |
|---|---|---|---|
| **EWMA of range-based variance** | **0.440** | **0.273** | **0.250** |
| Monthly average | 0.468 | 0.302 | 0.281 |
| RiskMetrics (EWMA of squared returns) | 0.478 | 0.317 | 0.298 |
| Naive (last value) | 0.781 | 0.324 | 0.281 |
| Historical mean | 0.728 | 0.528 | 0.414 |

![Out-of-sample QLIKE of the baselines](reports/figures/05_baseline_scores.png)

- **The bar to beat is the EWMA of the range-based variance**, with an average error of 7.1 volatility points at the 5-day horizon.
- **Measurement matters as much as the model.** EWMA and RiskMetrics are the same formula; feeding it the range-based variance of Step 4 instead of squared returns lowers QLIKE by 8% (1 day) to 16% (22 days).
- **Volatility is too persistent for its long-run average:** the historical mean is the worst baseline beyond one day, with errors almost twice as large as the others.
- **No baseline wins everywhere.** In crisis periods (19% of forecasts) the naive forecast, which reacts fastest, has the lowest 5-day QLIKE (0.271 against 0.297 for EWMA); in calm periods EWMA is clearly better (0.267 against 0.337).

![Forecasts during the COVID-19 crash](reports/figures/05_forecast_example.png)

Simple averages only react **after** volatility has risen and stay too high once markets calm down. Reacting faster in both directions is what the models of the next steps must achieve.

## Methodology (next steps)

| Level | Models | Status |
|---|---|---|
| Baselines | Naive, monthly average, historical mean, EWMA, RiskMetrics | ✅ |
| Econometrics | GARCH(1,1), GJR-GARCH, HAR-RV | Step 6 |
| Machine learning | LightGBM (per-stock and pooled) | Step 7 |
| Deep learning | LSTM / GRU, PatchTST or TFT | Step 8 |

**Further evaluation:** Diebold-Mariano tests, regime analysis, ablation studies, and VaR backtesting with the Kupiec test.

## Tech stack

- **Language:** Python
- **Data:** pandas, NumPy, yfinance
- **Database:** PostgreSQL, SQLAlchemy, psycopg2
- **Econometrics:** arch, statsmodels
- **Machine learning:** scikit-learn, LightGBM, Optuna, SHAP
- **Deep learning:** PyTorch, neuralforecast
- **Results tracking and testing:** forecasts and scores stored in PostgreSQL, pytest
- **Visualisation and dashboard:** matplotlib, plotly, Power BI (connected to PostgreSQL)

## Repository structure

```
european-volatility-forecasting/
├── data/
│   └── universe.csv    # list of tickers with country, sector, currency
├── sql/
│   ├── schema.sql      # PostgreSQL tables, keys, constraints and views
│   └── score_models.sql  # loss functions computed in SQL from the stored forecasts
├── notebooks/          # 01_data_quality, 02_eda, 03_features, 04_baselines, ...
├── src/
│   ├── db.py           # database connection (reads .env)
│   ├── viz.py          # shared chart style
│   ├── data/           # database setup, download and cleaning
│   ├── features/       # volatility estimators, targets, features
│   ├── models/         # common model interface, baselines (then econometric, ML, DL)
│   └── evaluation/     # walk-forward splits, backtest, loss functions, storage of results
├── tests/              # unit tests (incl. data-leakage checks)
├── reports/figures/    # charts used in this README
├── powerbi/            # Power BI dashboard (.pbix)
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
python -m src.features.build_features  # daily variance, features and targets
python -m src.models.run_baselines     # walk-forward backtest of the baselines (forecasts + scores)
python -m pytest                 # run the unit tests
```

Deep learning models (Step 8) need extra packages: `pip install -r requirements-dl.txt` (a GPU, e.g. Google Colab, is recommended).

## Project status

| Step | Description | Status |
|---|---|---|
| 0 | Project setup | ✅ Done |
| 1 | Data collection | ✅ Done |
| 2 | Data cleaning and quality checks | ✅ Done |
| 3 | Exploratory analysis: stylised facts of volatility | ✅ Done |
| 4 | Volatility targets and feature engineering | ✅ Done |
| 5 | Evaluation framework and baselines | ✅ Done |
| 6 | Econometric models (GARCH, GJR-GARCH, HAR) | ⏳ Next |
| 7 | Machine learning (LightGBM, SHAP) | ⬜ |
| 8 | Deep learning (LSTM, transformer) | ⬜ |
| 9 | Model comparison and statistical analysis | ⬜ |
| 10 | Application: Value-at-Risk backtesting | ⬜ |
| 11 | Interactive dashboard (Power BI) | ⬜ |
| 12 | Final report and documentation | ⬜ |

## Author

**Thanh Vinh Dang**

[LinkedIn](https://www.linkedin.com/in/thanhvinhdang2001) · thanhvinhdang.work@gmail.com
