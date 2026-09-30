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

- **Source:** Yahoo Finance (via `yfinance`), daily OHLCV data
- **Period:** 2000 to present, covering the dot-com crash, 2008, the euro debt crisis, COVID-19 and the 2022 rate shock
- **Universe:**
  - **50 stocks** across 13 countries and 11 sectors (France, Germany, Switzerland, UK, Netherlands, Luxembourg, Spain, Italy, Denmark, Norway, Sweden, Finland, Belgium)
  - **9 indices:** Euro Stoxx 50, CAC 40, DAX, FTSE 100, SMI, AEX, IBEX 35, FTSE MIB, OMX Stockholm 30
  - **VIX** as an external market-fear indicator

*Details will be added in Step 1.*

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
```

Deep learning models (Step 8) need extra packages: `pip install -r requirements-dl.txt` (a GPU, e.g. Google Colab, is recommended).

## Project status

| Step | Description | Status |
|---|---|---|
| 0 | Project setup | ✅ Done |
| 1 | Data collection | ⏳ Next |
| 2 | Data cleaning and quality checks | ⬜ |
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

[LinkedIn](https://www.linkedin.com/in/thanhvinhdang2001) · [GitHub](https://github.com/vinhdang111) · thanhvinhdang.work@gmail.com

## License

MIT. See [LICENSE](LICENSE).
