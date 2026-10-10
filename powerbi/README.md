# Power BI dashboard

Four pages that summarise the project for a reader who will not open the notebooks: which model forecasts best and how stable it is, how the forecasts behave through crises, what they are worth in risk management, and what to conclude.

| Page | Question | Main visuals |
|---|---|---|
| 1. Model performance | Which model is best, by how much, and is it stable? | Two KPI cards, ranking of the 15 models by QLIKE and by MAE, QLIKE in calm and stress periods, best model per family, gap to the best model by test year |
| 2. Forecast explorer | How do the forecasts follow realised volatility? | Forecast and realised volatility of any series through time, three KPI cards, QLIKE and bias of each forecast over the selected period |
| 3. Value-at-Risk | Do better forecasts make better risk limits? | Best VaR model, daily returns against three VaR limits for the Euro Stoxx 50, violations per year, VaR size vs violations, Basel traffic light |
| 4. Summary | What should a risk team conclude? | A recommendation and three findings, each stated in the title of its chart; the VaR backtest table |

The data comes from thirteen PostgreSQL views (`sql/powerbi_views.sql`), imported into Power BI. All the statistics are computed in Python and SQL; the dashboard only aggregates them, with the measures of [`measures.dax`](measures.dax).

## 1. Prepare the data

After Steps 5 to 10, from the project root:

```bash
python -m src.dashboard.setup_views
```

It creates the views and prints their number of rows. The largest, `pbi_forecast_paths`, has about 5.5 million rows (59 series × 3 horizons × 21 years × 6 lines).

## 2. Import the views

1. Power BI Desktop → **Get data** → **PostgreSQL database**. Server `localhost:5432`, database `volatility_db`, mode **Import**.
2. Credentials: tab **Database**, user `postgres` and your password. Power BI keeps them on your computer, not in the `.pbix` file. If it asks about an encrypted connection, accept the unencrypted one (local server).
3. In the Navigator, tick the thirteen `public.pbi_...` views and click **Transform data**.
4. Rename the queries (double-click the name), then **Close & Apply**:

| View | Name in Power BI | Type |
|---|---|---|
| `pbi_series` | Series | Dimension |
| `pbi_models` | Models | Dimension |
| `pbi_horizons` | Horizons | Dimension |
| `pbi_confidence` | Confidence | Dimension |
| `pbi_calendar` | Calendar | Dimension |
| `pbi_model_scores` | Model scores | Fact |
| `pbi_model_tests` | Model tests | Fact |
| `pbi_regime_scores` | Regime scores | Fact |
| `pbi_forecast_paths` | Forecast paths | Fact |
| `pbi_var_backtest` | VaR backtest | Fact |
| `pbi_var_yearly` | VaR yearly | Fact |
| `pbi_var_tests` | VaR tests | Fact |
| `pbi_var_daily` | VaR daily | Fact |

## 3. Data model

**Relationships** (Model view; all many-to-one, single direction, from the dimension to the fact). Delete any relationship that Power BI created automatically and is not in this list.

| Dimension column | Fact tables |
|---|---|
| Series[ticker] | Model scores, Forecast paths, VaR backtest, VaR yearly, VaR daily |
| Models[model] | Model scores, Model tests, Regime scores, VaR backtest, VaR yearly, VaR tests |
| Horizons[horizon] | Model scores, Model tests, Regime scores, Forecast paths |
| Confidence[confidence] | VaR backtest, VaR yearly, VaR tests, VaR daily |
| Calendar[date] | Forecast paths, VaR daily |

**Sort by column** (select the column → Column tools → Sort by column), so that legends and axes follow the order of the project rather than the alphabet:

| Column | Sort by |
|---|---|
| Models[model_label] | Models[model_order] |
| Models[family_label] | Models[family_order] |
| Horizons[horizon_label] | Horizons[horizon_order] |
| Forecast paths[line] | Forecast paths[line_order] |
| VaR yearly[zone_label] | VaR yearly[zone_order] |
| Calendar[month_label] | Calendar[month_key] |

**Date table:** select Calendar → Table tools → **Mark as date table** → column `date`.

**Theme:** View → Themes → **Browse for themes** → [`theme.json`](theme.json): the colours of the European Union flag (blue `#003399`, gold `#FFCC00`) on a soft blue page, white visuals with an EU-blue title bar, KPI cards in EU blue. Only the colours are used, not the flag or its stars, so that the dashboard does not look like an official EU publication.

**Measures:** [`measures.dax`](measures.dax) explains all 60 measures and their format. To add them all at once, open **DAX query view** (fourth icon on the left), paste [`measures_query.dax`](measures_query.dax) and click **Update model with changes** above `DEFINE`; then set the formats (Measure tools → Format).

**Colour rule:** a good model is **blue**, a bad one is **orange**, with a neutral grey-blue in between. Gradients are computed on a gap to the best model (0% = best), so no single model is used as the reference for the colours.

| Element | Colour |
|---|---|
| Good / neutral / bad (bars and points) | `#003399` / `#B7C1D6` / `#F39C12` |
| Good / neutral / bad (cell backgrounds, lighter so the text stays readable) | `#6A8FDF` / `#F2F4F8` / `#F7B24A` |
| One colour per forecast (lines) | Realised `#1A1A1A`, Ensemble `#003399`, HAR-X `#3FA7D6`, Transformer `#8E6BBF`, GJR-GARCH `#A9B4C8`, EWMA `#F39C12` |
| VaR limits | ensemble `#003399`, HAR-X `#3FA7D6`, historical simulation `#F39C12`, daily returns `#B0B7C3` |
| Basel zones green / yellow / red | `#2E9E5B` / `#F5C518` / `#C8102E` |
| Page / KPI cards / notes | `#D9E1F2` / `#003399` with gold title / `#FFF4CC` |

## 4. The four pages

The models are judged in two ways: as **forecasts** of volatility (pages 1 and 2: QLIKE and MAE against the realised volatility) and as **risk limits** (page 3: the 1-day VaR built from each forecast, judged by its violations and its quantile loss). The KPI cards say which: "Best forecast model" on page 1, "Best VaR model" on page 3.

Canvas 1920 × 1080 (Full HD). Every page starts with the same header: an EU-blue band with the title in white and a thin gold line under it. The slicers sit in one row under the header; tile slicers (horizon, confidence) are button slicers with the selected tile in EU blue.

### Page 1: Model performance

| Visual | Content |
|---|---|
| Horizon (button slicer) | 1 day, 1 week, 1 month |
| 2 KPI cards | best forecast model by QLIKE and best forecast model by MAE |
| QLIKE by model | bar chart of `Mean QLIKE` (raw value, sorted, lower = better), coloured by `QLIKE vs best model` |
| Gap to the best model of each year | matrix model × test year of `QLIKE vs best model` in that year (0% = best that year), combinations on top; QLIKE levels change a lot between calm and crisis years, so the raw value would mostly show the crises |
| Best model in each family | baselines, econometric models, machine learning, deep learning, combinations |
| MAE by model | bar chart of `MAE (vol points)` (average error in volatility points, sorted), coloured by `MAE vs best model`: the Transformer is first at 1 week and 1 month |
| Calm and stress periods | table of the 15 models: QLIKE in calm periods and in stress periods (the crises of 2007-09, 2011-12, 2020 and 2022) with the rank in each; the Transformer is among the best in calm markets and falls behind in crises |
| Key findings | in the **?** icon of the ranking chart (hover over the chart): four sentences |

### Page 2: Forecast explorer

| Visual | Content |
|---|---|
| Slicers | company / index (dropdown, titled "Company/Index"), horizon (buttons), models shown on the chart (with "Select all"), period (between two dates) |
| Forecast vs realised volatility | line chart: the forecast made on each date and the volatility actually realised over the horizon |
| 3 KPI cards | average and peak realised volatility, best forecast in the period (`Best forecast (period)`) |
| QLIKE in the period | raw `QLIKE (period)` of the five forecasts, sorted, coloured by `QLIKE vs best (period)` |
| Bias | `Forecast bias (pts)`: forecast minus realised, coloured by sign: orange below zero (forecasts too low, the dangerous side for risk), blue above. Over long periods a small positive bias is normal; zoom on a crisis to see who falls behind |
| How to read | in the **?** icon of the line chart: short reading note (March 2020: the Transformer stays too low) |

The text boxes with the key findings (page 1) and the reading note (page 2) are kept but hidden: View → Selection to show them again. The visuals removed from page 1 are left as empty hidden shapes and can be deleted there.

On page 2, the bars and cards ignore the "Models" slicer (Format → Edit interactions → None), so they always compare all five forecasts.

### Page 3: Value-at-Risk

| Visual | Content |
|---|---|
| Slicers | confidence (95% / 99%, buttons), period |
| Best VaR model (KPI card) | `Best VaR model`: among HAR-X and the models whose quantile loss is significantly lower than HAR-X's (Diebold-Mariano test, p < 0.05), the one with the lowest quantile loss at the selected confidence, all 50 companies and 9 indices over the whole test period: Ensemble (Mean) at 95% and 99%. Without the test, LightGBM would come first at 99%, but only 0.5% ahead and not significantly better than HAR-X |
| Euro Stoxx 50: daily return vs VaR limits | columns = daily return of the index, lines = the three limits with daily values (HAR-X, Ensemble (Mean), historical simulation) on the same axis (secondary y-axis off): a bar below a line is a violation |
| Violations per year | all companies and indices: columns = violations of each limit, black line = expected number |
| VaR size vs violation rate | scatter of all 16 VaR models (the 15 forecasting models and historical simulation): a good limit is small (left) and close to the promised rate (bottom); colour = overall quality (quantile loss) |
| Basel traffic light, 99% VaR | share of years (of each company or index) in the green, yellow and red zones for all 16 models, sorted by `Green zone share` (best first) |

The card, the scatter and the traffic light cover all companies and indices and the whole test period, so the date slicer does not filter them; the traffic light is always at 99% (the Basel rule is defined for the 99% VaR).

### Page 4: Summary

Each chart states its finding in its title, with the key numbers underneath.

| Visual | Finding |
|---|---|
| Recommendation | For 1 to 5 days, the average of the five advanced models; for a month, HAR-X alone. If only one model can be maintained, HAR-X: within 2.4% of the best at every horizon, estimated in seconds, 18 readable coefficients, no bad year |
| QLIKE relative to HAR-X | Beyond HAR-X, more complex models gain little: within 2.5% of HAR-X at 1 day and 1 week, while the classical models are 6 to 44% worse |
| Worst year vs HAR-X (`Worst year vs HAR-X`) | Single models have blind spots (the LSTM in 2009, the Transformer in 2020); the ensemble is the steadiest |
| Years in the top 3 (`Years in top 3`) | The ensemble is always among the best |
| 99% VaR backtest | violation rate, share passing Kupiec, average VaR, quantile loss (bp) and significance vs HAR-X for every model (moved from page 3) |

The horizon slicer applies to the three charts; the backtest table uses the 1-day 99% VaR.

## 5. Save and share

* The report is saved as a **Power BI project** (`european_volatility_dashboard.pbip`): the pages and visuals are JSON files in `european_volatility_dashboard.Report/`, the tables, relationships and measures are TMDL files in `european_volatility_dashboard.SemanticModel/`. Git ignores the `.pbi/` folders (local settings and the cache of imported data), so after a clone open the `.pbip` and click **Refresh**.
* To share a single file: File → Save as → `.pbix` (it contains the imported data, no password).
* Screenshots of the four pages for the main README: `reports/figures/11_model_performance.png`, `11_forecast_explorer.png`, `11_value_at_risk.png`, `11_summary.png`.
* To refresh after re-running the models: re-run `python -m src.dashboard.setup_views`, then **Refresh** in Power BI.
* Optional: publish to the Power BI service and use **Publish to web** to embed the dashboard in a portfolio site (the data is public market data).
