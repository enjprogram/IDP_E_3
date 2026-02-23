# Liquidity and Free Cash Flow: by Industries

An econometric and deep learning pipeline analysing the impact of liquidity shocks on Free Cash Flow (FCF) across 19 Russian industries using the RFSD (Russian Financial Statements Database) panel dataset.

---

## Research Question

> **Do Liquidity changes impact Free Cash Flow across industries, and does the effect vary by sector?**

The analysis combines classical time series econometrics (ARIMA/SARIMA/SARIMAX), Vector Autoregression (VAR) with Impulse Response Functions (IRF), and deep learning methods (Prophet, LSTM, GRU, N-BEATS) to answer this question from multiple methodological angles.

---

## Project Structure

```
Week3_Project/
├── data/
│   └── raw/
│       └── rfsd_2011_2024.parquet          # RFSD panel data (firm-level)
├── src/
│   ├── data_pipeline.py                    # Data loading, variable derivation, aggregation
│   ├── factor_models.py                    # DFM liquidity factor extraction (Kalman filter)
│   ├── utils.py                            # Lagged features, train/test split, rolling CV
│   ├── forecasting.py                      # ARIMA/SARIMA/SARIMAX fitting (pmdarima)
│   ├── var_analysis.py                     # VAR estimation and IRF computation
│   ├── evaluation.py                       # Metrics, Diebold-Mariano test
│   ├── adf_tests.py                        # ADF stationarity tests, differencing order
│   ├── diagnostics.py                      # Full diagnostic suite (ACF/PACF, residuals, etc.)
│   └── deep_learning.py                    # Prophet, LSTM, GRU, N-BEATS forecasting
├── results/
│   ├── lagged_df.pkl                       # Engineered feature panel (cached)
│   ├── performance_all.pkl                 # All model metrics (cached)
│   ├── forecasts_all.pkl                   # All model forecasts (cached)
│   ├── models_all.pkl                      # Fitted model objects (cached)
│   ├── irf_all.pkl                         # IRF results (cached)
│   ├── adf_results.pkl                     # ADF test results (cached)
│   ├── performance_summary.csv             # Flat metrics table
│   ├── diagnostics/                        # Per-industry diagnostic plots and CSVs
│   │   ├── {industry}/                     # ACF/PACF, decomposition, residuals, heatmaps
│   │   └── aic_bic_comparison.csv          # Cross-industry information criteria
│   ├── dl/                                 # Deep learning results
│   │   ├── {industry}_{model}.pkl          # Cached DL results per industry/model
│   │   ├── models/                         # Saved Keras model files
│   │   ├── tensorboard_logs/               # TensorBoard training logs
│   │   └── dl_performance_summary.csv      # Cross-industry DL metrics
│   └── figures/                            # Publication-ready PNG figures (300 DPI)
├── main.py                                 # Classical econometric pipeline
├── run_dl.py                               # Deep learning training script
├── export_figures.py                       # Export publication-ready figures
└── app.py                                  # Streamlit interactive dashboard
```

---

## Data

**Source:** RFSD (Russian Financial Statements Database), 2011–2024  
**Level:** Firm-level financial statements → aggregated to industry-year medians  
**Industries:** 19 OKVED2 sections (A–T), industry T skipped (insufficient data)  
**Final panel:** 18 industries × ~14 years = ~252 industry-year observations

### Key Variables Derived from Raw Line Codes

| Variable | Definition | Line Codes |
|---|---|---|
| `free_cash_flow` | Operating CF + Investing CF | `line_4100 + line_4200` |
| `current_ratio` | Current assets / Current liabilities | `line_1200 / line_1500` |
| `quick_ratio` | (Current assets − Inventory) / Current liabilities | `(line_1200 − line_1210) / line_1500` |
| `cash_ratio` | Cash / Current liabilities | `line_1250 / line_1500` |
| `working_capital` | Current assets − Current liabilities | `line_1200 − line_1500` |
| `total_assets` | Total assets | `line_1600` |
| `total_liabilities` | Total liabilities | `line_1400 + line_1500` |
| `revenue` | Revenue | `line_2110` |
| `net_income` | Net income | `line_2400` |

> `operating_cash_flow` is computed but excluded from all models — it is endogenous by construction (FCF = operating CF + investing CF).

---

## Methodology

### 1. Liquidity Factor Extraction (DFM)

A **Dynamic Factor Model** is estimated via the Kalman filter (`statsmodels DynamicFactorMQ`) to extract a latent common liquidity factor from the three liquidity ratios:

```
current_ratio_t  = λ₁ · F_t + ε₁_t
quick_ratio_t    = λ₂ · F_t + ε₂_t
cash_ratio_t     = λ₃ · F_t + ε₃_t

F_t = φ · F_{t-1} + η_t
```

- **`liquidity_factor_dfm`** — the smoothed latent factor capturing common liquidity conditions
- **`liquidity_shock_dfm`** — the innovation `η_t`, representing genuinely unexpected changes in industry liquidity (analogous to structural shocks in the VAR literature)

The DFM approach compresses three correlated liquidity ratios into one clean orthogonal factor, avoiding multicollinearity in downstream SARIMAX models.

### 2. Feature Engineering

Lagged features (lags 1–2) are created for FCF and all exogenous variables to encode temporal structure. The train/test split is 80/20 by year, consistent across all models.

### 3. Classical Time Series Models

Three models are estimated per industry via `auto_arima` (pmdarima):

| Model | Specification | Purpose |
|---|---|---|
| **ARIMA** | Univariate autoregressive | Baseline — captures FCF own dynamics |
| **SARIMA** | ARIMA + seasonal component | Tests for seasonal structure in annual data |
| **SARIMAX** | SARIMA + exogenous variables | Tests whether liquidity variables add predictive power |

**Exogenous variables in SARIMAX** (lagged to avoid endogeneity):
- `current_ratio_lag1`, `quick_ratio_lag1`, `cash_ratio_lag1`
- `liquidity_factor_dfm_lag1`, `liquidity_shock_dfm_lag1`

Model selection uses AIC/BIC minimisation. Rolling origin cross-validation (3 folds) provides additional validation.

### 4. VAR and Impulse Response Functions

A **Vector Autoregression** is estimated on `[free_cash_flow, liquidity_factor_dfm, current_ratio, quick_ratio, cash_ratio]` to capture dynamic interdependencies. **Impulse Response Functions** trace the response of FCF to a one-standard-deviation liquidity shock over a 10-period horizon.

### 5. Model Evaluation

- **Metrics:** RMSE, MAE, MAPE, SMAPE, R²
- **Diebold-Mariano test:** Formal statistical test of whether SARIMAX significantly outperforms ARIMA — if yes, liquidity variables contain genuine predictive information
- **Heteroskedasticity tests:** Breusch-Pagan, White, Ljung-Box on squared residuals (ARCH effects), Durbin-Watson, Jarque-Bera

### 6. Deep Learning Models

Four deep learning models are trained on the full feature set from `lagged_df` (all numeric columns including DFM factors and lags):

| Model | Features | Notes |
|---|---|---|
| **Prophet** | Top 5 features (DFM + lags prioritised) | Best for short annual series; handles trend + regressors |
| **LSTM** | All lagged_df columns | Multivariate sequence model |
| **GRU** | All lagged_df columns | Lighter alternative to LSTM; often better for short series |
| **N-BEATS** | FCF series only (univariate) | Pure basis expansion; no feature engineering required |

Training evidence is available via TensorBoard and stored loss curves in the dashboard.

> **Note:** With ~9 training observations per industry, deep learning models are data-starved. Results should be treated as robustness checks rather than primary findings. Classical ARIMA/SARIMAX models are more appropriate for this sample size.

### 7. Liquidity Shock Identification

Shocks are identified as industry-specific deviations exceeding one standard deviation:

```python
threshold = np.nanstd(liquidity_shock_dfm)
red   = shock < -threshold   # unexpected liquidity tightening
blue  = shock > +threshold   # unexpected liquidity improvement
```

The threshold is relative to each industry's own shock distribution — capturing sector-specific tightening rather than imposing a common threshold.

> **Limitation:** Industries with monotonically declining liquidity across the full sample period may show the entire period as red shading, conflating secular trend with genuine surprise. A detrended shock robustness check is recommended for these cases.

---

## Installation

```bash
# Clone / navigate to project
cd Project

uv venv
uv sync

# Install classical pipeline dependencies
uv add polars pandas numpy statsmodels pmdarima scipy matplotlib

# Install deep learning dependencies
uv add prophet tensorflow scikit-learn tensorboard

# Install dashboard dependencies
uv add streamlit plotly
```

---

## Usage

### Step 1 — Run the classical econometric pipeline

```bash
uv run python main.py
```

Results are cached in `results/` — subsequent runs load from cache. To rerun from scratch:

```bash
rm results/lagged_df.pkl results/performance_all.pkl \
   results/forecasts_all.pkl results/models_all.pkl \
   results/irf_all.pkl results/adf_results.pkl
uv run python main.py
```

### Step 2 — Run the deep learning pipeline

```bash
# All industries, all models
uv run python run_dl.py

# Specific industries or models
python run_dl.py --industries C G K
python run_dl.py --models prophet lstm
python run_dl.py --industries C --models prophet lstm
```

To retrain from scratch:

```bash
rm -rf results/dl/
uv run python run_dl.py
```

### Step 3 — Launch the interactive dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Step 4 — Launch TensorBoard (optional, in separate terminal)

```bash
tensorboard --logdir results/dl/tensorboard_logs
```

Opens at `http://localhost:6006`

### Step 5 — Export publication-ready figures

```bash
uv run python export_figures.py
```

Figures saved to `results/figures/` at 300 DPI.

---

## Dashboard Views

The Streamlit dashboard (`app.py`) provides 14 interactive views selectable from the left sidebar:

| View | Content |
|---|---|
| Forecast Comparison | Full-timeline FCF forecasts with train/test split and liquidity shock shading |
| Model Performance | RMSE, MAE, MAPE, R² by model and split |
| Information Criteria | AIC, BIC, AICc across industries |
| Rolling CV Metrics | 3-fold rolling origin cross-validation results |
| Diebold–Mariano Tests | Statistical test: does SARIMAX beat ARIMA? |
| Impulse Response Functions | FCF response to liquidity shock over 10 periods |
| Seasonal Decomposition | Trend, seasonal, residual decomposition |
| ACF / PACF | Autocorrelation and partial autocorrelation |
| Residual Diagnostics | Residuals, histogram, Q-Q plot, ACF of residuals |
| Heteroskedasticity Tests | Breusch-Pagan, White, ARCH, Durbin-Watson, Jarque-Bera |
| Covariance & Correlation | Liquidity variable correlation matrices |
| ADF Stationarity Tests | Unit root tests and differencing order |
| Cross-Industry Summary | All-industry performance heatmap |
| Deep Learning Forecasts | Prophet/LSTM/GRU/N-BEATS forecasts with training evidence |

---

## Output Files

### Figures (`results/figures/`)

| File | Description |
|---|---|
| `fig1_fcf_forecasts_all_industries.png` | Grid of FCF forecasts — all industries, classical models |
| `fig2_irf_all_industries.png` | IRF grid — FCF response to liquidity shock |
| `fig3_test_rmse_comparison.png` | Grouped bar chart — test RMSE by industry and model |
| `fig4_information_criteria.png` | AIC/BIC line plots across industries |
| `fig5_dl_forecasts_all_industries.png` | Grid of FCF forecasts — deep learning models |

### Diagnostics (`results/diagnostics/{industry}/`)

Each industry folder contains: seasonal decomposition, ACF/PACF plots and CSVs, correlation heatmap, covariance and correlation CSVs, and per-model subfolders with residual diagnostics, heteroskedasticity tests, and model summaries.

---

## How to Read the Results

| Evidence | What it shows |
|---|---|
| **IRF positive and persistent** | Liquidity shocks have lasting impact on FCF |
| **SARIMAX beats ARIMA (DM test p < 0.05)** | Liquidity variables have genuine predictive power |
| **Significant SARIMAX lag coefficients** | Direction and magnitude of the liquidity-FCF relationship |
| **ARCH effects significant** | Shocks affect FCF volatility, not just level |
| **Varies across industries** | Heterogeneous transmission by sector |

Industries where all four point in the same direction are the strongest cases for the research hypothesis.

---

## Industry Coverage

| Code | Industry | Code | Industry |
|---|---|---|---|
| A | Agriculture & Forestry | K | Financial & Insurance |
| B | Mining & Quarrying | L | Real Estate |
| C | Manufacturing | M | Professional & Scientific |
| D | Electricity & Gas Supply | N | Administrative Services |
| E | Water Supply & Waste | O | Public Administration |
| F | Construction | P | Education |
| G | Wholesale & Retail Trade | Q | Healthcare & Social |
| H | Transportation & Storage | R | Arts & Entertainment |
| I | Accommodation & Food | S | Other Services |
| J | Information & Communications | T | Skipped (insufficient data) |

---

## Known Limitations

- **Short panel:** ~14 annual observations per industry limits reliable ACF/PACF to 3–4 lags and constrains deep learning model capacity
- **Firm composition change:** RFSD coverage varies by year, making median ratios sensitive to entry/exit of reporting firms
- **Monotonic shock problem:** Industries with secular liquidity decline may show the entire period as negative shocks — detrending is recommended as a robustness check
- **N-BEATS and LSTM** are included as robustness checks only — the sample size is insufficient for reliable deep learning inference
- **Industry F** (Construction) skipped in some runs due to constant FCF series

---

## Dependencies

| Package | Purpose |
|---|---|
| `polars` | Fast data loading and transformation |
| `pandas` | Panel data manipulation |
| `numpy` | Numerical computation |
| `statsmodels` | DFM, VAR, ADF, diagnostic tests |
| `pmdarima` | Auto ARIMA/SARIMA/SARIMAX fitting |
| `scipy` | Statistical tests |
| `matplotlib` | Diagnostic plots and figure export |
| `plotly` | Interactive dashboard charts |
| `streamlit` | Interactive dashboard |
| `prophet` | Prophet forecasting |
| `tensorflow` | LSTM, GRU, N-BEATS |
| `scikit-learn` | Data scaling for DL models |
| `tensorboard` | Training visualisation |

---

## Citation

If using this pipeline or methodology, please cite the RFSD data source and relevant methodological references:

- Kalman, R.E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45. — DFM estimation
- Diebold, F.X. and Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. — DM test
- Oreshkin, B.N. et al. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR 2020*. — N-BEATS architecture
- Taylor, S.J. and Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45. — Prophet
