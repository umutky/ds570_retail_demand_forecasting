# Retail Demand Forecasting

End-to-end retail demand forecasting pipeline built on the M5 Accuracy Competition dataset.
This is my DS570 final project - an attempt to show how gradient-boosted trees with a Tweedie
loss function handle zero-inflated retail sales better than standard regression.

**Scope:** CA_1 store × 3,049 products × 3 categories (HOBBIES, HOUSEHOLD, FOODS)  
**Model:** LightGBM - Tweedie objective vs. Gaussian (L2) comparison  
**Dashboard:** Interactive Streamlit app with data exploration, 28-day forecast, model insights, and SHAP analysis

---

## Quickstart

The only requirement is Docker. No data download, no account, no local setup needed.

### Option A - Pull pre-built image (recommended, skips build step)

```bash
docker pull umutky/retail-forecast:latest

docker run -p 8501:8501 \
  -v retail-forecast-data:/app/data \
  -v retail-forecast-outputs:/app/outputs \
  umutky/retail-forecast:latest
```

### Option B - Build from source

```bash
docker build -t retail-forecast .

docker run -p 8501:8501 \
  -v retail-forecast-data:/app/data \
  -v retail-forecast-outputs:/app/outputs \
  retail-forecast
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

On first run the container will:
1. Download the CA_1 subset (~22 MB) from GitHub Releases
2. Download the pre-built feature matrix (~76 MB) from GitHub Releases - skips ~8 min of local computation
3. Train both Tweedie and Gaussian LightGBM models (~3 minutes)
4. Generate the 28-day demand forecast
5. Launch the Streamlit dashboard

Subsequent runs reuse the cached data and models from the named volumes, so all steps
are skipped and the dashboard launches in seconds.

> **Timing:** First run takes roughly 4–5 minutes total. Subsequent runs launch in seconds.

---

## Data

**Source:** [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy),
Makridakis Open Forecasting Center, 2020.

> Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition:
> Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364.

**License / Terms of Use:** The original M5 dataset is provided by Walmart via the Kaggle
competition platform. This project uses a CA_1 store subset solely for academic and
educational purposes (DS570 coursework). The subset is redistributed as a GitHub Release
asset under educational fair use. Original data rights belong to Walmart and the
Makridakis Open Forecasting Center. If you intend to use this data beyond academic
purposes, download it directly from the
[official Kaggle competition page](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

**Subset:** `store_id == 'CA_1'` × all products × all dates
- 3,049 time series (unique items)
- 3 categories: HOBBIES, HOUSEHOLD, FOODS (7 departments)
- 1,941 days (2011-01-29 to 2016-05-22)
- ~5.9M rows in long format

**Why CA_1?** Including all 10 M5 stores would produce ~59M rows and may require 60+ minutes
of training - incompatible with the Docker time budget. CA_1 was selected as a
representative high-volume store to keep the pipeline fast while still covering all three
product categories (HOBBIES, HOUSEHOLD, FOODS).

**Runtime fetch:** The raw subset and the pre-built feature matrix are both hosted as
[GitHub Release assets](https://github.com/umutky/ds570_retail_demand_forecasting/releases/tag/v0.1.0-data).
Both are downloaded automatically on first Docker run. No Kaggle account needed.

**Reproducibility:** To rebuild the subset and feature matrix from scratch:
```bash
# Download the full M5 data into data/raw/ from Kaggle first, then:
uv run rf-process   # rebuilds sales_long.parquet and feature_matrix.parquet
```

---

## Methods

### The problem: zero-inflated demand

Retail sales data is full of zeros. In the CA_1 subset, roughly 64% of all daily
item-level sales are zero - ~73% in HOBBIES, ~68% in HOUSEHOLD, and ~57% in FOODS.
Standard L2 (Gaussian) regression minimizes squared error, which pulls predicted values
toward the mean and systematically over-predicts zero-demand days. The Tweedie distribution
is a better fit for this kind of count data: it assigns explicit probability mass to zero
and models the heavy right tail of sporadic high-demand days.

### Baseline models

Three naive benchmarks are implemented in `src/retail_forecast/models/baseline.py`:

| Model | Description |
|---|---|
| **Seasonal Naive** | Repeat the same weekday from the previous week |
| **Moving Average** | 28-day trailing mean |
| **Zero Forecast** | Always predict 0 - a surprisingly competitive lower bound given ~64% zero rate |

All three are evaluated against LightGBM on the same holdout period.

### LightGBM

Two LightGBM models are trained with identical hyperparameters, differing only in objective:

- **Tweedie** (`objective: tweedie`, `tweedie_variance_power: 1.1`) - the main model
- **Gaussian** (`objective: regression`) - comparison baseline

Features are built in `src/retail_forecast/features.py` across eight families (**37 features total**):

| Family | Features | Count | Motivation |
|---|---|---|---|
| Lag | `sales_lag_1/2/3/4/5/6/7/14/28/365` | 10 | Weekly, monthly, and yearly seasonality |
| Rolling mean | `rolling_mean_7/14/28/30/60` | 5 | Local trend and demand momentum |
| Rolling std | `rolling_std_7/28` | 2 | Demand volatility |
| Intermittency | `zero_streak`, `days_since_last_sale` | 2 | Consecutive zero-sales runs; key for Tweedie |
| Hierarchical | `dept_rolling_mean_7/28` | 2 | Cross-item department-level demand signal |
| Calendar | wday, month, is_weekend, is_month_start, is_month_end, has_event, event_type_encoded, has_snap, doy_sin, doy_cos, week_of_year | 11 | Recurring patterns, promotions, SNAP days |
| Price | `sell_price`, `price_change_pct`, `price_rel_year` | 3 | Demand elasticity |
| Categorical | `dept_id`, `cat_id` | 2 | Hierarchical grouping (LightGBM native category) |

No future information leaks into the feature matrix.

### SHAP analysis

I use SHAP (SHapley Additive exPlanations) via `shap.TreeExplainer` to interpret both models.
Unlike gain-based feature importance, SHAP values are direction-aware: they show whether a
feature pushed a specific prediction up or down and by how much.

Key findings:
- **`sell_price`** is the #1 driver for Tweedie - higher-priced items tend to have fewer
  but larger sales events, which aligns with the Tweedie distribution's structure.
- **`zero_streak`** is the #2 driver - consecutive zero-sales days strongly suppresses
  next-day predicted demand.
- **`sales_rolling_mean_14`** captures recent momentum and is consistently in the top-3
  for both models.
- Gaussian weights recent rolling averages (`rolling_mean_7`) more heavily, while Tweedie
  leans on intermittency signals - this reflects the difference in loss function geometry.

### Train / validation / test split

Data is split chronologically (no shuffling):
- **Train:** 2012-01-29 to 2015-07-12 (~80%)
- **Validation:** 2015-07-12 to 2015-12-17 (~10%) - used for early stopping only
- **Test:** 2015-12-17 to 2016-05-22 (~10%) - held out, never seen during training

### Evaluation metrics

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error - penalizes large errors heavily |
| **MAE** | Mean Absolute Error - interpretable in units (daily item sales) |
| **WMAPE** | Weighted MAE / total sales - avoids undefined MAPE on zero-sales days |
| **WRMSSE** | Weighted RMSE Scaled Score - the official M5 competition metric; evaluated at item, department, category, and total store level |

---

## Results

### Tweedie vs. Gaussian on the test set

| Model | RMSE | MAE | WMAPE |
|---|---|---|---|
| **LightGBM Tweedie** | **1.9632** | **0.9989** | **68.7%** |
| LightGBM Gaussian | 1.9744 | 1.0111 | 69.5% |

Tweedie outperforms Gaussian across all three metrics. The margin is modest at item level
but compounds at higher aggregation levels.

### WRMSSE by aggregation level

| Level | Tweedie | Gaussian | Δ (Tweedie better by) |
|---|---|---|---|
| Item | 0.9477 | 0.9537 | +0.006 |
| Department | 0.4567 | 0.5322 | +0.076 |
| Category | 0.4063 | 0.4793 | +0.073 |
| **Total store** | **0.3394** | **0.4185** | **+0.079** |

The Tweedie advantage grows with aggregation - at total store level WRMSSE improves by 0.079.
This is consistent with the theoretical expectation: Tweedie's better handling of zeros and
the heavy tail reduces systematic bias, and that bias cancellation compounds when summing
across all items.

### LightGBM Tweedie results by category

| Category | Zero rate | RMSE | MAE | WMAPE |
|---|---|---|---|---|
| FOODS | ~57% | 2.273 | 1.242 | 61.1% |
| HOUSEHOLD | ~68% | 1.248 | 0.725 | 77.4% |
| HOBBIES | ~73% | 2.184 | 0.887 | 94.3% |

HOBBIES shows the highest WMAPE - items in this category are highly intermittent
(many zero-sales days punctuated by occasional bursts), which makes any point forecast
inherently uncertain. FOODS, with the lowest zero rate, is the most predictable category.

---

## Dashboard

The Streamlit app has five pages:

| Page | What it shows |
|---|---|
| **Data Explorer** | Historical sales time series per item, event overlays, zero-inflation by category, sales distribution, weekday pattern |
| **28-Day Forecast** | Item-level and category-level demand forecast for the 28 days after the training horizon |
| **Model Insights** | Tweedie vs. Gaussian metric comparison at all aggregation levels; link to SHAP Analysis |
| **Forecast Charts** | Actual vs. predicted time series at total store / category / department / item level |
| **SHAP Analysis** | Direction-aware feature attributions: global bar + beeswarm, feature dependence plots with LOWESS trend, single-prediction waterfall, SHAP distribution by category |

---

## Design Decisions

**LightGBM over deep learning:** For tabular time-series data with hand-crafted lag features, LightGBM is faster to train, easier to interpret via SHAP, and does not require GPU resources.

**Tweedie objective:** Tweedie loss is the correct statistical choice for zero-inflated
non-negative count data. The variance power parameter `p = 1.1` keeps the model close
to Poisson (which is appropriate for count data) while allowing some extra flexibility
for the heavy right tail in FOODS items.

**Pre-built feature matrix on GitHub Releases:** Building the 37-feature matrix from scratch
takes ~8 minutes on a typical machine. I pre-built it locally, verified its SHA256, and
uploaded it alongside the raw data subset. The Docker entrypoint downloads it on first run,
cutting total setup time to ~4–5 minutes. Rebuilding from scratch with `rf-process` is
always possible after feature changes.

**`uv` over pip:** Deterministic installs via `uv.lock`. The same lock file is used in
the Dockerfile and in local development, so the environment is guaranteed to be identical.

**Single-store focus (CA_1):** Restricting to one store removes store-level heterogeneity
from the analysis. The interesting comparison in this project is between categories
(HOBBIES vs. HOUSEHOLD vs. FOODS), and that comparison is cleaner when store effects
are held constant.

---

## Repository Structure

```
retail-demand-forecasting/
├── app/
│   ├── streamlit_app.py          # Main dashboard entry point (st.navigation)
│   └── pages/
│       ├── 1_Data_Explorer.py    # Historical sales, zero-inflation, event effects
│       ├── 2_Forecast.py         # 28-day demand forecast by item and category
│       ├── 3_Model_Insights.py   # Tweedie vs. Gaussian metrics by aggregation level
│       ├── 4_Forecast_Charts.py  # Actual vs. predicted time series
│       └── 5_SHAP_Analysis.py    # SHAP bar, beeswarm, dependence, waterfall, category box
├── notebooks/
│   ├── 2026-04-22-uk-01-eda-global.ipynb       # Comparative EDA across all 10 M5 stores → CA_1 selection
│   ├── 2026-04-22-uk-02-eda-ca1-focused.ipynb  # Detailed EDA on CA_1 subset
│   ├── 2026-05-11-uk-03-baseline-vs-lgbm.ipynb # Baseline models vs. LightGBM Tweedie
│   └── 2026-05-18-uk-04-tweedie-vs-gaussian.ipynb # Tweedie vs. Gaussian deep-dive + SHAP
├── scripts/
│   └── build_subset.py           # Reproducible script to rebuild the CA_1 subset from raw M5 CSVs
├── src/retail_forecast/
│   ├── config.py                 # Paths, data URLs, SHA256 checksums
│   ├── data/
│   │   ├── fetch.py              # HTTP download with SHA256 verification and caching
│   │   └── load.py               # Parquet loading, wide-to-long transformation
│   ├── features.py               # 37-feature engineering pipeline (8 families)
│   ├── models/
│   │   ├── baseline.py           # SeasonalNaive, MovingAverage, ZeroForecast
│   │   └── lgbm.py               # LightGBM wrappers (Tweedie + Gaussian + Forecast)
│   ├── evaluate.py               # RMSE, MAE, WMAPE, WRMSSE
│   ├── forecast.py               # 28-day recursive forecasting
│   ├── tracker.py                # Model run logging (params, metrics, feature importance)
│   └── cli.py                    # Entry points: rf-fetch, rf-fetch-features, rf-process, rf-train, rf-predict
├── tests/
│   ├── test_fetch.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
├── Dockerfile                    # Multi-stage build (builder + runtime)
├── entrypoint.sh                 # fetch → feature matrix → train → predict → streamlit
├── pyproject.toml
└── uv.lock
```

---

## Local Development

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync --all-extras

# Fetch raw data
uv run rf-fetch

# Option A: Download pre-built feature matrix (~80 MB, skips ~8 min of computation)
uv run rf-fetch-features

# Option B: Build feature matrix from scratch (use this after adding new features)
uv run rf-process

# Train both models
uv run rf-train

# Generate 28-day forecast
uv run rf-predict

# Launch dashboard
uv run streamlit run app/streamlit_app.py

# Run tests
uv run pytest
```

---

## Limitations & Future Work

- **CA_1 focus:** The model is trained and evaluated on the CA_1 store only. The pipeline
  is store-agnostic - repointing `DATA_URL` in `config.py` to a different store's subset
  and rerunning `rf-fetch → rf-process → rf-train → rf-predict` applies the same methodology
  to any M5 store.
- **Recursive forecast error accumulation:** The 28-day forecast is generated step-by-step:
  each day's prediction is fed back as a lag feature for the next day. Errors compound over
  the horizon - day-28 forecasts build on 27 prior predictions rather than observed sales.
  Short-lag features (lag_1, lag_7) are affected most; lag_28 and lag_365 always draw from
  historical data within the 28-day horizon.
- **Cold-start items:** Items with very few historical sales have sparse lag features. The model
  tends to underfit these - a known weakness of lag-based approaches.
- **Rare events:** Events like Super Bowl or Christmas appear very few times in the training data.
  The model learns an average effect but cannot capture how an unusual edition of the event
  might shift demand differently.
- **Point forecasts only:** The pipeline produces single-value predictions. Prediction intervals
  (e.g., quantile regression with `objective: quantile`) would make uncertainty explicit and
  would be a natural next step.
- **SHAP computation cost:** SHAP values are computed on a 3,000-row sample of the test set at
  dashboard load time (~30 seconds on first open). Full test set SHAP would require pre-computation during training.

---
