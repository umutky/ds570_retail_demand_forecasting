# Retail Demand Forecasting

End-to-end retail demand forecasting pipeline built on the M5 Accuracy Competition dataset.
This is my DS570 final project — an attempt to show how gradient-boosted trees with a Tweedie
loss function handle zero-inflated retail sales better than standard regression.

**Scope:** CA_1 store × 3,049 products × 3 categories (HOBBIES, HOUSEHOLD, FOODS)  
**Model:** LightGBM — Tweedie objective vs. Gaussian (L2) comparison  
**Dashboard:** Interactive Streamlit app with data exploration, 28-day forecast, and model insights

---

## Quickstart

The only requirement is Docker. No data download, no account, no local setup needed.

```bash
# Build the image (~2-3 minutes on first run)
docker build -t retail-forecast .

# Run the dashboard (data fetch + training happen automatically on first run)
docker run --rm -p 8501:8501 retail-forecast
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

On first run the container will:
1. Download the CA_1 subset (~60 MB) from GitHub Releases
2. Process the data and build the feature matrix
3. Train both Tweedie and Gaussian LightGBM models (~2-3 minutes)
4. Launch the Streamlit dashboard

Subsequent runs skip steps 1–3 if the data and models are already cached in the container.

> **Timing:** First run takes roughly 4–5 minutes total. Subsequent runs launch in seconds.

---

## Data

**Source:** [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy),
Makridakis Open Forecasting Center, 2020. Kaggle competition dataset, open for academic use.

**Subset:** `store_id == 'CA_1'` × all products × all dates
- 3,049 time series (unique items)
- 3 categories: HOBBIES, HOUSEHOLD, FOODS (7 departments)
- 1,941 days (2011-01-29 to 2016-06-19)
- ~5.9M rows in long format

**Why CA_1?** I ran a comparative EDA across all 10 M5 stores first
(see `notebooks/2026-04-22-uk-01-eda-global.ipynb`). CA_1 was selected because it has
good data completeness across all three categories, a representative range of zero-inflation
rates (56–73% depending on category), and a size that fits comfortably within the
5-minute Docker build+run budget. Focusing on a single store also keeps the intermittency
analysis clean — cross-store heterogeneity does not muddy the category-level comparison
that is the core novelty of this project.

**Runtime fetch:** The subset is hosted as a
[GitHub Release asset](https://github.com/umutky/ds570-term-project/releases/tag/v0.1.0-data).
`rf-fetch` downloads it automatically on first run and caches it at
`data/raw/m5_ca1_subset.parquet`. No Kaggle account needed.

**Reproducibility:** To rebuild the subset from the original M5 CSVs:
```bash
# Download the full M5 data into data/raw/ from Kaggle first, then:
uv run python scripts/build_subset.py
```

---

## Methods

### The problem: zero-inflated demand

Retail sales data is full of zeros. In the CA_1 subset, roughly 64% of all daily
item-level sales are zero — ~72% in HOBBIES, ~68% in HOUSEHOLD, and ~56% in FOODS.
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
| **Zero Forecast** | Always predict 0 — a surprisingly competitive lower bound given ~64% zero rate |

All three are evaluated against LightGBM on the same holdout period.

### LightGBM

Two LightGBM models are trained with identical hyperparameters, differing only in objective:

- **Tweedie** (`objective: tweedie`, `tweedie_variance_power: 1.1`) — the main model
- **Gaussian** (`objective: regression`) — comparison baseline

Features are built in `src/retail_forecast/features.py` across four families:

| Family | Features | Motivation |
|---|---|---|
| Lag | `sales_lag_7/14/28/365` | Weekly, monthly, yearly seasonality |
| Rolling | `rolling_mean/std` over 7 and 28 days | Local trend and demand volatility |
| Calendar | day-of-week, month, is_weekend, event flags, SNAP flags | Recurring patterns and promotions |
| Price | sell_price, price change %, price relative to yearly mean | Demand elasticity |

All lag and rolling features use `shift(1)` or greater — no future information leaks into
the feature matrix.

### Train / validation / test split

Data is split chronologically (no shuffling):
- **Train:** 2011-01-29 to 2015-04-30 (~80%)
- **Validation:** 2015-05-01 to 2015-11-10 (~10%) — used for early stopping
- **Test:** 2015-11-11 to 2016-05-22 (~10%) — held out, never seen during training

### Evaluation metrics

- **RMSE** — penalizes large errors heavily; primary metric
- **MAE** — easier to interpret in units (daily item sales)
- **WMAPE** — Weighted Mean Absolute Percentage Error; used instead of MAPE because
  individual zero-sales days make per-row percentage errors undefined

Accuracy is not reported — this is a regression problem.

---

## Results

### Tweedie vs. Gaussian on the test set

| Model | RMSE | MAE | WMAPE |
|---|---|---|---|
| **LightGBM Tweedie** | **1.9718** | **1.0061** | **69.2%** |
| LightGBM Gaussian | 1.9821 | 1.0163 | 69.9% |

Tweedie outperforms Gaussian across all three metrics. The gap is modest at item level
but compounds at higher aggregation levels — at total store level WRMSSE improves from
0.4165 (Gaussian) to 0.3476 (Tweedie), a +0.069 gain.

### Tweedie results by category

| Category | Zero rate | RMSE | MAE | WMAPE |
|---|---|---|---|---|
| FOODS | ~56% | 2.284 | 1.252 | 61.6% |
| HOUSEHOLD | ~68% | 1.254 | 0.730 | 77.9% |
| HOBBIES | ~72% | 2.192 | 0.893 | 95.0% |

HOBBIES shows the highest WMAPE — items in this category are highly intermittent
(many zero-sales days punctuated by occasional bursts), which makes any point forecast
inherently uncertain. FOODS, with the lowest zero rate, is the most predictable category.

---

## Design Decisions

**LightGBM over deep learning:** The M5 competition winners used gradient-boosted trees,
not neural networks. For tabular time-series data with hand-crafted lag features, LightGBM
is faster to train, easier to interpret via feature importance, and does not require GPU
resources. Given the 5-minute Docker constraint, this was also the practical choice.

**Tweedie objective:** Tweedie loss is the correct statistical choice for zero-inflated
non-negative count data. The variance power parameter `p = 1.1` keeps the model close
to Poisson (which is appropriate for count data) while allowing some extra flexibility
for the heavy right tail in FOODS items.

**GitHub Release for data hosting:** The subset parquet is versioned and publicly accessible
without any login. It satisfies the guideline requirement of fully automated data fetch
without requiring the user to sign up for any service.

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
│   ├── streamlit_app.py        # Main dashboard entry point (st.navigation)
│   └── pages/
│       ├── 1_Data_Explorer.py  # Historical sales, zero-inflation, event effects
│       ├── 2_Forecast.py       # 28-day demand forecast by item and category
│       ├── 3_Model_Insights.py # Feature importance, Tweedie vs Gaussian metrics by level
│       └── 4_Forecast_Charts.py# Actual vs predicted time-series at total/cat/dept/item level
├── notebooks/
│   ├── 2026-04-22-uk-01-eda-global.ipynb      # Comparative EDA across all 10 M5 stores → CA_1 selection
│   ├── 2026-04-22-uk-02-eda-ca1-focused.ipynb # Detailed EDA on the CA_1 subset
│   ├── 2026-05-11-uk-03-baseline-vs-lgbm.ipynb
│   └── 2026-05-18-uk-04-tweedie-vs-gaussian.ipynb
├── scripts/
│   └── build_subset.py         # Reproducible script to rebuild the CA_1 subset from raw M5 CSVs
├── src/retail_forecast/
│   ├── config.py               # Paths, data URL, SHA256
│   ├── data/
│   │   ├── fetch.py            # HTTP download with SHA256 verification and caching
│   │   └── load.py             # Parquet loading, wide-to-long transformation
│   ├── features.py             # Lag, rolling, calendar, and price features
│   ├── models/
│   │   ├── baseline.py         # SeasonalNaive, MovingAverage, ZeroForecast
│   │   └── lgbm.py             # LightGBM wrappers (Tweedie + Gaussian)
│   ├── evaluate.py             # RMSE, MAE, WMAPE, expanding-window backtest
│   ├── forecast.py             # 28-day recursive forecasting
│   ├── pipeline.py             # sklearn ColumnTransformer + LightGBM Pipeline
│   └── cli.py                  # Entry points: rf-fetch, rf-process, rf-train, rf-predict
├── tests/
│   ├── test_fetch.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
├── Dockerfile                  # Multi-stage build (builder + runtime)
├── entrypoint.sh               # Runs fetch → process → train → predict → streamlit
├── pyproject.toml
└── uv.lock
```

---

## Local Development

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync --all-extras

# Fetch data
uv run rf-fetch

# Process
uv run rf-process

# Train models
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

- **CA_1 only:** The model is trained on a single store. Performance on TX or WI stores
  is not evaluated.
- **Cold-start items:** Items with very few historical sales (new products or seasonal
  items just entering the catalog) have sparse lag features. The model tends to underfit
  these — a known weakness of lag-based approaches.
- **Rare events:** Events like Super Bowl or Christmas appear very few times in the
  training data. The model learns an average effect but cannot capture
  how an unusual edition of the event might shift demand differently.
- **Point forecasts only:** The pipeline produces single-value predictions.
  Prediction intervals (e.g., quantile regression) would make the uncertainty explicit
  and would be a natural next step.
- **No cross-store generalization:** Adding store embeddings or training a global model
  across all 10 stores would be a meaningful extension.
