#!/bin/bash
set -e

MODEL_PATH=/app/outputs/models/lgbm_tweedie.pkl
FEATURE_MATRIX_PATH=/app/data/processed/feature_matrix.parquet
SALES_LONG_PATH=/app/data/processed/sales_long.parquet

if [ ! -f "$MODEL_PATH" ]; then
    echo "First run: setting up data and models..."

    # Step 1: Download raw CA_1 subset (~22 MB)
    rf-fetch

    # Step 2: Build sales_long.parquet (fast, needed by rf-predict & forecast)
    if [ ! -f "$SALES_LONG_PATH" ]; then
        echo "Processing raw data into long format..."
        python3 -c "
from retail_forecast.data.fetch import fetch
from retail_forecast.data.load import load_subset, save_processed
from retail_forecast.config import PROCESSED_DATA_DIR
path = fetch()
df = load_subset(path)
save_processed(df, 'sales_long')
print(f'sales_long.parquet saved ({len(df):,} rows)')
"
    fi

    # Step 3: Download pre-built feature matrix (~76 MB, skips ~8 min of rf-process)
    if [ ! -f "$FEATURE_MATRIX_PATH" ]; then
        rf-fetch-features
    fi

    # Step 4: Train models (~3 min)
    rf-train

    # Step 5: Generate 28-day forecast
    rf-predict
fi

exec streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
