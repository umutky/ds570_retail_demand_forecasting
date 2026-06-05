from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# CA_1 subset - hosted as a GitHub Release asset.
SUBSET_FILE = "m5_ca1_subset.parquet"
DATA_URL = "https://github.com/umutky/ds570_retail_demand_forecasting/releases/download/v0.1.0-data/m5_ca1_subset.parquet"
DATA_SHA256 = "47e7029ffa1478d4127562a9765cd3b08fc03c5be91cd44ccd3a0c982e50b8b1"

# Pre-built feature matrix - skips rf-process on first Docker run (~8 min saved).
# Regenerate locally with `rf-process` if features change.
FEATURE_MATRIX_FILE = "feature_matrix.parquet"
FEATURE_MATRIX_URL = "https://github.com/umutky/ds570_retail_demand_forecasting/releases/download/v0.1.0-data/feature_matrix.parquet"
FEATURE_MATRIX_SHA256 = "0f87a5789f53e471276033416deba6f8811f0042fb0d363e524780f1223701d3"
