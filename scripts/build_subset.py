"""One-time script: build the CA_1 subset parquet from the full M5 dataset.

Prerequisites:
    data/raw/sales_train_evaluation.csv
    data/raw/calendar.csv
    data/raw/sell_prices.csv

    Download the full M5 dataset manually from Kaggle:
    https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Usage:
    uv run python scripts/build_subset.py

Output:
    data/raw/m5_ca1_subset.parquet   (~50-80 MB, long format with calendar + prices)
    SHA256 hash printed to stdout - copy this value into config.py DATA_SHA256.

After running:
    1. Create a GitHub Release tagged v0.1.0-data.
    2. Upload m5_ca1_subset.parquet as a release asset.
    3. Update config.py: DATA_URL and DATA_SHA256.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
STORE_ID = "CA_1"
OUT_FILE = RAW / "m5_ca1_subset.parquet"

REQUIRED = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]


def _check_inputs() -> None:
    missing = [f for f in REQUIRED if not (RAW / f).exists()]
    if missing:
        print(f"Missing files in {RAW}:")
        for f in missing:
            print(f"  {f}")
        print("\nDownload from Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        sys.exit(1)


def build() -> Path:
    _check_inputs()

    print("Reading sales_train_evaluation.csv ...")
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    print(f"  Full shape: {sales.shape}")

    print(f"Filtering store_id == '{STORE_ID}' ...")
    subset = sales[sales["store_id"] == STORE_ID].reset_index(drop=True)
    print(f"  Subset shape: {subset.shape}  ({len(subset):,} items)")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in subset.columns if c.startswith("d_")]
    print("Melting to long format ...")
    long = subset.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
    long["_day_num"] = long["d"].str[2:].astype(int)
    long = long.sort_values(["id", "_day_num"]).drop(columns=["_day_num"]).reset_index(drop=True)
    print(f"  Long shape: {long.shape}")

    print("Joining calendar ...")
    calendar = pd.read_csv(RAW / "calendar.csv")
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    keep = [c for c in cal_cols if c in calendar.columns]
    long = long.merge(calendar[keep], on="d", how="left")
    long["date"] = pd.to_datetime(long["date"])

    print("Joining sell prices ...")
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices_ca1 = prices[prices["store_id"] == STORE_ID]
    long = long.merge(prices_ca1[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
                      on=["store_id", "item_id", "wm_yr_wk"], how="left")

    print(f"Writing to {OUT_FILE} ...")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT_FILE, index=False, compression="snappy")

    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    sha256 = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()

    print(f"\nDone!")
    print(f"  Rows:    {len(long):,}")
    print(f"  Columns: {list(long.columns)}")
    print(f"  Size:    {size_mb:.1f} MB")
    print(f"  SHA256:  {sha256}")
    print(f"\nNext steps:")
    print(f"  1. Upload {OUT_FILE.name} to GitHub Release v0.1.0-data")
    print(f"  2. Set DATA_SHA256 = '{sha256}' in src/retail_forecast/config.py")

    return OUT_FILE


if __name__ == "__main__":
    build()
