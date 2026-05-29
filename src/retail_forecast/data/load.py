import pandas as pd
from pathlib import Path

from retail_forecast.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SUBSET_FILE


def load_subset(path: Path | None = None) -> pd.DataFrame:
    """Load the CA_1 subset parquet file.

    Args:
        path: Path to parquet file (defaults to data/raw/m5_ca1_subset.parquet).

    Returns:
        DataFrame with the subset data.
    """
    if path is None:
        path = RAW_DATA_DIR / SUBSET_FILE
    return pd.read_parquet(path)


def melt_to_long(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert wide-format sales data to long format.

    Input columns:  id, item_id, dept_id, cat_id, store_id, state_id, d_1 ... d_1941
    Output columns: same id columns + d + sales, optionally joined with calendar.

    Args:
        df: Wide-format sales DataFrame (one row per item, one column per day).
        calendar_df: Optional calendar DataFrame to join date and event metadata.

    Returns:
        Long-format DataFrame sorted by (id, d).
    """
    id_cols = [c for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] if c in df.columns]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    long = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    if calendar_df is not None:
        cal_cols = [
            "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        keep = [c for c in cal_cols if c in calendar_df.columns]
        long = long.merge(calendar_df[keep], on="d", how="left")
        long["date"] = pd.to_datetime(long["date"])

    # Sort by numeric day index - string sort gives d_1, d_10, d_100 which breaks time order
    long["_day_num"] = long["d"].str[2:].astype(int)
    long = long.sort_values(["id", "_day_num"]).drop(columns=["_day_num"])
    return long.reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame to data/processed/<name>.parquet.

    Args:
        df: DataFrame to persist.
        name: Base filename without extension, e.g. 'sales_long'.

    Returns:
        Path to the written parquet file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")
    return path
