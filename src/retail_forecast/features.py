"""Feature engineering for the CA_1 retail demand dataset.

All lag and rolling features use shift(1) or greater to prevent target leakage —
no future sales information leaks into the feature matrix.

Input: long-format DataFrame (from load.py), sorted by (id, date).
Output: DataFrame with FEATURE_COLS + sales + date + id, NaN rows dropped.
"""

import numpy as np
import pandas as pd


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add per-item lag features shifted by n days within each item group."""
    if lags is None:
        lags = [1, 2, 3, 4, 5, 6, 7, 14, 28, 365]
    df = _sort(df)
    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby("id")["sales"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Add per-item rolling statistics, computed on lag-1 sales to prevent leakage."""
    if windows is None:
        windows = [7, 28]
    if stats is None:
        stats = ["mean", "std"]
    df = _sort(df)
    # Shift by 1 within each item before rolling so day N's feature uses days ≤ N-1
    df["_lag1"] = df.groupby("id")["sales"].shift(1)
    for window in windows:
        for stat in stats:
            min_p = 1 if stat == "mean" else 2
            df[f"sales_rolling_{stat}_{window}"] = df.groupby("id")["_lag1"].transform(
                lambda s, w=window, mp=min_p, st=stat: s.rolling(w, min_periods=mp).agg(st)
            )
    return df.drop(columns=["_lag1"])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional calendar features from the existing date and event columns."""
    df = df.copy()
    df["is_weekend"] = df["wday"].isin([1, 2]).astype(int)  # wday=1 Sat, 2 Sun in M5
    df["is_month_start"] = (df["date"].dt.day <= 3).astype(int)
    df["is_month_end"] = (df["date"].dt.day >= 28).astype(int)
    df["has_event"] = df["event_type_1"].notna().astype(int)
    df["has_snap"] = df["snap_CA"].astype(int)
    doy = df["date"].dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price momentum and trailing-mean relative price features."""
    df = _sort(df)
    df["price_change_pct"] = df.groupby("id")["sell_price"].pct_change().fillna(0)
    # Trailing 365-day mean of the *previous* day's price to avoid any future leakage.
    trailing_mean = df.groupby("id")["sell_price"].transform(
        lambda s: s.shift(1).rolling(365, min_periods=30).mean()
    )
    df["price_rel_year"] = df["sell_price"] / trailing_mean.replace(0, np.nan)
    return df


FEATURE_COLS: list[str] = [
    # Lag features
    "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_4", "sales_lag_5", "sales_lag_6",
    "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_365",
    # Rolling statistics
    "sales_rolling_mean_7", "sales_rolling_mean_28",
    "sales_rolling_std_7", "sales_rolling_std_28",
    # Calendar
    "wday", "month", "is_weekend", "is_month_start", "is_month_end",
    "has_event", "has_snap", "doy_sin", "doy_cos", "week_of_year",
    # Price
    "sell_price", "price_change_pct", "price_rel_year",
    # Categorical IDs — LightGBM handles these natively as category dtype
    "dept_id", "cat_id",
]

TARGET_COL = "sales"
DATE_COL = "date"
ID_COL = "id"


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering and return a clean, model-ready matrix.

    Drops rows where any lag feature is NaN (items' early history before lag window).
    Encodes dept_id and cat_id as pd.Categorical for LightGBM native handling.

    Returns:
        DataFrame with FEATURE_COLS + TARGET_COL + DATE_COL + ID_COL columns.
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_price_features(df)

    for col in ["dept_id", "cat_id"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    keep = FEATURE_COLS + [TARGET_COL, DATE_COL, ID_COL]
    available = [c for c in keep if c in df.columns]
    lag_cols = [c for c in FEATURE_COLS if "sales_lag_" in c and c in available]
    df = df[available].dropna(subset=lag_cols)
    return df.reset_index(drop=True)
