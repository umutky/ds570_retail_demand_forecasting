"""Recursive 28-day ahead demand forecast.

For a 28-day horizon the recursion boundary is:
  - sales_lag_28  on forecast day N: always from historical data (N-28 <= last_date).
  - sales_lag_365 on any forecast day: always from historical data.
  - sales_lag_7   days 1-7: historical; days 8-28: prior predictions.
  - sales_lag_14  days 1-14: historical; days 15-28: prior predictions.

Rolling features are recomputed each step from a growing pivot table that
accumulates the predictions from previous steps alongside the historical data.

Calendar features for future dates:
  - Arithmetic fields (wday, month, doy_*) are derived directly from the date.
  - SNAP and event flags are approximated from the same calendar day one year prior.

Prices are carried forward from the last known value per item.
"""

import numpy as np
import pandas as pd

from retail_forecast.features import DATE_COL, FEATURE_COLS, ID_COL, TARGET_COL

# M5 wday convention: Sat=1, Sun=2, Mon=3, Tue=4, Wed=5, Thu=6, Fri=7
# Python dayofweek:   Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
_M5_WDAY = [3, 4, 5, 6, 7, 1, 2]

# Derive lag numbers directly from FEATURE_COLS so forecast.py stays in sync
# with features.py automatically when lags are added or removed.
_LAG_NUMS: list[int] = sorted(
    int(c.replace("sales_lag_", ""))
    for c in FEATURE_COLS
    if c.startswith("sales_lag_")
)


def forecast_future(
    model,
    df: pd.DataFrame,
    horizon: int = 28,
) -> pd.DataFrame:
    """Generate recursive demand forecasts for the next `horizon` days.

    Args:
        model: Fitted LGBMForecast (or any object with .predict(df)).
        df: Processed sales DataFrame in long format (output of rf-process).
            Required columns: id, date, sales, wday, month, year, dept_id,
            cat_id, event_type_1, snap_CA, sell_price.
        horizon: Number of calendar days to forecast ahead (default 28).

    Returns:
        DataFrame with columns [id, date, y_pred] sorted by (id, date).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()

    # Build pivot table: rows=items, columns=dates, values=sales.
    # Keep enough history to cover the furthest lag (365 days) plus the rolling
    # window (28 days). 10-day buffer for safety.
    hist_start = last_date - pd.Timedelta(days=380)
    pivot = (
        df[df["date"] >= hist_start]
        .pivot_table(index=ID_COL, columns="date", values=TARGET_COL, fill_value=0.0)
    )
    item_ids: list[str] = pivot.index.tolist()
    n_items = int(len(item_ids))

    # Item metadata: dept_id, cat_id, last known price
    meta = df.drop_duplicates(ID_COL).set_index(ID_COL).reindex(item_ids)

    # Carry forward the last non-null sell_price per item
    price_df = df.dropna(subset=["sell_price"]).sort_values("date")
    last_price_series = (
        price_df.groupby(ID_COL)["sell_price"]
        .last()
        .reindex(item_ids)
        .fillna(1.0)
    )
    last_price: np.ndarray = last_price_series.values.astype(float)

    # Trailing 365-day price mean per item (mirrors the feature engineering logic)
    trailing_mean_series = (
        price_df.groupby(ID_COL)["sell_price"]
        .apply(lambda s: s.tail(365).mean())
        .reindex(item_ids)
        .fillna(1.0)
    )
    price_rel_year: np.ndarray = (last_price / trailing_mean_series.values).astype(float)

    # Preserve category encoding from training data
    dept_cats = df["dept_id"].astype("category").cat.categories
    cat_cats = df["cat_id"].astype("category").cat.categories

    dept_values = pd.Categorical(meta["dept_id"].values, categories=dept_cats)
    cat_values = pd.Categorical(meta["cat_id"].values, categories=cat_cats)

    # Calendar approximation for future dates
    cal = (
        df[["date", "snap_CA", "event_type_1"]]
        .drop_duplicates("date")
        .set_index("date")
    )

    def _snap(date: pd.Timestamp) -> int:
        proxy = date - pd.Timedelta(days=364)
        return int(cal["snap_CA"].get(proxy, 0))

    def _has_event(date: pd.Timestamp) -> int:
        proxy = date - pd.Timedelta(days=364)
        val = cal["event_type_1"].get(proxy, None)
        return 0 if (val is None or (isinstance(val, float) and np.isnan(val))) else 1

    # Recursive forecasting loop
    all_steps: list[pd.DataFrame] = []

    for step in range(1, horizon + 1):
        fdate: pd.Timestamp = last_date + pd.Timedelta(days=step)
        doy = int(fdate.day_of_year)

        # Vectorised lag lookups (one pivot column per lag)
        def _lag(days: int) -> np.ndarray:
            d = fdate - pd.Timedelta(days=days)
            return pivot[d].values if d in pivot.columns else np.zeros(n_items)

        lag_arrays = {f"sales_lag_{n}": _lag(n) for n in _LAG_NUMS}

        # Rolling window: 28 lagged dates, vectorised via column_stack
        roll_cols = [fdate - pd.Timedelta(days=i) for i in range(1, 29)]
        roll_matrix = np.column_stack([
            pivot[d].values if d in pivot.columns else np.zeros(n_items)
            for d in roll_cols
        ])  # shape: (n_items, 28)

        rm7  = roll_matrix[:, :7].mean(axis=1)
        rm28 = roll_matrix.mean(axis=1)
        rs7  = roll_matrix[:, :7].std(axis=1)
        rs28 = roll_matrix.std(axis=1)

        # Calendar scalars (broadcast across all items in the DataFrame constructor)
        step_df = pd.DataFrame(
            {
                ID_COL:   item_ids,
                DATE_COL: fdate,
                "dept_id": dept_values,
                "cat_id":  cat_values,
                **lag_arrays,
                # Rolling statistics
                "sales_rolling_mean_7":  rm7,
                "sales_rolling_mean_28": rm28,
                "sales_rolling_std_7":   rs7,
                "sales_rolling_std_28":  rs28,
                # Calendar
                "wday":             _M5_WDAY[fdate.dayofweek],
                "month":            fdate.month,
                "is_weekend":       int(fdate.dayofweek >= 5),
                "is_month_start":   int(fdate.day <= 3),
                "is_month_end":     int(fdate.day >= 28),
                "has_event":        _has_event(fdate),
                "has_snap":         _snap(fdate),
                "doy_sin":          float(np.sin(2 * np.pi * doy / 365.25)),
                "doy_cos":          float(np.cos(2 * np.pi * doy / 365.25)),
                "week_of_year":     int(fdate.isocalendar().week),
                # Price (carry forward, no change assumed)
                "sell_price":       last_price,
                "price_change_pct": 0.0,
                "price_rel_year":   price_rel_year,
            }
        )

        preds: np.ndarray = np.maximum(model.predict(step_df), 0.0)

        # Add predictions to pivot so subsequent steps can use them as lags
        pivot[fdate] = preds

        all_steps.append(
            pd.DataFrame(
                {ID_COL: item_ids, DATE_COL: fdate, "y_pred": preds}
            )
        )

    result = pd.concat(all_steps, ignore_index=True)
    return result.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
