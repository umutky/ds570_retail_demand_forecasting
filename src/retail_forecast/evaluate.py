from typing import Any, Callable

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error.

    denom = sum(|y_true|) so the metric is well-defined even when individual
    actuals are zero — unlike standard MAPE which divides per-row.
    """
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def wrmsse(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    id_col: str = "id",
    sales_col: str = "sales",
    price_col: str = "sell_price",
    date_col: str = "date",
) -> float:
    """Weighted Root Mean Squared Scaled Error (M5 competition metric).

    For each series i:
      scale_i = mean squared naive (day-to-day diff) forecast error on training period.
      RMSSE_i = sqrt(MSE_i / scale_i)
      w_i     = sum(sell_price_i * actual_i) / total_test_revenue

    WRMSSE = sum_i(w_i * RMSSE_i)
    """
    y_pred_clipped = np.maximum(y_pred, 0)

    # Naive MSE scale per series from training history
    train_sorted = train_df[[id_col, date_col, sales_col]].sort_values([id_col, date_col])
    scales: dict[str, float] = {}
    for item_id, grp in train_sorted.groupby(id_col, observed=True):
        sales = grp[sales_col].values.astype(float)
        if len(sales) < 2:
            scales[str(item_id)] = 1.0
            continue
        s = float(np.mean(np.diff(sales) ** 2))
        scales[str(item_id)] = s if s > 0 else 1.0

    # RMSSE and revenue weight per series
    test_copy = test_df[[id_col, sales_col, price_col]].copy()
    test_copy = test_copy.assign(y_pred=y_pred_clipped)

    rmsse_vals: dict[str, float] = {}
    revenue: dict[str, float] = {}

    for item_id, grp in test_copy.groupby(id_col, observed=True):
        key = str(item_id)
        actuals = grp[sales_col].values.astype(float)
        preds = grp["y_pred"].values
        scale = scales.get(key, 1.0)
        rmsse_vals[key] = float(np.sqrt(np.mean((actuals - preds) ** 2) / scale))
        revenue[key] = float(np.sum(grp[price_col].fillna(0).values * actuals))

    total_revenue = sum(revenue.values())
    if total_revenue == 0:
        weights = {k: 1.0 / len(revenue) for k in revenue}
    else:
        weights = {k: v / total_revenue for k, v in revenue.items()}

    return float(sum(weights.get(k, 0.0) * v for k, v in rmsse_vals.items()))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and WMAPE after clipping predictions to >= 0."""
    y_pred_clipped = np.maximum(y_pred, 0)
    return {
        "rmse": rmse(y_true, y_pred_clipped),
        "mae": mae(y_true, y_pred_clipped),
        "wmape": wmape(y_true, y_pred_clipped),
    }


def evaluate_by_level(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    id_col: str = "id",
    sales_col: str = "sales",
    date_col: str = "date",
) -> dict[str, dict[str, float]]:
    """RMSE/MAE/WMAPE at item, department, category, and total aggregation levels.

    Each level sums sales and predictions across items before computing metrics,
    so dept-level RMSE measures accuracy of total department demand forecasts.
    """
    y_pred_clipped = np.maximum(y_pred, 0)
    test_copy = test_df.copy().reset_index(drop=True)
    test_copy["_pred"] = y_pred_clipped

    levels: list[tuple[str, str | None]] = [("item", None)]
    for col in ("dept_id", "cat_id"):
        if col in test_df.columns:
            levels.append((col, col))
    levels.append(("total", "__total__"))

    output: dict[str, dict[str, float]] = {}
    for level_name, grp_col in levels:
        if grp_col is None:
            # item level: raw granularity, no further aggregation
            y_true_agg = test_copy[sales_col].values.astype(float)
            y_pred_agg = test_copy["_pred"].values
        elif grp_col == "__total__":
            agg = (
                test_copy.groupby(date_col, observed=True)
                .agg(**{sales_col: (sales_col, "sum"), "_pred": ("_pred", "sum")})
                .reset_index()
            )
            y_true_agg = agg[sales_col].values.astype(float)
            y_pred_agg = agg["_pred"].values
        else:
            agg = (
                test_copy.groupby([grp_col, date_col], observed=True)
                .agg(**{sales_col: (sales_col, "sum"), "_pred": ("_pred", "sum")})
                .reset_index()
            )
            y_true_agg = agg[sales_col].values.astype(float)
            y_pred_agg = agg["_pred"].values

        output[level_name] = {
            k: round(v, 6) for k, v in evaluate(y_true_agg, y_pred_agg).items()
        }

    return output


def wrmsse_by_level(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    id_col: str = "id",
    sales_col: str = "sales",
    price_col: str = "sell_price",
    date_col: str = "date",
) -> dict[str, float]:
    """WRMSSE at item, department, category, and store (total) aggregation levels.

    For each level the item sales and predictions are summed to form aggregated
    time series, then RMSSE is computed on each aggregate and weighted by its
    revenue share at that level.

    Returns a dict keyed by level name (e.g. 'item', 'dept_id', 'cat_id', 'total').
    """
    y_pred_clipped = np.maximum(y_pred, 0)
    test_with_pred = test_df.copy()
    test_with_pred["y_pred"] = y_pred_clipped

    # Determine which grouping columns exist in both DataFrames
    candidate_levels: list[tuple[str, str]] = [("item", id_col)]
    for col in ("dept_id", "cat_id"):
        if col in test_df.columns and col in train_df.columns:
            candidate_levels.append((col, col))
    candidate_levels.append(("total", "__total__"))

    def _compute_level(grp_col: str, is_total: bool) -> float:
        if is_total:
            tr = train_df.copy()
            te = test_with_pred.copy()
            tr[grp_col] = "total"
            te[grp_col] = "total"
        else:
            tr = train_df
            te = test_with_pred

        # Aggregate sales / preds by (group, date)
        tr_agg = (
            tr.groupby([grp_col, date_col], observed=True)[sales_col]
            .sum()
            .reset_index()
        )
        te_agg = (
            te.groupby([grp_col, date_col], observed=True)
            .agg(**{sales_col: (sales_col, "sum"), "y_pred": ("y_pred", "sum")})
            .reset_index()
        )

        # Revenue weight per group
        if price_col in te.columns:
            rev = (
                te.assign(_rev=te[sales_col] * te[price_col].fillna(0))
                .groupby(grp_col, observed=True)["_rev"]
                .sum()
            )
        else:
            rev = te_agg.groupby(grp_col, observed=True)[sales_col].sum()

        total_rev = float(rev.sum())
        n_groups = max(len(rev), 1)
        weights = (
            (rev / total_rev).to_dict()
            if total_rev > 0
            else {k: 1.0 / n_groups for k in rev.index}
        )

        # Naive MSE scale per group from training history
        scales: dict[str, float] = {}
        for gid, grp in tr_agg.groupby(grp_col, observed=True):
            vals = grp.sort_values(date_col)[sales_col].values.astype(float)
            s = float(np.mean(np.diff(vals) ** 2)) if len(vals) > 1 else 1.0
            scales[str(gid)] = s if s > 0 else 1.0

        # Weighted RMSSE
        result = 0.0
        for gid, grp in te_agg.groupby(grp_col, observed=True):
            actuals = grp[sales_col].values.astype(float)
            preds = grp["y_pred"].values
            scale = scales.get(str(gid), 1.0)
            rmsse_i = float(np.sqrt(np.mean((actuals - preds) ** 2) / scale))
            w = weights.get(gid, weights.get(str(gid), 0.0))
            result += w * rmsse_i
        return result

    output: dict[str, float] = {}
    for level_name, col in candidate_levels:
        is_total = col == "__total__"
        output[level_name] = _compute_level(col, is_total)
    return output


def time_series_backtest(
    df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame, pd.DataFrame], Any],
    date_col: str = "date",
    n_folds: int = 3,
    horizon: int = 28,
) -> pd.DataFrame:
    """Expanding-window time-series cross-validation.

    For each fold:
      - Train on all data up to the cutoff date (expanding window).
      - Evaluate on the next `horizon` days.
      - model_factory(train_df, val_df) must return a fitted model with .predict(df).

    Cutoffs are spaced evenly over the last 60% of available dates, leaving
    the first 40% always in training (ensures lag_365 features are populated).

    Args:
        df: Feature matrix from build_feature_matrix(), sorted by (id, date).
        model_factory: callable(train_df, val_df) -> fitted model.
        date_col: Name of the date column.
        n_folds: Number of expanding-window folds.
        horizon: Forecast horizon in days per fold.

    Returns:
        DataFrame with columns [fold, cutoff, rmse, mae, wmape, n_rows].
    """
    dates = np.sort(df[date_col].unique())
    n_dates = len(dates)

    # Reserve the first 40% strictly for training history to ensure lag features exist
    eval_start_idx = int(n_dates * 0.40)
    eval_dates = dates[eval_start_idx:]

    if len(eval_dates) < n_folds * horizon:
        raise ValueError(
            f"Not enough dates for {n_folds} folds × {horizon}-day horizon. "
            f"Available eval window: {len(eval_dates)} days."
        )

    step = len(eval_dates) // (n_folds + 1)
    cutoffs = [eval_dates[i * step] for i in range(1, n_folds + 1)]

    results = []
    for fold, cutoff in enumerate(cutoffs, start=1):
        cutoff_ts = pd.Timestamp(cutoff)
        val_start = cutoff_ts + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=horizon - 1)

        train_df = df[df[date_col] <= cutoff_ts]
        val_df = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)]

        if len(val_df) == 0:
            continue

        model = model_factory(train_df, val_df)
        preds = model.predict(val_df)
        metrics = evaluate(val_df["sales"].values, preds)

        wrmsse_val = None
        if "sell_price" in val_df.columns and "sell_price" in train_df.columns:
            wrmsse_val = wrmsse(train_df, val_df, np.maximum(preds, 0))
            metrics["wrmsse"] = wrmsse_val

        row = {
            "fold": fold,
            "cutoff": cutoff_ts.date(),
            "n_train": len(train_df),
            "n_val": len(val_df),
            **metrics,
        }
        results.append(row)
        wrmsse_str = f" | WRMSSE={wrmsse_val:.4f}" if wrmsse_val is not None else ""
        print(
            f"  Fold {fold} | cutoff={cutoff_ts.date()} | "
            f"RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f} | WMAPE={metrics['wmape']:.4f}"
            f"{wrmsse_str}"
        )

    return pd.DataFrame(results)
