"""Smoke tests for feature engineering — leakage checks and column presence."""

import numpy as np
import pandas as pd
import pytest

from retail_forecast.features import (
    FEATURE_COLS,
    add_calendar_features,
    add_lag_features,
    add_price_features,
    add_rolling_features,
    build_feature_matrix,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal long-format DataFrame: 2 items × 31 days."""
    dates = pd.date_range("2020-01-01", periods=31, freq="D")
    rng = np.random.default_rng(42)
    rows = []
    for item in ["ITEM_A", "ITEM_B"]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "id": item,
                    "item_id": item,
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "sales": int(rng.integers(0, 6)),
                    "date": date,
                    "wday": (i % 7) + 1,
                    "month": date.month,
                    "year": date.year,
                    "event_type_1": None,
                    "snap_CA": 0,
                    "sell_price": 2.5,
                    "wm_yr_wk": 11101 + i // 7,
                }
            )
    return pd.DataFrame(rows)


class TestLagFeatures:
    def test_lag7_matches_7_days_prior(self, sample_df: pd.DataFrame) -> None:
        """sales_lag_7 on day N must equal sales on day N-7 for the same item."""
        df = add_lag_features(sample_df, lags=[7])
        for item_id, grp in df.sort_values("date").groupby("id"):
            grp = grp.reset_index(drop=True)
            for idx in range(7, len(grp)):
                assert grp.loc[idx, "sales_lag_7"] == grp.loc[idx - 7, "sales"], (
                    f"Lag mismatch for {item_id} at idx={idx}"
                )

    def test_first_rows_are_nan(self, sample_df: pd.DataFrame) -> None:
        """First `lag` rows per item must be NaN (no history available)."""
        df = add_lag_features(sample_df, lags=[7])
        for _, grp in df.sort_values("date").groupby("id"):
            grp = grp.reset_index(drop=True)
            assert grp.iloc[:7]["sales_lag_7"].isna().all()

    def test_no_cross_item_leakage(self, sample_df: pd.DataFrame) -> None:
        """Lag features must be computed per item, not across the whole dataframe."""
        df = add_lag_features(sample_df, lags=[1])
        for item_id, grp in df.sort_values("date").groupby("id"):
            grp = grp.reset_index(drop=True)
            # Row 0 (first day of this item) must be NaN — no prior day within this item
            assert np.isnan(grp.iloc[0]["sales_lag_1"])


class TestRollingFeatures:
    def test_no_future_leakage_row0(self, sample_df: pd.DataFrame) -> None:
        """Rolling mean for the very first row of each item must be NaN (no history)."""
        df = add_rolling_features(sample_df, windows=[3], stats=["mean"])
        for _, grp in df.sort_values("date").groupby("id"):
            grp = grp.reset_index(drop=True)
            assert np.isnan(grp.iloc[0]["sales_rolling_mean_3"])

    def test_rolling_columns_created(self, sample_df: pd.DataFrame) -> None:
        df = add_rolling_features(sample_df, windows=[7], stats=["mean", "std"])
        assert "sales_rolling_mean_7" in df.columns
        assert "sales_rolling_std_7" in df.columns


class TestCalendarFeatures:
    def test_required_columns_present(self, sample_df: pd.DataFrame) -> None:
        df = add_calendar_features(sample_df)
        for col in ["is_weekend", "is_month_start", "is_month_end",
                    "has_event", "has_snap", "doy_sin", "doy_cos", "week_of_year"]:
            assert col in df.columns, f"Missing: {col}"

    def test_doy_sin_cos_range(self, sample_df: pd.DataFrame) -> None:
        df = add_calendar_features(sample_df)
        assert df["doy_sin"].between(-1.0, 1.0).all()
        assert df["doy_cos"].between(-1.0, 1.0).all()

    def test_has_event_binary(self, sample_df: pd.DataFrame) -> None:
        df = add_calendar_features(sample_df)
        assert set(df["has_event"].unique()).issubset({0, 1})


class TestPriceFeatures:
    def test_required_columns_present(self, sample_df: pd.DataFrame) -> None:
        df = add_price_features(sample_df)
        assert "price_change_pct" in df.columns
        assert "price_rel_year" in df.columns

    def test_price_rel_year_near_one_for_constant_price(self, sample_df: pd.DataFrame) -> None:
        """When price is constant, price_rel_year should be 1.0 everywhere."""
        df = add_price_features(sample_df)
        assert (df["price_rel_year"].dropna() - 1.0).abs().max() < 1e-9


class TestBuildFeatureMatrix:
    def test_drops_lag_nan_rows(self, sample_df: pd.DataFrame) -> None:
        """With 30 days data, lag_365 is always NaN — build_feature_matrix drops all rows."""
        fm = build_feature_matrix(sample_df)
        # All rows dropped because lag_365 not available in a 30-day sample
        assert len(fm) == 0

    def test_no_lag_nan_in_surviving_rows(self) -> None:
        """Rows that survive dropna must have no NaN in any lag feature."""
        dates = pd.date_range("2019-01-01", periods=400, freq="D")
        rng = np.random.default_rng(7)
        rows = []
        for item in ["A", "B"]:
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "id": item, "item_id": item, "dept_id": "FOODS_1", "cat_id": "FOODS",
                        "store_id": "CA_1", "state_id": "CA",
                        "sales": int(rng.integers(0, 10)),
                        "date": date, "wday": (i % 7) + 1, "month": date.month, "year": date.year,
                        "event_type_1": None, "snap_CA": 0, "sell_price": 2.5,
                        "wm_yr_wk": 11101 + i // 7,
                    }
                )
        df = pd.DataFrame(rows)
        fm = build_feature_matrix(df)
        lag_cols = [c for c in fm.columns if "sales_lag_" in c]
        assert len(fm) > 0, "Expected some rows with 400-day history"
        assert fm[lag_cols].notna().all().all()
