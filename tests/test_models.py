"""Smoke tests for baseline and LightGBM models."""

import numpy as np
import pandas as pd
import pytest

from retail_forecast.evaluate import evaluate, mae, rmse, wmape
from retail_forecast.features import build_feature_matrix
from retail_forecast.models.baseline import MovingAverage, SeasonalNaive, ZeroForecast
from retail_forecast.models.lgbm import LGBMGaussian, LGBMTweedie


@pytest.fixture()
def two_item_400d() -> pd.DataFrame:
    """400-day two-item dataset - long enough for lag_365 features to exist."""
    dates = pd.date_range("2019-01-01", periods=400, freq="D")
    rng = np.random.default_rng(0)
    rows = []
    for item in ["ITEM_A", "ITEM_B"]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "id": item, "item_id": item, "dept_id": "FOODS_1", "cat_id": "FOODS",
                    "store_id": "CA_1", "state_id": "CA",
                    "sales": int(rng.integers(0, 10)),
                    "date": date, "wday": (i % 7) + 1, "month": date.month, "year": date.year,
                    "event_type_1": None, "snap_CA": 0,
                    "sell_price": 2.5 + rng.uniform(-0.1, 0.1),
                    "wm_yr_wk": 11101 + i // 7,
                }
            )
    return pd.DataFrame(rows)


# Metric unit tests
class TestMetrics:
    def test_rmse_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mae_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_wmape_zero_actuals(self) -> None:
        """WMAPE must return 0.0 when all actuals are zero (avoid division by zero)."""
        assert wmape(np.zeros(5), np.ones(5)) == 0.0

    def test_wmape_nonzero(self) -> None:
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([1.0, 3.0])
        # |2-1| + |4-3| = 2; sum(|y_true|) = 6 -> WMAPE = 2/6 ≈ 0.333
        assert wmape(y_true, y_pred) == pytest.approx(2 / 6, rel=1e-6)

    def test_evaluate_clips_negatives(self) -> None:
        """evaluate() must clip predictions to >= 0 before computing metrics."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([-1.0, 2.0])  # first pred is negative
        metrics = evaluate(y_true, y_pred)
        # Clipped pred = [0, 2]; errors = [1, 0]; RMSE = sqrt(0.5) ≈ 0.707
        assert metrics["rmse"] == pytest.approx(np.sqrt(0.5), rel=1e-6)


# Baseline model tests

class TestSeasonalNaive:
    def test_output_length(self, two_item_400d: pd.DataFrame) -> None:
        train = two_item_400d[two_item_400d["date"] < "2020-01-01"]
        test = two_item_400d[two_item_400d["date"] >= "2020-01-01"]
        preds = SeasonalNaive(season=7).fit(train).predict(test)
        assert len(preds) == len(test)

    def test_no_nan_in_preds(self, two_item_400d: pd.DataFrame) -> None:
        train = two_item_400d[two_item_400d["date"] < "2020-01-01"]
        test = two_item_400d[two_item_400d["date"] >= "2020-01-01"]
        preds = SeasonalNaive(season=7).fit(train).predict(test)
        assert not np.isnan(preds).any()

    def test_predicts_last_cycle(self, two_item_400d: pd.DataFrame) -> None:
        """All predictions come from the finite set of last-season training sales."""
        train = two_item_400d[two_item_400d["date"] < "2020-01-01"].copy()
        test = two_item_400d[two_item_400d["date"] >= "2020-01-01"].copy()
        model = SeasonalNaive(season=7).fit(train)
        full_test = test.reset_index(drop=True)
        preds = model.predict(full_test)
        for item_id in ["ITEM_A", "ITEM_B"]:
            item_train = train[train["id"] == item_id].sort_values("date")
            last_season = set(item_train["sales"].values[-7:].tolist())
            item_mask = full_test["id"] == item_id
            item_preds = set(preds[item_mask.values].tolist())
            assert item_preds.issubset(last_season)


class TestMovingAverage:
    def test_output_finite(self, two_item_400d: pd.DataFrame) -> None:
        train = two_item_400d[two_item_400d["date"] < "2020-01-01"]
        test = two_item_400d[two_item_400d["date"] >= "2020-01-01"]
        preds = MovingAverage(window=28).fit(train).predict(test)
        assert np.isfinite(preds).all()

    def test_unknown_item_returns_zero(self, two_item_400d: pd.DataFrame) -> None:
        model = MovingAverage().fit(two_item_400d)
        dummy = pd.DataFrame([{"id": "UNKNOWN_ITEM"}])
        preds = model.predict(dummy)
        assert preds[0] == 0.0


class TestZeroForecast:
    def test_always_zero(self, two_item_400d: pd.DataFrame) -> None:
        preds = ZeroForecast().fit(two_item_400d).predict(two_item_400d)
        assert np.all(preds == 0)


# LightGBM smoke tests

@pytest.fixture()
def feature_matrix(two_item_400d: pd.DataFrame) -> pd.DataFrame:
    fm = build_feature_matrix(two_item_400d)
    if len(fm) == 0:
        pytest.skip("Feature matrix empty - dataset too short for lag_365.")
    return fm


class TestLGBMTweedie:
    def test_fits_and_predicts(self, feature_matrix: pd.DataFrame) -> None:
        split = feature_matrix["date"].quantile(0.8)
        train = feature_matrix[feature_matrix["date"] <= split]
        val = feature_matrix[feature_matrix["date"] > split]
        model = LGBMTweedie(num_boost_round=10, early_stopping_rounds=5)
        model.fit(train, val)
        preds = model.predict(val)
        assert len(preds) == len(val)
        assert np.isfinite(preds).all()
        assert (preds >= 0).all(), "Tweedie output must be non-negative"

    def test_feature_importance_available(self, feature_matrix: pd.DataFrame) -> None:
        split = feature_matrix["date"].quantile(0.8)
        train = feature_matrix[feature_matrix["date"] <= split]
        val = feature_matrix[feature_matrix["date"] > split]
        model = LGBMTweedie(num_boost_round=10).fit(train, val)
        fi = model.feature_importance
        assert len(fi) > 0
        assert fi.index[0] in model.feature_names_

    def test_save_and_load(self, feature_matrix: pd.DataFrame, tmp_path) -> None:
        split = feature_matrix["date"].quantile(0.8)
        train = feature_matrix[feature_matrix["date"] <= split]
        val = feature_matrix[feature_matrix["date"] > split]
        model = LGBMTweedie(num_boost_round=5).fit(train, val)
        path = tmp_path / "lgbm_tweedie.pkl"
        model.save(path)
        loaded = LGBMTweedie.load(path)
        preds_orig = model.predict(val)
        preds_loaded = loaded.predict(val)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestLGBMGaussian:
    def test_fits_and_predicts(self, feature_matrix: pd.DataFrame) -> None:
        split = feature_matrix["date"].quantile(0.8)
        train = feature_matrix[feature_matrix["date"] <= split]
        val = feature_matrix[feature_matrix["date"] > split]
        model = LGBMGaussian(num_boost_round=10, early_stopping_rounds=5)
        model.fit(train, val)
        preds = model.predict(val)
        assert len(preds) == len(val)
        assert np.isfinite(preds).all()
