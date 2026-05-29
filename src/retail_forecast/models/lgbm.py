"""LightGBM wrappers for Tweedie and Gaussian objectives.

LGBMTweedie: suited for zero-inflated demand (the novelty angle).
LGBMGaussian: standard L2 regression used as a comparison baseline.

Both expose a uniform fit/predict/save/load API and a feature_importance property.
"""

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecast.features import FEATURE_COLS, TARGET_COL

_DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


class LGBMForecast:
    """Base LightGBM wrapper - subclass to set `objective`."""

    objective: str = "regression"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.params = {**_DEFAULT_PARAMS, "objective": self.objective, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self._booster: lgb.Booster | None = None
        self.feature_names_: list[str] = []

    def _prep(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        available = [c for c in FEATURE_COLS if c in df.columns]
        return df[available], available

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "LGBMForecast":
        X_train, feats = self._prep(train_df)
        y_train = train_df[TARGET_COL].values
        self.feature_names_ = feats

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        callbacks: list = [lgb.log_evaluation(period=100)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if val_df is not None:
            X_val, _ = self._prep(val_df)
            y_val = val_df[TARGET_COL].values
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self._booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted - call fit() first.")
        X, _ = self._prep(df)
        return self._booster.predict(X)

    @property
    def feature_importance(self) -> pd.Series:
        if self._booster is None:
            raise RuntimeError("Model not fitted.")
        return pd.Series(
            self._booster.feature_importance("gain"),
            index=self._booster.feature_name(),
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved -> {path}")

    @classmethod
    def load(cls, path: Path) -> "LGBMForecast":
        with open(Path(path), "rb") as f:
            return pickle.load(f)


class LGBMTweedie(LGBMForecast):
    """LightGBM with Tweedie objective - designed for zero-inflated demand.

    Tweedie loss (1 < p < 2) lies between Poisson (p=1) and Gamma (p=2),
    making it well-suited for retail data that is a mix of zeros and positive counts.
    M5 competition winners used this objective for exactly this reason.
    """

    objective = "tweedie"

    def __init__(self, tweedie_variance_power: float = 1.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.params["tweedie_variance_power"] = tweedie_variance_power


class LGBMGaussian(LGBMForecast):
    """LightGBM with L2 (Gaussian) objective - comparison baseline.

    Standard squared-error regression. Compared against LGBMTweedie on the
    same feature set to quantify the benefit of the Tweedie loss on intermittent demand.
    """

    objective = "regression"
