"""Baseline forecasting models: SeasonalNaive, MovingAverage, ZeroForecast.

All implement a sklearn-compatible fit/predict API operating on long-format DataFrames.
These serve as the required benchmarks against which LightGBM is evaluated.
"""

import numpy as np
import pandas as pd


class SeasonalNaive:
    """Seasonal naive: repeat the last `season` days of training history cyclically.

    For weekly seasonality (season=7), predicts Monday's sales using last Monday's value.
    Simple but competitive for datasets with strong weekly patterns.
    """

    def __init__(self, season: int = 7) -> None:
        self.season = season
        self._history: dict[str, np.ndarray] = {}

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        sales_col: str = "sales",
        date_col: str = "date",
    ) -> "SeasonalNaive":
        for item_id, grp in df.sort_values(date_col).groupby(id_col):
            self._history[item_id] = grp[sales_col].values
        return self

    def predict(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        **_,
    ) -> np.ndarray:
        preds = np.zeros(len(df))
        df = df.reset_index(drop=True)
        for item_id, grp in df.groupby(id_col):
            history = self._history.get(item_id, np.array([0.0]))
            tail = history[-self.season :]
            n = len(grp)
            repeated = np.tile(tail, n // self.season + 1)[:n]
            preds[grp.index] = repeated
        return preds


class MovingAverage:
    """Predict using the trailing `window`-day mean of training sales."""

    def __init__(self, window: int = 28) -> None:
        self.window = window
        self._means: dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        sales_col: str = "sales",
        date_col: str = "date",
    ) -> "MovingAverage":
        for item_id, grp in df.sort_values(date_col).groupby(id_col):
            vals = grp[sales_col].values
            self._means[item_id] = float(np.mean(vals[-self.window :]))
        return self

    def predict(self, df: pd.DataFrame, id_col: str = "id", **_) -> np.ndarray:
        return np.array([self._means.get(i, 0.0) for i in df[id_col].values])


class ZeroForecast:
    """Trivial lower bound: always predict zero.

    Useful for measuring how much a model improves over doing nothing -
    particularly revealing in high-intermittency categories (HOBBIES).
    """

    def fit(self, *_, **__) -> "ZeroForecast":
        return self

    def predict(self, df: pd.DataFrame, **_) -> np.ndarray:
        return np.zeros(len(df))
