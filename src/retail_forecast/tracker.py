"""Experiment tracker for model training runs.

Saves a full JSON log per run and appends one row to a cumulative summary CSV,
making it easy to compare experiments over time.

Usage:
    tracker = ModelTracker("LGBMTweedie")
    tracker.log_data(feat, train_end, val_end)
    tracker.log_params(model.params, num_boost_round=1000)
    tracker.log_training(model)
    tracker.log_metrics(model, train_df, val_df, test_df)
    tracker.log_feature_importance(model.feature_importance)
    tracker.save()
    tracker.print_report()
"""

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.config import REPORTS_DIR
from retail_forecast.evaluate import evaluate, evaluate_by_level, wrmsse, wrmsse_by_level
from retail_forecast.features import FEATURE_COLS

RUNS_DIR = REPORTS_DIR / "model_runs"

_FEATURE_GROUPS: dict[str, list[str]] = {
    "lag":      [c for c in FEATURE_COLS if "sales_lag_" in c],
    "rolling":  [c for c in FEATURE_COLS if "rolling_" in c],
    "calendar": ["wday", "month", "is_weekend", "is_month_start", "is_month_end",
                 "has_event", "has_snap", "doy_sin", "doy_cos", "week_of_year"],
    "price":    ["sell_price", "price_change_pct", "price_rel_year"],
    "id":       ["dept_id", "cat_id"],
}


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


class ModelTracker:
    """Records and persists one model training run."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.ts = datetime.now()
        self.run_id = self.ts.strftime("%Y%m%d_%H%M%S")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._d: dict[str, Any] = {
            "run_id":     self.run_id,
            "model_name": model_name,
            "timestamp":  self.ts.isoformat(timespec="seconds"),
            "git_hash":   _git_hash(),
        }

    # - data & split ---

    def log_data(
        self,
        feat: pd.DataFrame,
        train_end,
        val_end,
        date_col: str = "date",
        id_col: str = "id",
        cat_col: str = "cat_id",
    ) -> "ModelTracker":
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts   = pd.Timestamp(val_end)

        train_mask = feat[date_col] <= train_end_ts
        val_mask   = (feat[date_col] > train_end_ts) & (feat[date_col] <= val_end_ts)
        test_mask  = feat[date_col] > val_end_ts

        active = [c for c in FEATURE_COLS if c in feat.columns]

        cat_test_rows: dict[str, int] = {}
        if cat_col in feat.columns:
            for cat in sorted(feat[cat_col].unique(), key=str):
                cat_test_rows[str(cat)] = int((test_mask & (feat[cat_col] == cat)).sum())

        self._d["split"] = {
            "train_end":    str(train_end_ts.date()),
            "val_end":      str(val_end_ts.date()),
            "test_end":     str(feat[date_col].max().date()),
            "n_train":      int(train_mask.sum()),
            "n_val":        int(val_mask.sum()),
            "n_test":       int(test_mask.sum()),
            "n_items":      int(feat[id_col].nunique()),
            "n_days_train": int(feat.loc[train_mask, date_col].nunique()),
        }
        self._d["features"] = {
            "n_active": len(active),
            "active":   active,
            "groups": {
                grp: [f for f in cols if f in active]
                for grp, cols in _FEATURE_GROUPS.items()
            },
            "test_rows_by_category": cat_test_rows,
        }
        return self

    # - params -

    def log_params(
        self,
        params: dict[str, Any],
        num_boost_round: int = 0,
        early_stopping_rounds: int = 0,
    ) -> "ModelTracker":
        self._d["params"] = {
            **{k: v for k, v in params.items() if k != "verbosity"},
            "num_boost_round":       num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
        }
        return self

    # - training outcome -

    def log_training(self, model) -> "ModelTracker":
        booster = getattr(model, "_booster", None)
        best_score: dict = {}
        if booster and booster.best_score:
            best_score = {
                split: {metric: round(float(v), 6) for metric, v in scores.items()}
                for split, scores in booster.best_score.items()
            }
        self._d["training"] = {
            "best_iteration": int(booster.best_iteration) if booster else None,
            "best_score":     best_score,
        }
        return self

    # - metrics ---

    def log_metrics(
        self,
        model,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> "ModelTracker":
        """Compute RMSE / MAE / WMAPE / WRMSSE at item, dept, cat, and total levels."""

        val_preds  = np.maximum(model.predict(val_df), 0)
        test_preds = np.maximum(model.predict(test_df), 0)

        # Validation: overall only
        val_metrics: dict[str, float] = {
            k: round(v, 6) for k, v in evaluate(val_df["sales"].values, val_preds).items()
        }
        val_metrics["wrmsse"] = round(wrmsse(train_df, val_df, val_preds), 6)

        # Test: all metrics at all aggregation levels
        eval_lvl   = evaluate_by_level(test_df, test_preds)
        wrmsse_lvl = {k: round(v, 6) for k, v in wrmsse_by_level(train_df, test_df, test_preds).items()}

        test_by_level: dict[str, dict[str, float]] = {}
        for lvl, metrics in eval_lvl.items():
            test_by_level[lvl] = dict(metrics)
            if lvl in wrmsse_lvl:
                test_by_level[lvl]["wrmsse"] = wrmsse_lvl[lvl]

        self._d["metrics"] = {
            "val":  val_metrics,
            "test": {"by_level": test_by_level},
        }
        return self

    # - feature importance -

    def log_feature_importance(
        self,
        importance: "pd.Series",
        top_n: int = 20,
    ) -> "ModelTracker":
        total = float(importance.sum()) or 1.0
        self._d["feature_importance"] = {
            feat: round(float(gain) / total * 100, 3)
            for feat, gain in importance.head(top_n).items()
        }
        return self

    # - persistence -

    def save(self) -> Path:
        """Write full JSON log and append one row to summary.csv."""
        json_path = RUNS_DIR / f"{self.run_id}_{self.model_name}.json"
        json_path.write_text(
            json.dumps(self._d, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        csv_path = RUNS_DIR / "summary.csv"
        row = self._flat_row()
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return json_path

    def _flat_row(self) -> dict[str, Any]:
        d = self._d
        row: dict[str, Any] = {
            "run_id":    d["run_id"],
            "timestamp": d["timestamp"],
            "model":     d["model_name"],
            "git_hash":  d["git_hash"],
        }
        for k, v in d.get("split", {}).items():
            row[f"split_{k}"] = v
        for k, v in d.get("params", {}).items():
            row[f"param_{k}"] = v
        row["best_iteration"] = d.get("training", {}).get("best_iteration")
        for metric, val in d.get("metrics", {}).get("val", {}).items():
            row[f"val_{metric}"] = val
        for lvl, metrics in d.get("metrics", {}).get("test", {}).get("by_level", {}).items():
            for metric, val in metrics.items():
                row[f"test_{lvl}_{metric}"] = val
        fi = d.get("feature_importance", {})
        for i, (feat, pct) in enumerate(list(fi.items())[:5], 1):
            row[f"fi_top{i}"] = f"{feat}={pct:.1f}%"
        return row

    # - console report -

    def print_report(self) -> None:
        d = self._d
        W = 66

        def line(text: str = "") -> None:
            print(f"║  {text:<{W - 4}}║")

        def sep() -> None:
            print(f"╠{'═' * (W - 2)}╣")

        print(f"╔{'═' * (W - 2)}╗")
        line(f"Model Run : {d['model_name']}  ·  {d['timestamp']}")
        line(f"Run ID    : {d['run_id']}   git: {d['git_hash']}")
        sep()

        # Split
        sp = d.get("split", {})
        line("DATA")
        line(f"  Items      : {sp.get('n_items', '?'):,}")
        line(f"  Train      : up to {sp.get('train_end')}  "
             f"({sp.get('n_train', 0):>9,} rows  /  {sp.get('n_days_train', '?')} days)")
        line(f"  Validation : {sp.get('train_end')} → {sp.get('val_end')}  "
             f"({sp.get('n_val', 0):>7,} rows)")
        line(f"  Test       : {sp.get('val_end')} → {sp.get('test_end')}  "
             f"({sp.get('n_test', 0):>7,} rows)")
        cat_rows = sp  # already in split, but we use features.test_rows_by_category
        for cat, n in d.get("features", {}).get("test_rows_by_category", {}).items():
            line(f"    {cat:<12}: {n:,} test rows")
        sep()

        # Features
        ft = d.get("features", {})
        line(f"FEATURES  ({ft.get('n_active', 0)} active)")
        for grp, cols in ft.get("groups", {}).items():
            if cols:
                shown = ", ".join(cols[:5])
                suffix = f"  +{len(cols)-5} more" if len(cols) > 5 else ""
                line(f"  {grp:<10}: {shown}{suffix}")
        sep()

        # Params
        line("HYPERPARAMETERS")
        for k, v in d.get("params", {}).items():
            line(f"  {k:<34} {v}")
        tr = d.get("training", {})
        if tr.get("best_iteration"):
            line(f"  {'best_iteration':<34} {tr['best_iteration']}")
        # Best score from LightGBM
        for split_name, scores in tr.get("best_score", {}).items():
            for metric, val in scores.items():
                line(f"  best_score[{split_name}][{metric}]  {val:.6f}")
        sep()

        # Metrics
        m = d.get("metrics", {})
        line("METRICS")

        def mrow(label: str, metrics: dict) -> None:
            r  = metrics.get("rmse",   float("nan"))
            a  = metrics.get("mae",    float("nan"))
            w  = metrics.get("wmape",  float("nan"))
            wr = metrics.get("wrmsse", float("nan"))
            wr_str = f"{wr:>10.4f}" if wr == wr else f"{'n/a':>10}"
            line(f"  {label:<12} {r:>10.4f} {a:>10.4f} {w:>10.4f}{wr_str}")

        # Validation row
        line(f"  {'':12} {'RMSE':>10} {'MAE':>10} {'WMAPE':>10} {'WRMSSE':>10}")
        line("  " + "─" * 54)
        mrow("Validation", m.get("val", {}))
        sep()

        # Test by aggregation level
        line(f"  TEST SET — by aggregation level")
        line(f"  {'Level':<12} {'RMSE':>10} {'MAE':>10} {'WMAPE':>10} {'WRMSSE':>10}")
        line("  " + "─" * 54)
        for lvl, lvl_m in m.get("test", {}).get("by_level", {}).items():
            mrow(lvl, lvl_m)
        sep()

        # Feature importance
        fi = d.get("feature_importance", {})
        if fi:
            line("TOP 10 FEATURES  (% of total gain)")
            for i, (feat, pct) in enumerate(list(fi.items())[:10], 1):
                bar = "█" * max(1, int(pct / 2))
                line(f"  {i:>2}. {feat:<28} {pct:>6.1f}%  {bar}")

        print(f"╚{'═' * (W - 2)}╝")
        json_path = RUNS_DIR / f"{self.run_id}_{self.model_name}.json"
        print(f"  Saved → {json_path}")
        print(f"  CSV   → {RUNS_DIR / 'summary.csv'}")
