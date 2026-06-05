"""Pre-demo preflight check: verify every artifact the dashboard needs is present,
loadable, and internally consistent. Run this right before a live demo.

Usage:
    uv run python scripts/preflight_check.py

Exit code 0 = all critical checks passed (safe to demo).
Exit code 1 = at least one critical check failed (fix before demoing).

Checks are CRITICAL (block the demo) or INFO (nice-to-know, never blocks).
No network is required: every check reads local files only. The one network
check (GitHub Release reachability) is INFO, because once the data is cached
locally the live demo never touches the network.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

from retail_forecast import config
from retail_forecast.features import FEATURE_COLS

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

results: list[tuple[str, str, str]] = []  # (status, name, detail)


def ok(name: str, detail: str = "") -> None:
    results.append(("PASS", name, detail))


def fail(name: str, detail: str = "") -> None:
    results.append(("FAIL", name, detail))


def info(name: str, detail: str = "") -> None:
    results.append(("INFO", name, detail))


def _mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


# --- 1. Required files exist + non-trivial size ------------------------------
REQUIRED = {
    "raw subset":          config.RAW_DATA_DIR / "m5_ca1_subset.parquet",
    "sales_long":          config.PROCESSED_DATA_DIR / "sales_long.parquet",
    "feature_matrix":      config.PROCESSED_DATA_DIR / "feature_matrix.parquet",
    "model: tweedie":      config.MODELS_DIR / "lgbm_tweedie.pkl",
    "model: gaussian":     config.MODELS_DIR / "lgbm_gaussian.pkl",
    "test preds tweedie":  config.REPORTS_DIR / "test_predictions_tweedie.parquet",
    "test preds gaussian": config.REPORTS_DIR / "test_predictions_gaussian.parquet",
    "forecast 28d":        config.REPORTS_DIR / "forecast_28d.parquet",
}
for label, path in REQUIRED.items():
    if path.exists() and path.stat().st_size > 1024:
        ok(f"file · {label}", f"{_mb(path):.1f} MB  {path.relative_to(config.PROJECT_ROOT)}")
    else:
        fail(f"file · {label}", f"missing or empty: {path}")

# model_runs (for Model Insights page)
runs = sorted((config.REPORTS_DIR / "model_runs").glob("*.json")) if (config.REPORTS_DIR / "model_runs").exists() else []
if len(runs) >= 2:
    ok("file · model_runs json", f"{len(runs)} run(s) + summary.csv"
       f"{'' if (config.REPORTS_DIR / 'model_runs' / 'summary.csv').exists() else ' (summary.csv MISSING)'}")
else:
    fail("file · model_runs json", f"expected >=2 run JSONs, found {len(runs)}")


# --- 2. sales_long loads with the columns the dashboard reads -----------------
try:
    sl = pd.read_parquet(config.PROCESSED_DATA_DIR / "sales_long.parquet")
    need = {"id", "date", "sales", "cat_id", "dept_id", "sell_price", "wday"}
    missing = need - set(sl.columns)
    if missing:
        fail("schema · sales_long", f"missing columns: {missing}")
    else:
        d = pd.to_datetime(sl["date"])
        ok("schema · sales_long",
           f"{len(sl):,} rows · {sl['id'].nunique():,} items · {d.min().date()}→{d.max().date()}")
except Exception as e:
    fail("schema · sales_long", f"load error: {e}")


# --- 3. feature_matrix has every FEATURE_COL the models expect ----------------
try:
    # read only the schema (fast — avoids loading the full 80 MB matrix)
    import pyarrow.parquet as pq
    fm_schema = set(pq.read_schema(config.PROCESSED_DATA_DIR / "feature_matrix.parquet").names)
    missing = [c for c in FEATURE_COLS if c not in fm_schema]
    if missing:
        fail("schema · feature_matrix", f"missing {len(missing)} feature cols: {missing}")
    else:
        ok("schema · feature_matrix", f"all {len(FEATURE_COLS)} FEATURE_COLS present")
except Exception as e:
    fail("schema · feature_matrix", f"load error: {e}")


# --- 4. Both models load and predict on a real sample ------------------------
try:
    from retail_forecast.models.lgbm import LGBMForecast
    sample = pd.read_parquet(config.PROCESSED_DATA_DIR / "feature_matrix.parquet").head(200)
    for name in ("tweedie", "gaussian"):
        m = LGBMForecast.load(config.MODELS_DIR / f"lgbm_{name}.pkl")
        preds = m.predict(sample)
        if len(preds) == len(sample) and not pd.isna(preds).any():
            ok(f"model · {name} predict", f"{len(preds)} preds, mean={float(preds.mean()):.3f}")
        else:
            fail(f"model · {name} predict", "length mismatch or NaN in predictions")
except Exception as e:
    fail("model · predict", f"error: {e}")


# --- 5. test predictions schema (Model Insights / Forecast Charts) ------------
for name in ("tweedie", "gaussian"):
    try:
        tp = pd.read_parquet(config.REPORTS_DIR / f"test_predictions_{name}.parquet")
        need = {"id", "date", "sales", "cat_id", "dept_id", "y_pred"}
        missing = need - set(tp.columns)
        if missing:
            fail(f"schema · test_preds {name}", f"missing: {missing}")
        else:
            ok(f"schema · test_preds {name}", f"{len(tp):,} rows")
    except Exception as e:
        fail(f"schema · test_preds {name}", f"load error: {e}")


# --- 6. forecast = exactly 28 days × all items -------------------------------
try:
    fc = pd.read_parquet(config.REPORTS_DIR / "forecast_28d.parquet")
    need = {"id", "date", "y_pred", "cat_id", "dept_id"}
    missing = need - set(fc.columns)
    n_days = pd.to_datetime(fc["date"]).nunique()
    n_items = fc["id"].nunique()
    if missing:
        fail("schema · forecast_28d", f"missing: {missing}")
    elif n_days != 28:
        fail("schema · forecast_28d", f"expected 28 forecast days, got {n_days}")
    elif (fc["y_pred"] < 0).any():
        fail("schema · forecast_28d", "negative predictions present (should be clipped >=0)")
    else:
        ok("schema · forecast_28d",
           f"{n_items:,} items × {n_days} days · total={fc['y_pred'].sum():,.0f} units")
except Exception as e:
    fail("schema · forecast_28d", f"load error: {e}")


# --- 7. Streamlit app files present ------------------------------------------
app_entry = config.PROJECT_ROOT / "app" / "streamlit_app.py"
pages = sorted((config.PROJECT_ROOT / "app" / "pages").glob("[0-9]*.py"))
if app_entry.exists() and len(pages) == 5:
    ok("app · streamlit files", f"entry + {len(pages)} pages")
else:
    fail("app · streamlit files", f"entry exists={app_entry.exists()}, pages={len(pages)} (expected 5)")


# --- 8. INFO: external GitHub Release reachability (not needed once cached) ---
try:
    req = urllib.request.Request(config.DATA_URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=8) as r:
        info("network · release asset", f"reachable (HTTP {r.status}) — not required for the demo")
except Exception as e:
    info("network · release asset", f"NOT reachable ({type(e).__name__}) — OK, data is cached locally")


# --- Report ------------------------------------------------------------------
n_fail = sum(1 for s, _, _ in results if s == "FAIL")
n_pass = sum(1 for s, _, _ in results if s == "PASS")

print()
print("=" * 78)
print("  PREFLIGHT CHECK — Retail Demand Forecasting live demo")
print("=" * 78)
for status, name, detail in results:
    if status == "PASS":
        tag = f"{GREEN}✓ PASS{RESET}"
    elif status == "FAIL":
        tag = f"{RED}✗ FAIL{RESET}"
    else:
        tag = f"{YELLOW}• INFO{RESET}"
    print(f"  {tag}  {name:<28} {DIM}{detail}{RESET}")
print("-" * 78)

if n_fail == 0:
    print(f"  {GREEN}ALL CRITICAL CHECKS PASSED{RESET}  ({n_pass} pass)  →  safe to demo ✅")
    print("=" * 78)
    sys.exit(0)
else:
    print(f"  {RED}{n_fail} CRITICAL CHECK(S) FAILED{RESET}  →  fix before demoing ❌")
    print(f"  {DIM}Tip: run  uv run rf-fetch && uv run rf-process && "
          f"uv run rf-fetch-features && uv run rf-train && uv run rf-predict{RESET}")
    print("=" * 78)
    sys.exit(1)
