from retail_forecast import config


def fetch_data():
    """Download the CA_1 subset parquet from GitHub Release (cached after first run)."""
    from retail_forecast.data.fetch import fetch
    fetch()


def process_data():
    """Fetch the CA_1 subset, load it, and save as processed parquet."""
    from retail_forecast.data.fetch import fetch
    from retail_forecast.data.load import load_subset, save_processed

    path = fetch()
    print(f"Loading subset from {path} ...")
    df = load_subset(path)
    print(f"  Shape: {df.shape}")
    save_processed(df, "sales_long")
    print("Done. Run the EDA notebook to explore the data.")


def train():
    """Load processed data, build features, train Tweedie + Gaussian models, save both."""
    import pandas as pd
    from retail_forecast.features import build_feature_matrix
    from retail_forecast.models.lgbm import LGBMGaussian, LGBMTweedie
    from retail_forecast.tracker import ModelTracker

    processed = config.PROCESSED_DATA_DIR / "sales_long.parquet"
    if not processed.exists():
        print("Processed data not found. Running rf-process first...")
        process_data()

    print(f"Loading {processed} ...")
    df = pd.read_parquet(processed)
    print(f"  Shape: {df.shape}")

    feat_cache = config.PROCESSED_DATA_DIR / "feature_matrix.parquet"
    if feat_cache.exists():
        print(f"Loading cached feature matrix ...")
        fm = pd.read_parquet(feat_cache)
        for col in ["cat_id", "dept_id"]:
            if col in fm.columns:
                fm[col] = fm[col].astype("category")
    else:
        print("Building feature matrix (this may take a few minutes)...")
        fm = build_feature_matrix(df)
        fm.to_parquet(feat_cache, index=False)
    print(f"  Feature matrix shape: {fm.shape}")

    # Chronological 80/10/10 split
    dates = fm["date"].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * 0.80)]
    val_end   = dates[int(n * 0.90)]

    train_df = fm[fm["date"] <= train_end]
    val_df   = fm[(fm["date"] > train_end) & (fm["date"] <= val_end)]
    test_df  = fm[fm["date"] > val_end]
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} | Test: {len(test_df):,}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    trained_models = {}
    for ModelClass, name in [(LGBMTweedie, "tweedie"), (LGBMGaussian, "gaussian")]:
        print(f"\nTraining LightGBM ({name}) ...")
        model = ModelClass(num_boost_round=1000, early_stopping_rounds=50)
        model.fit(train_df, val_df)
        model.save(config.MODELS_DIR / f"lgbm_{name}.pkl")
        trained_models[name] = model

        tracker = ModelTracker(f"LGBM_{name.capitalize()}")
        tracker.log_data(fm, train_end, val_end)
        tracker.log_params(model.params,
                           num_boost_round=model.num_boost_round,
                           early_stopping_rounds=model.early_stopping_rounds)
        tracker.log_training(model)
        tracker.log_metrics(model, train_df, val_df, test_df)
        tracker.log_feature_importance(model.feature_importance)
        tracker.save()
        tracker.print_report()

    # Save test-set predictions for both models (used by the dashboard)
    import numpy as np
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in trained_models.items():
        preds = np.maximum(model.predict(test_df), 0)
        out = test_df[["id", "date", "sales", "cat_id", "dept_id"]].copy()
        out["y_pred"] = preds
        out.to_parquet(config.REPORTS_DIR / f"test_predictions_{name}.parquet", index=False)
    print(f"\nTest predictions saved to {config.REPORTS_DIR}")
    print(f"Done. Models saved to {config.MODELS_DIR}")


def predict():
    """Load the Tweedie model and produce a 28-day ahead demand forecast."""
    import pandas as pd
    from retail_forecast.forecast import forecast_future
    from retail_forecast.models.lgbm import LGBMForecast

    model_path = config.MODELS_DIR / "lgbm_tweedie.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run rf-train first.")

    processed = config.PROCESSED_DATA_DIR / "sales_long.parquet"
    if not processed.exists():
        raise FileNotFoundError("Processed data not found. Run rf-process first.")

    print(f"Loading model from {model_path} ...")
    model = LGBMForecast.load(model_path)

    print("Loading processed data ...")
    df = pd.read_parquet(processed)
    last_date = df["date"].max()
    print(f"  Last training date: {last_date.date()}")
    print(f"  Forecasting: {(last_date + pd.Timedelta(days=1)).date()} "
          f"to {(last_date + pd.Timedelta(days=28)).date()}")

    forecast = forecast_future(model, df, horizon=28)

    # Merge item metadata for richer output
    meta = df.drop_duplicates("id")[["id", "item_id", "dept_id", "cat_id"]].copy()
    forecast = forecast.merge(meta, on="id", how="left")

    out_path = config.REPORTS_DIR / "forecast_28d.parquet"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(out_path, index=False)
    print(f"\nForecast saved to {out_path}")
    print(f"  Rows : {len(forecast):,}  ({forecast['id'].nunique():,} items × 28 days)")
    print(f"  Total predicted demand: {forecast['y_pred'].sum():,.0f} units")

    # Summary by category
    if "cat_id" in forecast.columns:
        summary = (
            forecast.groupby("cat_id")["y_pred"]
            .sum()
            .rename("total_demand")
            .reset_index()
        )
        print("\n  28-day forecast by category:")
        for _, row in summary.iterrows():
            print(f"    {row['cat_id']:<12}: {row['total_demand']:>10,.0f} units")
