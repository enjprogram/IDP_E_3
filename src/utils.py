# src/utils.py
# adjusted split - 80:10:10 to have more observations, N-lags swtiched rom 3 to 2

import numpy as np
import pandas as pd
from src.evaluation import compute_metrics


def create_lagged_features(df, target_col, exog_cols, n_lags=2):
    # Accept either polars or pandas
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    lagged = df.copy()

    for lag in range(1, n_lags + 1):
        lagged[f"{target_col}_lag{lag}"] = (
            lagged.groupby("industry")[target_col].shift(lag)
        )
        for col in exog_cols:
            if col in lagged.columns:
                lagged[f"{col}_lag{lag}"] = (
                    lagged.groupby("industry")[col].shift(lag)
                )

    return lagged.dropna().reset_index(drop=True)


# the train test and val is not working well for the avaialble data
# def time_series_split(df, test_ratio=0.15, val_ratio=0.15):
#     years = sorted(df["year"].unique())
#     n     = len(years)

#     # For small panels (< 10 years), widen the train split
#     if n < 10:
#         test_ratio = 0.1
#         val_ratio  = 0.1

#     train_end = max(1, int(n * (1 - val_ratio - test_ratio)))
#     val_end   = max(train_end + 1, int(n * (1 - test_ratio)))

#     train = df[df["year"].isin(years[:train_end])]
#     val   = df[df["year"].isin(years[train_end:val_end])]
#     test  = df[df["year"].isin(years[val_end:])]

#     return train, val, test

# the split into train and test only
def time_series_split(df, test_ratio=0.2, val_ratio=0.0):
    years = sorted(df["year"].unique())
    n     = len(years)

    train_end = max(1, int(n * (1 - test_ratio)))

    train = df[df["year"].isin(years[:train_end])]
    val   = pd.DataFrame(columns=df.columns)  # empty
    test  = df[df["year"].isin(years[train_end:])]

    return train, val, test

def rolling_origin_cv(df,
                      target_col,
                      exog_cols=None,
                      seasonal_period=1,
                      n_splits=3,
                      model_type="arima"):
    """
    Rolling-origin CV using fit_ts_model for consistency with main pipeline.
    Automatically reduces n_splits if dataset is too small.
    """
    from src.forecasting import fit_ts_model, forecast_model

    n_obs     = len(df)

    # Auto-reduce splits for small panels
    max_splits = max(1, (n_obs - 2) // 2)
    n_splits   = min(n_splits, max_splits)
    fold_size  = n_obs // (n_splits + 1)

    if fold_size == 0:
        raise ValueError(
            f"Too few observations ({n_obs}) for CV with {n_splits} splits."
        )

    seasonal = model_type in ("sarima", "sarimax")
    results  = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        train_cv  = df.iloc[:train_end]
        test_cv   = df.iloc[train_end:train_end + fold_size]

        if len(test_cv) == 0 or len(train_cv) < 3:
            continue

        exog_train = train_cv[exog_cols].copy().astype(float).fillna(train_cv[exog_cols].median()).fillna(0.0) if exog_cols and model_type == "sarimax" else None
        exog_test  = test_cv[exog_cols].copy().astype(float).fillna(test_cv[exog_cols].median()).fillna(0.0)   if exog_cols and model_type == "sarimax" else None

        try:
            model, order, seasonal_order = fit_ts_model(
                train_cv[target_col],
                exog_train=exog_train,
                seasonal=seasonal,
                s=seasonal_period,
            )
            fc      = forecast_model(model, steps=len(test_cv), exog_future=exog_test)
            metrics = compute_metrics(test_cv[target_col], fc)

            results.append({
                "fold":           i,
                "train_end":      train_end,
                "forecast":       fc,
                "actual":         test_cv[target_col],
                "metrics":        metrics,
                "order":          order,
                "seasonal_order": seasonal_order,
            })

        except Exception as e:
            results.append({
                "fold":    i,
                "error":   str(e),
                "metrics": {k: np.nan for k in
                            ["MSE", "RMSE", "MAE", "MAPE", "SMAPE", "MSRE", "R2"]}
            })

    return results

