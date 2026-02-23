# main.py

import os
import pickle
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from src.data_pipeline   import load_rfsd_2011_2024
from src.factor_models   import extract_global_liquidity_factor
from src.utils           import create_lagged_features, time_series_split, rolling_origin_cv
from src.evaluation      import diebold_mariano, compute_metrics
from src.forecasting     import fit_ts_model, forecast_model
from src.var_analysis    import fit_var, compute_irf
from src.adf_tests       import determine_differencing, adf_test
from src.diagnostics     import (
    run_full_diagnostics,
    save_ic_comparison,
    plot_acf_pacf,
    save_acf_pacf_values,
)

# -----------------------------
# Config
# -----------------------------
n_lags          = 2   # reduced from 3 — annual data has ~14 obs per industry
seasonal_period = 1
save_dir        = "results"
diag_dir        = "results/diagnostics"

for d in [save_dir, diag_dir, "data/raw", "data/hf_cache"]:
    os.makedirs(d, exist_ok=True)

# -----------------------------
# Paths
# -----------------------------
LAGGED_DF_PATH   = os.path.join(save_dir, "lagged_df.pkl")
PERFORMANCE_PATH = os.path.join(save_dir, "performance_all.pkl")
FORECASTS_PATH   = os.path.join(save_dir, "forecasts_all.pkl")
MODELS_PATH      = os.path.join(save_dir, "models_all.pkl")
IRF_PATH         = os.path.join(save_dir, "irf_all.pkl")
ADF_PATH         = os.path.join(save_dir, "adf_results.pkl")

liquidity_cols = ["current_ratio", "quick_ratio", "cash_ratio"] 
exog_cols      = liquidity_cols + ["liquidity_factor_dfm", "liquidity_shock_dfm"]

# =============================================================================
# Step 1 — Data + Feature Engineering
# =============================================================================
if os.path.exists(LAGGED_DF_PATH):
    print(" Loading cached lagged_df...")
    lagged_df = pickle.load(open(LAGGED_DF_PATH, "rb"))
else:
    print(" Building dataset from scratch...")
    df = load_rfsd_2011_2024()
    print(f"   Loaded shape: {df.shape}  |  columns: {df.columns}")

    df, dfm_res = extract_global_liquidity_factor(df.to_pandas(), liquidity_cols)
    print(f"   Factors extracted. Shape: {df.shape}")

    lagged_df = create_lagged_features(df, "free_cash_flow", exog_cols, n_lags)
    print(f"   Lagged features created. Shape: {lagged_df.shape}")

    pickle.dump(lagged_df, open(LAGGED_DF_PATH, "wb"))
    print("   Saved lagged_df.")

# =============================================================================
# Step 2 — ADF tests (global, run once)
# =============================================================================
if os.path.exists(ADF_PATH):
    print(" Loading cached ADF results...")
    adf_results = pickle.load(open(ADF_PATH, "rb"))
else:
    print(" Running ADF tests...")
    adf_results = {}
    # aggregate to annual median first, then compute ACF/PACF
    for col in ["free_cash_flow"] + liquidity_cols:
        if col in lagged_df.columns:
            series = lagged_df[col].dropna()
            adf_results[col] = adf_test(series)

            # Aggregate to annual median across all industries
            global_series = (lagged_df.groupby("year")[col]
                                    .median()
                                    .sort_index()
                                    .dropna())
            nlags = max(2, len(global_series) // 2 - 1)
            plot_acf_pacf(global_series, industry="global",
                        nlags=nlags, out_dir=diag_dir, label=col)
            save_acf_pacf_values(global_series, industry="global",
                                nlags=nlags, out_dir=diag_dir, label=col)

    pickle.dump(adf_results, open(ADF_PATH, "wb"))
    pd.DataFrame(adf_results).T.to_csv(os.path.join(save_dir, "adf_results.csv"))
    print(" ADF results saved.")

# =============================================================================
# Step 3 — Load existing results
# =============================================================================
performance_all = pickle.load(open(PERFORMANCE_PATH, "rb")) if os.path.exists(PERFORMANCE_PATH) else {}
forecasts_all   = pickle.load(open(FORECASTS_PATH,   "rb")) if os.path.exists(FORECASTS_PATH)   else {}
models_all      = pickle.load(open(MODELS_PATH,      "rb")) if os.path.exists(MODELS_PATH)       else {}
irf_all         = pickle.load(open(IRF_PATH,         "rb")) if os.path.exists(IRF_PATH)          else {}

# =============================================================================
# Step 4 — Per-industry loop
# =============================================================================
industries = lagged_df["industry"].unique()

for ind in industries:

    # Skip if fully cached
    if ind in models_all and ind in performance_all and ind in irf_all:
        print(f"Skipping (cached): {ind}")
        continue

    print(f"\n{'='*60}")
    print(f"  Processing industry: {ind}")
    print(f"{'='*60}")

    subset = lagged_df[lagged_df["industry"] == ind].reset_index(drop=True)
    train, val, test = time_series_split(subset)

    if len(train) < 5:
        print(f" Skipping {ind} — insufficient training data ({len(train)} rows)")
        continue

    # ADF-based differencing order
    d_order = determine_differencing(train["free_cash_flow"])
    print(f"  ADF: d={d_order}")

    performance_all[ind] = {}
    forecasts_all[ind]   = {}
    models_all[ind]      = {}

    # -------------------------------------------------------------------------
    # Model loop: ARIMA, SARIMA, SARIMAX
    # -------------------------------------------------------------------------
    for model_name in ["arima", "sarima", "sarimax"]:
        print(f"  Fitting {model_name.upper()}...")

        seasonal   = model_name in ("sarima", "sarimax")
        use_exog   = model_name == "sarimax"
        lag_cols   = [c for c in subset.columns if "lag" in c and c != "free_cash_flow"]

        def clean_df(df_slice):
            df_slice = df_slice.copy().astype(float)
            # Fill with median, then 0 as final fallback
            medians = df_slice.median()
            medians = medians.fillna(0.0)
            return df_slice.fillna(medians).fillna(0.0)

        exog_train = clean_df(train[lag_cols]) if use_exog else None
        exog_val   = clean_df(val[lag_cols])   if use_exog else None
        exog_test  = clean_df(test[lag_cols])  if use_exog else None

        try:
            model, order, seasonal_order = fit_ts_model(
                train["free_cash_flow"],
                exog_train=exog_train,
                seasonal=seasonal,
                s=seasonal_period,
                d=d_order,
            )
        except Exception as e:
            print(f" {model_name.upper()} fit failed: {e}")
            continue

        # Forecasts
        train_fc = forecast_model(model, len(train), exog_future=exog_train)
        #val_fc   = forecast_model(model, len(val),   exog_future=exog_val)
        test_fc  = forecast_model(model, len(test),  exog_future=exog_test)

        # Metrics
        train_metrics = compute_metrics(train["free_cash_flow"], train_fc)
        #val_metrics   = compute_metrics(val["free_cash_flow"],   val_fc)
        test_metrics  = compute_metrics(test["free_cash_flow"],  test_fc)

        # Rolling CV
        try:
            cv_results = rolling_origin_cv(
                train,
                target_col="free_cash_flow",
                exog_cols=lag_cols if use_exog else None,
                seasonal_period=seasonal_period,
                n_splits=3,
                model_type=model_name,
            )
            cv_df  = pd.DataFrame([f["metrics"] for f in cv_results if "metrics" in f])
            cv_avg = cv_df.mean().to_dict() if not cv_df.empty else {}
        except Exception as e:
            print(f" CV failed for {model_name}: {e}")
            cv_df, cv_avg = pd.DataFrame(), {}

        # Model info — support both pmdarima and statsmodels objects
        def _get(obj, *attrs, default=np.nan):
            for a in attrs:
                try:
                    v = getattr(obj, a)
                    return float(v() if callable(v) else v)
                except Exception:
                    pass
            return default

        model_info = {
            "order":          str(order),
            "seasonal_order": str(seasonal_order),
            "aic":            _get(model, "aic"),
            "bic":            _get(model, "bic"),
            "aicc":           _get(model, "aicc"),
            "hqic":           _get(model, "hqic"),
            "llf":            _get(model, "llf"),
        }

        performance_all[ind][model_name] = {
            "train":      train_metrics,
            #"validation": val_metrics,
            "test":       test_metrics,
            "cv_folds":   cv_df.to_dict("records") if not cv_df.empty else [],
            "cv_average": cv_avg,
            "model_info": model_info,
        }

        forecasts_all[ind][model_name] = {
            "train_fc":     train_fc,
            #"val_fc":       val_fc,
            "test_fc":      test_fc,
            "train_actual": train["free_cash_flow"].values,
            "val_actual":   val["free_cash_flow"].values,
            "test_actual":  test["free_cash_flow"].values,
            "train_years":  train["year"].values,
            "val_years":    val["year"].values,
            "test_years":   test["year"].values,
            "train_shock":  train["liquidity_shock_dfm"].values if "liquidity_shock_dfm" in train.columns else None,
            "val_shock":    val["liquidity_shock_dfm"].values   if "liquidity_shock_dfm" in val.columns   else None,
            "test_shock":   test["liquidity_shock_dfm"].values  if "liquidity_shock_dfm" in test.columns  else None,
        }

        models_all[ind][model_name] = model
        print(f" AIC={model_info['aic']:.2f}  BIC={model_info['bic']:.2f}")

    # -------------------------------------------------------------------------
    # Diebold–Mariano
    # -------------------------------------------------------------------------
    try:
        dm_arima_sarimax  = diebold_mariano(
            test["free_cash_flow"],
            forecasts_all[ind]["arima"]["test_fc"],
            forecasts_all[ind]["sarimax"]["test_fc"],
        )
        dm_sarima_sarimax = diebold_mariano(
            test["free_cash_flow"],
            forecasts_all[ind]["sarima"]["test_fc"],
            forecasts_all[ind]["sarimax"]["test_fc"],
        )
        performance_all[ind]["diebold_mariano"] = {
            "arima_vs_sarimax":  dm_arima_sarimax,
            "sarima_vs_sarimax": dm_sarima_sarimax,
        }
    except Exception as e:
        print(f" Diebold-Mariano failed: {e}")

    # -------------------------------------------------------------------------
    # VAR + IRF
    # -------------------------------------------------------------------------
    try:
        var_cols = ["free_cash_flow", "liquidity_factor_dfm"] + liquidity_cols
        var_res  = fit_var(train[var_cols], var_cols)
        irf_all[ind]           = compute_irf(var_res, steps=10, response_col="free_cash_flow")
        models_all[ind]["var"] = var_res
        print(f" VAR fitted.  Lag order: {var_res.k_ar}")
    except Exception as e:
        print(f" VAR/IRF failed: {e}")
        irf_all[ind] = {}

    # -------------------------------------------------------------------------
    # Full diagnostics suite
    # -------------------------------------------------------------------------
    try:
        run_full_diagnostics(
            industry=ind,
            train_series=train["free_cash_flow"].reset_index(drop=True),
            models_dict={k: v for k, v in models_all[ind].items() if k != "var"},
            liquidity_df=train,
            liquidity_cols=liquidity_cols,
            seasonal_period=seasonal_period,
            out_dir=diag_dir,
        )
        print(f" Diagnostics saved: results/diagnostics/{ind}/")
    except Exception as e:
        print(f" Diagnostics failed: {e}")

    # -------------------------------------------------------------------------
    # Incremental save — protect against mid-run crash
    # -------------------------------------------------------------------------
    pickle.dump(performance_all, open(PERFORMANCE_PATH, "wb"))
    pickle.dump(forecasts_all,   open(FORECASTS_PATH,   "wb"))
    pickle.dump(models_all,      open(MODELS_PATH,      "wb"))
    pickle.dump(irf_all,         open(IRF_PATH,         "wb"))
    print(f" Saved results for: {ind}")

# =============================================================================
# Step 5 — Cross-industry summary tables
# =============================================================================
print("\n Building summary tables...")
ic_df, ic_path = save_ic_comparison(performance_all, out_dir=diag_dir)
print(f"   AIC/BIC table: {ic_path}")

# Flat performance summary CSV
rows = []
for ind, models in performance_all.items():
    for model_name in ["arima", "sarima", "sarimax"]:
        if model_name not in models:
            continue
        for split in ["train", "validation", "test"]:
            m = models[model_name].get(split, {})
            rows.append({"industry": ind, "model": model_name, "split": split, **m})

pd.DataFrame(rows).to_csv(os.path.join(save_dir, "performance_summary.csv"), index=False)
print(f"   Performance summary: {save_dir}/performance_summary.csv")

print("\n FULL ECONOMETRIC PIPELINE COMPLETE")
print(f"   Results: {save_dir}/")
print(f"   Diagnostics: {diag_dir}/")
