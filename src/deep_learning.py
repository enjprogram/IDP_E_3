# src/deep_learning.py
"""
Deep learning forecasting module for FCF prediction by industry.
Methods: Prophet, GRU, LSTM, N-BEATS
Uses lagged_df (includes DFM factors, PCA factors, all lagged liquidity variables).
Saves TensorBoard logs + loss history + model results for caching.
"""

import os
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

DL_RESULTS_DIR   = "results/dl"
DL_MODELS_DIR    = "results/dl/models"
DL_TBLOG_DIR     = "results/dl/tensorboard_logs"

INDUSTRY_NAMES = {
    "A": "Agriculture & Forestry",   "B": "Mining & Quarrying",
    "C": "Manufacturing",            "D": "Electricity & Gas Supply",
    "E": "Water Supply & Waste",     "F": "Construction",
    "G": "Wholesale & Retail Trade", "H": "Transportation & Storage",
    "I": "Accommodation & Food",     "J": "Information & Communications",
    "K": "Financial & Insurance",    "L": "Real Estate",
    "M": "Professional & Scientific","N": "Administrative Services",
    "O": "Public Administration",    "P": "Education",
    "Q": "Healthcare & Social",      "R": "Arts & Entertainment",
    "S": "Other Services",           "T": "Household Employers",
}


def _ensure_dirs():
    for d in [DL_RESULTS_DIR, DL_MODELS_DIR, DL_TBLOG_DIR]:
        os.makedirs(d, exist_ok=True)


def _result_path(industry, model_name):
    return os.path.join(DL_RESULTS_DIR, f"{industry}_{model_name}.pkl")


def _model_path(industry, model_name):
    return os.path.join(DL_MODELS_DIR, f"{industry}_{model_name}")


def _tb_log_dir(industry, model_name):
    return os.path.join(DL_TBLOG_DIR, f"{industry}_{model_name}")


def _results_exist(industry, model_name):
    return os.path.exists(_result_path(industry, model_name))


def _load_result(industry, model_name):
    with open(_result_path(industry, model_name), "rb") as f:
        return pickle.load(f)


def _save_result(industry, model_name, result):
    with open(_result_path(industry, model_name), "wb") as f:
        pickle.dump(result, f)


# =============================================================================
# Data preparation
# =============================================================================

def load_industry_data(parquet_path="data/raw/rfsd_2011_2024.parquet",
                        lagged_df_path="results/lagged_df.pkl",
                        industry=None,
                        target_col="free_cash_flow",
                        test_ratio=0.2):
    """
    Load lagged_df (preferred — includes DFM factors, PCA factors,
    and all lagged liquidity variables) or fall back to raw parquet.
    """
    if os.path.exists(lagged_df_path):
        df = pickle.load(open(lagged_df_path, "rb"))
        print(f"    Using lagged_df ({df.shape[1]} columns "
              f"including lags + DFM factors)")
    else:
        print("    ⚠ lagged_df not found, falling back to raw parquet")
        df = pd.read_parquet(parquet_path)

    if industry:
        df = df[df["industry"] == industry].copy()

    df = df.sort_values("year").reset_index(drop=True)

    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in (target_col, "year")]

    df[feat_cols]  = (df[feat_cols]
                      .fillna(df[feat_cols].median())
                      .fillna(0.0))
    df[target_col] = df[target_col].fillna(0.0)

    years     = sorted(df["year"].unique())
    n         = len(years)
    train_end = max(1, int(n * (1 - test_ratio)))

    train = df[df["year"].isin(years[:train_end])].reset_index(drop=True)
    test  = df[df["year"].isin(years[train_end:])].reset_index(drop=True)

    print(f"    Train: {len(train)} years  |  Test: {len(test)} years")
    print(f"    Features: {len(feat_cols)} "
          f"(lags: {sum('lag' in c for c in feat_cols)}, "
          f"DFM: {sum('dfm' in c for c in feat_cols)})")

    return train, test, feat_cols, target_col


def _compute_metrics(actual, predicted):
    actual    = np.array(actual,    dtype=float).flatten()
    predicted = np.array(predicted, dtype=float).flatten()
    n         = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]

    mse  = float(np.mean((actual - predicted) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(actual - predicted)))

    nonzero = actual != 0
    mape = (float(np.mean(np.abs(
                (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
            )) * 100) if nonzero.any() else float("nan"))

    denom = np.abs(actual) + np.abs(predicted)
    smape = float(100 * np.mean(
        2 * np.abs(predicted - actual) / np.where(denom == 0, 1e-8, denom)
    ))

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae,
            "MAPE": mape, "SMAPE": smape, "R2": r2}


def _build_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


def _inverse_target(preds_scaled, ref_rows, scaler, target_ix):
    dummy = ref_rows.copy()
    n = min(len(preds_scaled), len(dummy))
    dummy[:n, target_ix] = preds_scaled[:n]
    return scaler.inverse_transform(dummy)[:n, target_ix]


# =============================================================================
# 1. Prophet
# =============================================================================

def run_prophet(industry, train, test, target_col, feat_cols):
    if _results_exist(industry, "prophet"):
        print(f"    Loading cached Prophet for {industry}")
        return _load_result(industry, "prophet")

    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Run: pip install prophet")

    train_df = pd.DataFrame({
        "ds": pd.to_datetime(train["year"], format="%Y"),
        "y":  train[target_col].values,
    })
    test_df = pd.DataFrame({
        "ds": pd.to_datetime(test["year"], format="%Y"),
        "y":  test[target_col].values,
    })

    # Prioritise DFM factors and lags
    priority  = [c for c in feat_cols if "dfm" in c or "lag" in c]
    others    = [c for c in feat_cols if c not in priority]
    top_feats = (priority + others)[:5]

    for col in top_feats:
        train_df[col] = train[col].values
        test_df[col]  = test[col].values

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    for col in top_feats:
        model.add_regressor(col)

    model.fit(train_df)

    # Capture component decomposition for training evidence
    forecast_train = model.predict(train_df)
    train_pred     = forecast_train["yhat"].values
    test_pred      = model.predict(test_df)["yhat"].values

    result = {
        "model_name":    "Prophet",
        "train_years":   train["year"].tolist(),
        "test_years":    test["year"].tolist(),
        "train_actual":  train[target_col].tolist(),
        "test_actual":   test[target_col].tolist(),
        "train_pred":    train_pred.tolist(),
        "test_pred":     test_pred.tolist(),
        "train_metrics": _compute_metrics(train[target_col], train_pred),
        "test_metrics":  _compute_metrics(test[target_col],  test_pred),
        "features_used": top_feats,
        "note":          "Prophet uses top 5 features prioritising DFM factors and lags",
        # Training evidence — component decomposition
        "prophet_components": {
            "years":    train["year"].tolist(),
            "yhat":     forecast_train["yhat"].tolist(),
            "trend":    forecast_train["trend"].tolist(),
            "yhat_lower": forecast_train["yhat_lower"].tolist(),
            "yhat_upper": forecast_train["yhat_upper"].tolist(),
        },
    }

    with open(_model_path(industry, "prophet") + ".pkl", "wb") as f:
        pickle.dump(model, f)
    _save_result(industry, "prophet", result)
    return result


# =============================================================================
# 2. LSTM
# =============================================================================

def run_lstm(industry, train, test, target_col, feat_cols,
             n_steps=2, epochs=300):
    if _results_exist(industry, "lstm"):
        print(f"    Loading cached LSTM for {industry}")
        return _load_result(industry, "lstm")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError("Run: pip install tensorflow scikit-learn")

    all_cols  = [target_col] + feat_cols
    all_data  = pd.concat([train, test])[all_cols].values.astype(float)
    target_ix = 0

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(all_data)

    n_steps      = min(n_steps, max(1, len(train) - 1))
    train_scaled = scaled[:len(train)]
    X_train, y_train = _build_sequences(train_scaled, n_steps)

    if len(X_train) == 0:
        raise ValueError(f"Not enough data for LSTM (n_steps={n_steps})")

    model = Sequential([
        LSTM(32, input_shape=(n_steps, X_train.shape[2])),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    log_dir = _tb_log_dir(industry, "lstm")
    os.makedirs(log_dir, exist_ok=True)

    es = EarlyStopping(monitor = 'loss', patience=20, restore_best_weights=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        X_train, y_train[:, target_ix],
        epochs=epochs, batch_size=4,
        #validation_split=0.1 if len(X_train) > 5 else 0.0,
        callbacks=[es, tb], verbose=0
    )

    # In-sample predictions
    train_preds_scaled = [
        model.predict(
            train_scaled[i - n_steps:i].reshape(1, n_steps, -1),
            verbose=0)[0, 0]
        for i in range(n_steps, len(train_scaled))
    ]

    # Out-of-sample rolling forecast
    context = list(train_scaled[-n_steps:])
    test_preds_scaled = []
    for i in range(len(test)):
        x    = np.array(context[-n_steps:]).reshape(1, n_steps, -1)
        pred = model.predict(x, verbose=0)[0, 0]
        test_preds_scaled.append(pred)
        next_row = scaled[len(train) + i].copy()
        next_row[target_ix] = pred
        context.append(next_row)

    train_pred = _inverse_target(
        train_preds_scaled, train_scaled[n_steps:].copy(), scaler, target_ix)
    test_pred  = _inverse_target(
        test_preds_scaled, scaled[len(train):].copy(), scaler, target_ix)

    train_actual = train[target_col].values[n_steps:]
    test_actual  = test[target_col].values

    result = {
        "model_name":    "LSTM",
        "train_years":   train["year"].tolist()[n_steps:],
        "test_years":    test["year"].tolist(),
        "train_actual":  train_actual.tolist(),
        "test_actual":   test_actual.tolist(),
        "train_pred":    train_pred.tolist(),
        "test_pred":     test_pred.tolist(),
        "train_metrics": _compute_metrics(train_actual, train_pred),
        "test_metrics":  _compute_metrics(test_actual,  test_pred),
        "features_used": all_cols,
        "n_steps":       n_steps,
        "note":          "LSTM uses all lagged_df columns including DFM factors and lags",
        # Training evidence
        "train_loss":    [float(v) for v in history.history.get("loss",     [])],
        #"val_loss":      [float(v) for v in history.history.get("val_loss", [])],
        "epochs_run":    len(history.history.get("loss", [])),
        "epochs_max":    epochs,
        "tb_log_dir":    log_dir,
    }

    model.save(_model_path(industry, "lstm") + ".keras")
    with open(_model_path(industry, "lstm_scaler") + ".pkl", "wb") as f:
        pickle.dump(scaler, f)
    _save_result(industry, "lstm", result)
    return result


# =============================================================================
# 3. GRU
# =============================================================================

def run_gru(industry, train, test, target_col, feat_cols,
            n_steps=2, epochs=300):
    if _results_exist(industry, "gru"):
        print(f"    Loading cached GRU for {industry}")
        return _load_result(industry, "gru")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError("Run: pip install tensorflow scikit-learn")

    all_cols  = [target_col] + feat_cols
    all_data  = pd.concat([train, test])[all_cols].values.astype(float)
    target_ix = 0

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(all_data)

    n_steps      = min(n_steps, max(1, len(train) - 1))
    train_scaled = scaled[:len(train)]
    X_train, y_train = _build_sequences(train_scaled, n_steps)

    if len(X_train) == 0:
        raise ValueError(f"Not enough data for GRU (n_steps={n_steps})")

    model = Sequential([
        GRU(32, input_shape=(n_steps, X_train.shape[2])),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    log_dir = _tb_log_dir(industry, "gru")
    os.makedirs(log_dir, exist_ok=True)

    es = EarlyStopping(monitor = 'loss', patience=20, restore_best_weights=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        X_train, y_train[:, target_ix],
        epochs=epochs, batch_size=4,
        #validation_split=0.1 if len(X_train) > 5 else 0.0,
        callbacks=[es, tb], verbose=0
    )

    # In-sample predictions
    train_preds_scaled = [
        model.predict(
            train_scaled[i - n_steps:i].reshape(1, n_steps, -1),
            verbose=0)[0, 0]
        for i in range(n_steps, len(train_scaled))
    ]

    # Out-of-sample rolling forecast
    context = list(train_scaled[-n_steps:])
    test_preds_scaled = []
    for i in range(len(test)):
        x    = np.array(context[-n_steps:]).reshape(1, n_steps, -1)
        pred = model.predict(x, verbose=0)[0, 0]
        test_preds_scaled.append(pred)
        next_row = scaled[len(train) + i].copy()
        next_row[target_ix] = pred
        context.append(next_row)

    train_pred = _inverse_target(
        train_preds_scaled, train_scaled[n_steps:].copy(), scaler, target_ix)
    test_pred  = _inverse_target(
        test_preds_scaled, scaled[len(train):].copy(), scaler, target_ix)

    train_actual = train[target_col].values[n_steps:]
    test_actual  = test[target_col].values

    result = {
        "model_name":    "GRU",
        "train_years":   train["year"].tolist()[n_steps:],
        "test_years":    test["year"].tolist(),
        "train_actual":  train_actual.tolist(),
        "test_actual":   test_actual.tolist(),
        "train_pred":    train_pred.tolist(),
        "test_pred":     test_pred.tolist(),
        "train_metrics": _compute_metrics(train_actual, train_pred),
        "test_metrics":  _compute_metrics(test_actual,  test_pred),
        "features_used": all_cols,
        "n_steps":       n_steps,
        "note":          "GRU uses all lagged_df columns including DFM factors and lags",
        # Training evidence
        "train_loss":    [float(v) for v in history.history.get("loss",     [])],
        #"val_loss":      [float(v) for v in history.history.get("val_loss", [])],
        "epochs_run":    len(history.history.get("loss", [])),
        "epochs_max":    epochs,
        "tb_log_dir":    log_dir,
    }

    model.save(_model_path(industry, "gru") + ".keras")
    with open(_model_path(industry, "gru_scaler") + ".pkl", "wb") as f:
        pickle.dump(scaler, f)
    _save_result(industry, "gru", result)
    return result


# =============================================================================
# 4. N-BEATS
# =============================================================================

def run_nbeats(industry, train, test, target_col, feat_cols,
               n_steps=2, epochs=300):
    if _results_exist(industry, "nbeats"):
        print(f"    Loading cached N-BEATS for {industry}")
        return _load_result(industry, "nbeats")

    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Subtract, Add
        from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError("Run: pip install tensorflow scikit-learn")

    # N-BEATS is univariate by design
    target_series = train[target_col].values.astype(float)
    test_series   = test[target_col].values.astype(float)

    scaler        = MinMaxScaler()
    target_scaled = scaler.fit_transform(
        target_series.reshape(-1, 1)).flatten()
    test_scaled   = scaler.transform(
        test_series.reshape(-1, 1)).flatten()

    n_steps = min(n_steps, max(1, len(target_scaled) - 1))

    X, y = [], []
    for i in range(len(target_scaled) - n_steps):
        X.append(target_scaled[i:i + n_steps])
        y.append(target_scaled[i + n_steps])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        raise ValueError("Not enough data for N-BEATS")

    def _nbeats_block(x_in, units=64, n_layers=4, theta_dim=8):
        h = x_in
        for _ in range(n_layers):
            h = Dense(units, activation="relu")(h)
        theta_b  = Dense(theta_dim, activation="linear")(h)
        theta_f  = Dense(theta_dim, activation="linear")(h)
        backcast = Dense(n_steps, activation="linear")(theta_b)
        forecast = Dense(1,       activation="linear")(theta_f)
        return backcast, forecast

    inp      = Input(shape=(n_steps,))
    residual = inp
    forecasts_list = []
    for _ in range(3):
        backcast, forecast = _nbeats_block(residual)
        residual = Subtract()([residual, backcast])
        forecasts_list.append(forecast)

    final = Add()(forecasts_list) if len(forecasts_list) > 1 else forecasts_list[0]
    model = Model(inputs=inp, outputs=final)
    model.compile(optimizer="adam", loss="mse")

    log_dir = _tb_log_dir(industry, "nbeats")
    os.makedirs(log_dir, exist_ok=True)

    es = EarlyStopping(monitor = 'loss', patience=20, restore_best_weights=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        X, y,
        epochs=epochs, batch_size=4,
        #validation_split=0.1 if len(X) > 5 else 0.0,
        callbacks=[es, tb], verbose=0
    )

    # In-sample
    train_pred_scaled = model.predict(X, verbose=0).flatten()
    train_pred = scaler.inverse_transform(
        train_pred_scaled.reshape(-1, 1)).flatten()
    train_actual = target_series[n_steps:]

    # Out-of-sample rolling
    context = list(target_scaled[-n_steps:])
    test_pred_scaled = []
    for i in range(len(test_scaled)):
        x_in = np.array(context[-n_steps:]).reshape(1, -1)
        pred = model.predict(x_in, verbose=0)[0, 0]
        test_pred_scaled.append(pred)
        context.append(test_scaled[i])

    test_pred = scaler.inverse_transform(
        np.array(test_pred_scaled).reshape(-1, 1)).flatten()

    result = {
        "model_name":    "N-BEATS",
        "train_years":   train["year"].tolist()[n_steps:],
        "test_years":    test["year"].tolist(),
        "train_actual":  train_actual.tolist(),
        "test_actual":   test_series.tolist(),
        "train_pred":    train_pred.tolist(),
        "test_pred":     test_pred.tolist(),
        "train_metrics": _compute_metrics(train_actual, train_pred),
        "test_metrics":  _compute_metrics(test_series,  test_pred),
        "features_used": [target_col],
        "n_steps":       n_steps,
        "note":          "N-BEATS is univariate by design — uses FCF series only",
        # Training evidence
        "train_loss":    [float(v) for v in history.history.get("loss",     [])],
        #"val_loss":      [float(v) for v in history.history.get("val_loss", [])],
        "epochs_run":    len(history.history.get("loss", [])),
        "epochs_max":    epochs,
        "tb_log_dir":    log_dir,
    }

    model.save(_model_path(industry, "nbeats") + ".keras")
    with open(_model_path(industry, "nbeats_scaler") + ".pkl", "wb") as f:
        pickle.dump(scaler, f)
    _save_result(industry, "nbeats", result)
    return result


# =============================================================================
# Master runner
# =============================================================================

def run_dl_pipeline(parquet_path="data/raw/rfsd_2011_2024.parquet",
                    lagged_df_path="results/lagged_df.pkl",
                    industries=None,
                    test_ratio=0.2,
                    models=("prophet", "lstm", "gru", "nbeats")):
    _ensure_dirs()

    if os.path.exists(lagged_df_path):
        ref_df = pickle.load(open(lagged_df_path, "rb"))
    else:
        ref_df = pd.read_parquet(parquet_path)

    if industries is None:
        industries = sorted(ref_df["industry"].dropna().unique().tolist())

    all_results = {}

    for ind in industries:
        print(f"\n{'='*55}")
        print(f"  DL: {ind} — {INDUSTRY_NAMES.get(ind, ind)}")
        print(f"{'='*55}")

        try:
            train, test, feat_cols, target_col = load_industry_data(
                parquet_path=parquet_path,
                lagged_df_path=lagged_df_path,
                industry=ind,
                test_ratio=test_ratio,
            )
        except Exception as e:
            print(f" Data load failed: {e}")
            continue

        if len(train) < 4:
            print(f" Skipping — only {len(train)} training rows")
            continue

        all_results[ind] = {}

        for model_name in models:
            print(f"  Fitting {model_name.upper()}...")
            try:
                if model_name == "prophet":
                    res = run_prophet(ind, train, test, target_col, feat_cols)
                elif model_name == "lstm":
                    res = run_lstm(ind, train, test, target_col, feat_cols)
                elif model_name == "gru":
                    res = run_gru(ind, train, test, target_col, feat_cols)
                elif model_name == "nbeats":
                    res = run_nbeats(ind, train, test, target_col, feat_cols)
                else:
                    continue

                all_results[ind][model_name] = res
                m = res["test_metrics"]
                print(f"RMSE={m['RMSE']:.2f}  "
                      f"MAE={m['MAE']:.2f}  R2={m['R2']:.3f}")

            except Exception as e:
                print(f"{model_name.upper()} failed: {e}")

    # Save cross-industry summary CSV
    rows = []
    for ind, models_res in all_results.items():
        for mname, res in models_res.items():
            for split in ["train", "test"]:
                row = {
                    "industry":      ind,
                    "industry_name": INDUSTRY_NAMES.get(ind, ind),
                    "model":         mname,
                    "split":         split,
                }
                row.update(res[f"{split}_metrics"])
                rows.append(row)

    if rows:
        out = os.path.join(DL_RESULTS_DIR, "dl_performance_summary.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n DL summary saved → {out}")

    print(f"\n TensorBoard logs: {DL_TBLOG_DIR}")
    print(f"   Launch with: tensorboard --logdir {DL_TBLOG_DIR}")

    return all_results