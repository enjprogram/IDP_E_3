# src/evaluation.py
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


def diebold_mariano(actual, fc1, fc2):
    """
    Diebold-Mariano test for equal predictive accuracy.
    H0: no difference between fc1 and fc2 forecast accuracy.
    """
    actual = np.array(actual, dtype=float).flatten()
    fc1    = np.array(fc1,    dtype=float).flatten()
    fc2    = np.array(fc2,    dtype=float).flatten()

    # Align lengths
    n   = min(len(actual), len(fc1), len(fc2))
    actual, fc1, fc2 = actual[:n], fc1[:n], fc2[:n]

    e1 = actual - fc1
    e2 = actual - fc2
    d  = e1**2 - e2**2

    if np.std(d) == 0:
        return {"DM_stat": 0.0, "p_value": 1.0}

    DM_stat = np.mean(d) / np.sqrt(np.var(d, ddof=1) / n)
    p_value = float(2 * (1 - stats.norm.cdf(abs(DM_stat))))

    return {"DM_stat": float(DM_stat), "p_value": p_value}


def compute_metrics(actual, forecast):
    actual   = np.array(actual,   dtype=float).flatten()
    forecast = np.array(forecast, dtype=float).flatten()

    # Align lengths
    n = min(len(actual), len(forecast))
    actual, forecast = actual[:n], forecast[:n]

    mse  = float(np.mean((actual - forecast) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(actual - forecast)))

    # Guard division by zero for ratio metrics
    nonzero = actual != 0
    denom   = np.abs(actual) + np.abs(forecast)
    denom_safe = np.where(denom == 0, 1e-8, denom)

    mape  = float(np.mean(np.abs((actual[nonzero] - forecast[nonzero]) /
                                  actual[nonzero])) * 100) if nonzero.any() else float("nan")
    smape = float(100 * np.mean(2 * np.abs(forecast - actual) / denom_safe))
    msre  = float(np.mean(((actual[nonzero] - forecast[nonzero]) /
                            actual[nonzero]) ** 2)) if nonzero.any() else float("nan")

    try:
        r2 = float(r2_score(actual, forecast))
    except Exception:
        r2 = float("nan")

    return {
        "MSE":   mse,
        "RMSE":  rmse,
        "MAE":   mae,
        "MAPE":  mape,
        "SMAPE": smape,
        "MSRE":  msre,
        "R2":    r2,
    }