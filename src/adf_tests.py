# src/adf_tests.py
# added safeguards
import numpy as np
from statsmodels.tsa.stattools import adfuller


def adf_test(series, signif=0.05):
    series = np.array(series).flatten()
    series = series[~np.isnan(series)]

    if len(series) < 4:
        return {"error": "too few observations", "p_value": None, "stationary": None}

    if np.std(series) == 0:
        return {"error": "constant series", "p_value": None, "stationary": None}

    try:
        res = adfuller(series, autolag="AIC")
        return {
            "test_statistic":  res[0],
            "p_value":         res[1],
            "n_lags_used":     res[2],
            "n_obs":           res[3],
            "critical_values": res[4],
            "icbest":          res[5],
            "stationary":      res[1] < signif,
        }
    except Exception as e:
        return {"error": str(e), "p_value": None, "stationary": None}


def determine_differencing(series, signif=0.05, max_d=2):
    series = np.array(series).flatten()
    series = series[~np.isnan(series)]

    for d in range(max_d + 1):
        if len(series) < 4:
            return d
        if np.std(series) == 0:
            return d  # constant — no differencing will help, just return current d
        try:
            pval = adfuller(series, autolag="AIC")[1]
            if pval < signif:
                return d
        except Exception:
            return d
        series = np.diff(series)

    return max_d



# from statsmodels.tsa.stattools import adfuller

# def adf_test(series, signif=0.05):
#     res = adfuller(series, autolag='AIC')
#     return {
#         "test_statistic": res[0],
#         "p_value": res[1],
#         "n_lags_used": res[2],
#         "n_obs": res[3],
#         "critical_values": res[4],
#         "icbest": res[5]
#     }

# def determine_differencing(series, signif=0.05):
#     pval = adfuller(series, autolag='AIC')[1]
#     return 0 if pval < signif else 1