# src/forecasting.py

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pmdarima import auto_arima


def fit_ts_model(y_train,
                 exog_train=None,
                 seasonal=False,
                 s=1,
                 d=None,
                 criterion="aic"):
    from src.adf_tests import determine_differencing

    y_train = np.array(y_train).flatten().astype(float)
    y_train = y_train[~np.isnan(y_train)]

    if len(y_train) < 4:
        raise ValueError(f"Too few observations ({len(y_train)}) to fit model.")

    if np.std(y_train) == 0 or len(np.unique(y_train)) < 2:
        raise ValueError("Training series is constant — cannot fit ARIMA.")

    if d is None:
        d = determine_differencing(y_train)

    if len(y_train) < 8:
        d = min(d, 1)

    if exog_train is not None:
        exog_train = _clean_exog(exog_train)

    if not seasonal:
        model = auto_arima(
            y_train,
            exogenous=exog_train,
            d=d,
            seasonal=False,
            start_p=0, max_p=2,
            start_q=0, max_q=2,
            start_P=0, max_P=0,
            start_Q=0, max_Q=0,
            max_d=min(d, 2),
            information_criterion=criterion,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            n_fits=10,
        )
    else:
        model = auto_arima(
            y_train,
            exogenous=exog_train,
            d=d,
            seasonal=True,
            m=s,
            start_p=0, max_p=2,
            start_q=0, max_q=2,
            start_P=0, max_P=1,
            start_Q=0, max_Q=1,
            max_d=min(d, 2),
            information_criterion=criterion,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            n_fits=10,
        )

    return model, model.order, model.seasonal_order


def forecast_model(model, steps, exog_future=None):
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    if exog_future is not None:
        exog_future = _clean_exog(exog_future)
        if len(exog_future) != steps:
            raise ValueError(
                f"exog_future has {len(exog_future)} rows but steps={steps}. Must match."
            )

    try:
        if hasattr(model, "predict"):
            result = model.predict(n_periods=steps, exogenous=exog_future)
        else:
            result = model.forecast(steps=steps, exog=exog_future)

        result = np.array(result, dtype=float)
        # If prediction contains NaN/inf, fall back to last training value
        if not np.all(np.isfinite(result)):
            last_val = float(model.arima_res_.fittedvalues[-1])
            result = np.full(steps, last_val if np.isfinite(last_val) else 0.0)
        return result

    except Exception:
        # Last resort fallback — return zeros
        return np.zeros(steps)


def _clean_exog(exog):
    """Convert exog to clean float numpy array, filling NaNs with column medians."""
    if isinstance(exog, pd.DataFrame):
        exog = exog.copy().astype(float)
        exog = exog.fillna(exog.median()).fillna(0.0)
        return exog.values
    exog = np.array(exog, dtype=float)
    if exog.ndim == 1:
        median = np.nanmedian(exog)
        exog[np.isnan(exog)] = median if not np.isnan(median) else 0.0
    else:
        for col in range(exog.shape[1]):
            col_vals = exog[:, col]
            median   = np.nanmedian(col_vals)
            col_vals[np.isnan(col_vals)] = median if not np.isnan(median) else 0.0
            exog[:, col] = col_vals
    return exog



# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import itertools
# import warnings
# warnings.filterwarnings("ignore")


# def fit_ts_model(y_train,
#                  exog_train=None,
#                  seasonal=False,
#                  s=1,
#                  max_order=2,
#                  criterion="aic"):

#     from src.adf_tests import determine_differencing
#     d = determine_differencing(y_train)

#     p = q = range(0, max_order+1)
#     P = Q = range(0, 2) if seasonal else [0]
#     D = range(0, 2) if seasonal else [0]   # ← improved

#     best_score = float("inf")
#     best_model = None
#     best_info = None

#     for param in itertools.product(p, [d], q):
#         for seasonal_param in itertools.product(P, D, Q):
#             try:
#                 seasonal_tuple = (seasonal_param[0],
#                                   seasonal_param[1],
#                                   seasonal_param[2],
#                                   s)

#                 res = SARIMAX(
#                     y_train,
#                     exog=exog_train,
#                     order=param,
#                     seasonal_order=seasonal_tuple,
#                     enforce_stationarity=False,
#                     enforce_invertibility=False
#                 ).fit(disp=False)

#                 score = res.aic if criterion=="aic" else res.bic

#                 if score < best_score:
#                     best_score = score
#                     best_model = res
#                     best_info = {
#                         "order": param,
#                         "seasonal_order": seasonal_tuple,
#                         "aic": res.aic,
#                         "bic": res.bic,
#                         "llf": res.llf
#                     }

#             except:
#                 continue

#     return best_model, best_info


# def forecast_model(results, steps, exog_future=None):
#     return results.forecast(steps=steps, exog=exog_future)



# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import itertools
# import warnings
# warnings.filterwarnings("ignore")

# def fit_ts_model(y_train, exog_train=None, seasonal=False, s=1, max_order=2):
#     from src.adf_tests import determine_differencing
#     d = determine_differencing(y_train)

#     p = q = range(0, max_order+1)
#     P = Q = range(0, 2) if seasonal else [0]
#     D = [1] if seasonal else [0]

#     best_aic = float("inf")
#     best_model = None
#     best_order = None

#     for param in itertools.product(p,[d],q):
#         for seasonal_param in itertools.product(P,D,Q):
#             try:
#                 res = SARIMAX(y_train,
#                               exog=exog_train,
#                               order=param,
#                               seasonal_order=(seasonal_param[0],
#                                               seasonal_param[1],
#                                               seasonal_param[2],
#                                               s),
#                               enforce_stationarity=False,
#                               enforce_invertibility=False).fit(disp=False)
#                 if res.aic < best_aic:
#                     best_aic = res.aic
#                     best_model = res
#                     best_order = (param, seasonal_param)
#             except:
#                 continue
#     return best_model, best_order, best_aic

# def forecast_model(results, steps, exog_future=None):
#     return results.forecast(steps=steps, exog=exog_future)


# import pickle
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# def fit_sarimax(train, exog_train, order=(1,0,0)):
#     model = SARIMAX(
#         train,
#         exog=exog_train,
#         order=order,
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     )
#     results = model.fit(disp=False)
#     return results

# def forecast_model(results, steps, exog_future=None):
#     return results.forecast(steps=steps, exog=exog_future)

# def save_model(results, path):
#     with open(path, "wb") as f:
#         pickle.dump(results, f)