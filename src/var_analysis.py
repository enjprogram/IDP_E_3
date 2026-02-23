# src/var_analysis.py
import numpy as np
from statsmodels.tsa.api import VAR

def fit_var(train_df, cols, maxlags=5):
    data = train_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n, k = len(data), len(cols)

    if n < k + 2:
        raise ValueError(f"Too few observations ({n}) for VAR with {k} variables.")

    max_allowed = max(1, (n - 1) // k - 1)
    maxlags     = min(maxlags, max_allowed)

    model        = VAR(data)
    lag_order    = model.select_order(maxlags)
    selected_lag = max(1, lag_order.selected_orders["bic"])
    return model.fit(selected_lag)


def compute_irf(var_results, steps=10, response_col="free_cash_flow"):
    """
    Extract IRF of all variables - response_col as a plain dict of numpy arrays.
    """
    names = list(var_results.names)

    if response_col not in names:
        raise ValueError(
            f"response_col '{response_col}' not found in VAR variables: {names}"
        )

    irf     = var_results.irf(steps)
    resp_ix = names.index(response_col)

    # irf.irfs shape: (steps+1, n_vars, n_vars) — [horizon, response, impulse]
    return {
        impulse: irf.irfs[:, resp_ix, imp_ix].copy()
        for imp_ix, impulse in enumerate(names)
        if impulse != response_col
    }

# from statsmodels.tsa.api import VAR

# def fit_var(train_df, cols, maxlags=5):
#     model = VAR(train_df[cols])
#     lag_order = model.select_order(maxlags)
#     selected_lag = lag_order.selected_orders["bic"]
#     results = model.fit(selected_lag)
#     return results

# def compute_irf(var_results, steps=10):
#     irf = var_results.irf(steps)
#     return irf