# factor models

# src/factor_models.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

MIN_OBS = 20

def extract_global_liquidity_factor(df, liquidity_cols):
    """
    Compute DFM and PCA liquidity factors from liquidity_cols.
    Returns enriched DataFrame and DFM results object.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Keep only liquidity_cols that actually exist in the dataframe
    # ------------------------------------------------------------------
    available_cols = [c for c in liquidity_cols if c in df.columns]
    missing = [c for c in liquidity_cols if c not in df.columns]
    if missing:
        print(f"   ⚠  Missing liquidity columns (will skip): {missing}")
    if not available_cols:
        raise ValueError(f"None of the liquidity columns found in dataframe: {liquidity_cols}")

    # ------------------------------------------------------------------
    # 2. Imputing NaNs — median per column, then forward/back fill
    #    To handle both sparse industries and structural zeros
    # ------------------------------------------------------------------
    for col in available_cols:
        # Replace inf with NaN first
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Median imputation via sklearn (handles all-NaN columns gracefully)
    imputer = SimpleImputer(strategy="median")
    df[available_cols] = imputer.fit_transform(df[available_cols])

    # If any column is still all-NaN after imputation (all-zero industry),
    # fill with 0
    for col in available_cols:
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    if len(df) < MIN_OBS:
        raise ValueError(
            f"Too few observations ({len(df)}) for factor extraction (need {MIN_OBS})"
        )

    # ------------------------------------------------------------------
    # 3. Scale
    # ------------------------------------------------------------------
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(df[available_cols])
    df_scaled = pd.DataFrame(X_scaled, columns=available_cols, index=df.index)

    # Final NaN check — should never trigger after imputation above
    if np.isnan(X_scaled).any():
        # Column-wise fallback: replace any remaining NaN with 0
        X_scaled  = np.nan_to_num(X_scaled, nan=0.0)
        df_scaled = pd.DataFrame(X_scaled, columns=available_cols, index=df.index)

    # ------------------------------------------------------------------
    # 4. PCA — pin sign to first liquidity column for consistency
    # ------------------------------------------------------------------
    pca        = PCA(n_components=1)
    pca_factor = pca.fit_transform(df_scaled).squeeze()

    if len(available_cols) > 0 and np.std(pca_factor) > 0:
        if np.corrcoef(pca_factor, df[available_cols[0]])[0, 1] < 0:
            pca_factor = -pca_factor

    df["liquidity_factor_pca"] = pca_factor

    # ------------------------------------------------------------------
    # 5. DFM
    # ------------------------------------------------------------------
    try:
        model   = DynamicFactorMQ(df_scaled, factors=1, factor_orders=1)
        results = model.fit(disp=False, maxiter=500)

        dfm_factor = results.factors.filtered.squeeze()
        dfm_shock  = results.filter_results.standardized_forecasts_error.squeeze()

        # Handle shape mismatches from squeeze()
        if np.ndim(dfm_factor) == 0:
            dfm_factor = np.full(len(df), float(dfm_factor))
        if np.ndim(dfm_shock) == 0:
            dfm_shock = np.full(len(df), float(dfm_shock))

        # Trim or pad to match df length
        dfm_factor = np.array(dfm_factor).flatten()[:len(df)]
        dfm_shock  = np.array(dfm_shock).flatten()[:len(df)]

        df["liquidity_factor_dfm"] = dfm_factor
        df["liquidity_shock_dfm"]  = dfm_shock

    except Exception as e:
        print(f"DFM failed ({e}), falling back to PCA factor for dfm columns")
        df["liquidity_factor_dfm"] = pca_factor
        df["liquidity_shock_dfm"]  = pd.Series(pca_factor).diff().fillna(0).values
        results = None

    return df, results

# from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# def extract_global_liquidity_factor(df, liquidity_cols):
#     """
#     Compute DFM and PCA liquidity factors.
#     """
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df[liquidity_cols])
#     df_scaled = pd.DataFrame(X_scaled, columns=liquidity_cols, index=df.index)

#     # PCA
#     pca = PCA(n_components=1)
#     df["liquidity_factor_pca"] = pca.fit_transform(df_scaled)

#     # DFM
#     model = DynamicFactor(df_scaled, k_factors=1, factor_order=1)
#     results = model.fit(disp=False)
#     df["liquidity_factor_dfm"] = results.factors.filtered[0]
#     df["liquidity_shock_dfm"] = results.filter_results.standardized_forecasts_error[0]

#     return df, results


# # src/factor_models.py
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# def extract_global_liquidity_factor(df: pd.DataFrame, liquidity_cols: list):
#     """
#     Compute global liquidity factor and liquidity shocks.
    
#     Args:
#         df: pandas DataFrame with liquidity columns
#         liquidity_cols: list of column names to use as liquidity indicators
    
#     Returns:
#         df: original DataFrame with added columns:
#             - liquidity_factor_pca
#             - liquidity_factor_dfm
#             - liquidity_shock_dfm
#         dfm_results: fitted DynamicFactor model (statsmodels)
#     """
#     df = df.copy()
    
#     # Standardize liquidity indicators
#     scaler = StandardScaler()
#     df_scaled = pd.DataFrame(
#         scaler.fit_transform(df[liquidity_cols]),
#         columns=liquidity_cols,
#         index=df.index
#     )
    
#     # ---- PCA for quick overview (optional) ----
#     pca = PCA(n_components=1)
#     df["liquidity_factor_pca"] = pca.fit_transform(df_scaled)
    
#     # ---- Dynamic Factor Model (DFM) ----
#     dfm_model = DynamicFactor(df_scaled, k_factors=1, factor_order=1)
#     dfm_results = dfm_model.fit(disp=False)
    
#     # Add the filtered factor as global liquidity factor
#     df["liquidity_factor_dfm"] = dfm_results.factors.filtered[0]
    
#     # Liquidity shocks (standardized forecast errors)
#     df["liquidity_shock_dfm"] = dfm_results.filter_results.standardized_forecasts_error[0]
    
#     return df, dfm_results