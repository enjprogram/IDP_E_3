# src/diagnostics.py
"""
Full diagnostic suite for ARIMA/SARIMA/SARIMAX models and time series.
Covers: model summaries, seasonal decomposition, ACF/PACF, AIC/BIC,
covariance matrices, heteroskedasticity tests, residual diagnostics.
All outputs saved to results/diagnostics/.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

warnings.filterwarnings("ignore")

DIAG_DIR = "results/diagnostics"


def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# =============================================================================
# 1. Model Summary (text + JSON)
# =============================================================================

def save_model_summary(model, model_name: str, industry: str, out_dir: str = DIAG_DIR):
    """
    Save full statsmodels / pmdarima model summary as .txt and key stats as .json.
    """
    ind_dir = os.path.join(out_dir, _safe(industry), model_name)
    _ensure_dirs(ind_dir)

    # --- Text summary ---
    txt_path = os.path.join(ind_dir, "summary.txt")
    try:
        # pmdarima wraps statsmodels — .summary() works for both
        summary_str = str(model.summary())
    except Exception as e:
        summary_str = f"Summary unavailable: {e}"
    with open(txt_path, "w") as f:
        f.write(summary_str)

    # --- JSON of key scalars ---
    json_path = os.path.join(ind_dir, "info.json")
    info = {}
    for attr in ("aic", "bic", "aicc", "hqic", "llf"):
        try:
            info[attr] = float(getattr(model, attr))
        except Exception:
            pass
    # pmdarima stores order differently
    for attr in ("order", "seasonal_order"):
        try:
            info[attr] = str(getattr(model, attr))
        except Exception:
            pass
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    return {"summary_txt": txt_path, "info_json": json_path}


# =============================================================================
# 2. Seasonal Decomposition
# =============================================================================

def plot_seasonal_decomposition(series: pd.Series,
                                 industry: str,
                                 period: int = 1,
                                 model: str = "additive",
                                 out_dir: str = DIAG_DIR):
    """
    Seasonal decompose and save a 4-panel plot (observed, trend, seasonal, residual).
    Falls back gracefully when period=1 (no seasonality in annual data).
    """
    ind_dir = os.path.join(out_dir, _safe(industry))
    _ensure_dirs(ind_dir)
    out_path = os.path.join(ind_dir, "seasonal_decomposition.png")

    series = series.dropna().reset_index(drop=True)

    # Annual data: period must be >=2 for decomposition
    effective_period = max(period, 2)
    if len(series) < 2 * effective_period:
        _save_placeholder(out_path, f"Not enough data for decomposition (need {2*effective_period}, got {len(series)})")
        return out_path

    try:
        decomp = seasonal_decompose(series, model=model, period=effective_period, extrapolate_trend="freq")
    except Exception as e:
        _save_placeholder(out_path, f"Decomposition failed: {e}")
        return out_path

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Seasonal Decomposition — {industry}", fontsize=13, fontweight="bold")

    components = [
        (decomp.observed,  "Observed",  "#2c7bb6"),
        (decomp.trend,     "Trend",     "#d7191c"),
        (decomp.seasonal,  "Seasonal",  "#1a9641"),
        (decomp.resid,     "Residual",  "#756bb1"),
    ]
    for ax, (data, label, color) in zip(axes, components):
        ax.plot(data, color=color, linewidth=1.2)
        ax.set_ylabel(label, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# 3. ACF / PACF Plots
# =============================================================================

def plot_acf_pacf(series: pd.Series,
                  industry: str,
                  nlags: int = 20,
                  out_dir: str = DIAG_DIR,
                  label: str = "series"):
    """
    Save ACF and PACF plots side-by-side.
    """
    ind_dir = os.path.join(out_dir, _safe(industry))
    _ensure_dirs(ind_dir)
    out_path = os.path.join(ind_dir, f"acf_pacf_{_safe(label)}.png")

    series = series.dropna()
    effective_lags = min(nlags, len(series) // 2 - 1)
    if effective_lags < 2:
        _save_placeholder(out_path, f"Not enough data for ACF/PACF (n={len(series)})")
        return out_path

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"ACF / PACF — {industry} — {label}", fontsize=12, fontweight="bold")

    plot_acf(series,  lags=effective_lags, ax=axes[0], alpha=0.05, color="#2c7bb6")
    plot_pacf(series, lags=effective_lags, ax=axes[1], alpha=0.05, color="#d7191c", method="ywm")

    axes[0].set_title("Autocorrelation Function (ACF)")
    axes[1].set_title("Partial Autocorrelation Function (PACF)")
    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_acf_pacf_values(series: pd.Series,
                          industry: str,
                          nlags: int = 20,
                          out_dir: str = DIAG_DIR,
                          label: str = "series"):
    """Save ACF and PACF numeric values to CSV."""
    ind_dir = os.path.join(out_dir, _safe(industry))
    _ensure_dirs(ind_dir)
    out_path = os.path.join(ind_dir, f"acf_pacf_{_safe(label)}.csv")

    series = series.dropna()
    effective_lags = min(nlags, len(series) // 2 - 1)
    if effective_lags < 2:
        return None

    acf_vals,  acf_ci  = acf(series,  nlags=effective_lags, alpha=0.05)
    pacf_vals, pacf_ci = pacf(series, nlags=effective_lags, alpha=0.05, method="ywm")

    df = pd.DataFrame({
        "lag":        range(len(acf_vals)),
        "acf":        acf_vals,
        "acf_ci_low": acf_ci[:, 0],
        "acf_ci_hi":  acf_ci[:, 1],
        "pacf":       pacf_vals,
        "pacf_ci_low": pacf_ci[:, 0],
        "pacf_ci_hi":  pacf_ci[:, 1],
    })
    df.to_csv(out_path, index=False)
    return out_path


# =============================================================================
# 4. Residual Diagnostics
# =============================================================================

def plot_residual_diagnostics(residuals: np.ndarray,
                               model_name: str,
                               industry: str,
                               out_dir: str = DIAG_DIR):
    """
    4-panel residual plot: residuals over time, histogram + KDE,
    Q-Q plot, ACF of residuals.
    """
    ind_dir = os.path.join(out_dir, _safe(industry), model_name)
    _ensure_dirs(ind_dir)
    out_path = os.path.join(ind_dir, "residual_diagnostics.png")

    resid = np.array(residuals)
    resid = resid[~np.isnan(resid)]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Residual Diagnostics — {industry} — {model_name.upper()}",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Residuals over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(resid, color="#2c7bb6", linewidth=1)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title("Residuals over Time")
    ax1.set_xlabel("Time Step")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Histogram + KDE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(resid, bins=min(30, len(resid)//3 + 1), density=True,
             color="#2c7bb6", alpha=0.6, edgecolor="white")
    xmin, xmax = ax2.get_xlim()
    xgrid = np.linspace(xmin, xmax, 200)
    ax2.plot(xgrid, stats.norm.pdf(xgrid, resid.mean(), resid.std()),
             color="#d7191c", linewidth=1.5, label="Normal fit")
    ax2.set_title("Residual Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Q-Q plot
    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
    ax3.plot(osm, osr, "o", color="#2c7bb6", markersize=3, alpha=0.7)
    ax3.plot(osm, slope * np.array(osm) + intercept, color="#d7191c", linewidth=1.5)
    ax3.set_title(f"Q-Q Plot  (r={r:.3f})")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.grid(True, alpha=0.3)

    # Panel 4: ACF of residuals
    ax4 = fig.add_subplot(gs[1, 1])
    eff_lags = min(20, len(resid) // 2 - 1)
    if eff_lags >= 2:
        plot_acf(resid, lags=eff_lags, ax=ax4, alpha=0.05, color="#2c7bb6")
    ax4.set_title("ACF of Residuals")
    ax4.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# 5. Heteroskedasticity Tests
# =============================================================================

def test_heteroskedasticity(residuals: np.ndarray,
                             exog: np.ndarray = None,
                             model_name: str = "",
                             industry: str = "",
                             out_dir: str = DIAG_DIR):
    """
    Run Breusch-Pagan, White (if exog provided), and Ljung-Box tests.
    Save results to JSON.
    """
    ind_dir = os.path.join(out_dir, _safe(industry), model_name)
    _ensure_dirs(ind_dir)
    out_path = os.path.join(ind_dir, "heteroskedasticity.json")

    resid = np.array(residuals)
    resid = resid[~np.isnan(resid)]
    results = {}

    # Breusch-Pagan (requires exog; fall back to squared time trend)
    try:
        if exog is None or len(exog) != len(resid):
            _exog = np.column_stack([np.ones(len(resid)),
                                     np.arange(len(resid)),
                                     np.arange(len(resid))**2])
        else:
            _exog = np.column_stack([np.ones(len(resid)), exog])

        bp_stat, bp_pval, bp_f, bp_fpval = het_breuschpagan(resid, _exog)
        results["breusch_pagan"] = {
            "lm_stat":   float(bp_stat),
            "p_value":   float(bp_pval),
            "f_stat":    float(bp_f),
            "f_p_value": float(bp_fpval),
            "heteroskedastic": bool(bp_pval < 0.05),
        }
    except Exception as e:
        results["breusch_pagan"] = {"error": str(e)}

    # White test
    try:
        w_stat, w_pval, w_f, w_fpval = het_white(resid, _exog)
        results["white"] = {
            "lm_stat":   float(w_stat),
            "p_value":   float(w_pval),
            "f_stat":    float(w_f),
            "f_p_value": float(w_fpval),
            "heteroskedastic": bool(w_pval < 0.05),
        }
    except Exception as e:
        results["white"] = {"error": str(e)}

    # Ljung-Box on squared residuals (ARCH-type test)
    try:
        lags = min(10, len(resid) // 2 - 1)
        lb = acorr_ljungbox(resid**2, lags=[lags], return_df=True)
        results["ljung_box_squared"] = {
            "lb_stat":  float(lb["lb_stat"].iloc[-1]),
            "p_value":  float(lb["lb_pvalue"].iloc[-1]),
            "lags":     int(lags),
            "arch_effects":    bool(float(lb["lb_pvalue"].iloc[-1]) < 0.05),
        }
    except Exception as e:
        results["ljung_box_squared"] = {"error": str(e)}

    # Durbin-Watson
    try:
        dw = durbin_watson(resid)
        results["durbin_watson"] = {
            "statistic": float(dw),
            "interpretation": (
                "positive autocorrelation" if dw < 1.5 else
                "no autocorrelation"       if dw < 2.5 else
                "negative autocorrelation"
            )
        }
    except Exception as e:
        results["durbin_watson"] = {"error": str(e)}

    # Jarque-Bera normality test
    try:
        jb_stat, jb_pval = stats.jarque_bera(resid)
        results["jarque_bera"] = {
            "statistic": float(jb_stat),
            "p_value":   float(jb_pval),
            "normal":    bool(jb_pval >= 0.05),
        }
    except Exception as e:
        results["jarque_bera"] = {"error": str(e)}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results, out_path


# =============================================================================
# 6. Covariance Matrix
# =============================================================================

def save_covariance_matrix(df: pd.DataFrame,
                            cols: list,
                            industry: str,
                            out_dir: str = DIAG_DIR,
                            label: str = "liquidity"):
    """
    Compute and save covariance and correlation matrices as CSV and heatmap PNG.
    """
    ind_dir = os.path.join(out_dir, _safe(industry))
    _ensure_dirs(ind_dir)

    subset = df[cols].dropna()
    cov_df  = subset.cov()
    corr_df = subset.corr()

    cov_csv  = os.path.join(ind_dir, f"covariance_{_safe(label)}.csv")
    corr_csv = os.path.join(ind_dir, f"correlation_{_safe(label)}.csv")
    cov_df.to_csv(cov_csv)
    corr_df.to_csv(corr_csv)

    # Heatmap
    png_path = os.path.join(ind_dir, f"correlation_heatmap_{_safe(label)}.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Covariance & Correlation — {industry}", fontsize=12, fontweight="bold")

    for ax, matrix, title, fmt in [
        (axes[0], cov_df,  "Covariance",   ".2e"),
        (axes[1], corr_df, "Correlation",  ".2f"),
    ]:
        im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=-1 if "Corr" in title else None,
                       vmax=1 if "Corr" in title else None, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, format(matrix.values[i, j], fmt),
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(matrix.values[i, j]) < 0.7 else "white")

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"covariance_csv": cov_csv, "correlation_csv": corr_csv, "heatmap_png": png_path}


# =============================================================================
# 7. AIC/BIC Comparison Table
# =============================================================================

def save_ic_comparison(performance_all: dict,
                        out_dir: str = DIAG_DIR):
    """
    Build a cross-industry AIC/BIC/AICc comparison table and save as CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for industry, models in performance_all.items():
        for model_name in ["arima", "sarima", "sarimax"]:
            if model_name not in models:
                continue
            info = models[model_name].get("model_info", {})
            rows.append({
                "industry":      industry,
                "model":         model_name.upper(),
                "order":         info.get("order", ""),
                "seasonal_order": info.get("seasonal_order", ""),
                "AIC":           info.get("aic", np.nan),
                "BIC":           info.get("bic", np.nan),
                "LLF":           info.get("llf", np.nan),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "aic_bic_comparison.csv")
    df.to_csv(out_path, index=False)
    return df, out_path


# =============================================================================
# 8. Master runner — call once per industry in main.py
# =============================================================================

def run_full_diagnostics(industry: str,
                          train_series: pd.Series,
                          models_dict: dict,
                          liquidity_df: pd.DataFrame,
                          liquidity_cols: list,
                          seasonal_period: int = 1,
                          out_dir: str = DIAG_DIR):
    """
    Run the full diagnostic suite for one industry.

    Parameters
    ----------
    industry       : industry label
    train_series   : training FCF series
    models_dict    : {model_name: fitted_model} e.g. {"arima": ..., "sarima": ..., "sarimax": ...}
    liquidity_df   : DataFrame slice for this industry (train split)
    liquidity_cols : list of liquidity column names
    seasonal_period: seasonal period (1 for annual)
    out_dir        : root output directory
    """
    manifest = {}

    # --- Series-level diagnostics (done once per industry) ---
    manifest["seasonal_decomp"] = plot_seasonal_decomposition(
        train_series, industry, period=seasonal_period, out_dir=out_dir
    )
    manifest["acf_pacf_fcf"] = plot_acf_pacf(
        train_series, industry, nlags=20, out_dir=out_dir, label="free_cash_flow"
    )
    manifest["acf_pacf_fcf_csv"] = save_acf_pacf_values(
        train_series, industry, nlags=20, out_dir=out_dir, label="free_cash_flow"
    )
    manifest["covariance"] = save_covariance_matrix(
        liquidity_df, liquidity_cols, industry, out_dir=out_dir, label="liquidity"
    )

    # --- Per-model diagnostics ---
    for model_name, model in models_dict.items():
        if model is None or model_name == "var":
            continue

        manifest[model_name] = {}

        # Summary
        manifest[model_name]["summary"] = save_model_summary(
            model, model_name, industry, out_dir=out_dir
        )

        # Residuals
        try:
            # pmdarima
            if hasattr(model, "resid"):
                resid = np.array(model.resid())
            else:
                resid = np.array(model.resid)
        except Exception:
            resid = np.array([])

        if len(resid) > 4:
            manifest[model_name]["residual_plot"] = plot_residual_diagnostics(
                resid, model_name, industry, out_dir=out_dir
            )
            manifest[model_name]["heteroskedasticity"] = test_heteroskedasticity(
                resid, model_name=model_name, industry=industry, out_dir=out_dir
            )
            manifest[model_name]["acf_pacf_resid"] = plot_acf_pacf(
                pd.Series(resid), industry, nlags=20,
                out_dir=out_dir, label=f"{model_name}_residuals"
            )

    # Save manifest
    ind_dir = os.path.join(out_dir, _safe(industry))
    _ensure_dirs(ind_dir)
    manifest_path = os.path.join(ind_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(_serialise(manifest), f, indent=2)

    return manifest


# =============================================================================
# Helpers
# =============================================================================

def _safe(s: str) -> str:
    """Filesystem-safe version of a string."""
    return str(s).replace("/", "_").replace(" ", "_").replace("\\", "_")


def _save_placeholder(path: str, msg: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="grey",
            transform=ax.transAxes)
    ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _serialise(obj):
    """Recursively convert numpy/python types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj
