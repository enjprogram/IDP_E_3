# app.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# -----------------------------
# Load results
# -----------------------------
@st.cache_data
def load_results():
    forecasts_all   = pickle.load(open("results/forecasts_all.pkl",   "rb"))
    performance_all = pickle.load(open("results/performance_all.pkl", "rb"))
    irf_all         = pickle.load(open("results/irf_all.pkl",         "rb"))
    return forecasts_all, performance_all, irf_all

@st.cache_data
def load_summary_csvs():
    perf_csv = adf_csv = ic_csv = None
    p = "results/performance_summary.csv"
    if os.path.exists(p): perf_csv = pd.read_csv(p)
    p = "results/adf_results.csv"
    if os.path.exists(p): adf_csv = pd.read_csv(p, index_col=0)
    p = "results/diagnostics/aic_bic_comparison.csv"
    if os.path.exists(p): ic_csv = pd.read_csv(p)
    return perf_csv, adf_csv, ic_csv

forecasts_all, performance_all, irf_all = load_results()
perf_csv, adf_csv, ic_csv = load_summary_csvs()

MODEL_COLORS = {"arima": "#2196F3", "sarima": "#4CAF50", "sarimax": "#F44336"}
DIAG_DIR     = "results/diagnostics"

INDUSTRY_NAMES = {
    "A": "A — Agriculture & Forestry",
    "B": "B — Mining & Quarrying",
    "C": "C — Manufacturing",
    "D": "D — Electricity & Gas Supply",
    "E": "E — Water Supply & Waste",
    "F": "F — Construction",
    "G": "G — Wholesale & Retail Trade",
    "H": "H — Transportation & Storage",
    "I": "I — Accommodation & Food Service",
    "J": "J — Information & Communications",
    "K": "K — Financial & Insurance",
    "L": "L — Real Estate",
    "M": "M — Professional & Scientific",
    "N": "N — Administrative Services",
    "O": "O — Public Administration",
    "P": "P — Education",
    "Q": "Q — Healthcare & Social Services",
    "R": "R — Arts & Entertainment",
    "S": "S — Other Services",
    "T": "T — Household Employers",
}

st.set_page_config(page_title="RFSD Liquidity–FCF Dashboard", layout="wide")
st.title("RFSD Liquidity–FCF Research Dashboard")
st.caption("Impact of liquidity shocks on Free Cash Flow — Russian industries 2011–2024")

# -----------------------------
# Sidebar
# -----------------------------
industry_options = {INDUSTRY_NAMES.get(k, k): k for k in sorted(forecasts_all.keys())}
industry_label   = st.sidebar.selectbox("Industry", list(industry_options.keys()))
industry         = industry_options[industry_label]

view = st.sidebar.radio("View", [
    "Forecast Comparison",
    "Model Performance",
    "Information Criteria",
    "Rolling CV Metrics",
    "Diebold–Mariano Tests",
    "Impulse Response Functions",
    "Seasonal Decomposition",
    "ACF / PACF",
    "Residual Diagnostics",
    "Heteroskedasticity Tests",
    "Covariance & Correlation",
    "ADF Stationarity Tests",
    "Cross-Industry Summary",
    "Deep Learning Forecasts"
])

def _safe(s): return str(s).replace("/", "_").replace(" ", "_").replace("\\", "_")
def ind_dir(): return os.path.join(DIAG_DIR, _safe(industry))
def model_dir(m): return os.path.join(ind_dir(), m)
def ind_name(): return INDUSTRY_NAMES.get(industry, industry)

def show_image(path, caption=""):
    if path and os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not found: {path}")

def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}

# =============================================================================
# Forecast Comparison
# =============================================================================
if view == "Forecast Comparison":
    st.subheader(f"{ind_name()} — Forecast Comparison (Full Timeline)")
    data = forecasts_all[industry]
    perf = performance_all[industry]

    available = [m for m in ["arima","sarima","sarimax"] if m in data and "train_fc" in data[m]]
    if not available:
        st.warning("No forecasts available.")
    else:
        m0 = available[0]

        # Build full timeline from all three splits
        all_years  = []
        all_actual = []
        all_shock  = []
        split_boundaries = {}  # track where each split starts/ends

        for split in ["train", "val", "test"]:
            yrs = list(data[m0].get(f"{split}_years", []))
            act = list(np.array(data[m0].get(f"{split}_actual", []), dtype=float))
            shk = data[m0].get(f"{split}_shock")
            shk = list(np.array(shk, dtype=float)) if shk is not None else [np.nan]*len(yrs)
            if yrs:
                split_boundaries[split] = (min(yrs), max(yrs))
            all_years  += yrs
            all_actual += act
            all_shock  += shk

        fig = go.Figure()

        # --- Background region shading ---
        region_colors = {
            "train": ("rgba(180,180,180,0.2)", "Training"),
            "val":   ("rgba(255,200,50,0.2)",  "Validation"),
            "test":  ("rgba(50,200,100,0.2)",   "Test / Forecast"),
        }
        for split, (color, label) in region_colors.items():
            if split in split_boundaries:
                x0, x1 = split_boundaries[split]
                fig.add_vrect(
                    x0=x0 - 0.5, x1=x1 + 0.5,
                    fillcolor=color, line_width=0,
                    layer="below"
                )
                # Label at top of region
                fig.add_annotation(
                    x=(x0 + x1) / 2, y=1.0, yref="paper",
                    text=f"<b>{label}</b>", showarrow=False,
                    font=dict(size=11, color="grey"),
                    xanchor="center"
                )

        # Boundary lines between splits
        for split in ["val", "test"]:
            if split in split_boundaries:
                fig.add_vline(
                    x=split_boundaries[split][0] - 0.5,
                    line_dash="dash", line_color="grey", line_width=1.5
                )

        # --- DFM shock shading (> 1 SD) ---
        shock_arr = np.array(all_shock, dtype=float)
        threshold = np.nanstd(shock_arr)
        for yr, shk in zip(all_years, shock_arr):
            if np.isfinite(shk) and abs(shk) > threshold:
                color = "rgba(200,0,0,0.25)" if shk < 0 else "rgba(0,0,200,0.25)"
                fig.add_vrect(
                    x0=yr - 0.4, x1=yr + 0.4,
                    fillcolor=color, line_width=1,
                    line_color=color,
                    annotation_text=f"{'↓' if shk < 0 else '↑'}{shk:.1f}σ",
                    annotation_position="bottom right",
                    annotation_font_size=9,
                    layer="below"
                )

        # --- Actual FCF ---
        fig.add_trace(go.Scatter(
            x=all_years, y=all_actual,
            mode="lines+markers", name="Actual FCF",
            line=dict(color="black", width=3),
            marker=dict(size=8, symbol="circle"),
            zorder=10
        ))

        # --- Model forecasts across full timeline ---
        # for m in available:
        #     fc_years = []
        #     fc_vals  = []
        #     for split in ["train", "val", "test"]:
        #         yrs = list(data[m].get(f"{split}_years", []))
        #         fc  = list(np.array(data[m].get(f"{split}_fc", []), dtype=float))
        #         # Only add if lengths match
        #         n = min(len(yrs), len(fc))
        #         fc_years += yrs[:n]
        #         fc_vals  += fc[:n]

        #     fig.add_trace(go.Scatter(
        #         x=fc_years, y=fc_vals,
        #         mode="lines+markers", name=m.upper(),
        #         line=dict(color=MODEL_COLORS[m], width=2, dash="dash"),
        #         marker=dict(size=5),
        #         visible=True
        #     ))

        # Define distinct styles per split
        SPLIT_STYLES = {
            "train": dict(dash="solid", symbol="circle",  size=5),
            #"val":   dict(dash="dot",   symbol="diamond", size=6),
            "test":  dict(dash="dash",  symbol="x",       size=8),
        }

        for m in available:
            for split in ["train", 
                          #"val",
                         "test"]:
                yrs = list(data[m].get(f"{split}_years", []))
                fc  = list(np.array(data[m].get(f"{split}_fc", []), dtype=float))
                n   = min(len(yrs), len(fc))
                if n == 0:
                    continue
                style = SPLIT_STYLES[split]
                show_legend = (split == "train")
                fig.add_trace(go.Scatter(
                    x=yrs[:n], y=fc[:n],
                    mode="lines+markers",
                    name=m.upper(),
                    showlegend=show_legend,
                    legendgroup=m,
                    line=dict(color=MODEL_COLORS[m], width=2, dash=style["dash"]),
                    marker=dict(size=style["size"], symbol=style["symbol"],
                                color=MODEL_COLORS[m]),  
                ))

           
        fig.update_layout(
            title=f"{ind_name()} — Full Period: Actual vs Forecasts",
            xaxis=dict(title="Year", tickmode="linear", dtick=1),
            yaxis_title="Free Cash Flow (median)",
            hovermode="x unified",
            height=500,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.08,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="grey", borderwidth=1
            ),
            margin=dict(t=100)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add legend explanation
        col1, col2, col3 = st.columns(3)
        col1.info("⬜ Grey = Training period")
        col2.warning("🟨 Yellow = Validation period")
        col3.success("🟩 Green = Test / Forecast period")

        # Per-split metrics below
        split_choice = st.radio("Show metrics for",
                                ["train", 
                                # "validation", 
                                 "test"], 
                                 horizontal=True)
        rows = {m: perf[m][split_choice] for m in available
                if m in perf and split_choice in perf[m]}
        if rows:
            df_m = pd.DataFrame(rows).T
            num_cols = df_m.select_dtypes("number").columns
            st.dataframe(df_m.style
                         .format({c: "{:.4f}" for c in num_cols})
                         .highlight_min(axis=0, subset=num_cols, color="#d4edda"))

# =============================================================================
# Model Performance
# =============================================================================
elif view == "Model Performance":
    st.subheader(f"{ind_name()} — Full Model Performance")
    perf = performance_all[industry]

    for split, label in [("train","Train"),("validation","Validation"),("test","Test")]:
        st.write(f"#### {label} Metrics")
        rows = {m: perf[m][split] for m in ["arima","sarima","sarimax"]
                if m in perf and split in perf[m]}
        if rows:
            df_m = pd.DataFrame(rows).T
            num_cols = df_m.select_dtypes("number").columns
            st.dataframe(df_m.style
                         .format({c: "{:.4f}" for c in num_cols})
                         .highlight_min(axis=0, subset=num_cols, color="#d4edda"))

# =============================================================================
# Information Criteria
# =============================================================================
elif view == "Information Criteria":
    st.subheader(f"{ind_name()} — Information Criteria")
    perf = performance_all[industry]
    rows = {}
    for m in ["arima","sarima","sarimax"]:
        if m not in perf: continue
        info = perf[m].get("model_info", {})
        rows[m.upper()] = {k: info.get(k, np.nan)
                           for k in ["order","seasonal_order","aic","bic","aicc","hqic","llf"]}
    if rows:
        df_ic    = pd.DataFrame(rows).T
        num_cols = [c for c in df_ic.columns if c not in ("order","seasonal_order")]
        st.dataframe(df_ic.style
                     .format({c: "{:.2f}" for c in num_cols})
                     .highlight_min(axis=0,
                                    subset=[c for c in num_cols if c in ("aic","bic","aicc","hqic")],
                                    color="#d4edda"))
        fig = go.Figure()
        for metric, color in [("aic","#2196F3"),("bic","#4CAF50")]:
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=list(rows.keys()),
                y=[rows[m].get(metric, np.nan) for m in rows],
                marker_color=color
            ))
        fig.update_layout(barmode="group", title="AIC / BIC Comparison",
                          height=350, xaxis_title="Model", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Rolling CV Metrics
# =============================================================================
elif view == "Rolling CV Metrics":
    st.subheader(f"{ind_name()} — Rolling-Origin Cross-Validation")
    perf = performance_all[industry]

    for m in ["arima","sarima","sarimax"]:
        if m not in perf: continue
        st.write(f"#### {m.upper()}")
        folds = perf[m].get("cv_folds", [])
        avg   = perf[m].get("cv_average", {})
        if folds:
            folds_df = pd.DataFrame(folds)
            avg_row  = pd.DataFrame([avg], index=["Average"])
            st.dataframe(pd.concat([folds_df, avg_row]).style.format("{:.4f}"))

            rmse_col = next((c for c in ["RMSE","rmse"] if c in folds_df.columns), None)
            if rmse_col:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Fold {i+1}" for i in range(len(folds_df))],
                    y=folds_df[rmse_col].tolist(),
                    marker_color=MODEL_COLORS[m]
                ))
                avg_rmse = avg.get(rmse_col, avg.get("RMSE", None))
                if avg_rmse:
                    fig.add_hline(y=avg_rmse, line_dash="dash",
                                  line_color="black", annotation_text="Average")
                fig.update_layout(title=f"{m.upper()} — CV RMSE per Fold",
                                  height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Diebold–Mariano
# =============================================================================
elif view == "Diebold–Mariano Tests":
    st.subheader(f"{ind_name()} — Diebold–Mariano Tests")
    st.caption("H₀: no difference in predictive accuracy between models")
    dm = performance_all[industry].get("diebold_mariano", {})
    if not dm:
        st.warning("No DM results found.")
    else:
        rows = [{"Comparison": k.replace("_"," ").title(), **v} for k,v in dm.items()]
        df_dm = pd.DataFrame(rows).set_index("Comparison")
        num_cols = df_dm.select_dtypes("number").columns
        st.dataframe(df_dm.style.format({c: "{:.4f}" for c in num_cols}))
        st.write("#### Interpretation")
        for k, v in dm.items():
            pval = v.get("p_value", None)
            if pval is not None:
                sig = "**significant **" if pval < 0.05 else "not significant"
                st.write(f"- **{k.replace('_',' ').title()}**: p = {pval:.4f} — {sig} at 5%")

# =============================================================================
# Impulse Response Functions
# =============================================================================
elif view == "Impulse Response Functions":
    st.subheader(f"{ind_name()} — Impulse Response Functions (VAR)")
    st.caption("FCF response to a 1-SD shock — 10-period horizon")
    irf = irf_all.get(industry, {})
    if not irf:
        st.warning("No IRF results — VAR could not be estimated (insufficient observations).")
    else:
        cols_ui = st.columns(2)
        for i, (var, values) in enumerate(irf.items()):
            x      = list(range(len(values)))
            values = [float(v) for v in values]
            fig = go.Figure()
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            pos = [v if v > 0 else 0 for v in values]
            neg = [v if v < 0 else 0 for v in values]
            fig.add_trace(go.Scatter(x=x, y=pos, fill="tozeroy",
                                     fillcolor="rgba(0,200,0,0.15)", line=dict(width=0),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=neg, fill="tozeroy",
                                     fillcolor="rgba(200,0,0,0.15)", line=dict(width=0),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=values, mode="lines+markers",
                                     line=dict(color="navy", width=2),
                                     marker=dict(size=6), name="Response"))
            fig.update_layout(title=f"Shock: {var} → FCF",
                              xaxis_title="Horizon", yaxis_title="FCF Response",
                              height=300, showlegend=False)
            with cols_ui[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Seasonal Decomposition
# =============================================================================
elif view == "Seasonal Decomposition":
    st.subheader(f"{ind_name()} — Seasonal Decomposition")
    show_image(os.path.join(ind_dir(), "seasonal_decomposition.png"))

# =============================================================================
# ACF / PACF
# =============================================================================
elif view == "ACF / PACF":
    st.subheader(f"{ind_name()} — ACF / PACF")
    series_choice = st.selectbox("Series", ["free_cash_flow"] +
                                  [f"{m}_residuals" for m in ["arima","sarima","sarimax"]])
    if "residuals" in series_choice:
        m_name = series_choice.replace("_residuals","")
        path   = os.path.join(model_dir(m_name), f"acf_pacf_{_safe(series_choice)}.png")
    else:
        path = os.path.join(ind_dir(), f"acf_pacf_{_safe(series_choice)}.png")
    show_image(path)
    csv_path = path.replace(".png", ".csv")
    if os.path.exists(csv_path):
        with st.expander("Show numeric values"):
            st.dataframe(pd.read_csv(csv_path).style.format("{:.4f}"))

# =============================================================================
# Residual Diagnostics
# =============================================================================
elif view == "Residual Diagnostics":
    st.subheader(f"{ind_name()} — Residual Diagnostics")
    model_choice = st.selectbox("Model", ["arima","sarima","sarimax"])
    show_image(os.path.join(model_dir(model_choice), "residual_diagnostics.png"))
    txt_path = os.path.join(model_dir(model_choice), "summary.txt")
    if os.path.exists(txt_path):
        with st.expander("Full model summary"):
            st.code(open(txt_path).read(), language="text")

# =============================================================================
# Heteroskedasticity Tests
# =============================================================================
elif view == "Heteroskedasticity Tests":
    st.subheader(f"{ind_name()} — Heteroskedasticity & Residual Tests")
    model_choice = st.selectbox("Model", ["arima","sarima","sarimax"])
    data = load_json(os.path.join(model_dir(model_choice), "heteroskedasticity.json"))
    if not data:
        st.warning("No results found — run main.py first.")
    else:
        for test_name, res in data.items():
            st.write(f"#### {test_name.replace('_',' ').title()}")
            if "error" in res:
                st.error(res["error"])
            else:
                cols = st.columns(min(len(res), 4))
                for col, (k, v) in zip(cols, res.items()):
                    col.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

# =============================================================================
# Covariance & Correlation
# =============================================================================
elif view == "Covariance & Correlation":
    st.subheader(f"{ind_name()} — Covariance & Correlation")
    show_image(os.path.join(ind_dir(), "correlation_heatmap_liquidity.png"))
    col1, col2 = st.columns(2)
    for label, col_ui in [("covariance", col1), ("correlation", col2)]:
        p = os.path.join(ind_dir(), f"{label}_liquidity.csv")
        if os.path.exists(p):
            df_mat = pd.read_csv(p, index_col=0)
            col_ui.write(f"**{label.title()} Matrix**")
            col_ui.dataframe(df_mat.style.format("{:.4f}"))
            fig = go.Figure(go.Heatmap(
                z=df_mat.values, x=df_mat.columns.tolist(), y=df_mat.index.tolist(),
                colorscale="RdYlGn",
                zmin=-1 if label=="correlation" else None,
                zmax=1  if label=="correlation" else None,
                text=np.round(df_mat.values, 3), texttemplate="%{text}",
            ))
            fig.update_layout(title=label.title(), height=350)
            col_ui.plotly_chart(fig, use_container_width=True)

# =============================================================================
# ADF Stationarity Tests
# =============================================================================
elif view == "ADF Stationarity Tests":
    st.subheader("ADF Stationarity Tests")
    if adf_csv is not None:
        num_cols = adf_csv.select_dtypes("number").columns.tolist()
        st.dataframe(adf_csv.style.format({c: "{:.4f}" for c in num_cols}))
    else:
        st.warning("ADF results CSV not found. Run main.py first.")

# =============================================================================
# Cross-Industry Summary
# =============================================================================
elif view == "Cross-Industry Summary":
    st.subheader("Cross-Industry Summary")
    if perf_csv is not None:
        split_filter = st.selectbox("Filter by split", ["all","train","validation","test"])
        metric       = st.selectbox("Metric", ["RMSE","MAE","MAPE","SMAPE","R2"])
        df_show = perf_csv if split_filter=="all" else perf_csv[perf_csv["split"]==split_filter]
        num_cols = df_show.select_dtypes("number").columns.tolist()
        st.dataframe(df_show.style.format({c: "{:.4f}" for c in num_cols}))

        test_df = perf_csv[perf_csv["split"]=="test"]
        if metric in test_df.columns:
            fig = go.Figure()
            for m, color in MODEL_COLORS.items():
                sub    = test_df[test_df["model"]==m]
                labels = [INDUSTRY_NAMES.get(i, i) for i in sub["industry"]]
                fig.add_trace(go.Bar(name=m.upper(), x=labels,
                                     y=sub[metric].tolist(), marker_color=color))
            fig.update_layout(
                barmode="group", title=f"Test {metric} by Industry",
                xaxis_tickangle=-45, height=420,
                xaxis_title="Industry", yaxis_title=metric,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

    if ic_csv is not None:
        st.write("#### AIC/BIC Comparison (all industries)")
        st.dataframe(ic_csv.style.format({"AIC":"{:.2f}","BIC":"{:.2f}","LLF":"{:.2f}"}))


# =============================================================================
# DEEP LEARNING SECTION
# =============================================================================

elif view == "Deep Learning Forecasts":
    st.subheader("Deep Learning FCF Forecasts")
    st.caption(
        "Prophet - LSTM - GRU - N-BEATS  |  "
        "Features: all lagged_df columns including DFM factors and lagged liquidity ratios"
    )

    DL_DIR = "results/dl"
    DL_TBLOG_DIR = "results/dl/tensorboard_logs"

    MODEL_COLORS_DL = {
        "prophet": "#FF6B35",
        "lstm":    "#7B2FBE",
        "gru":     "#009FB7",
        "nbeats":  "#E84855",
    }
    MODEL_LABELS = {
        "prophet": "Prophet",
        "lstm":    "LSTM",
        "gru":     "GRU",
        "nbeats":  "N-BEATS",
    }

    # ── Check results exist ──────────────────────────────────────────────────
    if not os.path.exists(DL_DIR) or not any(
        f.endswith(".pkl") for f in os.listdir(DL_DIR)
    ):
        st.warning("No deep learning results found.")
        st.info("Install dependencies and train the models:")
        st.code(
            "pip install prophet tensorflow scikit-learn\n"
            "python run_dl.py",
            language="bash"
        )
        st.stop()

    # ── Load results for selected industry ───────────────────────────────────
    dl_results = {}
    for model_name in ["prophet", "lstm", "gru", "nbeats"]:
        path = os.path.join(DL_DIR, f"{industry}_{model_name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                dl_results[model_name] = pickle.load(f)

    if not dl_results:
        st.warning(f"No DL results found for **{ind_name()}**.")
        st.info(f"Run:  `python run_dl.py --industries {industry}`")
        st.stop()

    available_models = list(dl_results.keys())

    # ── Four tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Forecast Plot",
        "Metrics",
        "Cross-Industry",
        "Training Evidence",
    ])

    # =========================================================================
    # Tab 1 — Forecast Plot
    # =========================================================================
    with tab1:
        st.write(f"#### {ind_name()} — DL Model Forecasts vs Actual FCF")

        m0   = available_models[0]
        res0 = dl_results[m0]

        all_years  = res0["train_years"] + res0["test_years"]
        all_actual = res0["train_actual"] + res0["test_actual"]

        fig = go.Figure()

        # Training region
        if res0["train_years"]:
            fig.add_vrect(
                x0=min(res0["train_years"]) - 0.5,
                x1=max(res0["train_years"]) + 0.5,
                fillcolor="rgba(180,180,180,0.15)",
                line_width=0, layer="below"
            )
            fig.add_annotation(
                x=np.mean(res0["train_years"]), y=1.0, yref="paper",
                text="<b>Training</b>", showarrow=False,
                font=dict(size=11, color="grey"), xanchor="center"
            )

        # Test region
        if res0["test_years"]:
            fig.add_vrect(
                x0=min(res0["test_years"]) - 0.5,
                x1=max(res0["test_years"]) + 0.5,
                fillcolor="rgba(50,200,100,0.12)",
                line_width=0, layer="below"
            )
            fig.add_annotation(
                x=np.mean(res0["test_years"]), y=1.0, yref="paper",
                text="<b>Test / Forecast</b>", showarrow=False,
                font=dict(size=11, color="grey"), xanchor="center"
            )
            fig.add_vline(
                x=min(res0["test_years"]) - 0.5,
                line_dash="dash", line_color="grey", line_width=1.5
            )

        # Actual FCF
        fig.add_trace(go.Scatter(
            x=all_years, y=all_actual,
            mode="lines+markers", name="Actual FCF",
            line=dict(color="black", width=3),
            marker=dict(size=8, symbol="circle"),
        ))

        # Model forecasts
        for mname in available_models:
            res   = dl_results[mname]
            color = MODEL_COLORS_DL[mname]
            label = MODEL_LABELS[mname]

            fig.add_trace(go.Scatter(
                x=res["train_years"], y=res["train_pred"],
                mode="lines+markers", name=f"{label} fit",
                legendgroup=mname,
                line=dict(color=color, width=1.8, dash="solid"),
                marker=dict(size=4, symbol="circle"),
            ))
            fig.add_trace(go.Scatter(
                x=res["test_years"], y=res["test_pred"],
                mode="lines+markers", name=f"{label} forecast",
                legendgroup=mname,
                line=dict(color=color, width=2.5, dash="dash"),
                marker=dict(size=10, symbol="x"),
            ))

        fig.update_layout(
            xaxis=dict(title="Year", tickmode="linear", dtick=1),
            yaxis_title="Free Cash Flow (median)",
            hovermode="x unified", height=520,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.08,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="grey", borderwidth=1
            ),
            margin=dict(t=110)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.info("Grey = Training  (solid lines)")
        col2.success("Green = Test  (dashed + ✕)")

        with st.expander("Features used per model"):
            for mname in available_models:
                res   = dl_results[mname]
                feats = res.get("features_used", [])
                note  = res.get("note", "")
                st.write(
                    f"**{MODEL_LABELS[mname]}** ({len(feats)} features): "
                    f"{', '.join(str(f) for f in feats[:8])}"
                    f"{'...' if len(feats) > 8 else ''}"
                )
                if note:
                    st.caption(note)

    # =========================================================================
    # Tab 2 — Metrics
    # =========================================================================
    with tab2:
        st.write(f"#### {ind_name()} — Test Set Metrics")
        test_rows = {
            MODEL_LABELS[m]: dl_results[m]["test_metrics"]
            for m in available_models
        }
        df_test  = pd.DataFrame(test_rows).T
        num_cols = df_test.select_dtypes("number").columns.tolist()
        st.dataframe(
            df_test.style
                   .format({c: "{:.4f}" for c in num_cols})
                   .highlight_min(
                       axis=0,
                       subset=[c for c in num_cols if c != "R2"],
                       color="#d4edda"
                   )
                   .highlight_max(
                       axis=0,
                       subset=["R2"] if "R2" in num_cols else [],
                       color="#d4edda"
                   )
        )

        st.write(f"#### {ind_name()} — Training Set Metrics")
        train_rows = {
            MODEL_LABELS[m]: dl_results[m]["train_metrics"]
            for m in available_models
        }
        df_train = pd.DataFrame(train_rows).T
        st.dataframe(df_train.style.format({c: "{:.4f}" for c in num_cols}))

        st.write("#### Test RMSE by Model")
        fig_bar = go.Figure(go.Bar(
            x=[MODEL_LABELS[m] for m in available_models],
            y=[dl_results[m]["test_metrics"]["RMSE"] for m in available_models],
            marker_color=[MODEL_COLORS_DL[m] for m in available_models],
            text=[f"{dl_results[m]['test_metrics']['RMSE']:.2f}"
                  for m in available_models],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=320, showlegend=False,
            yaxis_title="RMSE", xaxis_title="Model"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================================
    # Tab 3 — Cross-Industry
    # =========================================================================
    with tab3:
        st.write("#### Cross-Industry DL Performance (Test Set)")
        dl_summary_path = os.path.join(DL_DIR, "dl_performance_summary.csv")

        if not os.path.exists(dl_summary_path):
            st.info("Run all industries to see cross-industry comparison:")
            st.code("python run_dl.py", language="bash")
        else:
            dl_summary   = pd.read_csv(dl_summary_path)
            test_summary = dl_summary[dl_summary["split"] == "test"].copy()

            metric = st.selectbox(
                "Metric", ["RMSE", "MAE", "MAPE", "SMAPE", "R2"],
                key="dl_metric"
            )

            if metric in test_summary.columns:
                fig2 = go.Figure()
                for mname in ["prophet", "lstm", "gru", "nbeats"]:
                    sub    = test_summary[test_summary["model"] == mname]
                    labels = [INDUSTRY_NAMES.get(i, i) for i in sub["industry"]]
                    fig2.add_trace(go.Bar(
                        name=MODEL_LABELS[mname],
                        x=labels,
                        y=sub[metric].tolist(),
                        marker_color=MODEL_COLORS_DL[mname],
                    ))
                fig2.update_layout(
                    barmode="group",
                    title=f"Test {metric} — All Industries",
                    xaxis_tickangle=-45, height=450,
                    xaxis_title="Industry", yaxis_title=metric,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig2, use_container_width=True)

            num_cols = test_summary.select_dtypes("number").columns.tolist()
            st.dataframe(
                test_summary.style.format({c: "{:.4f}" for c in num_cols})
            )

    # =========================================================================
    # Tab 4 — Training Evidence
    # =========================================================================
    with tab4:
        st.write(f"#### {ind_name()} — Training Evidence")

        # TensorBoard section
        tb_logs_exist = (
            os.path.exists(DL_TBLOG_DIR) and
            any(f.startswith(industry + "_") for f in os.listdir(DL_TBLOG_DIR))
        )

        with st.expander("TensorBoard — Interactive Training Dashboard",
                          expanded=True):
            if tb_logs_exist:
                st.success("TensorBoard logs found for this industry.")
                st.write("Launch in a **separate terminal**:")
                st.code(
                    f"tensorboard --logdir {DL_TBLOG_DIR}",
                    language="bash"
                )
                st.write("Then open **http://localhost:6006** in your browser.")
                st.write("**What to look for:**")
                st.write(
                    "- **Scalars epoch_loss**: should decrease and flatten (convergence)\n"
                    "- **Scalars epoch_val_loss**: should track train loss — "
                    "if it rises while train loss falls, the model is overfitting\n"
                    "- **Histograms**: weight distributions should stabilise over epochs\n"
                    "- **Smoothing slider** (top left): set to 0.6–0.8 to see the trend"
                )
                runs = [f for f in os.listdir(DL_TBLOG_DIR)
                        if f.startswith(industry + "_")]
                st.write("**Available log runs:**")
                for r in sorted(runs):
                    parts = r.split("_")
                    mname = parts[1] if len(parts) > 1 else r
                    st.write(f"  - `{r}` — {MODEL_LABELS.get(mname, mname)}")
            else:
                st.warning("No TensorBoard logs found for this industry.")
                st.info(f"Run:  `python run_dl.py --industries {industry}`")

        st.divider()

        # Loss curves from stored history (Streamlit — no TensorBoard needed)
        st.write("#### Loss Curves (from stored training history)")

        for mname in available_models:
            res   = dl_results[mname]
            label = MODEL_LABELS[mname]
            color = MODEL_COLORS_DL[mname]

            st.write(f"##### {label}")

            if mname == "prophet":
                # Prophet: show component decomposition instead of loss curve
                comp = res.get("prophet_components")
                if comp:
                    fig_p = go.Figure()

                    # Confidence interval
                    fig_p.add_trace(go.Scatter(
                        x=comp["years"] + comp["years"][::-1],
                        y=comp["yhat_upper"] + comp["yhat_lower"][::-1],
                        fill="toself",
                        fillcolor=f"rgba(255,107,53,0.15)",
                        line=dict(width=0),
                        name="95% interval",
                        showlegend=True,
                    ))
                    fig_p.add_trace(go.Scatter(
                        x=comp["years"], y=comp["yhat"],
                        mode="lines+markers", name="Fitted (yhat)",
                        line=dict(color=color, width=2)
                    ))
                    fig_p.add_trace(go.Scatter(
                        x=comp["years"], y=comp["trend"],
                        mode="lines", name="Trend component",
                        line=dict(color="grey", width=1.5, dash="dash")
                    ))
                    fig_p.add_trace(go.Scatter(
                        x=comp["years"], y=res["train_actual"],
                        mode="lines+markers", name="Actual FCF",
                        line=dict(color="black", width=2),
                        marker=dict(size=6)
                    ))
                    fig_p.update_layout(
                        title="Prophet — Fitted Values, Trend & Uncertainty",
                        xaxis_title="Year", yaxis_title="FCF",
                        height=350, hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_p, use_container_width=True)
                    st.caption(
                        "Prophet uses L-BFGS optimisation internally — no "
                        "epoch-by-epoch loss curve. The plot above shows the "
                        "fitted values, extracted trend component, and 95% "
                        "uncertainty interval as training evidence."
                    )
                else:
                    st.info("Prophet component data not found. "
                            "Delete cached results and re-run to generate.")

            else:
                # LSTM / GRU / N-BEATS — loss curves
                train_loss = res.get("train_loss", [])
                val_loss   = res.get("val_loss",   [])
                epochs_run = res.get("epochs_run", len(train_loss))
                epochs_max = res.get("epochs_max", 300)

                if not train_loss:
                    st.info(
                        f"No loss history for {label}. "
                        "Delete cached results and re-run to generate: "
                        f"`python run_dl.py --industries {industry}`"
                    )
                    continue

                epochs_range = list(range(1, len(train_loss) + 1))

                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs_range, y=train_loss,
                    mode="lines", name="Training loss",
                    line=dict(color=color, width=2)
                ))
                if val_loss:
                    fig_loss.add_trace(go.Scatter(
                        x=epochs_range, y=val_loss,
                        mode="lines", name="Validation loss",
                        line=dict(color=color, width=1.5, dash="dash",
                                  ),
                    ))

                # Early stopping marker
                fig_loss.add_vline(
                    x=epochs_run,
                    line_dash="dot", line_color="red", line_width=1.5,
                    annotation_text=f"Early stop (epoch {epochs_run})",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="red",
                )

                fig_loss.update_layout(
                    title=f"{label} — MSE Loss per Epoch",
                    xaxis_title="Epoch",
                    yaxis_title="MSE Loss",
                    height=320,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_loss, use_container_width=True)

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Epochs run",       epochs_run)
                c2.metric("Max epochs",        epochs_max)
                c3.metric("Final train loss",  f"{train_loss[-1]:.6f}")
                if val_loss:
                    ratio = (train_loss[-1] / val_loss[-1]
                             if val_loss[-1] > 0 else 1.0)
                    c4.metric(
                        "Final val loss", f"{val_loss[-1]:.6f}",
                        delta=f"ratio {ratio:.2f}",
                        delta_color="normal" if 0.7 < ratio < 1.3 else "inverse"
                    )
