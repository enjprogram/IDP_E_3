# export_figures.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

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

MODEL_COLORS = {"arima": "#2196F3", "sarima": "#4CAF50", "sarimax": "#F44336"}
DL_COLORS    = {"prophet": "#FF6B35", "lstm": "#7B2FBE",
                "gru": "#009FB7",    "nbeats": "#E84855"}

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi":     300,
})

# Load all results
forecasts_all   = pickle.load(open("results/forecasts_all.pkl",   "rb"))
performance_all = pickle.load(open("results/performance_all.pkl", "rb"))
irf_all         = pickle.load(open("results/irf_all.pkl",         "rb"))

# ── Figure 1: FCF Forecasts — all industries grid ──────────────────────────
industries = sorted(forecasts_all.keys())
n          = len(industries)
ncols      = 4
nrows      = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
axes = axes.flatten()

for i, ind in enumerate(industries):
    ax   = axes[i]
    data = forecasts_all[ind]
    m0   = next(iter(data))

    all_years  = list(data[m0].get("train_years", [])) + \
                 list(data[m0].get("test_years",  []))
    all_actual = list(data[m0].get("train_actual", [])) + \
                 list(data[m0].get("test_actual",  []))

    ax.plot(all_years, all_actual, "k-o", lw=2, ms=5, label="Actual FCF", zorder=5)

    # for m, color in MODEL_COLORS.items():
    #     if m not in data:
    #         continue
    #     fc_years = list(data[m].get("train_years", [])) + \
    #                list(data[m].get("test_years",  []))
    #     fc_vals  = list(data[m].get("train_fc",    [])) + \
    #                list(data[m].get("test_fc",     []))
    #     n_pts    = min(len(fc_years), len(fc_vals))
    #     ax.plot(fc_years[:n_pts], fc_vals[:n_pts],
    #             "--", color=color, lw=1.5, label=m.upper())
    # Replace the existing model loop in Figure 1 with:
    for m, color in MODEL_COLORS.items():
        if m not in data:
            continue

        # Training fit — solid line, circle markers
        tr_yrs = list(data[m].get("train_years", []))
        tr_fc  = list(np.array(data[m].get("train_fc", []), dtype=float))
        n_tr   = min(len(tr_yrs), len(tr_fc))
        if n_tr:
            ax.plot(tr_yrs[:n_tr], tr_fc[:n_tr],
                    "-o", color=color, lw=1.5, ms=3,
                    label=f"{m.upper()} fit")

        # Test forecast — dashed line, X markers
        te_yrs = list(data[m].get("test_years", []))
        te_fc  = list(np.array(data[m].get("test_fc", []), dtype=float))
        n_te   = min(len(te_yrs), len(te_fc))
        if n_te:
            ax.plot(te_yrs[:n_te], te_fc[:n_te],
                    "--x", color=color, lw=2, ms=6, mew=1.5,
                    label=f"{m.upper()} forecast")

    # Shade test region
    test_yrs = list(data[m0].get("test_years", []))
    if test_yrs:
        ax.axvspan(min(test_yrs) - 0.5, max(test_yrs) + 0.5,
                   alpha=0.08, color="green")

    # Shade liquidity shocks
    train_shock = data[m0].get("train_shock")
    test_shock  = data[m0].get("test_shock")
    shock = (list(train_shock) if train_shock is not None else []) + \
            (list(test_shock)  if test_shock  is not None else [])
    if shock and all_years:
        shock_arr = np.array(shock, dtype=float)
        threshold = np.nanstd(shock_arr)
        for yr, shk in zip(all_years, shock_arr):
            if np.isfinite(shk) and abs(shk) > threshold:
                ax.axvspan(yr - 0.4, yr + 0.4,
                           alpha=0.2,
                           color="red" if shk < 0 else "blue")

    ax.set_title(INDUSTRY_NAMES.get(ind, ind), fontsize=10, pad=4)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("FCF (median)", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

# Hide unused axes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Single shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           ncol=4, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "FCF Forecasts by Industry: ARIMA, SARIMA, SARIMAX\n"
    "(Shaded: test period | Red/Blue: liquidity shocks > 1SD)",
    fontsize=13, y=1.01
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
path = os.path.join(OUT_DIR, "fig1_fcf_forecasts_all_industries.png")
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print(f"{path}")

# ── Figure 2: IRF — all industries ─────────────────────────────────────────
irf_industries = [ind for ind in industries if irf_all.get(ind)]
if irf_industries:
    n_irf  = len(irf_industries)
    nrows2 = (n_irf + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows2, ncols,
                                figsize=(16, nrows2 * 3))
    axes2 = axes2.flatten()

    for i, ind in enumerate(irf_industries):
        ax  = axes2[i]
        irf = irf_all[ind]
        for var, values in irf.items():
            ax.plot(range(len(values)), values,
                    lw=1.5, label=var, marker="o", ms=3)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.fill_between(range(len(list(irf.values())[0])),
                        0,
                        list(irf.values())[0],
                        alpha=0.1,
                        color="green" if list(irf.values())[0][-1] > 0
                        else "red")
        ax.set_title(INDUSTRY_NAMES.get(ind, ind), fontsize=10)
        ax.set_xlabel("Horizon (periods)", fontsize=9)
        ax.set_ylabel("FCF Response", fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)

    fig2.suptitle(
        "Impulse Response Functions: FCF Response to Liquidity Shock",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "fig2_irf_all_industries.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{path2}")

# ── Figure 3: Cross-industry test RMSE comparison ──────────────────────────
perf_csv = pd.read_csv("results/performance_summary.csv")
test_df  = perf_csv[perf_csv["split"] == "test"]

fig3, ax3 = plt.subplots(figsize=(14, 5))
x      = np.arange(len(industries))
width  = 0.25
models = ["arima", "sarima", "sarimax"]

for j, m in enumerate(models):
    sub    = test_df[test_df["model"] == m].set_index("industry")
    values = [sub.loc[ind, "RMSE"] if ind in sub.index else np.nan
              for ind in industries]
    labels = [INDUSTRY_NAMES.get(ind, ind) for ind in industries]
    ax3.bar(x + j * width, values, width,
            label=m.upper(), color=MODEL_COLORS[m], alpha=0.85)

ax3.set_xticks(x + width)
ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax3.set_ylabel("RMSE (test set)")
ax3.set_title("Test RMSE by Industry and Model", fontsize=13)
ax3.legend(fontsize=10)
plt.tight_layout()
path3 = os.path.join(OUT_DIR, "fig3_test_rmse_comparison.png")
plt.savefig(path3, dpi=300, bbox_inches="tight")
plt.close()
print(f"{path3}")

# ── Figure 4: AIC/BIC information criteria ─────────────────────────────────

ic_csv = pd.read_csv("results/diagnostics/aic_bic_comparison.csv")
if not ic_csv.empty:
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    for ax4, metric in zip(axes4, ["AIC", "BIC"]):
        for m, color in MODEL_COLORS.items():
            sub = ic_csv[ic_csv["model"].str.upper() == m.upper()]
            if sub.empty or metric not in sub.columns:
                continue
            labels = [INDUSTRY_NAMES.get(i, i) for i in sub["industry"]]
            ax4.plot(labels, sub[metric].values,
                     "o-", color=color, lw=1.5, ms=6, label=m.upper())
        ax4.set_title(f"{metric} by Industry", fontsize=12)
        ax4.set_xlabel("Industry")
        ax4.set_ylabel(metric)
        ax4.tick_params(axis="x", rotation=45, labelsize=8)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    path4 = os.path.join(OUT_DIR, "fig4_information_criteria.png")
    plt.savefig(path4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" {path4}")

# ── Figure 5: DL forecasts grid (if available) ─────────────────────────────
DL_DIR = "results/dl"
dl_industries = [
    ind for ind in industries
    if any(os.path.exists(os.path.join(DL_DIR, f"{ind}_{m}.pkl"))
           for m in ["prophet", "lstm", "gru", "nbeats"])
]

if dl_industries:
    n_dl   = len(dl_industries)
    nrows5 = (n_dl + ncols - 1) // ncols
    fig5, axes5 = plt.subplots(nrows5, ncols,
                                figsize=(16, nrows5 * 3.5))
    axes5 = axes5.flatten()

    for i, ind in enumerate(dl_industries):
        ax = axes5[i]
        res0 = None

        for mname in ["prophet", "lstm", "gru", "nbeats"]:
            path = os.path.join(DL_DIR, f"{ind}_{mname}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    res0 = pickle.load(f)
                break

        if res0 is None:
            continue

        all_years  = res0["train_years"] + res0["test_years"]
        all_actual = res0["train_actual"] + res0["test_actual"]
        ax.plot(all_years, all_actual,
                "k-o", lw=2, ms=5, label="Actual FCF", zorder=5)

        # for mname in ["prophet", "lstm", "gru", "nbeats"]:
        #     path = os.path.join(DL_DIR, f"{ind}_{mname}.pkl")
        #     if not os.path.exists(path):
        #         continue
        #     with open(path, "rb") as f:
        #         res = pickle.load(f)
        #     fc_years = res["train_years"] + res["test_years"]
        #     fc_vals  = res["train_pred"]  + res["test_pred"]
        #     n_pts    = min(len(fc_years), len(fc_vals))
        #     ax.plot(fc_years[:n_pts], fc_vals[:n_pts],
        #             "--", color=DL_COLORS[mname],
        #             lw=1.5, label=res["model_name"])

        for mname in ["prophet", "lstm", "gru", "nbeats"]:
            path = os.path.join(DL_DIR, f"{ind}_{mname}.pkl")
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                res = pickle.load(f)
            color = DL_COLORS[mname]
            label = res["model_name"]

            # Training fit
            tr_yrs = res["train_years"]
            tr_fc  = res["train_pred"]
            n_tr   = min(len(tr_yrs), len(tr_fc))
            if n_tr:
                ax.plot(tr_yrs[:n_tr], tr_fc[:n_tr],
                        "-o", color=color, lw=1.5, ms=3,
                        label=f"{label} fit")

            # Test forecast
            te_yrs = res["test_years"]
            te_fc  = res["test_pred"]
            n_te   = min(len(te_yrs), len(te_fc))
            if n_te:
                ax.plot(te_yrs[:n_te], te_fc[:n_te],
                        "--x", color=color, lw=2, ms=6, mew=1.5,
                        label=f"{label} forecast")

        test_yrs = res0["test_years"]
        if test_yrs:
            ax.axvspan(min(test_yrs) - 0.5, max(test_yrs) + 0.5,
                       alpha=0.08, color="green")

        ax.set_title(INDUSTRY_NAMES.get(ind, ind), fontsize=10)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("FCF (median)", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    for j in range(i + 1, len(axes5)):
        axes5[j].set_visible(False)

    handles5, labels5 = axes5[0].get_legend_handles_labels()
    fig5.legend(handles5, labels5, loc="lower center",
                ncol=5, fontsize=10, frameon=True,
                bbox_to_anchor=(0.5, 0.01))
    fig5.suptitle(
        "FCF Forecasts by Industry: Deep Learning Models\n"
        "(Shaded: test period)",
        fontsize=13, y=1.01
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path5 = os.path.join(OUT_DIR, "fig5_dl_forecasts_all_industries.png")
    plt.savefig(path5, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{path5}")

print(f"\n All figures saved to {OUT_DIR}/")
print("   Ready for PowerPoint / LaTeX / Word.")