"""
Figure: 预测值 vs 真实值散点图 — 时间保留分割 (2x2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_PredVsTrue_Temporal.csv"))
met = pd.read_csv(str(FIGURE_DATA / "Fig_PredVsTrue_Metrics.csv"))

cond_colors = {"C1 (Pure PBAT)": "#E74C3C",
               "C2 (Pure PLA)": "#3498DB",
               "C3 (70/30 PBAT/PLA)": "#2ECC71"}
cond_markers = {"C1 (Pure PBAT)": "o",
                "C2 (Pure PLA)": "^",
                "C3 (70/30 PBAT/PLA)": "s"}

models = [
    ("Persistence_Pred", "Persistence", "(a) Persistence predictor"),
    ("LSMPR_Pred",      "LS-MPR",      "(b) LS-MPR"),
    ("GPR_Pred",        "GPR",         "(c) GPR"),
    ("ExpDecay_Pred",   "Exp Decay",   "(d) Exp Decay"),
]

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 10, "axes.labelweight": "bold",
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.24, wspace=0.20, bottom=0.08, top=0.95,
                    left=0.08, right=0.96)

lim_min, lim_max = 185, 310

for idx, (ax, (col, model, title)) in enumerate(zip(axes.flat, models)):
    r2  = met[met["Model"] == model]["R2"].values[0]
    rmse = met[met["Model"] == model]["RMSE"].values[0]
    mae  = met[met["Model"] == model]["MAE"].values[0]

    for cond in ["C1 (Pure PBAT)", "C2 (Pure PLA)", "C3 (70/30 PBAT/PLA)"]:
        sub = df[df["Condition"] == cond]
        ax.scatter(sub["CO2_True"], sub[col],
                   s=22, color=cond_colors[cond], marker=cond_markers[cond],
                   alpha=0.65, edgecolors="none", zorder=3, rasterized=True)

    # y=x
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="gray",
            linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

    # 统计
    metric_str = f"$R^2$ = {r2:+.3f}\nRMSE = {rmse:.1f}\nMAE = {mae:.1f}"
    ax.text(0.95, 0.06, metric_str, transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color="#2C3E50",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDC3C7", alpha=0.88))

    # 模型特定标注
    if model == "Persistence":
        ax.annotate("Near-perfect\nplateau prediction",
                    xy=(250, 250), xytext=(215, 260),
                    fontsize=7.5, fontweight="bold", color="#2471A3",
                    arrowprops=dict(arrowstyle="->", color="#2471A3", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2471A3", alpha=0.85))
    elif model == "LS-MPR":
        ax.annotate("Systematic\nunder-prediction",
                    xy=(250, 210), xytext=(265, 230),
                    fontsize=7.5, fontweight="bold", color="#C0392B",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C0392B", alpha=0.85))
    elif model == "Exp Decay":
        ax.annotate("ALL points above y=x\n(over-estimation)",
                    xy=(245, 290), xytext=(205, 275),
                    fontsize=7.5, fontweight="bold", color="#C0392B",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C0392B", alpha=0.85))

    ax.set_title(title, pad=6)
    ax.set_xlabel("Observed CO$_2$ (g)")
    ax.set_ylabel("Predicted CO$_2$ (g)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 图例 ─────────────────────────────────────────────
legend_elements = []
for cond in ["C1 (Pure PBAT)", "C2 (Pure PLA)", "C3 (70/30 PBAT/PLA)"]:
    short = cond.split(" (")[0]
    legend_elements.append(
        Line2D([0],[0], marker=cond_markers[cond], color="none",
               markerfacecolor=cond_colors[cond], markersize=9,
               markeredgewidth=0, label=short)
    )
leg = fig.legend(handles=legend_elements, loc="upper center", ncol=3,
                 framealpha=0.90, edgecolor="#BDC3C7", fancybox=True,
                 fontsize=9, bbox_to_anchor=(0.5, 0.98))
for t in leg.get_texts(): t.set_fontweight("bold")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_pred_vs_true_temporal.png", dpi=600)
fig.savefig(OUT_DIR / "fig_pred_vs_true_temporal.svg")
print(f"Saved: {OUT_DIR / 'fig_pred_vs_true_temporal.png'}")
print(f"Saved: {OUT_DIR / 'fig_pred_vs_true_temporal.svg'}")
print("Done.")
plt.close(fig)