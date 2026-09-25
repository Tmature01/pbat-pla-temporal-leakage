"""
Figure V2: 预测值 vs 真实值散点图 — 时间保留分割 (画布不变 12x10, 文字x1.5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_PredVsTrue_Temporal.csv"))
met = pd.read_csv(str(FIGURE_DATA / "Fig_PredVsTrue_Metrics.csv"))

cond_colors = {"C1 (Pure PBAT)":"#E74C3C","C2 (Pure PLA)":"#3498DB","C3 (70/30 PBAT/PLA)":"#2ECC71"}
cond_markers = {"C1 (Pure PBAT)":"o","C2 (Pure PLA)":"^","C3 (70/30 PBAT/PLA)":"s"}

models = [
    ("Persistence_Pred","Persistence","(a) Persistence predictor"),
    ("LSMPR_Pred","LS-MPR","(b) LS-MPR"),
    ("GPR_Pred","GPR","(c) GPR"),
    ("ExpDecay_Pred","Exp Decay","(d) Exp Decay"),
]

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial"],
    "font.size":13.5,"font.weight":"bold",
    "axes.labelsize":15,"axes.labelweight":"bold",
    "axes.titlesize":16.5,"axes.titleweight":"bold",
    "xtick.labelsize":12,"ytick.labelsize":12,
    "figure.dpi":300,"savefig.dpi":600,
    "savefig.bbox":"tight","savefig.pad_inches":0.08,
})

fig, axes = plt.subplots(2,2,figsize=(12,10))
fig.subplots_adjust(hspace=0.26,wspace=0.22,bottom=0.06,top=0.95,left=0.09,right=0.96)

lim_min, lim_max = 185, 310

for idx, (ax, (col, model, title)) in enumerate(zip(axes.flat, models)):
    r2 = met[met["Model"]==model]["R2"].values[0]
    rmse = met[met["Model"]==model]["RMSE"].values[0]
    mae = met[met["Model"]==model]["MAE"].values[0]

    for cond in ["C1 (Pure PBAT)","C2 (Pure PLA)","C3 (70/30 PBAT/PLA)"]:
        sub = df[df["Condition"]==cond]
        ax.scatter(sub["CO2_True"],sub[col],s=28,color=cond_colors[cond],marker=cond_markers[cond],
                   alpha=0.60,edgecolors="none",zorder=3,rasterized=True)

    ax.plot([lim_min,lim_max],[lim_min,lim_max],color="gray",linewidth=0.7,linestyle="--",alpha=0.45,zorder=1)

    metric_str = f"$R^2$ = {r2:+.3f}\nRMSE = {rmse:.1f}\nMAE = {mae:.1f}"
    ax.text(0.96,0.04,metric_str,transform=ax.transAxes,
            fontsize=11,fontweight="bold",color="#2C3E50",
            ha="right",va="bottom",
            bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="#BDC3C7",alpha=0.88))

    if model == "Persistence":
        ax.annotate("Near-perfect\nplateau prediction",
                    xy=(250,250),xytext=(200,288),
                    fontsize=11,fontweight="bold",color="#2471A3",
                    arrowprops=dict(arrowstyle="->",color="#2471A3",lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="#2471A3",alpha=0.85))
    elif model == "LS-MPR":
        ax.annotate("Systematic\nunder-prediction",
                    xy=(250,210),xytext=(195,262),
                    fontsize=11,fontweight="bold",color="#C0392B",
                    arrowprops=dict(arrowstyle="->",color="#C0392B",lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="#C0392B",alpha=0.85))
    elif model == "Exp Decay":
        ax.annotate("ALL points above y=x\n(over-estimation)",
                    xy=(245,290),xytext=(195,294),
                    fontsize=11,fontweight="bold",color="#C0392B",
                    arrowprops=dict(arrowstyle="->",color="#C0392B",lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="#C0392B",alpha=0.85))

    ax.set_title(title,pad=8)
    ax.set_xlabel("Observed CO$_2$ (g)"); ax.set_ylabel("Predicted CO$_2$ (g)")
    ax.set_xlim(lim_min,lim_max); ax.set_ylim(lim_min,lim_max)
    ax.set_aspect("equal")
    ax.tick_params(which="both",direction="in",bottom=True,top=False,left=True,right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 图例 ─────────────────────────────────────────────
leg_els = []
for cond in ["C1 (Pure PBAT)","C2 (Pure PLA)","C3 (70/30 PBAT/PLA)"]:
    short = cond.split(" (")[0]
    leg_els.append(Line2D([0],[0],marker=cond_markers[cond],color="none",
                   markerfacecolor=cond_colors[cond],markersize=12,markeredgewidth=0,label=short))
leg = axes[0,1].legend(handles=leg_els,loc="upper left",ncol=1,
                 framealpha=0.85,edgecolor="#BDC3C7",fancybox=True,
                 fontsize=12,bbox_to_anchor=(0.02,1.0))
for t in leg.get_texts(): t.set_fontweight("bold")

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT_DIR / "fig_pred_vs_true_temporal_v2.png",dpi=600)
print(f"Saved: {OUT_DIR / 'fig_pred_vs_true_temporal_v2.png'}")
print("Done.")
plt.close(fig)