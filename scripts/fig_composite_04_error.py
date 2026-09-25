"""
Figure 4 (复合): 误差结构 — 失败如何展开
(a) 误差增长曲线 (b) 误差热图 (c) 残差符号一致性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

smooth   = pd.read_csv(str(FIGURE_DATA / "Fig_ErrorGrowth_Smooth.csv"))
windows  = pd.read_csv(str(FIGURE_DATA / "Fig_ErrorGrowth_Windows.csv"))
heatmap  = pd.read_csv(str(FIGURE_DATA / "Fig_ErrorHeatmap.csv"))
resid    = pd.read_csv(str(FIGURE_DATA / "Fig_Residual_Sign_Consistency.csv"))
resid_summ = pd.read_csv(str(FIGURE_DATA / "Fig_Residual_Summary.csv"))

# ── 2. 全局样式 ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "font.weight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold",
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.55, 1.0],
                      hspace=0.30, left=0.07, right=0.96, top=0.96, bottom=0.05)

axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC1 = fig.add_subplot(gs[2])

# ══════════════════════════════════════════════════════
# PANEL (a): 误差增长曲线
# ══════════════════════════════════════════════════════
models = [
    ("Persistence","Persistence","#2C3E50","-"),
    ("LSMPR","LS-MPR","#E74C3C","--"),
    ("GPR","GPR","#3498DB","-."),
    ("ExpDecay","Exp Decay","#27AE60",":"),
]
for col, name, color, ls in models:
    axA.plot(smooth["Distance"], smooth[f"{col}_RMSE"], color=color, linewidth=2.0, linestyle=ls, zorder=3)

# 窗口标记 (仅 Persistence + LS-MPR)
for col, mk in [("Persistence","o"),("LSMPR","s")]:
    c = "#2C3E50" if col=="Persistence" else "#E74C3C"
    axA.scatter(windows["Distance_Mid"], windows[f"{col}_RMSE"], s=20, color=c, marker=mk, facecolors="white", linewidth=1.2, zorder=4, alpha=0.6)

axA.axvline(x=0, color="black", linewidth=0.8, linestyle=(0,(5,4)), alpha=0.4, zorder=1)
axA.text(1, 49, "Training cutoff", fontsize=7.5, fontweight="bold", color="#7F8C8D")
axA.set_title("(a) RMSE growth with prediction distance", pad=6)
axA.set_xlabel("Distance from training cutoff (days)"); axA.set_ylabel("RMSE (CO$_2$ g)")
axA.set_xlim(-2, 62); axA.set_ylim(-1, 52)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

leg_els = [Line2D([0],[0], color=c, linewidth=2.0, linestyle=ls, label=n) for _,n,c,ls in models]
legA = axA.legend(handles=leg_els, loc="upper left", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=8, ncol=4)
for t in legA.get_texts(): t.set_fontweight("bold")

# ══════════════════════════════════════════════════════
# PANEL (b): 误差热图 (3 条水平带)
# ══════════════════════════════════════════════════════
cmap_hm = LinearSegmentedColormap.from_list("err", [
    (0.00,"#FFFFFF"),(0.05,"#FFF5EB"),(0.15,"#FDEBD0"),(0.35,"#F5B041"),(0.60,"#E74C3C"),(0.85,"#922B21"),(1.00,"#4A0E0E")])
days = heatmap["Day"].values
vmax = 68
cond_data = [("C1", heatmap["C1_AbsError"].values), ("C2", heatmap["C2_AbsError"].values), ("C3", heatmap["C3_AbsError"].values)]
cond_labels = ["C1 (T=50, H=70, R=70)", "C2 (T=58, H=60, R=100)", "C3 (T=58, H=70, R=0)"]

for idx, (cname, err) in enumerate(cond_data):
    y_off = 2 - idx
    X_mesh = np.arange(0.5, 181.5, 1)
    im = axB.pcolormesh(X_mesh, [y_off, y_off+1], err.reshape(1,-1), cmap=cmap_hm, vmin=0, vmax=vmax, rasterized=True)
    axB.axvline(x=119.5, color="black", linewidth=1.0, zorder=5)
    tr_mae = err[days<=119].mean(); te_mae = err[days>=120].mean()
    axB.text(60, y_off+0.5, f"Train MAE={tr_mae:.1f}", ha="center", va="center", fontsize=7, fontweight="bold", color="#2C3E50")
    axB.text(150, y_off+0.5, f"Test MAE={te_mae:.1f} (x{te_mae/tr_mae:.0f})", ha="center", va="center", fontsize=7, fontweight="bold", color="#C0392B")
    axB.text(2, y_off+0.85, cond_labels[idx], fontsize=6.5, fontweight="bold", color="#7F8C8D", va="top")

axB.set_title("(b) Per-condition per-day absolute error heatmap", pad=6)
axB.set_ylim(0, 3); axB.set_xlim(0.5, 180.5)
axB.set_xlabel("Days"); axB.set_yticks([])
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=False, right=False)

cbar = fig.colorbar(im, ax=axB, fraction=0.025, pad=0.01, ticks=[0,20,40,68])
cbar.set_label("Abs error (g)", fontweight="bold", fontsize=8); cbar.ax.tick_params(length=0)

# ══════════════════════════════════════════════════════
# PANEL (c): 残差符号一致性 (2 子面板)
# ══════════════════════════════════════════════════════
gs_c = gs[2].subgridspec(2, 1, hspace=0.18)
axC_top = fig.add_subplot(gs_c[0])
axC_bot = fig.add_subplot(gs_c[1])

def plot_resid(ax, df, resid_col, title, model_name):
    daily_mean = df.groupby("Day")[resid_col].mean()
    days = daily_mean.index.values; res = daily_mean.values
    pos = res>=0; neg = res<0
    if pos.any(): ax.fill_between(days, 0, res, where=pos, facecolor="#E74C3C", alpha=0.50, linewidth=0)
    if neg.any(): ax.fill_between(days, 0, res, where=neg, facecolor="#3498DB", alpha=0.50, linewidth=0)
    ax.axhline(y=0, color="black", linewidth=0.8, zorder=3)
    row = resid_summ[resid_summ["Model"]==model_name].iloc[0]
    ax.text(0.5, 0.93, f"Pos: {row['Pct_Positive']:.0f}% | Neg: {row['Pct_Negative']:.0f}% | Longest run: {int(row['Longest_Run'])} | MAE: {row['MAE']:.1f}",
            transform=ax.transAxes, fontsize=6.5, fontweight="bold", color="#2C3E50", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#BDC3C7", alpha=0.85))
    ax.set_title(title, pad=5, fontsize=9)
    ax.set_ylabel("Residual (g)")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

plot_resid(axC_top, resid, "LSMPR_Residual", "LS-MPR: Variance collapse + systematic bias", "LS-MPR")
axC_top.set_xticklabels([])
plot_resid(axC_bot, resid, "ExpDecay_Residual", "Exp Decay: Pure structural deviation (100% negative)", "Exp Decay")
axC_bot.set_xlabel("Days")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.01,
         "Figure 4. Error structure — how failure unfolds. (a) RMSE grows monotonically with prediction distance; "
         "Persistence is the most stable. (b) Error is systematic across all 3 conditions post Day 119. "
         "(c) Two distinct failure modes: LS-MPR variance explosion vs Exp Decay systematic overestimation.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_04_Error_Structure.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_04_Error_Structure.svg")
print(f"Saved: {OUT_DIR / 'Fig_04_Error_Structure.png'}")
print(f"Saved: {OUT_DIR / 'Fig_04_Error_Structure.svg'}")
print("Done.")
plt.close(fig)