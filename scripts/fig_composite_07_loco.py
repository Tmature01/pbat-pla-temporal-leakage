"""
Figure 7 (复合): LOCO — 三种不同的失败模式
(a) 残差 KDE (b) 残差 vs Days (c) LOCO 距离散点
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

resid   = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Residuals.csv"))
kde     = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Residuals_KDE.csv"))
summ    = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Residuals_Summary.csv"))
loco    = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Distance_R2.csv"))
curves  = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Curves.csv"))

folds = ["C1 (Pure PBAT)", "C2 (Pure PLA)", "C3 (70/30 PBAT/PLA)"]
colors_f = {"C1 (Pure PBAT)":"#E74C3C","C2 (Pure PLA)":"#3498DB","C3 (70/30 PBAT/PLA)":"#2ECC71"}
short = {"C1 (Pure PBAT)":"C1 (PBAT)","C2 (Pure PLA)":"C2 (PLA)","C3 (70/30 PBAT/PLA)":"C3 (70/30)"}

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

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.0],
                      wspace=0.28, left=0.05, right=0.97, top=0.92, bottom=0.10)

axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

# ══════════════════════════════════════════════════════
# PANEL (a): 残差 KDE
# ══════════════════════════════════════════════════════
for fold in folds:
    axA.plot(kde["Residual"], kde[fold], color=colors_f[fold], linewidth=2.0, zorder=3)
axA.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)

annot_pos_a = [(112, 0.035), (120, 0.005), (-66, 0.0042)]
for i, fold in enumerate(folds):
    row = summ[summ["Fold"]==fold].iloc[0]
    axA.annotate(f"{short[fold]}\nBias={row['Bias']:+.0f}\nStd={row['Std']:.0f}",
                xy=(row["Bias"], kde[fold].max()), xytext=annot_pos_a[i],
                fontsize=7, fontweight="bold", color=colors_f[fold],
                arrowprops=dict(arrowstyle="->", color=colors_f[fold], lw=0.7),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=colors_f[fold], alpha=0.85))

axA.set_title("(a) Residual distribution (KDE)", pad=6)
axA.set_xlabel("Residual (CO$_2$ g)"); axA.set_ylabel("Density")
axA.set_xlim(-200, 400)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# PANEL (b): 残差 vs Days + LOESS
# ══════════════════════════════════════════════════════
for fold in folds:
    sub = resid[resid["Fold"]==fold]
    axB.scatter(sub["Day"], sub["Residual"], s=8, color=colors_f[fold], alpha=0.30, edgecolors="none", rasterized=True, zorder=2)
    lo = np.asarray(lowess(sub["Residual"].values, sub["Day"].values, frac=0.25, return_sorted=True))
    axB.plot(lo[:,0], lo[:,1], color=colors_f[fold], linewidth=2.0, zorder=4)
    rho = summ[summ["Fold"]==fold]["rho"].values[0]
    axB.text(140, lo[-1,1]+5, f"{short[fold]} (r={rho:+.2f})", fontsize=6.5, fontweight="bold", color=colors_f[fold], va="bottom", ha="right")

axB.axhline(y=0, color="black", linewidth=0.8, alpha=0.4, zorder=1)
axB.set_title("(b) Residual vs Days | LOESS trend", pad=6)
axB.set_xlabel("Days"); axB.set_ylabel("Residual (CO$_2$ g)")
axB.set_xlim(-3, 195)
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

axB.text(20, 184, "C2: catastrophic PLA\nmechanism mismatch\n(r=+1.00)", fontsize=7, fontweight="bold", color="#2471A3",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#2471A3", alpha=0.85))
axB.text(130, -94, "C3: sign reversal\n(temp gap effect)\n(r=-1.00)", fontsize=7, fontweight="bold", color="#27AE60",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#27AE60", alpha=0.85))

# ══════════════════════════════════════════════════════
# PANEL (c): LOCO 距离散点 (机理距离 vs LOCO R^2)
# ══════════════════════════════════════════════════════
xc = loco["Mech_Dist"].values
yc = loco["LOCO_R2"].values
fc_c = {"C1 (Pure PBAT)":"#E74C3C","C2 (Pure PLA)":"#3498DB","C3 (70/30 PBAT/PLA)":"#2ECC71"}

for i in range(len(loco)):
    fc = fc_c.get(loco.iloc[i]["Fold"],"#333")
    axC.scatter(xc[i], yc[i], s=100, color=fc, edgecolors="white", linewidth=1.0, zorder=4)
    fs = loco.iloc[i]["Fold"].split(" (")[0]
    axC.annotate(fs, xy=(xc[i],yc[i]), xytext=(0,6), textcoords="offset points", fontsize=8, fontweight="bold", color=fc, ha="center", zorder=5)

rC, pC = stats.pearsonr(xc, yc)
xf = np.linspace(xc.min()-0.02, xc.max()+0.02, 50)
sl, ic, _, _, _ = stats.linregress(xc, yc)
axC.plot(xf, sl*xf+ic, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
axC.axhline(y=0, color="black", linewidth=0.5, alpha=0.3, zorder=0)

axC.set_title(f"(c) Mechanism distance vs LOCO $R^2$\n$r$ = {rC:.2f}", pad=6)
axC.set_xlabel("Mechanism distance"); axC.set_ylabel("LOCO $R^2$")
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# 统一图例
leg_els = [Line2D([0],[0], color=colors_f[f], linewidth=2.0, label=short[f]) for f in folds]
leg = fig.legend(handles=leg_els, loc="upper center", ncol=3, framealpha=0.90, edgecolor="#BDC3C7", fancybox=True, fontsize=8.5, bbox_to_anchor=(0.5, 0.97))
for t in leg.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "Figure 7. LOCO — three distinct failure modes. (a) C2 (PLA) has catastrophic bias (+194). "
         "(b) C2 explodes, C3 sign-reverses, C1 shifts. (c) Mechanism distance predicts LOCO failure (r=-0.82), "
         "CO2 curve distance does not (r=+0.60). LOCO failure is about mechanism compatibility, not feature similarity.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_07_LOCO_FailureModes.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_07_LOCO_FailureModes.svg")
print(f"Saved: {OUT_DIR / 'Fig_07_LOCO_FailureModes.png'}")
print(f"Saved: {OUT_DIR / 'Fig_07_LOCO_FailureModes.svg'}")
print("Done.")
plt.close(fig)