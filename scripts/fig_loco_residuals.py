"""
Figure: LOCO 三折残差分析 — Panel (a) KDE + Panel (b) Residual vs Days
"""

import pandas as pd
import numpy as np
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

resid = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Residuals.csv"))
kde   = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Residuals_KDE.csv"))
summ  = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Residuals_Summary.csv"))

folds = ["C1 (Pure PBAT)", "C2 (Pure PLA)", "C3 (70/30 PBAT/PLA)"]
colors = {"C1 (Pure PBAT)": "#E74C3C", "C2 (Pure PLA)": "#3498DB",
          "C3 (70/30 PBAT/PLA)": "#2ECC71"}
short = {"C1 (Pure PBAT)": "C1 (PBAT)", "C2 (Pure PLA)": "C2 (PLA)",
         "C3 (70/30 PBAT/PLA)": "C3 (70/30)"}

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.24, bottom=0.14, top=0.88, left=0.08, right=0.96)

# ══════════════════════════════════════════════════════
# Panel (a): KDE 分布
# ══════════════════════════════════════════════════════
for fold in folds:
    axA.plot(kde["Residual"], kde[fold], color=colors[fold], linewidth=2.0, zorder=3)

axA.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)

for i, fold in enumerate(folds):
    row = summ[summ["Fold"] == fold].iloc[0]
    bias, stdv = row["Bias"], row["Std"]
    # 手动指定标注位置避免重叠
    annot_pos = [(112, 0.035), (120, 0.005), (-66, 0.0042)]
    axA.annotate(f"{short[fold]}\nBias={bias:+.0f}\nStd={stdv:.0f}",
                xy=(bias, kde[fold].max()), xytext=annot_pos[i],
                fontsize=7.5, fontweight="bold", color=colors[fold],
                arrowprops=dict(arrowstyle="->", color=colors[fold], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colors[fold], alpha=0.85))

axA.set_title("(a) Residual distribution (KDE)", pad=8)
axA.set_xlabel("Residual (CO$_2$ g)"); axA.set_ylabel("Density")
axA.set_xlim(-200, 400)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axA.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ══════════════════════════════════════════════════════
# Panel (b): 残差 vs Days + LOESS
# ══════════════════════════════════════════════════════
for fold in folds:
    sub = resid[resid["Fold"] == fold]
    days, res = sub["Day"].values, sub["Residual"].values
    axB.scatter(days, res, s=10, color=colors[fold], alpha=0.35,
                edgecolors="none", rasterized=True, zorder=2)
    lo = np.asarray(lowess(res, days, frac=0.25, return_sorted=True))
    axB.plot(lo[:, 0], lo[:, 1], color=colors[fold], linewidth=2.0, zorder=4)
    rho = summ[summ["Fold"] == fold]["rho"].values[0]
    # 曲线内嵌短标签
    axB.text(140, lo[-1, 1] + 5, f"{short[fold]} (r={rho:+.2f})",
            fontsize=7, fontweight="bold", color=colors[fold], va="bottom", ha="right")

axB.axhline(y=0, color="black", linewidth=0.8, alpha=0.4, zorder=1)
axB.set_title("(b) Residual vs Days  |  LOESS trend", pad=8)
axB.set_xlabel("Days"); axB.set_ylabel("Residual (CO$_2$ g)")
axB.set_xlim(-3, 195)
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axB.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axB.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# 标注: 移到空白区
axB.text(20, 184, "C2: catastrophic PLA\nmechanism mismatch\n(r=+1.00)", fontsize=7.5,
         fontweight="bold", color="#2471A3",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2471A3", alpha=0.85))
axB.text(130, -94, "C3: sign reversal\n(temp gap effect)\n(r=-1.00)", fontsize=7.5,
         fontweight="bold", color="#27AE60",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#27AE60", alpha=0.85))

# 图例
leg_els = [Line2D([0],[0], color=colors[f], linewidth=2.0, label=short[f]) for f in folds]
leg = fig.legend(handles=leg_els, loc="upper center", ncol=3,
                 framealpha=0.90, edgecolor="#BDC3C7", fancybox=True,
                 fontsize=9, bbox_to_anchor=(0.5, 0.94))
for t in leg.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "Three distinct LOCO failure modes. C1: systematic under-prediction. "
         "C2: catastrophic — PLA mechanism mismatch. C3: sign reversal — temperature gap.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_loco_residuals.png", dpi=600)
fig.savefig(OUT_DIR / "fig_loco_residuals.svg")
print(f"Saved: {OUT_DIR / 'fig_loco_residuals.png'}")
print(f"Saved: {OUT_DIR / 'fig_loco_residuals.svg'}")
print("Done.")
plt.close(fig)