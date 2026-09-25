"""
Figure: LOCO 失败与条件距离的关系
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

loco = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Distance_R2.csv"))
curves = pd.read_csv(str(FIGURE_DATA / "Fig_LOCO_Curves.csv"))

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.subplots_adjust(wspace=0.24, bottom=0.15, top=0.90, left=0.07, right=0.96)

# ── (a) CO2 降解曲线 ────────────────────────────────
colors_c = {"C1 (Pure PBAT)": "#E74C3C",
            "C2 (Pure PLA)": "#3498DB",
            "C3 (70/30 PBAT/PLA)": "#2ECC71"}
labels_c = list(colors_c.keys())

for col_name in labels_c:
    axL.plot(curves["Day"], curves[col_name], color=colors_c[col_name],
             linewidth=1.6, label=col_name, zorder=2)

axL.set_title("(a) CO$_2$ degradation curves", pad=6)
axL.set_xlabel("Days")
axL.set_ylabel("CO$_2$ (g)")
axL.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axL.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axL.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
leg = axL.legend(loc="lower right", framealpha=0.90, edgecolor="#BDC3C7",
                 fancybox=True, fontsize=8)
for t in leg.get_texts():
    t.set_fontweight("bold")

# ── (b) Mech_Dist vs LOCO R^2 ────────────────────────
x = loco["Mech_Dist"].values
y = loco["LOCO_R2"].values
fold_colors = {"C1 (Pure PBAT)": "#E74C3C",
               "C2 (Pure PLA)": "#3498DB",
               "C3 (70/30 PBAT/PLA)": "#2ECC71"}

for i in range(len(loco)):
    fc = fold_colors.get(loco.iloc[i]["Fold"], "#333333")
    axR.scatter(x[i], y[i], s=140, color=fc, edgecolors="white",
                linewidth=1.2, zorder=4)

offsets = [(-25, -8), (20, 8), (-20, -12)]
for i in range(len(loco)):
    fc = fold_colors.get(loco.iloc[i]["Fold"], "#333333")
    name = loco.iloc[i]["HeldOut"]
    axR.annotate(name, xy=(x[i], y[i]), xytext=offsets[i],
                textcoords="offset points", fontsize=8, fontweight="bold",
                color=fc, ha="center",
                arrowprops=dict(arrowstyle="->", color=fc, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=fc, alpha=0.85),
                zorder=5)

# 趋势线
r, p = stats.pearsonr(x, y)
x_fit = np.linspace(x.min() - 0.02, x.max() + 0.02, 50)
slope, intercept, _, _, _ = stats.linregress(x, y)
axR.plot(x_fit, slope * x_fit + intercept, color="gray", linewidth=0.8,
         linestyle="--", alpha=0.6, zorder=1)

axR.axhline(y=0, color="black", linewidth=0.5, alpha=0.3, zorder=0)

axR.set_title("(b) Mechanism distance vs LOCO $R^2$  |  $r$ = {:.2f}".format(r), pad=6)
axR.set_xlabel("Mechanism distance (PLA fraction + Temp diff)")
axR.set_ylabel("LOCO $R^2$")
axR.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axR.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axR.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

axR.text(0.95, 0.06, "n = 3 folds only\n(hypothesis-generating)",
         transform=axR.transAxes, fontsize=7, fontweight="bold",
         color="#7F8C8D", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDC3C7", alpha=0.85))

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "C2 (Pure PLA): closest in CO2 curve shape but worst LOCO failure (R^2=-7.37). "
         "Mechanism incompatibility (hydrolytic vs enzymatic degradation) drives LOCO failure. "
         "n=3, exploratory only.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_loco_distance.png", dpi=600)
fig.savefig(OUT_DIR / "fig_loco_distance.svg")
print(f"Saved: {OUT_DIR / 'fig_loco_distance.png'}")
print(f"Saved: {OUT_DIR / 'fig_loco_distance.svg'}")
print(f"r (Mech_Dist vs LOCO R2) = {r:.3f}, p = {p:.3f}")
print("Done.")
plt.close(fig)