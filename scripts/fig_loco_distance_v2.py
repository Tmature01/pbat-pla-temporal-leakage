"""
Figure V2: LOCO 失败与条件距离 — 3 面板
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

fold_colors = {"C1 (Pure PBAT)": "#E74C3C",
               "C2 (Pure PLA)": "#3498DB",
               "C3 (70/30 PBAT/PLA)": "#2ECC71"}
fold_short = {"C1 (Pure PBAT)": "C1", "C2 (Pure PLA)": "C2",
              "C3 (70/30 PBAT/PLA)": "C3"}

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

# 布局: 左大 (A) + 右两小 (B上 C下)
fig = plt.figure(figsize=(14, 6.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1],
                      hspace=0.32, wspace=0.26)
axA = fig.add_subplot(gs[:, 0])   # 左列, 跨两行
axB = fig.add_subplot(gs[0, 1])   # 右上
axC = fig.add_subplot(gs[1, 1])   # 右下

fig.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.13)

# ═══════════════════════════════════════════════════════
# Panel A: CO₂ 曲线距离 vs LOCO R² (主图)
# ═══════════════════════════════════════════════════════
xA = loco["CO2_Integral_Diff"].values
yA = loco["LOCO_R2"].values

for i in range(len(loco)):
    fc = fold_colors[loco.iloc[i]["Fold"]]
    fs = fold_short[loco.iloc[i]["Fold"]]
    axA.scatter(xA[i], yA[i], s=160, color=fc, edgecolors="white",
                linewidth=1.2, zorder=4)
    axA.annotate(fs, xy=(xA[i], yA[i]), xytext=(0, 6),
                textcoords="offset points", fontsize=10, fontweight="bold",
                color=fc, ha="center", zorder=5)

# 趋势线
rA, pA = stats.pearsonr(xA, yA)
xA_fit = np.linspace(xA.min() - 200, xA.max() + 200, 50)
slopeA, intA, _, _, _ = stats.linregress(xA, yA)
axA.plot(xA_fit, slopeA * xA_fit + intA, color="gray", linewidth=0.8,
         linestyle="--", alpha=0.5, zorder=1)

axA.axhline(y=0, color="black", linewidth=0.6, alpha=0.35, zorder=0)
axA.set_title("(a) CO$_2$ curve distance vs LOCO $R^2$", pad=8)
axA.set_xlabel("CO$_2$ integral difference (g·day)")
axA.set_ylabel("LOCO $R^2$")

# 关键标注
# C2: 最近但最差
axA.annotate("Closest CO$_2$ curve shape,\nworst LOCO failure\n(mechanism mismatch)",
            xy=(xA[1], yA[1]), xytext=(xA[1] + 20, yA[1] + 1.2),
            fontsize=7.5, fontweight="bold", color="#2471A3",
            arrowprops=dict(arrowstyle="->", color="#2471A3", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2471A3", alpha=0.85))

# C1: 最远却唯一正
axA.annotate("Largest feature distance,\nonly positive LOCO $R^2$",
            xy=(xA[0], yA[0]), xytext=(xA[0] - 100, yA[0] - 1.2),
            fontsize=7.5, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C0392B", alpha=0.85))

# C3: 最远但中间表现
axA.annotate("Largest CO$_2$ distance,\nintermediate LOCO failure\n(blend: mechanism partially shared)",
            xy=(xA[2], yA[2]), xytext=(xA[2] - 1200, yA[2] - 0.5),
            fontsize=7.5, fontweight="bold", color="#27AE60",
            arrowprops=dict(arrowstyle="->", color="#27AE60", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#27AE60", alpha=0.85))

axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axA.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ═══════════════════════════════════════════════════════
# Panel B: 机理距离 vs LOCO R²
# ═══════════════════════════════════════════════════════
xB = loco["Mech_Dist"].values
yB = loco["LOCO_R2"].values

for i in range(len(loco)):
    fc = fold_colors[loco.iloc[i]["Fold"]]
    fs = fold_short[loco.iloc[i]["Fold"]]
    axB.scatter(xB[i], yB[i], s=120, color=fc, edgecolors="white",
                linewidth=1.0, zorder=4)
    axB.annotate(fs, xy=(xB[i], yB[i]), xytext=(0, 10),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color=fc, ha="center", zorder=5)

rB, pB = stats.pearsonr(xB, yB)
xB_fit = np.linspace(xB.min() - 0.02, xB.max() + 0.02, 50)
slopeB, intB, _, _, _ = stats.linregress(xB, yB)
axB.plot(xB_fit, slopeB * xB_fit + intB, color="gray", linewidth=0.8,
         linestyle="--", alpha=0.5, zorder=1)

axB.axhline(y=0, color="black", linewidth=0.6, alpha=0.35, zorder=0)
axB.set_title(f"(b) Mechanism distance vs LOCO $R^2$  |  $r$ = {rB:.2f}", pad=6)
axB.set_xlabel("Mechanism distance (PLA% + Temperature)")
axB.set_ylabel("LOCO $R^2$")
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axB.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axB.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ═══════════════════════════════════════════════════════
# Panel C: 3 条降解曲线
# ═══════════════════════════════════════════════════════
labels_c = list(fold_colors.keys())

for col_name in labels_c:
    axC.plot(curves["Day"], curves[col_name], color=fold_colors[col_name],
             linewidth=1.5, label=fold_short[col_name] + ": " +
             col_name.split("(")[-1].rstrip(")"), zorder=2)

axC.set_title("(c) CO$_2$ degradation curves", pad=6)
axC.set_xlabel("Days")
axC.set_ylabel("CO$_2$ (g)")
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axC.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axC.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
leg = axC.legend(loc="lower right", framealpha=0.90, edgecolor="#BDC3C7",
                 fancybox=True, fontsize=7.5)
for t in leg.get_texts():
    t.set_fontweight("bold")

# ── n=3 声明 (全局) ──────────────────────────────────
fig.text(0.96, 0.015, "n = 3 folds only — hypothesis-generating, not confirmatory",
         ha="right", fontsize=7, fontweight="bold", color="#7F8C8D", style="italic")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "LOCO failure is driven by mechanism incompatibility (hydrolytic vs enzymatic degradation), "
         "not by feature-space distance. C2 (Pure PLA) has the most similar CO2 curve shape but the worst R^2.",
         ha="center", fontsize=7.2, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_loco_distance_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_loco_distance_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_loco_distance_v2.png'}")
print(f"Saved: {OUT_DIR / 'fig_loco_distance_v2.svg'}")
print(f"Panel A: r={rA:.3f}, p={pA:.3f}")
print(f"Panel B: r={rB:.3f}, p={pB:.3f}")
print("Done.")
plt.close(fig)