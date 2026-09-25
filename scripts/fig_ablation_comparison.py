"""
Figure: 真实数据 vs 合成数据 特征消融对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

abl = pd.read_csv(str(FIGURE_DATA / "Fig_Ablation_Comparison.csv"))
cond = pd.read_csv(str(FIGURE_DATA / "Fig_Ablation_ConditionNumber.csv"))

features = abl["Feature"].values
real_dr2 = abl["Real_Delta_R2"].values
syn_dr2  = abl["Syn_Delta_R2"].values

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 10, "axes.labelweight": "bold",
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5.2))
fig.subplots_adjust(wspace=0.30, bottom=0.14, top=0.88, left=0.06, right=0.97)

C_DAYS = "#C0392B"
C_ENV  = "#3498DB"
C_SYN  = "#E67E22"

x = np.arange(len(features))
bar_w = 0.55

# ══════════════════════════════════════════════════════
# Panel (a): Real data
# ══════════════════════════════════════════════════════
colors_a = [C_DAYS if f == "Days" else C_ENV for f in features]
bars_a = axA.bar(x, real_dr2, bar_w, color=colors_a, edgecolor="white", linewidth=0.5, zorder=3)

# 标注
for i, (f, v) in enumerate(zip(features, real_dr2)):
    if abs(v) > 0.001:
        axA.text(i, v - 0.03, f"{v:+.4f}", ha="center", fontsize=8, fontweight="bold",
                color="white" if abs(v) > 0.5 else C_DAYS)
    else:
        axA.text(i, v + 0.02, "0.000", ha="center", fontsize=7, fontweight="bold", color=C_ENV)

axA.axhline(y=0, color="black", linewidth=0.5, zorder=1)
axA.set_title("(a) Real data ablation\n(VIF = inf)", pad=8)
axA.set_ylabel("$\\Delta R^2$")
axA.set_xticks(x)
axA.set_xticklabels(["Days", "Temp", "Humid", "Ratio", "CV"], rotation=20)
axA.set_ylim(-1.05, 0.08)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# 4 个环境特征为零的标注
axA.annotate("4 features:\n$\\Delta R^2 = 0$\n(complete collinearity)",
            xy=(2.5, 0), xytext=(3.3, -0.35),
            fontsize=7.5, fontweight="bold", color=C_ENV, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ENV, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_ENV, alpha=0.85))

# ══════════════════════════════════════════════════════
# Panel (b): Synthetic data
# ══════════════════════════════════════════════════════
colors_b = [C_DAYS if f == "Days" else C_SYN for f in features]
axB.bar(x, syn_dr2, bar_w, color=colors_b, edgecolor="white", linewidth=0.5, zorder=3)

for i, (f, v) in enumerate(zip(features, syn_dr2)):
    axB.text(i, v - 0.02, f"{v:+.3f}", ha="center", fontsize=7.5,
            fontweight="bold", color="white" if abs(v) > 0.15 else "#2C3E50")

axB.axhline(y=0, color="black", linewidth=0.5, zorder=1)
axB.set_title("(b) Synthetic data ablation\n(independent features)", pad=8)
axB.set_ylabel("$\\Delta R^2$")
axB.set_xticks(x)
axB.set_xticklabels(["Days", "Temp", "Humid", "Ratio", "CV"], rotation=20)
axB.set_ylim(-1.05, 0.08)
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

axB.annotate("All 5 features\ncontribute significantly",
            xy=(1.5, -0.28), xytext=(3.3, -0.35),
            fontsize=7.5, fontweight="bold", color=C_SYN, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_SYN, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_SYN, alpha=0.85))

# ══════════════════════════════════════════════════════
# Panel (c): Condition number
# ══════════════════════════════════════════════════════
labels_c = cond["Data"].values
log10_c  = cond["Log10_Condition"].values
eff_r    = cond["Effective_Rank"].values
full_r   = cond["Full_Rank"].values

colors_c = ["#C0392B", "#27AE60"]
axC.bar([0, 1], log10_c, 0.5, color=colors_c, edgecolor="white", linewidth=0.5, zorder=3)

for i, (lbl, lv, er, fr) in enumerate(zip(labels_c, log10_c, eff_r, full_r)):
    cn = cond["Condition_Number"].values[i]
    axC.text(i, lv + 1.2, f"log10 = {lv:.1f}\n$\\kappa$ = {cn:.1e}\nRank = {er}/{fr}",
            ha="center", fontsize=7.5, fontweight="bold", color=colors_c[i],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors_c[i], alpha=0.85))

axC.set_title("(c) Design matrix condition", pad=8)
axC.set_ylabel("Log$_{10}$ condition number")
axC.set_xticks([0, 1])
axC.set_xticklabels(["Real\n(PBAT/PLA)", "Synthetic\n(independent)"])
axC.set_ylim(0, 38)
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "Real data: only Days matters (4 environmental features contribute zero — VIF=inf). "
         "Synthetic data: all 5 independent features contribute. "
         "The limitation is the experimental design, not the method.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_ablation_comparison.png", dpi=600)
fig.savefig(OUT_DIR / "fig_ablation_comparison.svg")
print(f"Saved: {OUT_DIR / 'fig_ablation_comparison.png'}")
print(f"Saved: {OUT_DIR / 'fig_ablation_comparison.svg'}")
print("Done.")
plt.close(fig)