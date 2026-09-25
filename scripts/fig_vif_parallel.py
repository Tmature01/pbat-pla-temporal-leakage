"""
Figure V2: VIF=inf — 平行坐标图展示 3 条件在 4D 环境空间的退化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 数据 ──────────────────────────────────────────
features = ["Temperature\n(C)", "Humidity\n(%)", "Ratio\n(%)", "Compost\nVolume"]
raw = np.array([
    [50, 70, 70,  0.90],  # C1
    [58, 60, 100, 0.50],  # C2
    [58, 70, 0,   0.75],  # C3
])

# Min-max 归一化到 [0,1]
norm = (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0) + 1e-10)

cond_names = ["C1 (PBAT)", "C2 (PLA)", "C3 (70/30)"]
colors     = ["#C0392B", "#2471A3", "#27AE60"]
linewidths = [2.5, 2.5, 2.5]
linestyles = ["-", "--", "-."]

x = np.arange(len(features))

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 10, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(bottom=0.16, top=0.90, left=0.16, right=0.94)

for i in range(3):
    ax.plot(x, norm[i], color=colors[i], linewidth=linewidths[i],
            linestyle=linestyles[i], marker="o", markersize=10,
            markerfacecolor="white", markeredgewidth=2.0,
            markeredgecolor=colors[i], zorder=3, label=cond_names[i])

# 在每个轴标注原始值
for j in range(4):
    for i in range(3):
        val = raw[i, j]
        txt = f"{val:.0f}" if j < 3 else f"{val:.2f}"
        y_pos = norm[i, j]
        dy = 0.035 if i == 1 else -0.035
        ax.text(j, y_pos + dy, txt,
                fontsize=6.5, fontweight="bold", color=colors[i], ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          ec=colors[i], alpha=0.7, linewidth=0.5))

# ── 关键标注 ─────────────────────────────────────────
# 在 Temperature-Humidity 之间标注"同步"
ax.annotate("C2 & C3 share\nsame T (58 C)",
            xy=(0.1, 0.97), xytext=(0.5, 0.35),
            fontsize=7.5, fontweight="bold", color="#2471A3",
            arrowprops=dict(arrowstyle="->", color="#2471A3", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2471A3", alpha=0.85))

# 在 Ratio 轴标注分化
ax.annotate("3 lines in 4D:\nrank = 3 < 4\n-> VIF = inf",
            xy=(3.0, 0.5), xytext=(3.2, 0.75),
            fontsize=8.5, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C0392B", alpha=0.88))

# ── 坐标轴 ───────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=10)
ax.set_ylabel("Normalized value")
ax.set_ylim(-0.08, 1.12)
ax.set_xlim(-0.5, 3.5)
ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# 竖直虚线分隔各特征
for j in range(3):
    ax.axvline(x=j + 0.5, color="#BDC3C7", linewidth=0.4, linestyle=":", alpha=0.5, zorder=0)

# ── 图例 ─────────────────────────────────────────────
leg = ax.legend(loc="upper left", framealpha=0.85, edgecolor="#BDC3C7",
                fancybox=True, fontsize=8.5, borderpad=0.5,
                bbox_to_anchor=(-0.22, 1.0))
for t in leg.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.03,
         "3 lines across 4 axes can only span 3 independent dimensions. "
         "A 4th condition would be needed to uniquely determine all 4 environmental effects. "
         "This is why VIF = inf and the design matrix is rank-deficient.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_vif_parallel.png", dpi=600)
fig.savefig(OUT_DIR / "fig_vif_parallel.svg")
print(f"Saved: {OUT_DIR / 'fig_vif_parallel.png'}")
print(f"Saved: {OUT_DIR / 'fig_vif_parallel.svg'}")
print("Done.")
plt.close(fig)