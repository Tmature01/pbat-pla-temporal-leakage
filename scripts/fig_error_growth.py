"""
Figure: 时间预测误差增长曲线 — 简洁版
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

smooth  = pd.read_csv(str(FIGURE_DATA / "Fig_ErrorGrowth_Smooth.csv"))
windows = pd.read_csv(str(FIGURE_DATA / "Fig_ErrorGrowth_Windows.csv"))

models = [
    ("Persistence", "Persistence", "#2C3E50", "-"),
    ("LSMPR",       "LS-MPR",      "#E74C3C", "--"),
    ("GPR",         "GPR",         "#3498DB", "-."),
    ("ExpDecay",    "Exp Decay",   "#27AE60", ":"),
]

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.subplots_adjust(bottom=0.14, top=0.92, left=0.10, right=0.96)

# ── 平滑曲线 (主视觉) ────────────────────────────────
for col, name, color, ls in models:
    ax.plot(smooth["Distance"], smooth[f"{col}_RMSE"], color=color,
            linewidth=2.2, linestyle=ls, zorder=3, label=name)

# ── 窗口标记 (仅 Persistence + LS-MPR, 减少杂乱) ───
markers_list = [("Persistence", "o", 5), ("LSMPR", "s", 5)]
for col, mk, ms in markers_list:
    color = "#2C3E50" if col == "Persistence" else "#E74C3C"
    ax.scatter(windows["Distance_Mid"], windows[f"{col}_RMSE"],
              s=25, color=color, marker=mk, facecolors="white",
              linewidth=1.5, zorder=4, alpha=0.7)

# ── 参考线 ───────────────────────────────────────────
ax.axvline(x=0, color="black", linewidth=0.8, linestyle=(0, (5, 4)),
           alpha=0.4, zorder=1)
ax.text(1, ax.get_ylim()[1] * 0.92, "Training cutoff", fontsize=8,
        fontweight="bold", color="#7F8C8D", ha="left")

# Persistence 基线
pers_final = windows["Persistence_RMSE"].values[-1]

# ── 坐标轴 ───────────────────────────────────────────
ax.set_xlabel("Distance from training cutoff (days)")
ax.set_ylabel("RMSE (CO$_2$ g)")
ax.set_xlim(-2, 62)
ax.set_ylim(-1, 52)
ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 图例 ─────────────────────────────────────────────
leg_els = []
for col, name, color, ls in models:
    leg_els.append(Line2D([0],[0], color=color, linewidth=2.2, linestyle=ls,
                   label=name))
leg = ax.legend(handles=leg_els, loc="upper left", framealpha=0.90,
                edgecolor="#BDC3C7", fancybox=True, fontsize=9,
                bbox_to_anchor=(0.14, 1.0))
for t in leg.get_texts(): t.set_fontweight("bold")

# ── 简洁标注 ─────────────────────────────────────────
ax.annotate("LS-MPR & GPR RMSE\nexplode ~50x beyond\nDay 150",
            xy=(50, 45), xytext=(31, 42),
            fontsize=8, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C0392B", alpha=0.85))

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "RMSE grows monotonically with prediction distance. "
         "Persistence (0 params) is the most stable. "
         "Error explosion is structural, not a window-selection artifact.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_error_growth.png", dpi=600)
fig.savefig(OUT_DIR / "fig_error_growth.svg")
print(f"Saved: {OUT_DIR / 'fig_error_growth.png'}")
print(f"Saved: {OUT_DIR / 'fig_error_growth.svg'}")
print("Done.")
plt.close(fig)