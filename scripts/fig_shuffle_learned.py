"""
Figure: LS-MPR 洗牌后"学到"的函数 — 原始 vs 洗牌对比
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

scatter   = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_Scatter.csv"))
curves    = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_Curves.csv"))
orig_ts   = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_OriginalTS.csv"))

# R^2
r2_orig = r2_score(orig_ts["CO2"],
                   np.interp(orig_ts["Days"], curves["Day"], curves["Pred_Original"]))
r2_shuf = r2_score(scatter["CO2_Observed"],
                   np.interp(scatter["Day_Shuffled"], curves["Day"], curves["Pred_Shuffled"]))
print(f"R^2 Original: {r2_orig:.4f}")
print(f"R^2 Shuffled: {r2_shuf:.4f}")

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

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.subplots_adjust(wspace=0.22, bottom=0.14, top=0.90, left=0.08, right=0.96)

# ── (a) 原始数据 ────────────────────────────────────
ax0.scatter(orig_ts["Days"], orig_ts["CO2"], s=16, color="#34495E", alpha=0.55,
            edgecolors="none", rasterized=True, zorder=2)
ax0.plot(curves["Day"], curves["Pred_Original"], color="#C0392B", linewidth=2.2,
         zorder=3, label="LS-MPR fit")
ax0.set_title("(a) Original time series  |  LS-MPR fit", pad=8)
ax0.set_xlabel("Days")
ax0.set_ylabel("CO$_2$ (g)")
ax0.text(0.95, 0.08, f"$R^2$ = {r2_orig:.3f}\n(monotonic saturation curve)",
         transform=ax0.transAxes, fontsize=9, fontweight="bold",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

# ── (b) 洗牌数据 ────────────────────────────────────
ax1.scatter(scatter["Day_Shuffled"], scatter["CO2_Observed"], s=16, color="#7F8C8D",
            alpha=0.45, edgecolors="none", rasterized=True, zorder=2)
ax1.plot(curves["Day"], curves["Pred_Shuffled"], color="#E74C3C", linewidth=2.2,
         zorder=3, label="LS-MPR fit (shuffled)")

# 叠加原始时间序列作为参照
ax1.scatter(orig_ts["Days"], orig_ts["CO2"], s=12, color="#3498DB", alpha=0.30,
            edgecolors="none", rasterized=True, zorder=1)

ax1.set_title("(b) Shuffled data  |  LS-MPR 'learned' function", pad=8)
ax1.set_xlabel("Days (shuffled)")
ax1.set_ylabel("CO$_2$ (g)")
ax1.text(0.95, 0.08,
         f"$R^2$ = {r2_shuf:.3f}\n(near-constant ~196 g, no kinetics learned)",
         transform=ax1.transAxes, fontsize=9, fontweight="bold",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

# 标注洗牌曲线的窄范围
ymin_s = curves["Pred_Shuffled"].min()
ymax_s = curves["Pred_Shuffled"].max()
ax1.annotate(f"Range: [{ymin_s:.0f}, {ymax_s:.0f}] g\n(~mean CO$_2$)",
            xy=(90, (ymin_s + ymax_s) / 2),
            xytext=(130, ymax_s + 20),
            fontsize=8, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C0392B", alpha=0.85))

# ── 坐标轴 ───────────────────────────────────────────
for ax in (ax0, ax1):
    ax.set_xlim(-5, 185)
    ax.tick_params(which="both", direction="in", bottom=True, top=False,
                   left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# 图例
from matplotlib.lines import Line2D
leg_els = [
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#34495E",
           markersize=7, markeredgewidth=0, label="Observed (original order)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#7F8C8D",
           markersize=7, markeredgewidth=0, label="Observed (shuffled)"),
    Line2D([0],[0], color="#C0392B", linewidth=2.0, label="LS-MPR fit"),
]
leg = ax1.legend(handles=leg_els, loc="upper left", framealpha=0.90,
                 edgecolor="#BDC3C7", fancybox=True, fontsize=8,
                 borderpad=0.5, labelspacing=0.3)
for t in leg.get_texts():
    t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "After shuffling, LS-MPR 'learns' a near-constant function hovering around the mean CO2, "
         "R^2 ~ 0. It found no degradation kinetics — only noise. "
         "The original R^2=0.995 relied entirely on temporal proximity, not physical understanding.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_shuffle_learned.png", dpi=600)
fig.savefig(OUT_DIR / "fig_shuffle_learned.svg")
print(f"\nSaved: {OUT_DIR / 'fig_shuffle_learned.png'}")
print(f"Saved: {OUT_DIR / 'fig_shuffle_learned.svg'}")
print("Done.")
plt.close(fig)