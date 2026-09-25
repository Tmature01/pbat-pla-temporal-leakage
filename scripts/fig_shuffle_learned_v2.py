"""
Figure V2: LS-MPR 洗牌后"学到"的函数 (画布不变 12x5.2, 文字x1.5, LaTeX R²)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

scatter = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_Scatter.csv"))
curves  = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_Curves.csv"))
orig_ts = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_OriginalTS.csv"))

r2_orig = r2_score(orig_ts["CO2"], np.interp(orig_ts["Days"], curves["Day"], curves["Pred_Original"]))
r2_shuf = r2_score(scatter["CO2_Observed"], np.interp(scatter["Day_Shuffled"], curves["Day"], curves["Pred_Shuffled"]))
print(f"R2 Original: {r2_orig:.4f}, R2 Shuffled: {r2_shuf:.4f}")

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 13.5, "font.weight": "bold",
    "axes.labelsize": 16.5, "axes.labelweight": "bold",
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.subplots_adjust(wspace=0.26, bottom=0.08, top=0.94, left=0.09, right=0.96)

AFS = 12   # annotate font (9→12)

# ══════════════════════════════════════════════════════
# (a) 原始数据
# ══════════════════════════════════════════════════════
ax0.scatter(orig_ts["Days"], orig_ts["CO2"], s=20, color="#34495E", alpha=0.55,
            edgecolors="none", rasterized=True, zorder=2)
ax0.plot(curves["Day"], curves["Pred_Original"], color="#C0392B", linewidth=2.5, zorder=3)
ax0.set_title("(a) Original time series | LS-MPR fit", pad=10)
ax0.set_xlabel("Days"); ax0.set_ylabel("CO$_2$ (g)")
ax0.text(0.95, 0.06, f"$R^2$ = {r2_orig:.3f}\n(monotonic saturation curve)",
         transform=ax0.transAxes, fontsize=AFS, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
ax0.set_xlim(-5, 185)
ax0.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax0.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ══════════════════════════════════════════════════════
# (b) 洗牌数据
# ══════════════════════════════════════════════════════
ax1.scatter(scatter["Day_Shuffled"], scatter["CO2_Observed"], s=20, color="#7F8C8D",
            alpha=0.45, edgecolors="none", rasterized=True, zorder=2)
ax1.plot(curves["Day"], curves["Pred_Shuffled"], color="#E74C3C", linewidth=2.5, zorder=3)
ax1.scatter(orig_ts["Days"], orig_ts["CO2"], s=14, color="#3498DB", alpha=0.25,
            edgecolors="none", rasterized=True, zorder=1)

ax1.set_title("(b) Shuffled data | LS-MPR learned function", pad=10)
ax1.set_xlabel("Days (shuffled)"); ax1.set_ylabel("CO$_2$ (g)")

# R² 标注移到右下角，图例放左上角
ax1.text(0.96, 0.06, f"$R^2$ = {r2_shuf:.3f}\n(near-constant, no kinetics learned)",
         transform=ax1.transAxes, fontsize=AFS, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

# Range 标注 — 移到红线下方 (数据坐标)
ymin_s = curves["Pred_Shuffled"].min()
ymax_s = curves["Pred_Shuffled"].max()
ax1.annotate(f"Range: [{ymin_s:.0f}, {ymax_s:.0f}] g\n(~mean CO$_2$)",
            xy=(120, 175), fontsize=10.5, fontweight="bold", color="#C0392B",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C0392B", alpha=0.85))

ax1.set_xlim(-5, 185)
ax1.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# 图例 — 放在 (a) 图内 R² 框上面
leg_els = [
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#34495E", markersize=10, lw=0, label="Observed (original)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#7F8C8D", markersize=10, lw=0, label="Observed (shuffled)"),
    Line2D([0],[0], color="#C0392B", linewidth=2.5, label="LS-MPR fit"),
]
leg = ax0.legend(handles=leg_els, loc="upper left",
                 framealpha=0.85, edgecolor="#BDC3C7", fancybox=True,
                 fontsize=10.5, borderpad=0.5, labelspacing=0.3)
for t in leg.get_texts(): t.set_fontweight("bold")

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_shuffle_learned_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_shuffle_learned_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_shuffle_learned_v2.png'}")
print("Done.")
plt.close(fig)