"""
Figure: 预测误差随 Horizon 演化 (重新设计 — 简洁版)
LS-MPR vs Persistence 绝对误差, LOESS 平滑 + 95% 置信带
"""

import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 读取 & 按 horizon 聚合 ────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_Horizon_Error.csv"))

g = df.groupby("Horizon_days").agg(
    L_mean=("LSMPR_AbsError", "mean"),
    L_std=("LSMPR_AbsError", "std"),
    P_mean=("Persistence_AbsError", "mean"),
    P_std=("Persistence_AbsError", "std"),
).reset_index()

h_daily = g["Horizon_days"].values
lm_mean = g["L_mean"].values
lm_std  = g["L_std"].values
ps_mean = g["P_mean"].values
ps_std  = g["P_std"].values

# ── LOESS 平滑 + Bootstrap 置信带 ─────────────────────────
def loess_ci(x, y, frac=0.30, n_boot=500):
    lo = np.asarray(lowess(y, x, frac=frac, return_sorted=True))
    xs, ys = lo[:, 0], lo[:, 1]
    resid = y - np.interp(x, xs, ys)
    np.random.seed(42)
    yb = np.zeros((n_boot, len(xs)))
    for i in range(n_boot):
        yb[i, :] = ys + np.random.choice(resid, size=len(xs), replace=True)
    lo_b, hi_b = np.percentile(yb, [2.5, 97.5], axis=0)
    return xs, ys, lo_b, hi_b

xL, yL, loL, hiL = loess_ci(h_daily, lm_mean, frac=0.30)
xP, yP, loP, hiP = loess_ci(h_daily, ps_mean, frac=0.30)

# ── 样式 ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "font.weight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, ax = plt.subplots(figsize=(10, 6))

# ── 95% 置信带 (先画，zorder 最低) ───────────────────────
ax.fill_between(xL, loL, hiL, color="#D4E6F1", alpha=0.5, linewidth=0, zorder=1)
ax.fill_between(xP, loP, hiP, color="#FDEBD0", alpha=0.5, linewidth=0, zorder=1)

# ── 日均值点 (61 个点 × 2) ───────────────────────────────
ax.scatter(h_daily, lm_mean, s=30, color="#C0392B", edgecolors="white",
           linewidth=0.3, zorder=3, label="LS-MPR")
ax.scatter(h_daily, ps_mean, s=30, color="#2471A3", edgecolors="white",
           linewidth=0.3, zorder=3, label="Persistence")

# ── LOESS 趋势线 ─────────────────────────────────────────
ax.plot(xL, yL, color="#922B21", linewidth=2.2, zorder=4)
ax.plot(xP, yP, color="#1A5276", linewidth=2.2, zorder=4)

# ── 坐标轴 ───────────────────────────────────────────────
ax.set_xlabel("Horizon (days beyond training cutoff)")
ax.set_ylabel("Absolute prediction error (CO$_2$ g)")
ax.set_xlim(-1, 63)
ax.set_ylim(-2, 72)

# ── 简洁文字标注 ─────────────────────────────────────────
# 放在末端空白区域
ax.text(57, yL[-1] + 8, "LS-MPR\n(x9.6)", color="#922B21",
        fontsize=9, fontweight="bold", ha="center", va="bottom")
ax.text(57, yP[-1] + 5, "Persistence\n(x5.8)", color="#1A5276",
        fontsize=9, fontweight="bold", ha="center", va="bottom")

# ── 窗口竖线 (轻量) ─────────────────────────────────────
for wstart in [120, 130, 140, 150, 160, 170]:
    hval = wstart - 119
    ax.axvline(x=hval, color="gray", linewidth=0.3, linestyle=":",
               alpha=0.35, zorder=0)

# ── 刻度 ─────────────────────────────────────────────────
ax.tick_params(which="both", direction="in", bottom=True, top=False,
               left=True, right=False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 图例 ─────────────────────────────────────────────────
legend = ax.legend(loc="upper left", framealpha=0.9, edgecolor="gray",
                   fancybox=True, handlelength=1.2)
for t in legend.get_texts():
    t.set_fontweight("bold")

# ── 保存 ─────────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_horizon_error.png", dpi=600)
fig.savefig(OUT_DIR / "fig_horizon_error.svg")
print(f"Saved: {OUT_DIR / 'fig_horizon_error.png'}")
print(f"Saved: {OUT_DIR / 'fig_horizon_error.svg'}")
print("Done.")
plt.close(fig)