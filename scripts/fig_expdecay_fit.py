"""
Figure: Exp Decay 模型洗牌前后拟合对比 — 4 面板 (2x2)
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

scatter = pd.read_csv(str(FIGURE_DATA / "Fig_ExpDecay_Scatter.csv"))
curves  = pd.read_csv(str(FIGURE_DATA / "Fig_ExpDecay_Curves.csv"))

d_orig = scatter["Day_Orig"].values
d_shuf = scatter["Day_Shuf"].values
co2    = scatter["CO2"].values

# 插值曲线到散点 Day 位置以计算残差
f_orig = interp1d(curves["X"], curves["Y_Orig"], kind="linear")
f_shuf = interp1d(curves["X"], curves["Y_Shuf"], kind="linear")

resid_orig = co2 - f_orig(d_orig)
resid_shuf = co2 - f_shuf(d_shuf)

r2_orig = 1 - np.sum(resid_orig**2) / np.sum((co2 - co2.mean())**2)
r2_shuf = 1 - np.sum(resid_shuf**2) / np.sum((co2 - co2.mean())**2)

print(f"R^2 Original: {r2_orig:.4f}")
print(f"R^2 Shuffled: {r2_shuf:.4f}")

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

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.30, wspace=0.22, left=0.09, right=0.96,
                    top=0.94, bottom=0.08)
(ax0, ax1), (ax2, ax3) = axes

COLOR_SCATTER = "#34495E"
COLOR_CURVE   = "#C0392B"
COLOR_RESID   = "#3498DB"

# ── (a) 原始数据 + 拟合曲线 ──────────────────────────
ax0.scatter(d_orig, co2, s=14, color=COLOR_SCATTER, alpha=0.55,
            edgecolors="none", rasterized=True, zorder=2)
ax0.plot(curves["X"], curves["Y_Orig"], color=COLOR_CURVE, linewidth=2.0, zorder=3)
ax0.set_title("(a) Original data  |  $y = a(1-e^{-bx})$", pad=6)
ax0.set_xlabel("Day")
ax0.set_ylabel("CO$_2$ (g)")
ax0.text(0.95, 0.08, f"$R^2$ = {r2_orig:.3f}\na = 225, b = 0.017",
         transform=ax0.transAxes, fontsize=8.5, fontweight="bold",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

# ── (b) 洗牌数据 + 重拟合曲线 ────────────────────────
ax1.scatter(d_shuf, co2, s=14, color="#E67E22", alpha=0.55,
            edgecolors="none", rasterized=True, zorder=2)
ax1.plot(curves["X"], curves["Y_Shuf"], color="#8E44AD", linewidth=2.0, zorder=3)
ax1.set_title("(b) Shuffled data  |  $y = a(1-e^{-bx})$ re-fit", pad=6)
ax1.set_xlabel("Day (shuffled)")
ax1.set_ylabel("CO$_2$ (g)")
ax1.text(0.95, 0.08, f"$R^2$ = {r2_shuf:.3f}\na = 225, b = 0.017",
         transform=ax1.transAxes, fontsize=8.5, fontweight="bold",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

# ── (c) 原始残差 ────────────────────────────────────
ax2.axhline(y=0, color="black", linewidth=0.6, alpha=0.4, zorder=1)
ax2.scatter(d_orig, resid_orig, s=12, color=COLOR_RESID, alpha=0.55,
            edgecolors="none", rasterized=True, zorder=2)
ax2.set_title("(c) Residuals  |  Original", pad=6)
ax2.set_xlabel("Day")
ax2.set_ylabel("Residual (g)")
rmse_orig = np.sqrt(np.mean(resid_orig**2))
ax2.text(0.95, 0.08, f"RMSE = {rmse_orig:.1f} g",
         transform=ax2.transAxes, fontsize=8.5, fontweight="bold",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

# ── (d) 洗牌残差 ────────────────────────────────────
ax3.axhline(y=0, color="black", linewidth=0.6, alpha=0.4, zorder=1)
ax3.scatter(d_shuf, resid_shuf, s=12, color="#E67E22", alpha=0.55,
            edgecolors="none", rasterized=True, zorder=2)
ax3.set_title("(d) Residuals  |  Shuffled", pad=6)
ax3.set_xlabel("Day (shuffled)")
ax3.set_ylabel("Residual (g)")
rmse_shuf = np.sqrt(np.mean(resid_shuf**2))
ax3.text(0.95, 0.08, f"RMSE = {rmse_shuf:.1f} g",
         transform=ax3.transAxes, fontsize=8.5, fontweight="bold",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

# ── 坐标轴美化 ───────────────────────────────────────
for ax in axes.flat:
    ax.tick_params(which="both", direction="in", bottom=True, top=False,
                   left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "Exp Decay is immune to temporal shuffle: R^2 stays ~0.985 because the functional form "
         "y = a(1-e^{-bx}) fits the bounded-monotonic CO2 cloud regardless of time ordering. "
         "Data-driven models lose the temporal-proximity shortcut and collapse.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_expdecay_fit.png", dpi=600)
fig.savefig(OUT_DIR / "fig_expdecay_fit.svg")
print(f"Saved: {OUT_DIR / 'fig_expdecay_fit.png'}")
print(f"Saved: {OUT_DIR / 'fig_expdecay_fit.svg'}")
print("Done.")
plt.close(fig)