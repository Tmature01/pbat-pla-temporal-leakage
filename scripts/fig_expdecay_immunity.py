"""
Figure: Exp Decay 免疫性 — 4 面板 (修正版: 整行打乱)
(a) 原始拟合 (b) 整行打乱后重拟合 (c) 原始残差 (d) 打乱残差
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 & 拟合 ──────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_ExpDecay_Scatter.csv"))
x_orig = df["Day_Orig"].values
y_orig = df["CO2"].values

def exp_decay(t, a, b):
    return a * (1 - np.exp(-b * t))

# 原始拟合
popt_o, _ = curve_fit(exp_decay, x_orig, y_orig, p0=[225, 0.017], maxfev=10000)
y_pred_o = exp_decay(x_orig, *popt_o)
resid_o = y_orig - y_pred_o
r2_o = 1 - np.sum(resid_o**2) / np.sum((y_orig - y_orig.mean())**2)

# 整行打乱 (保持 Day-CO2 配对)
np.random.seed(42)
idx = np.random.permutation(len(x_orig))
x_shuf = x_orig[idx]
y_shuf = y_orig[idx]
popt_s, _ = curve_fit(exp_decay, x_shuf, y_shuf, p0=[225, 0.017], maxfev=10000)
y_pred_s = exp_decay(x_shuf, *popt_s)
resid_s = y_shuf - y_pred_s
r2_s = 1 - np.sum(resid_s**2) / np.sum((y_shuf - y_shuf.mean())**2)

# 平滑曲线 (用于绘制)
x_smooth = np.linspace(0, 180, 200)
y_smooth_o = exp_decay(x_smooth, *popt_o)
y_smooth_s = exp_decay(x_smooth, *popt_s)

print(f"Original:  a={popt_o[0]:.1f}, b={popt_o[1]:.4f}, R^2={r2_o:.4f}")
print(f"Shuffled:  a={popt_s[0]:.1f}, b={popt_s[1]:.4f}, R^2={r2_s:.4f}")

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 10, "axes.labelweight": "bold",
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.28, wspace=0.22, top=0.93, bottom=0.06, left=0.09, right=0.96)
(axA, axB), (axC, axD) = axes

C_SCAT = "#34495E"; C_CURVE = "#C0392B"; C_RESID = "#3498DB"

# ══════════════════════════════════════════════════════
# (a) 原始数据 + 拟合曲线
# ══════════════════════════════════════════════════════
axA.scatter(x_orig, y_orig, s=14, color=C_SCAT, alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axA.plot(x_smooth, y_smooth_o, color=C_CURVE, linewidth=2.2, zorder=3)
axA.set_title("(a) Exp Decay: original data", pad=6)
axA.set_xlabel("Days"); axA.set_ylabel("CO$_2$ (g)")
axA.text(0.95, 0.08, f"$R^2$ = {r2_o:.3f}\na = {popt_o[0]:.0f}, b = {popt_o[1]:.4f}",
         transform=axA.transAxes, fontsize=8.5, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (b) 整行打乱 + 重拟合曲线 (应与(a)完全相同)
# ══════════════════════════════════════════════════════
axB.scatter(x_shuf, y_shuf, s=14, color="#E67E22", alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axB.plot(x_smooth, y_smooth_s, color="#8E44AD", linewidth=2.2, zorder=3)
# 叠加原始曲线(虚线)证明重叠
axB.plot(x_smooth, y_smooth_o, color=C_CURVE, linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)
axB.set_title("(b) Exp Decay: row-shuffled data", pad=6)
axB.set_xlabel("Days (shuffled)"); axB.set_ylabel("CO$_2$ (g)")
axB.text(0.95, 0.08, f"$R^2$ = {r2_s:.3f}\na = {popt_s[0]:.0f}, b = {popt_s[1]:.4f}\n(unchanged)",
         transform=axB.transAxes, fontsize=8.5, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (c) 原始残差
# ══════════════════════════════════════════════════════
axC.axhline(y=0, color="black", linewidth=0.6, alpha=0.35, zorder=1)
axC.scatter(x_orig, resid_o, s=12, color=C_RESID, alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axC.set_title("(c) Residuals: original", pad=6)
axC.set_xlabel("Days"); axC.set_ylabel("Residual (g)")
rmse_o = np.sqrt(np.mean(resid_o**2))
axC.text(0.95, 0.08, f"RMSE = {rmse_o:.1f} g", transform=axC.transAxes,
         fontsize=8.5, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (d) 打乱残差
# ══════════════════════════════════════════════════════
axD.axhline(y=0, color="black", linewidth=0.6, alpha=0.35, zorder=1)
axD.scatter(x_shuf, resid_s, s=12, color="#E67E22", alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axD.set_title("(d) Residuals: row-shuffled", pad=6)
axD.set_xlabel("Days (shuffled)"); axD.set_ylabel("Residual (g)")
rmse_s = np.sqrt(np.mean(resid_s**2))
axD.text(0.95, 0.08, f"RMSE = {rmse_s:.1f} g", transform=axD.transAxes,
         fontsize=8.5, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axD.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_expdecay_immunity.png", dpi=600)
fig.savefig(OUT_DIR / "fig_expdecay_immunity.svg")
print(f"Saved: {OUT_DIR / 'fig_expdecay_immunity.png'}")
print("Done.")
plt.close(fig)