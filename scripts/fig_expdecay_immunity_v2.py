"""
Figure V2: Exp Decay 免疫性 (画布不变 12x9, 文字×1.5, LaTeX R²)
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_ExpDecay_Scatter.csv"))
x_orig = df["Day_Orig"].values
y_orig = df["CO2"].values

def exp_decay(t, a, b):
    return a * (1 - np.exp(-b * t))

popt_o, _ = curve_fit(exp_decay, x_orig, y_orig, p0=[225, 0.017], maxfev=10000)
resid_o = y_orig - exp_decay(x_orig, *popt_o)
r2_o = 1 - np.sum(resid_o**2) / np.sum((y_orig - y_orig.mean())**2)

np.random.seed(42)
idx = np.random.permutation(len(x_orig))
x_shuf, y_shuf = x_orig[idx], y_orig[idx]
popt_s, _ = curve_fit(exp_decay, x_shuf, y_shuf, p0=[225, 0.017], maxfev=10000)
resid_s = y_shuf - exp_decay(x_shuf, *popt_s)
r2_s = 1 - np.sum(resid_s**2) / np.sum((y_shuf - y_shuf.mean())**2)

x_smooth = np.linspace(0, 180, 200)
y_smooth_o = exp_decay(x_smooth, *popt_o)
y_smooth_s = exp_decay(x_smooth, *popt_s)

print(f"Original: a={popt_o[0]:.1f}, b={popt_o[1]:.4f}, R2={r2_o:.4f}")
print(f"Shuffled: a={popt_s[0]:.1f}, b={popt_s[1]:.4f}, R2={r2_s:.4f}")

# ── 字体 ×1.5, 画布不变 12×9 ─────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 13.5, "font.weight": "bold",           # 9→13.5
    "axes.labelsize": 15, "axes.labelweight": "bold",    # 10→15
    "axes.titlesize": 16.5, "axes.titleweight": "bold",  # 11→16.5
    "xtick.labelsize": 12, "ytick.labelsize": 12,        # 8→12
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.30, wspace=0.24, top=0.92, bottom=0.07, left=0.10, right=0.96)
(axA, axB), (axC, axD) = axes

C_SCAT = "#34495E"; C_CURVE = "#C0392B"; C_RESID = "#3498DB"
AFS = 12.5   # annotate font size (8.5→12.5)

# ══════════════════════════════════════════════════════
# (a) 原始数据 + 拟合曲线
# ══════════════════════════════════════════════════════
axA.scatter(x_orig, y_orig, s=18, color=C_SCAT, alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axA.plot(x_smooth, y_smooth_o, color=C_CURVE, linewidth=2.5, zorder=3)
axA.set_title("(a) Exp Decay: original data", pad=8)
axA.set_xlabel("Days"); axA.set_ylabel("CO$_2$ (g)")
axA.text(0.95, 0.06, f"$R^2$ = {r2_o:.3f}\na = {popt_o[0]:.0f}, b = {popt_o[1]:.4f}",
         transform=axA.transAxes, fontsize=AFS, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (b) 整行打乱 + 重拟合
# ══════════════════════════════════════════════════════
axB.scatter(x_shuf, y_shuf, s=18, color="#E67E22", alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axB.plot(x_smooth, y_smooth_s, color="#8E44AD", linewidth=2.5, zorder=3)
axB.plot(x_smooth, y_smooth_o, color=C_CURVE, linewidth=1.2, linestyle="--", alpha=0.5, zorder=2)
axB.set_title("(b) Exp Decay: row-shuffled data", pad=8)
axB.set_xlabel("Days (shuffled)"); axB.set_ylabel("CO$_2$ (g)")
axB.text(0.95, 0.06, f"$R^2$ = {r2_s:.3f}\na = {popt_s[0]:.0f}, b = {popt_s[1]:.4f}\n(unchanged)",
         transform=axB.transAxes, fontsize=AFS, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (c) 原始残差
# ══════════════════════════════════════════════════════
axC.axhline(y=0, color="black", linewidth=0.8, alpha=0.35, zorder=1)
axC.scatter(x_orig, resid_o, s=16, color=C_RESID, alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axC.set_title("(c) Residuals: original", pad=8)
axC.set_xlabel("Days"); axC.set_ylabel("Residual (g)")
rmse_o = np.sqrt(np.mean(resid_o**2))
axC.text(0.95, 0.06, f"RMSE = {rmse_o:.1f} g", transform=axC.transAxes,
         fontsize=AFS, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (d) 打乱残差
# ══════════════════════════════════════════════════════
axD.axhline(y=0, color="black", linewidth=0.8, alpha=0.35, zorder=1)
axD.scatter(x_shuf, resid_s, s=16, color="#E67E22", alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axD.set_title("(d) Residuals: row-shuffled", pad=8)
axD.set_xlabel("Days (shuffled)"); axD.set_ylabel("Residual (g)")
rmse_s = np.sqrt(np.mean(resid_s**2))
axD.text(0.95, 0.06, f"RMSE = {rmse_s:.1f} g", transform=axD.transAxes,
         fontsize=AFS, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))
axD.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_expdecay_immunity_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_expdecay_immunity_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_expdecay_immunity_v2.png'}")
print("Done.")
plt.close(fig)