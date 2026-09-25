"""
Figure V2: LS-MPR 残差自相关函数 ACF (画布不变 12x5, 文字x1.5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_ACF_Residuals.csv"))
lag = df["Lag"].values[1:]
acf_r = df["RandomSplit_ACF"].values[1:]
acf_t = df["TemporalHoldout_ACF"].values[1:]
ci = 0.145

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

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.24, bottom=0.08, top=0.88, left=0.09, right=0.96)

AFS = 12  # annotate font (8→12)

def plot_acf(ax, lags, acf, title, bar_color):
    ax.axhspan(-ci, ci, facecolor="#EAECEE", alpha=0.50, linewidth=0, zorder=0)
    ax.axhline(y=0, color="black", linewidth=0.7, alpha=0.45, zorder=1)
    ax.axhline(y=+ci, color="gray", linewidth=0.5, linestyle="--", alpha=0.45)
    ax.axhline(y=-ci, color="gray", linewidth=0.5, linestyle="--", alpha=0.45)

    sig = np.abs(acf) > ci
    if (~sig).any():
        ax.vlines(lags[~sig], 0, acf[~sig], color="#BDC3C7", linewidth=1.8, zorder=2)
    if sig.any():
        ax.vlines(lags[sig], 0, acf[sig], color=bar_color, linewidth=2.5, zorder=3)
    ax.scatter(lags, acf, s=10, color=bar_color, zorder=4)

    ax.set_title(title, pad=10)
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
    ax.set_xlim(-1, 31); ax.set_ylim(-0.38, 1.05)
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 峰值标注
    peak_idx = np.argmax(np.abs(acf))
    peak_lag, peak_val = lags[peak_idx], acf[peak_idx]
    if peak_val > 0.85:
        offset = -0.20; x_offset = 5   # 高位: 下移+右移
    elif peak_val > 0:
        offset = 0.15; x_offset = 3
    else:
        offset = -0.15; x_offset = 3
    ax.annotate(f"Lag {peak_lag}: {peak_val:+.2f}",
                xy=(peak_lag, peak_val),
                xytext=(peak_lag + x_offset, peak_val + offset),
                fontsize=AFS, fontweight="bold", color=bar_color,
                arrowprops=dict(arrowstyle="->", color=bar_color, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=bar_color, alpha=0.85))

# Left
plot_acf(ax0, lag, acf_r, "(a) Random split residuals", "#3498DB")
# Right
plot_acf(ax1, lag, acf_t, "(b) Temporal holdout residuals", "#C0392B")

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_acf_residuals_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_acf_residuals_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_acf_residuals_v2.png'}")
print("Done.")
plt.close(fig)