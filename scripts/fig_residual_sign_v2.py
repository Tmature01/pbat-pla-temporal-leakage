"""
Figure V2: 残差符号一致性 (画布不变 12x7, 文字x1.5, 无底注)
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

df = pd.read_csv(str(FIGURE_DATA / "Fig_Residual_Sign_Consistency.csv"))
summary = pd.read_csv(str(FIGURE_DATA / "Fig_Residual_Summary.csv"))

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 13.5, "font.weight": "bold",
    "axes.labelsize": 16.5, "axes.labelweight": "bold",
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7))
fig.subplots_adjust(hspace=0.14, bottom=0.06, top=0.94, left=0.11, right=0.96)

def plot_residuals(ax, resid_col, title, model_name):
    daily_mean = df.groupby("Day")[resid_col].mean()
    days = daily_mean.index.values
    res = daily_mean.values

    pos_mask = res >= 0
    if pos_mask.any():
        ax.fill_between(days, 0, res, where=pos_mask,
                        facecolor="#E74C3C", alpha=0.50, linewidth=0, zorder=2)
    neg_mask = res < 0
    if neg_mask.any():
        ax.fill_between(days, 0, res, where=neg_mask,
                        facecolor="#3498DB", alpha=0.50, linewidth=0, zorder=2)

    ax.axhline(y=0, color="black", linewidth=1.2, zorder=3)

    row = summary[summary["Model"]==model_name].iloc[0]
    stats_text = (
        f"Pos: {row['Pct_Positive']:.0f}%  |  Neg: {row['Pct_Negative']:.0f}%  |  "
        f"Longest run: {int(row['Longest_Run'])}  |  MAE: {row['MAE']:.1f}  |  "
        f"Max |resid|: {row['MaxAbsResidual']:.1f}"
    )
    ax.text(0.5, 0.93, stats_text, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#2C3E50",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

    ax.set_title(title, pad=10)
    ax.set_ylabel("Residual (g)")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

plot_residuals(ax0, "LSMPR_Residual",
               "(a) LS-MPR residuals | Variance collapse + systematic bias", "LS-MPR")
ax0.set_xticklabels([])

plot_residuals(ax1, "ExpDecay_Residual",
               "(b) Exp Decay residuals | Pure structural deviation", "Exp Decay")
ax1.set_xlabel("Days")

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_residual_sign_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_residual_sign_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_residual_sign_v2.png'}")
print("Done.")
plt.close(fig)