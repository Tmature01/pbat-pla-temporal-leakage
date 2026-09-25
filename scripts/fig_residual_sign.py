"""
Figure: 残差符号一致性 — LS-MPR vs Exp Decay 两种失败模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_Residual_Sign_Consistency.csv"))
summary = pd.read_csv(str(FIGURE_DATA / "Fig_Residual_Summary.csv"))

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7))
fig.subplots_adjust(hspace=0.10, bottom=0.12, top=0.94, left=0.09, right=0.96)

def plot_residuals(ax, resid_col, title, model_name):
    # 计算日均残差 (3 条件平均), 避免填充叠加
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

    # 零线
    ax.axhline(y=0, color="black", linewidth=1.0, zorder=3)

    # 统计
    row = summary[summary["Model"] == model_name].iloc[0]
    pct_pos = row["Pct_Positive"]
    pct_neg = row["Pct_Negative"]
    longest = int(row["Longest_Run"])

    stats_text = (
        f"Pos: {pct_pos:.0f}%  |  Neg: {pct_neg:.0f}%  |  "
        f"Longest run: {longest}  |  MAE: {row['MAE']:.1f}  |  "
        f"Max |resid|: {row['MaxAbsResidual']:.1f}"
    )
    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color="#2C3E50",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

    ax.set_title(title, pad=6)
    ax.set_ylabel("Residual (g)")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── (a) LS-MPR ──────────────────────────────────────
plot_residuals(ax0, "LSMPR_Residual",
               "(a) LS-MPR residuals  |  Variance collapse + systematic bias",
               "LS-MPR")
ax0.set_xticklabels([])

# ── (b) Exp Decay ──────────────────────────────────────
plot_residuals(ax1, "ExpDecay_Residual",
               "(b) Exp Decay residuals  |  Pure structural deviation",
               "Exp Decay")
ax1.set_xlabel("Days")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "Two distinct failure modes: Exp Decay — 100% negative residuals, systematic overestimation "
         "(correct functional form, wrong extrapolated parameters). "
         "LS-MPR — 87% positive, variance explodes beyond Day 140 (no physical constraint, wild extrapolation).",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_residual_sign.png", dpi=600)
fig.savefig(OUT_DIR / "fig_residual_sign.svg")
print(f"Saved: {OUT_DIR / 'fig_residual_sign.png'}")
print(f"Saved: {OUT_DIR / 'fig_residual_sign.svg'}")
print("Done.")
plt.close(fig)