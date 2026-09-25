"""
Figure: LS-MPR 残差自相关函数 (ACF) — Random Split vs Temporal Holdout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取数据 ──────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_ACF_Residuals.csv"))
lag = df["Lag"].values[1:]  # 跳过 lag=0
acf_r = df["RandomSplit_ACF"].values[1:]
acf_t = df["TemporalHoldout_ACF"].values[1:]
ci = 0.145  # 95% CI

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

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.22, bottom=0.15, top=0.90, left=0.08, right=0.96)

def plot_acf(ax, lags, acf, title, ci_val):
    # CI 区
    ax.axhspan(-ci_val, ci_val, facecolor="#EAECEE", alpha=0.55, linewidth=0, zorder=0)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, zorder=1)
    ax.axhline(y=+ci_val, color="gray", linewidth=0.4, linestyle="--", alpha=0.5, zorder=1)
    ax.axhline(y=-ci_val, color="gray", linewidth=0.4, linestyle="--", alpha=0.5, zorder=1)

    # 区分显著/不显著
    sig_mask = np.abs(acf) > ci_val
    # 不显著: 灰色
    if (~sig_mask).any():
        ax.vlines(lags[~sig_mask], 0, acf[~sig_mask], color="#BDC3C7", linewidth=1.5, zorder=2)
    # 显著: 彩色
    if sig_mask.any():
        ax.vlines(lags[sig_mask], 0, acf[sig_mask], color=bar_color, linewidth=2.0, zorder=3)

    # 端点标记
    ax.scatter(lags, acf, s=6, color=bar_color, zorder=4)

    ax.set_title(title, pad=8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xlim(-1, 31)
    ax.set_ylim(-0.38, 1.05)
    ax.tick_params(which="both", direction="in", bottom=True, top=False,
                   left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 关键标注
    # 找最大 |acf| (除 Lag 0)
    peak_idx = np.argmax(np.abs(acf))
    peak_lag = lags[peak_idx]
    peak_val = acf[peak_idx]
    offset = -0.12 if peak_val > 0.85 else (0.12 if peak_val > 0 else -0.12)
    ax.annotate(f"Lag {peak_lag}: {peak_val:+.2f}",
                xy=(peak_lag, peak_val),
                xytext=(peak_lag + 3, peak_val + offset),
                fontsize=8, fontweight="bold", color=bar_color,
                arrowprops=dict(arrowstyle="->", color=bar_color, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=bar_color, alpha=0.85))

# Left: Random Split
bar_color = "#3498DB"
plot_acf(ax0, lag, acf_r, "(a) Random split residuals", ci)

# Right: Temporal Holdout
bar_color = "#C0392B"
plot_acf(ax1, lag, acf_t, "(b) Temporal holdout residuals", ci)

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.025,
         "Random split: residuals ~ white noise (ACF decays to zero by lag 5). "
         "Temporal holdout: strong positive autocorrelation persisting to lag 15+ "
         "— model systematically errs in the same direction for consecutive days.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_acf_residuals.png", dpi=600)
fig.savefig(OUT_DIR / "fig_acf_residuals.svg")
print(f"Saved: {OUT_DIR / 'fig_acf_residuals.png'}")
print(f"Saved: {OUT_DIR / 'fig_acf_residuals.svg'}")
print("Done.")
plt.close(fig)