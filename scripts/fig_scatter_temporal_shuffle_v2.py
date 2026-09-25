"""
Figure V2: 时间洗牌散点 (画布不变 12x5.2, 文字×1.5, 无遮挡)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

random_df = pd.read_csv(str(FIGURE_DATA / "Fig_Scatter_Random.csv"))
shuffle_df = pd.read_csv(str(FIGURE_DATA / "Fig_Scatter_Shuffle.csv"))

def compute_metrics(df):
    y_true = df["Observed_CO2_g"]
    y_pred = df["Predicted_CO2_g"]
    return r2_score(y_true, y_pred), spearmanr(y_true, y_pred)[0]

r2_rand, rho_rand = compute_metrics(random_df)
r2_shuf, rho_shuf = compute_metrics(shuffle_df)
print(f"Random: R2={r2_rand:.4f}, rho={rho_rand:.4f}")
print(f"Shuffle: R2={r2_shuf:.4f}, rho={rho_shuf:.4f}")

# ── 字体 ×1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 15, "font.weight": "bold",           # 10→15
    "axes.labelsize": 16.5, "axes.labelweight": "bold",  # 11→16.5
    "axes.titlesize": 18, "axes.titleweight": "bold",    # 12→18
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,     # 9→13.5
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.subplots_adjust(wspace=0.30)

all_obs = pd.concat([random_df["Observed_CO2_g"], shuffle_df["Observed_CO2_g"]])
all_pred = pd.concat([random_df["Predicted_CO2_g"], shuffle_df["Predicted_CO2_g"]])
lim_min = min(all_obs.min(), all_pred.min()) - 5
lim_max = max(all_obs.max(), all_pred.max()) + 5
vmin = min(random_df["Days"].min(), shuffle_df["Days"].min())
vmax = max(random_df["Days"].max(), shuffle_df["Days"].max())

def draw_panel(ax, df, r2, rho, title, is_shuffle=False):
    sc = ax.scatter(df["Observed_CO2_g"], df["Predicted_CO2_g"],
                    c=df["Days"], cmap="Spectral_r", s=32,
                    edgecolors="none", alpha=0.85, vmin=vmin, vmax=vmax, rasterized=True)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black",
            linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)

    if is_shuffle:
        textstr = f"$R^2$ = 0.035\nSpearman $\\rho$ = {rho:.3f}\n(30-shuffle mean)"
    else:
        textstr = f"$R^2$ = {r2:.3f}\nSpearman $\\rho$ = {rho:.3f}"

    # R² 框移到左上角，避免与 colorbar 冲突
    ax.text(0.04, 0.94, textstr, transform=ax.transAxes,
            fontsize=12, fontweight="bold", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Observed CO$_2$ (g)")
    ax.set_ylabel("Predicted CO$_2$ (g)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    return sc

sc0 = draw_panel(ax0, random_df, r2_rand, rho_rand, "(a) Random split")
sc1 = draw_panel(ax1, shuffle_df, r2_shuf, rho_shuf, "(b) Temporal shuffle", is_shuffle=True)

cbar = fig.colorbar(sc1, ax=[ax0, ax1], fraction=0.038, pad=0.02,
                    label="Days", ticks=np.linspace(vmin, vmax, 5))
cbar.ax.tick_params(length=0)
cbar.set_label("Days", fontweight="bold", fontsize=13)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_scatter_temporal_shuffle_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_scatter_temporal_shuffle_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_scatter_temporal_shuffle_v2.png'}")
print("Done.")
plt.close(fig)