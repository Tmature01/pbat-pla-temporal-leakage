"""
Figure 1 (复合): 数据泄漏概念与核心发现
(a) 数据泄漏概念示意 (b) 洗牌散点 (c) ACF 残差
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取所有数据 ──────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df_proto  = pd.read_csv(str(FIGURE_DATA / "Fig_DataLeakage_Protocols.csv"))
kde       = pd.read_csv(str(FIGURE_DATA / "Fig_DataLeakage_KDE.csv"))
scat_r    = pd.read_csv(str(FIGURE_DATA / "Fig_Scatter_Random.csv"))
scat_s    = pd.read_csv(str(FIGURE_DATA / "Fig_Scatter_Shuffle.csv"))
acf_df    = pd.read_csv(str(FIGURE_DATA / "Fig_ACF_Residuals.csv"))

# ── 2. 全局样式 ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "font.weight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold",
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

fig = plt.figure(figsize=(16, 13))

# ── 3. 布局: (a) 占顶部, 1行3列; (b) 中间, 1行2列; (c) 底部, 1行2列 ──
gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1.0, 1.0],
                      hspace=0.38, wspace=0.32,
                      left=0.04, right=0.97, top=0.96, bottom=0.05)

# (a) 占 [0, :] → 3 小面板, 每列 2 格
axA1 = fig.add_subplot(gs[0, 0:2])
axA2 = fig.add_subplot(gs[0, 2:4])
axA3 = fig.add_subplot(gs[0, 4:6])

# (b) 占 [1, :] → 2 面板, 每列 3 格
axB1 = fig.add_subplot(gs[1, 0:3])
axB2 = fig.add_subplot(gs[1, 3:6])

# (c) 占 [2, :] → 2 面板, 每列 3 格
axC1 = fig.add_subplot(gs[2, 0:3])
axC2 = fig.add_subplot(gs[2, 3:6])

C_TRAIN = "#2471A3"
C_TEST  = "#C0392B"
C_BG    = "#BDC3C7"
C_GAP   = "#E67E22"

# ══════════════════════════════════════════════════════
# PANEL (a): 数据泄漏概念
# ══════════════════════════════════════════════════════
def draw_leakage_panel(ax, df, proto_col, title, boundary_day=None):
    ax.plot(df["Day"], df["CO2"], color=C_BG, linewidth=0.5, alpha=0.4, zorder=0)
    train = df[df[proto_col] == "Train"]
    test  = df[df[proto_col] == "Test"]
    ax.scatter(train["Day"], train["CO2"], s=8, color=C_TRAIN, alpha=0.55, zorder=2)
    ax.scatter(test["Day"], test["CO2"], s=18, facecolors="none",
               edgecolors=C_TEST, linewidth=1.0, zorder=3)
    if boundary_day:
        ax.axvline(x=boundary_day, color="#2C3E50", linewidth=1.0,
                   linestyle=(0, (5, 3)), alpha=0.6, zorder=4)
    ax.set_title(title, pad=5)
    ax.set_xlabel("Days"); ax.set_ylabel("CO$_2$ (g)")
    ax.set_xlim(-5, 185)
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

draw_leakage_panel(axA1, df_proto, "Random_Partition",
                   "Random split: train & test interleaved")
draw_leakage_panel(axA2, df_proto, "Temporal_Partition",
                   "Temporal holdout: clean separation", 120)

# KDE
axA3.plot(kde["Day"], kde["Random_Train_Density"], color=C_TRAIN, linewidth=1.4, label="Random Train")
axA3.plot(kde["Day"], kde["Random_Test_Density"], color=C_TEST, linewidth=1.4, linestyle="--", label="Random Test")
axA3.plot(kde["Day"], kde["Temporal_Train_Density"], color=C_TRAIN, linewidth=0.9, alpha=0.35, label="Temporal Train")
axA3.plot(kde["Day"], kde["Temporal_Test_Density"], color=C_TEST, linewidth=0.9, linestyle="--", alpha=0.35, label="Temporal Test")
axA3.set_title("Train/test temporal distribution overlap", pad=5)
axA3.set_xlabel("Days"); axA3.set_ylabel("Density")
axA3.text(80, 0.85, "Random:\nheavy overlap\n-> leakage", fontsize=6.5, fontweight="bold", color="#C0392B")
axA3.text(140, 0.20, "Temporal:\nfully separated", fontsize=6.5, fontweight="bold", color="#2471A3")
axA3.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ── (a) 内图例 ─────────────────────────────────────
leg_a = axA1.legend(handles=[
    Line2D([0],[0], marker="o", color=C_TRAIN, markersize=6, markeredgewidth=0, lw=0, label="Train"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor="none", markeredgecolor=C_TEST, markersize=8, markeredgewidth=1.2, lw=0, label="Test"),
], loc="lower right", framealpha=0.85, fontsize=7, borderpad=0.3)
for t in leg_a.get_texts(): t.set_fontweight("bold")

# ══════════════════════════════════════════════════════
# PANEL (b): 洗牌散点
# ══════════════════════════════════════════════════════
def compute_metrics(df):
    return r2_score(df["Observed_CO2_g"], df["Predicted_CO2_g"]), \
           spearmanr(df["Observed_CO2_g"], df["Predicted_CO2_g"])[0]

r2_r, rho_r = compute_metrics(scat_r)
r2_s, rho_s = compute_metrics(scat_s)

all_obs = pd.concat([scat_r["Observed_CO2_g"], scat_s["Observed_CO2_g"]])
all_pred = pd.concat([scat_r["Predicted_CO2_g"], scat_s["Predicted_CO2_g"]])
lim_min = min(all_obs.min(), all_pred.min()) - 5
lim_max = max(all_obs.max(), all_pred.max()) + 5
vmin = min(scat_r["Days"].min(), scat_s["Days"].min())
vmax = max(scat_r["Days"].max(), scat_s["Days"].max())

def draw_scatter(ax, df, r2, rho, title):
    sc = ax.scatter(df["Observed_CO2_g"], df["Predicted_CO2_g"],
                    c=df["Days"], cmap="Spectral_r", s=14, alpha=0.7,
                    edgecolors="none", vmin=vmin, vmax=vmax, rasterized=True, zorder=2)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=0.6,
            linestyle="--", alpha=0.4, zorder=1)
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nSpearman $\\rho$ = {rho:.3f}",
            transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))
    ax.set_title(title, pad=5)
    ax.set_xlabel("Observed CO$_2$ (g)"); ax.set_ylabel("Predicted CO$_2$ (g)")
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    return sc

sc0 = draw_scatter(axB1, scat_r, r2_r, rho_r, "LS-MPR: Random split")
sc1 = draw_scatter(axB2, scat_s, r2_s, rho_s, "LS-MPR: Temporal shuffle")

cbar = fig.colorbar(sc1, ax=[axB1, axB2], fraction=0.04, pad=0.015,
                    ticks=np.linspace(vmin, vmax, 5))
cbar.set_label("Days", fontweight="bold", fontsize=8)
cbar.ax.tick_params(length=0)

# ══════════════════════════════════════════════════════
# PANEL (c): ACF 残差
# ══════════════════════════════════════════════════════
lag = acf_df["Lag"].values[1:]
acf_rv = acf_df["RandomSplit_ACF"].values[1:]
acf_tv = acf_df["TemporalHoldout_ACF"].values[1:]
ci = 0.145

def draw_acf(ax, lags, acf, title, bar_color):
    ax.axhspan(-ci, ci, facecolor="#EAECEE", alpha=0.45, linewidth=0, zorder=0)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.35, zorder=1)
    ax.axhline(y=+ci, color="gray", linewidth=0.3, linestyle="--", alpha=0.4)
    ax.axhline(y=-ci, color="gray", linewidth=0.3, linestyle="--", alpha=0.4)
    sig = np.abs(acf) > ci
    if (~sig).any():
        ax.vlines(lags[~sig], 0, acf[~sig], color="#BDC3C7", linewidth=1.0, zorder=2)
    if sig.any():
        ax.vlines(lags[sig], 0, acf[sig], color=bar_color, linewidth=1.6, zorder=3)
    ax.scatter(lags, acf, s=5, color=bar_color, zorder=4)
    peak_idx = np.argmax(np.abs(acf))
    offset = -0.10 if acf[peak_idx] > 0.85 else 0.10
    ax.annotate(f"Lag {lags[peak_idx]}: {acf[peak_idx]:+.2f}",
                xy=(lags[peak_idx], acf[peak_idx]),
                xytext=(lags[peak_idx] + 3, acf[peak_idx] + offset),
                fontsize=7, fontweight="bold", color=bar_color,
                arrowprops=dict(arrowstyle="->", color=bar_color, lw=0.6),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=bar_color, alpha=0.85))
    ax.set_title(title, pad=5)
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
    ax.set_xlim(-1, 31); ax.set_ylim(-0.38, 1.05)
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

draw_acf(axC1, lag, acf_rv, "Random split residuals (white noise)", "#3498DB")
draw_acf(axC2, lag, acf_tv, "Temporal holdout residuals (strong autocorrelation)", "#C0392B")

# ── 全局标签 ─────────────────────────────────────────
fig.text(0.02, 0.98, "(a)", fontsize=12, fontweight="bold")
fig.text(0.02, 0.66, "(b)", fontsize=12, fontweight="bold")
fig.text(0.02, 0.34, "(c)", fontsize=12, fontweight="bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.01,
         "Figure 1. Data leakage concept and core evidence. (a) Random split interleaves future into training; "
         "temporal holdout isolates it. (b) LS-MPR R^2 collapses from 0.994 to 0.052 under shuffle. "
         "(c) Random-split residuals are white noise; temporal-holdout residuals show strong positive autocorrelation.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_01_DataLeakage_CoreEvidence.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_01_DataLeakage_CoreEvidence.svg")
print(f"Saved: {OUT_DIR / 'Fig_01_DataLeakage_CoreEvidence.png'}")
print(f"Saved: {OUT_DIR / 'Fig_01_DataLeakage_CoreEvidence.svg'}")
print("Done.")
plt.close(fig)