"""
Figure: 时间洗牌前后 LS-MPR 预测值 vs 真实值散点图
左面板: 随机分割 (Random split)
右面板: 时间洗牌 (Temporal shuffle)
点按 Days 着色, 标注 R² 和 Spearman ρ
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 读取数据 ──────────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(str(FIGURE_DATA))
random_df = pd.read_csv(DATA_DIR / "Fig_Scatter_Random.csv")
shuffle_df = pd.read_csv(DATA_DIR / "Fig_Scatter_Shuffle.csv")

# ── 计算指标 ──────────────────────────────────────────────
def compute_metrics(df):
    y_true = df["Observed_CO2_g"]
    y_pred = df["Predicted_CO2_g"]
    r2 = r2_score(y_true, y_pred)
    rho, p = spearmanr(y_true, y_pred)
    return r2, rho

r2_rand, rho_rand = compute_metrics(random_df)
r2_shuf, rho_shuf = compute_metrics(shuffle_df)

print(f"Random split:        R2={r2_rand:.4f}, Spearman rho={rho_rand:.4f}")
print(f"Temporal shuffle:     R2={r2_shuf:.4f}, Spearman rho={rho_shuf:.4f}")

# ── 绘图设置 ──────────────────────────────────────────────
# 学术期刊风格 (Polymer Testing 适配)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "font.weight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.subplots_adjust(wspace=0.28)

# ── 全局 y=x 范围 ─────────────────────────────────────────
all_obs = pd.concat([random_df["Observed_CO2_g"], shuffle_df["Observed_CO2_g"]])
all_pred = pd.concat([random_df["Predicted_CO2_g"], shuffle_df["Predicted_CO2_g"]])
lim_min = min(all_obs.min(), all_pred.min()) - 5
lim_max = max(all_obs.max(), all_pred.max()) + 5

# 公共 colorbar 用全局 Days 范围
vmin = min(random_df["Days"].min(), shuffle_df["Days"].min())
vmax = max(random_df["Days"].max(), shuffle_df["Days"].max())

# ── 面板绘制函数 ──────────────────────────────────────────
def draw_panel(ax, df, r2, rho, title, vmin, vmax):
    obs = df["Observed_CO2_g"]
    pred = df["Predicted_CO2_g"]
    days = df["Days"]

    sc = ax.scatter(obs, pred, c=days, cmap="Spectral_r",
                    s=28, edgecolors="none", alpha=0.85,
                    vmin=vmin, vmax=vmax, rasterized=True)

    # y = x 参考线
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)

    # 标注 (b 面板使用 30 次 Shuffle 均值 0.035)
    if "Shuffle" in title:
        textstr = f"$R^2$ = 0.035\nSpearman $\\rho$ = {rho:.3f}\n(30-shuffle mean)"
    else:
        textstr = f"$R^2$ = {r2:.3f}\nSpearman $\\rho$ = {rho:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, fontweight="bold", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Observed CO$_2$ (g)")
    ax.set_ylabel("Predicted CO$_2$ (g)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    return sc

sc0 = draw_panel(ax0, random_df, r2_rand, rho_rand,
                 "(a) Random split", vmin, vmax)
sc1 = draw_panel(ax1, shuffle_df, r2_shuf, rho_shuf,
                 "(b) Temporal shuffle", vmin, vmax)

# ── Colorbar ──────────────────────────────────────────────
cbar = fig.colorbar(sc1, ax=[ax0, ax1], fraction=0.038, pad=0.02,
                    label="Days", ticks=np.linspace(vmin, vmax, 5))
cbar.ax.tick_params(length=0)

# ── 保存 ──────────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_png = OUT_DIR / "fig_scatter_temporal_shuffle.png"
out_svg = OUT_DIR / "fig_scatter_temporal_shuffle.svg"
fig.savefig(out_png, dpi=600)
fig.savefig(out_svg)
print(f"\nSaved: {out_png}")
print(f"Saved: {out_svg}")

plt.close(fig)
print("Done.")