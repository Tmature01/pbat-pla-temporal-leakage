"""
Figure V2: 反事实验证多窗口预测轨迹 (高级配色版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 全曲线数据 ─────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

RAW = str(_REPO_ROOT / "data" / "pbat_pla_experimental_543.csv")
raw = pd.read_csv(RAW, header=None, skiprows=1)
days_full = raw.iloc[:, 0].values
co2_cols = {"C1": 6, "C2": 17, "C3": 26}
obs_full = {c: raw.iloc[:, col].values for c, col in co2_cols.items()}

# ── 2. 4 个测试 CSV ───────────────────────────────────
TEMP = Path(str(FIGURE_DATA))
configs = [
    ("Fig_Antifactual_Growth.csv",        "Rapid growth phase",    0, 29),
    ("Fig_Antifactual_Growth+Decay.csv",  "Growth + decay phase",  0, 59),
    ("Fig_Antifactual_Mixed.csv",         "Mixed phase",           0, 89),
    ("Fig_Antifactual_Plateau.csv",       "Plateau phase",         0, 119),
]
test_dfs = {f: pd.read_csv(TEMP / f) for f, *_ in configs}

r2_labels = {
    "Fig_Antifactual_Growth.csv":        "$R^2$: Persist = -3.67 / LS-MPR = +0.02",
    "Fig_Antifactual_Growth+Decay.csv":  "$R^2$: Persist = -1.81 / LS-MPR = -0.32",
    "Fig_Antifactual_Mixed.csv":         "$R^2$: Persist = +0.44 / LS-MPR = +0.85",
    "Fig_Antifactual_Plateau.csv":       "$R^2$: Persist = +0.95 / LS-MPR = -0.06",
}

# ── 3. 高级配色 ───────────────────────────────────────
# 灵感: Scientific Reports / Nature Communications 风格
# 柔和、低饱和度、色盲友好
C_OBS  = "#3D3D3D"   # 深炭灰 — 观测值
C_LS   = "#D4786E"   # 暖珊瑚 — LS-MPR (温暖、突出)
C_PERS = "#5B9E9B"   # 冷薄荷 — Persistence (冷静、稳定)
C_TRAIN_BG = "#EDF2F7"  # 冷淡蓝灰 — 训练区
C_TEST_BG  = "#FFFBF2"  # 暖象牙白 — 测试区
C_BOUNDARY = "#B0B0B0"  # 分界线

# ── 4. 绘图 ───────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
    "font.weight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.34, wspace=0.24)

for idx, (ax, (fname, label, t_start, t_end)) in enumerate(zip(axes.flat, configs)):
    df_test = test_dfs[fname]
    train_end = int(df_test["Train_End"].iloc[0])
    test_end  = int(df_test["Day"].max())

    # ── 背景色块 ────────────────────────────────────
    ax.axvspan(0, train_end, facecolor=C_TRAIN_BG, alpha=0.55, linewidth=0, zorder=0)
    ax.axvspan(train_end, test_end + 1, facecolor=C_TEST_BG, alpha=0.45, linewidth=0, zorder=0)

    # ── 训练区: 仅画 C1（三条几乎重合，简化视觉）───
    train_mask = days_full <= train_end
    ax.plot(days_full[train_mask], obs_full["C1"][train_mask],
            color=C_OBS, linewidth=1.1, alpha=0.85, zorder=2)

    # ── 测试区: C1 实测 + 两条 C2/C3 淡色 ──────────
    for cond, lw, alpha in [("C1", 1.3, 0.85), ("C2", 0.5, 0.30), ("C3", 0.5, 0.30)]:
        cdf = df_test[df_test["Condition"] == cond]
        ax.plot(cdf["Day"], cdf["Observed_CO2"],
                color=C_OBS, linewidth=lw, alpha=alpha, zorder=2)

    # ── 测试区: LS-MPR 预测 ─────────────────────────
    for cond, lw, alpha in [("C1", 1.6, 0.90), ("C2", 0.6, 0.30), ("C3", 0.6, 0.30)]:
        cdf = df_test[df_test["Condition"] == cond]
        ax.plot(cdf["Day"], cdf["LSMPR_Predicted"],
                color=C_LS, linewidth=lw, alpha=alpha, zorder=4)

    # ── 测试区: Persistence 预测 ─────────────────────
    for cond, lw, alpha in [("C1", 1.6, 0.90), ("C2", 0.6, 0.30), ("C3", 0.6, 0.30)]:
        cdf = df_test[df_test["Condition"] == cond]
        ax.plot(cdf["Day"], cdf["Persistence_Predicted"],
                color=C_PERS, linewidth=lw, alpha=alpha, zorder=4)

    # ── 分界线 ──────────────────────────────────────
    ax.axvline(x=train_end, color=C_BOUNDARY, linewidth=1.0,
               linestyle=(0, (5, 3)), alpha=0.7, zorder=3)

    # ── 区域文字标注 ─────────────────────────────────
    y_top = obs_full["C1"].max() + 5
    ax.text(train_end / 2, y_top * 0.97, "Training",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#7F8C8D")
    ax.text(train_end + (test_end - train_end) / 2, y_top * 0.97, "Test",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#B7950B")

    # ── 坐标轴 ───────────────────────────────────────
    ax.set_title(f"({chr(97+idx)}) {label}", pad=8)
    ax.set_xlabel("Days")
    ax.set_ylabel("CO$_2$ release (g)")
    ax.set_xlim(-2, test_end + 2)
    y_pad = (obs_full["C1"].max() - obs_full["C1"].min()) * 0.08
    ax.set_ylim(obs_full["C1"].min() - y_pad, obs_full["C1"].max() + y_pad * 2.2)
    ax.tick_params(which="both", direction="in", bottom=True, top=False,
                   left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── R² 标注 ──────────────────────────────────────
    ax.text(0.98, 0.05, r2_labels[fname], transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#D5D8DC", alpha=0.92))

# ── 统一图例 ─────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color=C_OBS,  linewidth=2.0, label="Observed CO$_2$"),
    Line2D([0], [0], color=C_LS,   linewidth=2.0, label="LS-MPR prediction"),
    Line2D([0], [0], color=C_PERS, linewidth=2.0, label="Persistence prediction"),
]
leg = fig.legend(handles=legend_elements, loc="upper center",
                 ncol=3, framealpha=0.92, edgecolor="#D5D8DC",
                 fancybox=True, fontsize=10, bbox_to_anchor=(0.5, 1.00))
for t in leg.get_texts():
    t.set_fontweight("bold")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_antifactual_trajectories_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_antifactual_trajectories_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_antifactual_trajectories_v2.png'}")
print(f"Saved: {OUT_DIR / 'fig_antifactual_trajectories_v2.svg'}")
print("Done.")
plt.close(fig)