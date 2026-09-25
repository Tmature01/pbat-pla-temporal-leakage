"""
Figure: 反事实验证多窗口预测轨迹 (4 面板, 2x2)
表 4 的视觉化: 证明持久化预测器的优势是相位条件性的
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 从宽格式原始数据提取全曲线 ──────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

RAW = str(_REPO_ROOT / "data" / "pbat_pla_experimental_543.csv")
raw = pd.read_csv(RAW, header=None, skiprows=1)
days_full = raw.iloc[:, 0].values  # Day 始终在第 0 列
# 3 个条件的 CO2 通过列位置提取
co2_cols = {"C1": 6, "C2": 17, "C3": 26}
obs_full = {cond: raw.iloc[:, col].values for cond, col in co2_cols.items()}

# ── 2. 读取 4 个测试 CSV ────────────────────────────────
TEMP = Path(str(FIGURE_DATA))
configs = [
    ("Fig_Antifactual_Growth.csv",        "Rapid growth phase",    0, 29),
    ("Fig_Antifactual_Growth+Decay.csv",  "Growth + decay phase",  0, 59),
    ("Fig_Antifactual_Mixed.csv",         "Mixed phase",           0, 89),
    ("Fig_Antifactual_Plateau.csv",       "Plateau phase",         0, 119),
]

test_dfs = {}
for fname, label, t_start, t_end in configs:
    df = pd.read_csv(TEMP / fname)
    test_dfs[fname] = df

# ── R² 值 (来自用户表) ──────────────────────────────────
r2_labels = {
    "Fig_Antifactual_Growth.csv":        "$R^2$: Persist=$-$3.67 / LS-MPR=+0.02",
    "Fig_Antifactual_Growth+Decay.csv":  "$R^2$: Persist=$-$1.81 / LS-MPR=$-$0.32",
    "Fig_Antifactual_Mixed.csv":         "$R^2$: Persist=+0.44 / LS-MPR=+0.85",
    "Fig_Antifactual_Plateau.csv":       "$R^2$: Persist=+0.95 / LS-MPR=$-$0.06",
}

# ── 3. 绘图设置 ─────────────────────────────────────────
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

COLOR_OBS  = "#2C3E50"   # 深灰蓝 — 真实值
COLOR_LS   = "#C0392B"   # 红 — LS-MPR
COLOR_PERS = "#2471A3"   # 蓝 — Persistence
COLOR_TRAIN_BG = "#EAECEE"  # 训练区底色
COLOR_TEST_BG  = "#FDFEFE"  # 测试区底色 (几乎白)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.32, wspace=0.22)

for idx, (ax, (fname, label, t_start, t_end)) in enumerate(zip(axes.flat, configs)):
    df_test = test_dfs[fname]
    train_end = int(df_test["Train_End"].iloc[0])
    test_end  = int(df_test["Day"].max())

    # ── 背景色块 ──────────────────────────────────────
    ax.axvspan(0, train_end, facecolor="#E8F0FE", alpha=0.30, linewidth=0, zorder=0)
    ax.axvspan(train_end, test_end + 1, facecolor="#FEF9E7", alpha=0.25, linewidth=0, zorder=0)

    # ── 训练区 (0 → train_end) ───────────────────────
    # --- 仅画 C1 的训练期数据 (3 条件几乎重合) ---
    train_mask = days_full <= train_end
    for cond in ["C1", "C2", "C3"]:
        ax.plot(days_full[train_mask], obs_full[cond][train_mask],
                color=COLOR_OBS, linewidth=0.7, alpha=0.5, zorder=1)

    # ── 测试区: 真实值 ───────────────────────────────
    for cond in ["C1", "C2", "C3"]:
        cond_test = df_test[df_test["Condition"] == cond]
        ax.plot(cond_test["Day"], cond_test["Observed_CO2"],
                color=COLOR_OBS, linewidth=1.0, alpha=0.6, zorder=1)

    # ── 测试区: LS-MPR 预测 ──────────────────────────
    for cond in ["C1", "C2", "C3"]:
        cond_test = df_test[df_test["Condition"] == cond]
        ax.plot(cond_test["Day"], cond_test["LSMPR_Predicted"],
                color=COLOR_LS, linewidth=1.2, alpha=0.8, zorder=3,
                linestyle="--" if cond != "C1" else "-")
        if cond == "C1":
            ax.plot(cond_test["Day"], cond_test["LSMPR_Predicted"],
                    color=COLOR_LS, linewidth=1.5, zorder=3)

    # ── 测试区: Persistence 预测 ──────────────────────
    for cond in ["C1", "C2", "C3"]:
        cond_test = df_test[df_test["Condition"] == cond]
        ax.plot(cond_test["Day"], cond_test["Persistence_Predicted"],
                color=COLOR_PERS, linewidth=1.2, alpha=0.8, zorder=3,
                linestyle="--" if cond != "C1" else "-")
        if cond == "C1":
            ax.plot(cond_test["Day"], cond_test["Persistence_Predicted"],
                    color=COLOR_PERS, linewidth=1.5, zorder=3)

    # ── 训练/测试分界线 ──────────────────────────────
    ax.axvline(x=train_end, color="black", linewidth=1.0, linestyle=(0, (5, 3)),
               alpha=0.6, zorder=2)

    # ── 区域标注 ─────────────────────────────────────
    ax.text(train_end / 2, ax.get_ylim()[1] * 0.97, "Train",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#5D6D7E", alpha=0.7)
    ax.text(train_end + (test_end - train_end) / 2, ax.get_ylim()[1] * 0.97, "Test",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#D4AC0D", alpha=0.7)

    # ── 坐标轴 ───────────────────────────────────────
    ax.set_title(f"({chr(97+idx)}) {label}", pad=6)
    ax.set_xlabel("Days")
    ax.set_ylabel("CO$_2$ release (g)")
    ax.set_xlim(-2, test_end + 2)
    ax.tick_params(which="both", direction="in", bottom=True, top=False,
                   left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # ── R² 标注 ──────────────────────────────────────
    ax.text(0.98, 0.06, r2_labels[fname], transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85))

# ── 统一图例 (仅第 1 面板) ───────────────────────────
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLOR_OBS,  linewidth=1.5, label="Observed"),
    Line2D([0], [0], color=COLOR_LS,   linewidth=1.5, label="LS-MPR"),
    Line2D([0], [0], color=COLOR_PERS, linewidth=1.5, label="Persistence"),
]
leg = fig.legend(handles=legend_elements, loc="upper center",
                 ncol=3, framealpha=0.9, edgecolor="gray",
                 fancybox=True, fontsize=10, bbox_to_anchor=(0.5, 0.995))
for t in leg.get_texts():
    t.set_fontweight("bold")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_antifactual_trajectories.png", dpi=600)
fig.savefig(OUT_DIR / "fig_antifactual_trajectories.svg")
print(f"Saved: {OUT_DIR / 'fig_antifactual_trajectories.png'}")
print(f"Saved: {OUT_DIR / 'fig_antifactual_trajectories.svg'}")
print("Done.")
plt.close(fig)