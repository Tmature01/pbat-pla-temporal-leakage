"""
Figure: 逐条件、逐时段预测误差热图
3 面板 (C1/C2/C3) x 180 天, 统一色阶 0-68 g
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── 1. 读取数据 ──────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_ErrorHeatmap.csv"))
days = df["Day"].values
c1_err = df["C1_AbsError"].values
c2_err = df["C2_AbsError"].values
c3_err = df["C3_AbsError"].values
vmax = 68

# 统计标注
for name, err in [("C1", c1_err), ("C2", c2_err), ("C3", c3_err)]:
    train_mae = err[days <= 119].mean()
    test_mae  = err[days >= 120].mean()
    print(f"{name}: train MAE={train_mae:.1f}, test MAE={test_mae:.1f}, x{test_mae/train_mae:.0f}")

# ── 2. 配色 ──────────────────────────────────────────
# 自定义: 白→浅黄→橙→深红
cmap = LinearSegmentedColormap.from_list("error_cmap", [
    (0.00, "#FFFFFF"),
    (0.05, "#FFF5EB"),
    (0.15, "#FDEBD0"),
    (0.35, "#F5B041"),
    (0.60, "#E74C3C"),
    (0.85, "#922B21"),
    (1.00, "#4A0E0E"),
])

# ── 3. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, axes = plt.subplots(3, 1, figsize=(13, 4.5))
fig.subplots_adjust(left=0.10, right=0.92, top=0.93, bottom=0.12, hspace=0.18)

cond_data = [("C1", c1_err), ("C2", c2_err), ("C3", c3_err)]
cond_labels = [
    "C1 (T=50, H=70, R=70)",
    "C2 (T=58, H=60, R=100)",
    "C3 (T=58, H=70, R=0)",
]

for idx, (ax, (cname, err_data)) in enumerate(zip(axes, cond_data)):
    # 构造 1x180 的色块网格
    X = np.tile(days.reshape(1, -1), (2, 1))  # 2 rows for pcolormesh
    Y = np.array([[0], [1]])
    C = np.tile(err_data.reshape(1, -1), (1, 1))  # use 1 row since pcolormesh needs (ny, nx)

    # pcolormesh 需要 C 是 (ny-1) x (nx-1)
    X_mesh = np.arange(0.5, 181.5, 1)  # cell edges
    Y_mesh = np.array([0, 1])
    # 把 1D error 变成 2D for pcolormesh: (1, 180) -> pcolormesh needs (1, 180)
    err_2d = err_data.reshape(1, -1)

    im = ax.pcolormesh(X_mesh, Y_mesh, err_2d, cmap=cmap, vmin=0, vmax=vmax,
                        edgecolors="none", linewidth=0, rasterized=True)

    # 训练/测试分界线
    ax.axvline(x=119.5, color="black", linewidth=1.2, zorder=5)

    # 标注
    train_mae = err_data[days <= 119].mean()
    test_mae  = err_data[days >= 120].mean()
    ratio = test_mae / train_mae

    ax.text(60, 0.5, f"Train MAE={train_mae:.1f}", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#2C3E50")
    ax.text(150, 0.5, f"Test MAE={test_mae:.1f}  (x{ratio:.0f})",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#C0392B")

    # 坐标轴
    ax.set_ylabel(cname, fontsize=10, labelpad=8, rotation=0, ha="right", va="center")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, 180.5)
    ax.set_yticks([])
    if idx == 2:
        ax.set_xlabel("Days")
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    else:
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.set_xticklabels([])

    ax.tick_params(which="both", direction="in", bottom=True, top=False,
                   left=False, right=False, labelsize=8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 面板标签
    ax.text(0.5, 0.95, cond_labels[idx], transform=ax.transAxes,
            fontsize=7, fontweight="bold", color="#7F8C8D", ha="left", va="top")

# ── Colorbar ──────────────────────────────────────────
cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015,
                    ticks=[0, 10, 20, 30, 40, 50, 60, 68])
cbar.set_label("Absolute error (CO$_2$ g)", fontweight="bold", fontsize=10)
cbar.ax.tick_params(length=0)

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.025,
         "Error in all 3 conditions escalates systematically after Day 119. "
         "The dark-red contiguous blocks in the test region rule out 'occasional bad points' as an explanation.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_error_heatmap.png", dpi=600)
fig.savefig(OUT_DIR / "fig_error_heatmap.svg")
print(f"Saved: {OUT_DIR / 'fig_error_heatmap.png'}")
print(f"Saved: {OUT_DIR / 'fig_error_heatmap.svg'}")
print("Done.")
plt.close(fig)