"""
Figure: 5x5 特征相关矩阵热图 + VIF 标注
可视化 VIF=inf 和设计矩阵秩亏问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 相关系数矩阵 ──────────────────────────────────
features = ["Days", "Temperature", "Humidity", "Ratio", "Compost\nVolume"]
n = len(features)

corr = np.array([
    [ 1.000,  0.174,  0.167,  0.095,  0.201],
    [ 0.174,  1.000, -0.786,  0.655, -0.684],
    [ 0.167, -0.786,  1.000, -0.549,  0.929],
    [ 0.095,  0.655, -0.549,  1.000, -0.643],
    [ 0.201, -0.684,  0.929, -0.643,  1.000],
])

vif_values = ["1.0", "∞", "∞", "∞", "∞"]

# ── 2. 绘图设置 ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
    "font.weight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

fig, ax = plt.subplots(figsize=(7.5, 6.2))

# ── 3. 热图 (下三角屏蔽) ─────────────────────────────
# 使用上三角
mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
corr_display = np.ma.array(corr, mask=mask)

cmap = plt.cm.RdBu_r
im = ax.imshow(corr_display, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

# ── 4. 单元格文字 ────────────────────────────────────
for i in range(n):
    for j in range(n):
        if i <= j:  # 对角线 + 上三角
            val = corr[i, j]
            if i == j:
                text = "1.000"
            else:
                text = f"{val:+.3f}" if val != 1.0 else "1.000"
            # 文字颜色: 深色背景用白色
            text_color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

# ── 5. 坐标轴 ────────────────────────────────────────
ax.set_xticks(range(n))
ax.set_xticklabels(features, fontsize=9.5)
ax.set_yticks(range(n))
ax.set_yticklabels(features, fontsize=9.5, rotation=30, va="center")

# 隐藏刻度
ax.tick_params(which="both", length=0, bottom=False, left=False)

# ── 6. VIF 标注 (右侧额外列) ─────────────────────────
vif_colors = ["#2C3E50", "#C0392B", "#C0392B", "#C0392B", "#C0392B"]
for i, (vif_val, vif_color) in enumerate(zip(vif_values, vif_colors)):
    ax.text(n + 0.35, i, f"VIF = {vif_val}", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=vif_color)

# VIF 列标题
ax.text(n + 0.35, -0.65, "VIF", ha="left", va="center",
        fontsize=9.5, fontweight="bold", color="#7F8C8D")

# ── 7. 右侧竖线分隔 ──────────────────────────────────
ax.axvline(x=n - 0.55, color="#BDC3C7", linewidth=1.2, zorder=5)

# ── 8. Colorbar ──────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02,
                    ticks=[-1, -0.5, 0, 0.5, 1])
cbar.set_label("Pearson r", fontweight="bold", fontsize=10, labelpad=-4)
cbar.ax.tick_params(length=0)
cbar.outline.set_visible(False)

# ── 9. 标题 ──────────────────────────────────────────
ax.set_title("Feature Correlation Matrix & VIF", pad=14, fontsize=13)

# ── 10. 底部注释 ─────────────────────────────────────
fig.subplots_adjust(bottom=0.17)
fig.text(0.5, 0.04,
         "VIF = inf:  3 experimental conditions < 4 environmental features  →  rank-deficient design",
         ha="center", fontsize=8.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_correlation_vif.png", dpi=600)
fig.savefig(OUT_DIR / "fig_correlation_vif.svg")
print(f"Saved: {OUT_DIR / 'fig_correlation_vif.png'}")
print(f"Saved: {OUT_DIR / 'fig_correlation_vif.svg'}")
print("Done.")
plt.close(fig)