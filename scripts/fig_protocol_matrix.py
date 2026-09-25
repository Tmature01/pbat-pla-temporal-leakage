"""
Figure: 评估协议-模型性能矩阵热图 (V2 — 简洁学术风格)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_ProtocolMatrix.csv"))
methods = ["Persistence","LS-MPR","Ridge-MPR","Lasso-MPR","SVR-RBF",
           "RF","GPR-RBF","Exp Decay","Mean Predictor"]
protocols = ["Random Split", "Temporal Holdout", "LOCO (mean)", "Temporal Shuffle"]
p_short  = ["Random\nSplit", "Temporal\nHoldout", "LOCO\n(mean)", "Temporal\nShuffle"]

r2_mat  = np.full((9, 4), np.nan)
rho_mat = np.full((9, 4), np.nan)
for i, m in enumerate(methods):
    for j, p in enumerate(protocols):
        r2_col  = f"{p}_R2"
        rho_col = f"{p}_rho"
        row = df[df["Method"] == m]
        if r2_col in df.columns and pd.notna(row[r2_col].values[0]):
            r2_mat[i, j] = row[r2_col].values[0]
        if rho_col in df.columns and pd.notna(row[rho_col].values[0]):
            rho_mat[i, j] = row[rho_col].values[0]

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 13.5, "font.weight": "bold",
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, ax = plt.subplots(figsize=(8.5, 6.5))
fig.subplots_adjust(left=0.26, right=0.90, top=0.88, bottom=0.06)

# Crameri 'roma' 风格: 暖灰-橙 → 白 → 蓝绿
# Nature Geoscience / Science Advances 常用
cmap = LinearSegmentedColormap.from_list("roma_style", [
    (0.000, "#3B1F0B"),  # 深棕
    (0.100, "#7B4A2C"),
    (0.200, "#B07B52"),
    (0.300, "#D7AE89"),
    (0.400, "#EDD7C3"),
    (0.500, "#F6F5F4"),  # 暖白
    (0.600, "#C9DFE2"),
    (0.700, "#8CC3CC"),
    (0.800, "#4EA0AB"),
    (0.900, "#1B7885"),
    (1.000, "#004D5A"),  # 深青
], N=256)

# 剪裁到合理的 R² 范围用于颜色映射
vmin, vmax = -5, 1.0
norm = Normalize(vmin=vmin, vmax=vmax)

# 画热图
for i in range(9):
    for j in range(4):
        r2  = r2_mat[i, j]
        rho = rho_mat[i, j]
        x, y = j, 8 - i

        if np.isnan(r2):
            ax.add_patch(plt.Rectangle((j - 0.47, 8 - i - 0.47), 0.94, 0.94,
                         facecolor="#F2F3F5", edgecolor="#D5D8DC",
                         linewidth=0.4, alpha=0.7))
            ax.text(x, y, "—", ha="center", va="center", fontsize=16,
                    fontweight="bold", color="#B0B8C1")
            continue

        # 剪裁颜色
        r2_clipped = np.clip(r2, vmin, vmax)
        fc = cmap(norm(r2_clipped))

        ax.add_patch(plt.Rectangle((j - 0.47, 8 - i - 0.47), 0.94, 0.94,
                     facecolor=fc, edgecolor="#F7F7F7", linewidth=0.8))

        # R² 主数字
        r2_str = f"{r2:+.3f}" if abs(r2) < 10 else f"{r2:+.1f}"
        txt_c = "white" if (r2_clipped < -2.5 or r2_clipped > 0.85) else "#1A1A1A"
        ax.text(x, y + 0.12, r2_str, ha="center", va="center", fontsize=13.5,
                fontweight="bold", color=txt_c)

        # ρ 小字
        if not np.isnan(rho):
            rho_str = f"rho={rho:+.2f}"
            ax.text(x, y - 0.22, rho_str, ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="#7F8C8D")

# ── 坐标轴 ───────────────────────────────────────────
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 8.5)
ax.set_xticks(range(4))
ax.set_xticklabels(p_short, fontsize=13.5)
ax.set_yticks(range(9)[::-1])
ax.set_yticklabels(methods, fontsize=13.5)
ax.tick_params(length=0)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

# 列标注
col_labels = ["All methods\nconverge", "Generalization\nrevealed", "Extreme\nextrapolation", "Shuffle\nimmunity"]
for j, txt in enumerate(col_labels):
    ax.text(j, -0.55, txt, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color="#7F8C8D", style="italic")

# ── Colorbar ──────────────────────────────────────────
cbar_ax = fig.add_axes([0.91, 0.22, 0.025, 0.55])
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("$R^2$", fontweight="bold", fontsize=15, labelpad=5)
cbar.ax.tick_params(length=0)
cbar.set_ticks([-5, -2, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(["<= -5", "-2", "0", "0.5", "1.0"], fontsize=11)

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_protocol_matrix.png", dpi=600)
fig.savefig(OUT_DIR / "fig_protocol_matrix.svg")
print(f"Saved: {OUT_DIR / 'fig_protocol_matrix.png'}")
print(f"Saved: {OUT_DIR / 'fig_protocol_matrix.svg'}")
print("Done.")
plt.close(fig)