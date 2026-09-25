"""
Figure: 学习曲线 — 超参数稳定性验证 (极简版, 零遮挡)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

with open(str(FIGURE_DATA / "Fig_LearningCurve.csv"), "r") as f:
    lines = f.readlines()
col_names = ["TrainRatio"] + [l.strip().strip(",").strip('"') for l in lines[1:10]]
data_rows = []
for line in lines[10:]:
    parts = line.strip().split(",")
    if len(parts) >= 10:
        data_rows.append([float(x) for x in parts[:10]])
ratios = np.array([r[0] * 100 for r in data_rows])
data = {col_names[i+1]: np.array([r[i+1] for r in data_rows]) for i in range(9)}

ridge_cols = ["Ridge (alpha=0.01)", "Ridge (alpha=0.1)", "Ridge (alpha=1.0)"]
svr_cols   = ["SVR (C=1)", "SVR (C=10)", "SVR (C=100)"]
rf_cols    = ["RF (100 trees)", "RF (200 trees)", "RF (500 trees)"]

# ── 2. 配色 ──────────────────────────────────────────
R = {"Ridge (alpha=0.01)": "#6C8EBF", "Ridge (alpha=0.1)": "#4A6FA5", "Ridge (alpha=1.0)": "#2B4C7E"}
S = {"SVR (C=1)": "#F0A860",        "SVR (C=10)": "#E8853B",       "SVR (C=100)": "#C0601A"}
F = {"RF (100 trees)": "#5DAA8C",   "RF (200 trees)": "#3D8B6E",   "RF (500 trees)": "#1F6C4F"}
colors = {**R, **S, **F}
lstyles = {**{k:"-" for k in ridge_cols[:1]+svr_cols[:1]+rf_cols[:1]},
           **{ridge_cols[1]:"--", svr_cols[1]:"--", rf_cols[1]:"--"},
           **{ridge_cols[2]:":",  svr_cols[2]:":",  rf_cols[2]:":"}}

# ── 3. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})

fig, ax = plt.subplots(figsize=(10, 5.8))
fig.subplots_adjust(left=0.09, right=0.96, top=0.92, bottom=0.25)

# ── 背景 ─────────────────────────────────────────────
ax.axhspan(-42, 0, facecolor="#FADBD8", alpha=0.10, linewidth=0, zorder=0)
ax.axhline(y=0, color="black", linewidth=0.7, linestyle="-", alpha=0.3, zorder=1)

# ── 学习曲线 ─────────────────────────────────────────
for col in ridge_cols + svr_cols + rf_cols:
    ax.plot(ratios, data[col], color=colors[col], linestyle=lstyles.get(col, "-"),
            linewidth=1.3, marker="o", markersize=4,
            markerfacecolor="white", markeredgewidth=0.8,
            markeredgecolor=colors[col], zorder=2)

# ── 时序留出参考线 ───────────────────────────────────
tmp_refs = [(-0.058, "#2B4C7E"), (-0.203, "#C0601A"), (-0.312, "#1F6C4F")]
for yv, clr in tmp_refs:
    ax.axhline(y=yv, color=clr, linewidth=0.5, linestyle=(0, (4, 6)), alpha=0.5, zorder=1)

# ── 坐标轴 ───────────────────────────────────────────
ax.set_xlabel("Training ratio (%)")
ax.set_ylabel("$R^2$ (time-series CV)")
ax.set_xlim(28, 92)
ax.set_ylim(-42, 8)
ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 图例: 底部 3 组 x 3 列, 整齐排列 ──────────────────
# 构造: 每组一个标题行 + 3 个方法行, 共 4 行 x 3 列
# 使用 ncol=3 的 legend, 每列放一组
group_data = [
    ("Ridge",  [("alpha=0.01", "#6C8EBF", "-"),
                ("alpha=0.1",  "#4A6FA5", "--"),
                ("alpha=1.0",  "#2B4C7E", ":")]),
    ("SVR",    [("C=1",   "#F0A860", "-"),
                ("C=10",  "#E8853B", "--"),
                ("C=100", "#C0601A", ":")]),
    ("RF",     [("100 trees", "#5DAA8C", "-"),
                ("200 trees", "#3D8B6E", "--"),
                ("500 trees", "#1F6C4F", ":")]),
]

handles = []
labels = []
for group_name, variants in group_data:
    handles.append(Line2D([0],[0], color="none", linewidth=0))
    labels.append(group_name)
    for vname, vclr, vls in variants:
        handles.append(Line2D([0],[0], color=vclr, linewidth=2.0, linestyle=vls,
                              marker="o", markersize=5, markerfacecolor="white",
                              markeredgewidth=1.0, markeredgecolor=vclr))
        labels.append(vname)

leg = fig.legend(handles, labels, loc="upper center", ncol=3,
                framealpha=0.90, edgecolor="#BDC3C7", fancybox=True,
                fontsize=7.5, borderpad=0.6, labelspacing=0.4,
                columnspacing=2.0, handlelength=2.2, handletextpad=0.6,
                bbox_to_anchor=(0.5, 0.175))
for t in leg.get_texts():
    t.set_fontweight("bold")
for t in leg.get_texts():
    if t.get_text() in ["Ridge", "SVR", "RF"]:
        t.set_fontsize(8.5)

# 左上角: 参考线小标注 (ax 内部, 不遮挡数据)
ref_handle = [Line2D([0],[0], color="gray", linewidth=1.2,
                     linestyle=(0, (4, 6)))]
ref_leg = ax.legend(ref_handle, ["Temporal holdout R^2"],
                    loc="upper left", framealpha=0.85, edgecolor="#BDC3C7",
                    fancybox=True, fontsize=7, borderpad=0.4,
                    handlelength=1.6, handletextpad=0.5)
for t in ref_leg.get_texts():
    t.set_fontweight("bold")

# ── 轻量标注 ─────────────────────────────────────────
# "全部为负" 区域标注 (红区内右侧空白处)
ax.text(75, -18, "All temporal holdout\n$R^2$ negative", fontsize=7.5,
        fontweight="bold", color="#C0392B", alpha=0.65, ha="center")

# RF 关键对照 (右端): CV vs Temporal
rf200_end = data["RF (200 trees)"][-1]  # ~0.999
ax.text(93, rf200_end + 0.8, "CV: +0.999", fontsize=7, fontweight="bold",
        color="#1F6C4F", ha="right", va="bottom")
ax.text(93, -0.312 - 1.2, "Temporal: -0.312", fontsize=7, fontweight="bold",
        color="#1F6C4F", ha="right", va="top")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.025,
         "Hyperparameters span 2 orders of magnitude (0.01-1.0 for alpha, 1-100 for C, 100-500 for trees). "
         "All CV R^2 trend negative. Not a hyperparameter issue -- it is an evaluation protocol issue.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_learning_curve.png", dpi=600)
fig.savefig(OUT_DIR / "fig_learning_curve.svg")
print(f"Saved: {OUT_DIR / 'fig_learning_curve.png'}")
print(f"Saved: {OUT_DIR / 'fig_learning_curve.svg'}")
print("Done.")
plt.close(fig)