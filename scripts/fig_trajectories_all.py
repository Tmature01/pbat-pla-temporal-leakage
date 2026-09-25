"""
Figure: 七种方法时序留出预测轨迹叠加 — "集体失效" 视觉叙事
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from pathlib import Path

# ── 1. 读取数据 ──────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_Trajectories_All.csv"))
train = df[df["Region"] == "Train"]
test  = df[df["Region"] == "Test"]

methods = ["LS_MPR", "Ridge", "Lasso", "SVR", "RF", "GPR"]
colors_m = {
    "LS_MPR": "#E74C3C",
    "Ridge":   "#E67E22",
    "Lasso":   "#2ECC71",
    "SVR":     "#9B59B6",
    "RF":      "#3498DB",
    "GPR":     "#1ABC9C",
}
# ── 计算 R² ──────────────────────────────────────────
r2_vals = {}
for m in methods:
    r2_vals[m] = r2_score(test["Observed_CO2"], test[m])
r2_vals["Persistence"] = r2_score(test["Observed_CO2"], test["Persistence"])

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.subplots_adjust(left=0.08, right=0.68, top=0.94, bottom=0.13)

# ── 背景色块 ─────────────────────────────────────────
ax.axvspan(0, 119, facecolor="#EAECEE", alpha=0.45, linewidth=0, zorder=0)
ax.axvspan(119, 180, facecolor="white", alpha=0.0, linewidth=0, zorder=0)
ax.axvline(x=119, color="black", linewidth=1.0, linestyle=(0, (5, 4)),
           alpha=0.55, zorder=2)

# ── 区域标注 ─────────────────────────────────────────
ax.text(59, df["Observed_CO2"].max() * 0.97, "Training\n(Days 0–119)",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color="#7F8C8D")
ax.text(149, df["Observed_CO2"].max() * 0.88, "Test\n(Days 120–180)",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color="#2C3E50")

# ── 实测 CO₂ 曲线 ────────────────────────────────────
ax.plot(df["Day"], df["Observed_CO2"], color="black", linewidth=1.8,
        zorder=3, label="Observed CO$_2$")

# ── 7 种方法 + Persistence ───────────────────────────
for m in methods:
    ax.plot(test["Day"], test[m], color=colors_m[m], linewidth=1.2,
            alpha=0.85, zorder=4)

# Persistence: 蓝色虚线
ax.plot(test["Day"], test["Persistence"], color="#2C3E50", linewidth=1.3,
        linestyle="--", alpha=0.8, zorder=4)

# ── 坐标轴 ───────────────────────────────────────────
ax.set_xlabel("Days")
ax.set_ylabel("CO$_2$ release (g)")
ax.set_xlim(-3, 183)
ax.tick_params(which="both", direction="in", bottom=True, top=False,
               left=True, right=False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 图例: 右侧外部 ───────────────────────────────────
legend_elements = [
    Line2D([0],[0], color="black", linewidth=2.0, label="Observed"),
    Line2D([0],[0], color="#2C3E50", linewidth=1.5, linestyle="--",
           label=f"Persist  R$^2$={r2_vals['Persistence']:.1f}"),
]
for m in methods:
    r2_str = f"{r2_vals[m]:.0f}" if r2_vals[m] < -10 else f"{r2_vals[m]:.1f}"
    name = m.replace("_","-")
    legend_elements.append(
        Line2D([0],[0], color=colors_m[m], linewidth=2.0,
               label=f"{name}  R$^2$={r2_str}")
    )

leg = fig.legend(handles=legend_elements, loc="center left",
                 framealpha=0.92, edgecolor="#BDC3C7", fancybox=True,
                 fontsize=8, borderpad=0.6, labelspacing=0.4,
                 handlelength=1.8, bbox_to_anchor=(0.705, 0.5))
for t in leg.get_texts():
    t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "All 7 data-driven methods diverge from observed CO$_2$ in the test region. "
         "None captures the plateau behavior. This is a protocol-level failure, not a model-specific one.",
         ha="center", fontsize=8, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_trajectories_all.png", dpi=600)
fig.savefig(OUT_DIR / "fig_trajectories_all.svg")
print(f"Saved: {OUT_DIR / 'fig_trajectories_all.png'}")
print(f"Saved: {OUT_DIR / 'fig_trajectories_all.svg'}")
print("Done.")
plt.close(fig)