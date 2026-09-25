"""
Figure: Bootstrap R^2 分布对比 — Random Split vs Temporal Holdout
"""

import pandas as pd
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

df = pd.read_csv(str(FIGURE_DATA / "Fig_Violin_Bootstrap.csv"))
methods = ["LS-MPR", "Ridge", "SVR", "RF"]
colors_m = {"LS-MPR": "#E74C3C", "Ridge": "#3498DB", "SVR": "#E67E22", "RF": "#2ECC71"}

# 统计
for m in methods:
    for p in ["Random", "Temporal"]:
        sub = df[(df.Method == m) & (df.Protocol == p)]
        if len(sub) > 0:
            print(f"{m:8s} {p:10s}: median={sub.R2.median():.4f}, IQR=[{sub.R2.quantile(0.25):.3f}, {sub.R2.quantile(0.75):.3f}]")

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 10, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.12, bottom=0.15, top=0.90, left=0.08, right=0.96)

def plot_violins(ax, protocol, ylim, title):
    data_list = []
    positions = []
    method_list = []
    for i, m in enumerate(methods):
        sub = df[(df.Method == m) & (df.Protocol == protocol)]
        if len(sub) == 0:
            continue
        data_list.append(sub["R2"].values)
        positions.append(i)
        method_list.append(m)

    vp = ax.violinplot(data_list, positions=positions, showmeans=False,
                       showmedians=True, widths=0.65)

    # 美化
    for i, body in enumerate(vp["bodies"]):
        m = method_list[i]
        body.set_facecolor(colors_m[m])
        body.set_alpha(0.40)
        body.set_edgecolor(colors_m[m])
        body.set_linewidth(1.2)
    for part in ["cbars", "cmins", "cmaxes"]:
        if part in vp:
            vp[part].set_color("#7F8C8D")
            vp[part].set_linewidth(0.8)
    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.8)

    # 中位数标注
    for i, (m, d) in enumerate(zip(method_list, data_list)):
        med = np.median(d)
        ax.annotate(f"{med:+.2f}", xy=(i + 0.35, med),
                    fontsize=7.5, fontweight="bold", color=colors_m[m],
                    va="center", ha="left")

    # 持久化参考线
    if protocol == "Temporal":
        ax.axhline(y=-4.99, color="#2C3E50", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.text(len(method_list) - 0.5, -4.99, "Persistence", fontsize=7,
                fontweight="bold", color="#2C3E50", va="bottom", ha="right", alpha=0.7)

    ax.set_title(title, pad=8)
    ax.set_ylabel("$R^2$")
    ax.set_xticks(range(len(method_list)))
    ax.set_xticklabels(method_list)
    ax.set_ylim(ylim)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3, zorder=0)
    ax.tick_params(which="both", direction="in", bottom=True, top=False,
                   left=True, right=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Left: Random Split
plot_violins(axL, "Random", (-0.05, 1.08), "(a) Random split bootstrap")

# Right: Temporal Holdout
plot_violins(axR, "Temporal", (-5.5, 1.05), "(b) Temporal holdout bootstrap")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "Random split: tight distributions, high certainty (illusory). "
         "Temporal holdout: wide distributions, many below zero — real predictive uncertainty exposed. "
         "300 bootstrap samples per method per protocol.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = _REPO_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_violin_bootstrap.png", dpi=600)
fig.savefig(OUT_DIR / "fig_violin_bootstrap.svg")
print(f"\nSaved: {OUT_DIR / 'fig_violin_bootstrap.png'}")
print(f"Saved: {OUT_DIR / 'fig_violin_bootstrap.svg'}")
print("Done.")
plt.close(fig)
