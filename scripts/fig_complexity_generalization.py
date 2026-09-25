"""
Figure: 模型复杂度 vs 泛化性能 — 参数越多不等于越好
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_Complexity_Generalization.csv"))
df = df.dropna(subset=["n_params"])

print(df[["Method", "n_params", "R2_Temporal"]].to_string(index=False))

# ── 2. 配色 ──────────────────────────────────────────
colors = {
    "Persistence": "#2C3E50",
    "Exp Decay":   "#8E44AD",
    "GPR-RBF":     "#1ABC9C",
    "Lasso-MPR":   "#2ECC71",
    "SVR-RBF":     "#E67E22",
    "LS-MPR":      "#E74C3C",
    "Ridge-MPR":   "#3498DB",
    "RF":          "#27AE60",
}

# ── 3. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, ax = plt.subplots(figsize=(10.5, 6.5))
fig.subplots_adjust(left=0.11, right=0.94, top=0.90, bottom=0.14)

# ── 参考线 ───────────────────────────────────────────
persist_r2 = df[df["Method"] == "Persistence"]["R2_Temporal"].values[0]
ax.axhline(y=persist_r2, color="#2C3E50", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
ax.text(120000, -6.0, f"Persistence baseline ($R^2$={persist_r2:+.3f})",
        fontsize=8, fontweight="bold", color="#2C3E50", ha="right", va="bottom", alpha=0.7)
ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3, zorder=0)

# ── 散点 ─────────────────────────────────────────────
for _, row in df.iterrows():
    m = row["Method"]
    x = row["n_params"]
    y = row["R2_Temporal"]
    ax.scatter(x, y, s=100, color=colors[m], edgecolors="white",
               linewidth=1.0, zorder=4)

    offsets = {
        "Persistence": (40, -30),  "Exp Decay": (40, -15),
        "GPR-RBF": (50, -15),     "Lasso-MPR": (-15, -15),
        "SVR-RBF": (85, -18),     "LS-MPR": (35, -15),
        "Ridge-MPR": (60, -30),   "RF": (-100, -8),
    }
    dx, dy = offsets.get(m, (15, 10))
    ax.annotate(row["Label"], xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, fontweight="bold",
                color=colors[m], ha="center",
                arrowprops=dict(arrowstyle="->", color=colors[m], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[m], alpha=0.85),
                zorder=5)

# ── 坐标轴 ───────────────────────────────────────────
ax.set_xlabel("Number of parameters (log scale)")
ax.set_ylabel("$R^2$ (temporal holdout)")
ax.set_xscale("log")
ax.set_xlim(0.8, 150000)
ax.set_ylim(-6.5, 1.15)
ax.tick_params(which="both", direction="in", bottom=True, top=False,
               left=True, right=False)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
ax.set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "0 params (Persistence) outperforms or matches models with up to ~91,000 params. "
         "Complexity does not improve temporal generalization. "
         "The bottleneck is the evaluation protocol, not model capacity.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_complexity_generalization.png", dpi=600)
fig.savefig(OUT_DIR / "fig_complexity_generalization.svg")
print(f"\nSaved: {OUT_DIR / 'fig_complexity_generalization.png'}")
print(f"Saved: {OUT_DIR / 'fig_complexity_generalization.svg'}")
print("Done.")
plt.close(fig)