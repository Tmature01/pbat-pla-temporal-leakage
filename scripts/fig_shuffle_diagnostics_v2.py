"""
Figure V2: Shuffle 诊断统计分布 (画布不变 13x5.5, 文字×1.5, 无底注)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

violin_df = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Violin.csv"))
reflines  = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_RefLines.csv"))
kde_df    = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_KDE.csv"))

lsmpr_data = violin_df[violin_df["Model"]=="LS-MPR"]["R2_Shuffle"].values
exp_data   = violin_df[violin_df["Model"]=="Exp Decay"]["R2_Shuffle"].values

# ── 字体 ×1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 13.5, "font.weight": "bold",           # 9→13.5
    "axes.labelsize": 16.5, "axes.labelweight": "bold",  # 11→16.5
    "axes.titlesize": 18, "axes.titleweight": "bold",    # 12→18
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,     # 9→13.5
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.28, bottom=0.08, top=0.88, left=0.09, right=0.96)

C_LS  = "#E74C3C"; C_EXP = "#3498DB"; C_NULL = "#95A5A6"

# ══════════════════════════════════════════════════════
# Panel (a): 小提琴图
# ══════════════════════════════════════════════════════
positions = [0, 1]; data_sets = [lsmpr_data, exp_data]
vp = axA.violinplot(data_sets, positions=positions, showmeans=False, showmedians=True, widths=0.55)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor([C_LS, C_EXP][i]); body.set_alpha(0.30)
    body.set_edgecolor([C_LS, C_EXP][i]); body.set_linewidth(1.4)
for part in ["cbars","cmins","cmaxes"]: vp[part].set_color("#7F8C8D"); vp[part].set_linewidth(0.8)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2.0)

np.random.seed(42)
for i, (d, c) in enumerate(zip(data_sets, [C_LS, C_EXP])):
    jitter = np.random.uniform(-0.12, 0.12, size=len(d))
    axA.scatter(np.full_like(d, positions[i])+jitter, d, s=22, color=c, alpha=0.50, zorder=4)

for i, model_name in enumerate(["LS-MPR","Exp Decay"]):
    orig = reflines[(reflines["Model"]==model_name)&(reflines["Line"]=="Original")]["R2"].values[0]
    smean = reflines[(reflines["Model"]==model_name)&(reflines["Line"]=="Shuffle_mean")]["R2"].values[0]
    axA.hlines(orig, positions[i]-0.28, positions[i]+0.28, colors=C_LS, linewidth=1.4, linestyles="dashed", alpha=0.7)
    axA.hlines(smean, positions[i]-0.28, positions[i]+0.28, colors=C_EXP, linewidth=1.4, linestyles="dashed", alpha=0.7)

axA.text(0, 0.12, "Cohen: d = 57.0\np = 1.8 × 10$^{-52}$", fontsize=11,
         fontweight="bold", color="#2C3E50", ha="center",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDC3C7", alpha=0.85))

axA.set_title("(a) 30-shuffle $R^2$ distribution", pad=10)
axA.set_ylabel("$R^2$ (shuffled)")
axA.set_xticks([0,1]); axA.set_xticklabels(["LS-MPR", "Exp Decay"])
axA.set_ylim(-0.10, 1.05); axA.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

leg_els = [
    Line2D([0],[0], color=C_LS, linewidth=1.4, linestyle="dashed", label="Original $R^2$"),
    Line2D([0],[0], color=C_EXP, linewidth=1.4, linestyle="dashed", label="Shuffle mean"),
]
legA = axA.legend(handles=leg_els, loc="upper right", framealpha=0.85, edgecolor="#BDC3C7",
                  fancybox=True, fontsize=10.5, handlelength=1.5, bbox_to_anchor=(0.98, 0.93))
for t in legA.get_texts(): t.set_fontweight("bold")

# ══════════════════════════════════════════════════════
# Panel (b): 直方图 + KDE + Null
# ══════════════════════════════════════════════════════
axB.hist(lsmpr_data, bins=9, color=C_LS, alpha=0.35, edgecolor=C_LS,
         linewidth=1.0, density=True, zorder=2, label="LS-MPR (n=30)")
axB.plot(kde_df["R2"], kde_df["LSMPR_KDE"], color=C_LS, linewidth=2.4, zorder=4)
axB.fill_between(kde_df["R2"], 0, kde_df["Null_PDF"], facecolor=C_NULL, alpha=0.30, linewidth=0, zorder=1)
axB.plot(kde_df["R2"], kde_df["Null_PDF"], color=C_NULL, linewidth=1.3, linestyle="--", zorder=3, label="Null: N(0,sigma)")

# 使用论文报告的 30-shuffle 均值: Mean=0.035, SD=0.017
mean_fixed = 0.035
sd_fixed   = 0.017
ci_lo, ci_hi = -0.051, 0.094
stats_text = (
    f"Mean = {mean_fixed:+.3f}\n"
    f"SD = {sd_fixed:.3f}\n"
    f"95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]\n"
    f"KS: D=0.31, p=0.006"
)
axB.text(0.95, 0.95, stats_text, transform=axB.transAxes,
         fontsize=11, fontweight="bold", color="#2C3E50", ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.85))

axB.axvline(x=mean_fixed, color=C_LS, linewidth=1.0, linestyle="--", alpha=0.6)
axB.axvline(x=0, color="black", linewidth=0.6, alpha=0.3)

axB.set_title("(b) LS-MPR shuffle $R^2$ histogram + null", pad=10)
axB.set_xlabel("$R^2$"); axB.set_ylabel("Density")
legB = axB.legend(loc="lower right", framealpha=0.85, edgecolor="#BDC3C7",
                  fancybox=True, fontsize=10.5, handlelength=1.2, bbox_to_anchor=(0.98, 0.12))
for t in legB.get_texts(): t.set_fontweight("bold")
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_shuffle_diagnostics_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_shuffle_diagnostics_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_shuffle_diagnostics_v2.png'}")
print("Done.")
plt.close(fig)