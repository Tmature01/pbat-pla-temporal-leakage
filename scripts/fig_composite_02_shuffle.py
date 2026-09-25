"""
Figure 2 (复合): 96% 崩塌 —— Shuffle 统计证据
(a) 小提琴图 (b) 直方图+Null (c) 洗牌后"学到"的函数
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取所有数据 ──────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

violin_df = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Violin.csv"))
reflines  = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_RefLines.csv"))
kde_df    = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_KDE.csv"))
scatter   = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_Scatter.csv"))
curves    = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_Curves.csv"))
orig_ts   = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Learned_OriginalTS.csv"))

lsmpr_data = violin_df[violin_df["Model"] == "LS-MPR"]["R2_Shuffle"].values
exp_data   = violin_df[violin_df["Model"] == "Exp Decay"]["R2_Shuffle"].values

r2_shuf = r2_score(scatter["CO2_Observed"],
                   np.interp(scatter["Day_Shuffled"], curves["Day"], curves["Pred_Shuffled"]))

# ── 2. 全局样式 ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "font.weight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold",
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0],
                      hspace=0.28, wspace=0.30,
                      left=0.05, right=0.97, top=0.95, bottom=0.06)

# (a) 小提琴: 占上半部左3列  (b) 直方图: 占上半部右1列
axA = fig.add_subplot(gs[0, 0:3])  # 小提琴宽面板
axB = fig.add_subplot(gs[0, 3])    # 直方图窄面板
# (c) 洗牌函数: 占下半部全宽, 2 列
axC1 = fig.add_subplot(gs[1, 0:2])
axC2 = fig.add_subplot(gs[1, 2:4])

C_LS  = "#E74C3C"
C_EXP = "#3498DB"
C_NULL = "#95A5A6"

# ══════════════════════════════════════════════════════
# PANEL (a): 小提琴图 + 散点
# ══════════════════════════════════════════════════════
positions = [0, 1]
data_sets = [lsmpr_data, exp_data]

vp = axA.violinplot(data_sets, positions=positions, showmeans=False,
                     showmedians=True, widths=0.55)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor([C_LS, C_EXP][i])
    body.set_alpha(0.30)
    body.set_edgecolor([C_LS, C_EXP][i])
    body.set_linewidth(1.2)
for part in ["cbars", "cmins", "cmaxes"]:
    vp[part].set_color("#7F8C8D"); vp[part].set_linewidth(0.6)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.8)

np.random.seed(42)
for i, (d, c) in enumerate(zip(data_sets, [C_LS, C_EXP])):
    jitter = np.random.uniform(-0.12, 0.12, size=len(d))
    axA.scatter(np.full_like(d, positions[i]) + jitter, d, s=14, color=c, alpha=0.45, zorder=4)

# 参考线
for i, model_name in enumerate(["LS-MPR", "Exp Decay"]):
    orig = reflines[(reflines["Model"]==model_name)&(reflines["Line"]=="Original")]["R2"].values[0]
    smean = reflines[(reflines["Model"]==model_name)&(reflines["Line"]=="Shuffle_mean")]["R2"].values[0]
    axA.hlines(orig, positions[i]-0.28, positions[i]+0.28, colors=C_LS, linewidth=1.2, linestyles="dashed", alpha=0.6)
    axA.hlines(smean, positions[i]-0.28, positions[i]+0.28, colors=C_EXP, linewidth=1.2, linestyles="dashed", alpha=0.6)

axA.text(0, 0.10, "Cohen's d = 57.0\np = 1.8e-52", fontsize=7.5, fontweight="bold", color="#2C3E50", ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.85))
axA.set_title("(a) 30-shuffle R^2 distribution", pad=6)
axA.set_ylabel("$R^2$ (shuffled)")
axA.set_xticks([0,1]); axA.set_xticklabels(["LS-MPR", "Exp Decay"])
axA.set_ylim(-0.10, 1.05)
axA.axhline(y=0, color="black", linewidth=0.5, alpha=0.3, zorder=0)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

leg_els = [
    Line2D([0],[0], color=C_LS, linewidth=1.2, linestyle="dashed", label="Original R^2"),
    Line2D([0],[0], color=C_EXP, linewidth=1.2, linestyle="dashed", label="Shuffle mean"),
]
legA = axA.legend(handles=leg_els, loc="upper right", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=7, handlelength=1.5)
for t in legA.get_texts(): t.set_fontweight("bold")

# ══════════════════════════════════════════════════════
# PANEL (b): 直方图 + KDE + Null
# ══════════════════════════════════════════════════════
axB.hist(lsmpr_data, bins=9, color=C_LS, alpha=0.35, edgecolor=C_LS, linewidth=0.6, density=True, zorder=2)
axB.plot(kde_df["R2"], kde_df["LSMPR_KDE"], color=C_LS, linewidth=1.8, zorder=4)
axB.fill_between(kde_df["R2"], 0, kde_df["Null_PDF"], facecolor=C_NULL, alpha=0.30, linewidth=0, zorder=1)
axB.plot(kde_df["R2"], kde_df["Null_PDF"], color=C_NULL, linewidth=1.0, linestyle="--", zorder=3)

mean_val = np.mean(lsmpr_data)
std_val = np.std(lsmpr_data, ddof=1)
axB.text(0.95, 0.95, f"Mean={mean_val:+.3f}\nSD={std_val:.3f}\nCI:[{-0.051:+.3f},{+0.094:+.3f}]\nKS: D=0.31, p=0.006",
         transform=axB.transAxes, fontsize=6.5, fontweight="bold", color="#2C3E50", ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.85))
axB.axvline(x=mean_val, color=C_LS, linewidth=0.6, linestyle="--", alpha=0.5)
axB.axvline(x=0, color="black", linewidth=0.4, alpha=0.25)
axB.set_title("(b) LS-MPR shuffle R^2\nhistogram + null", pad=6)
axB.set_xlabel("$R^2$"); axB.set_ylabel("Density")
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# PANEL (c): 洗牌后"学到"的函数
# ══════════════════════════════════════════════════════
axC1.scatter(orig_ts["Days"], orig_ts["CO2"], s=10, color="#34495E", alpha=0.50, edgecolors="none", rasterized=True, zorder=2)
axC1.plot(curves["Day"], curves["Pred_Original"], color="#C0392B", linewidth=1.8, zorder=3)
axC1.set_title("Original data  |  LS-MPR fit", pad=5)
axC1.set_xlabel("Days"); axC1.set_ylabel("CO$_2$ (g)")
axC1.text(0.95, 0.06, "$R^2 = 0.995$\n(monotonic saturation curve)", transform=axC1.transAxes,
         fontsize=7.5, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.85))
axC1.set_xlim(-5, 185)
axC1.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

axC2.scatter(scatter["Day_Shuffled"], scatter["CO2_Observed"], s=10, color="#7F8C8D", alpha=0.40, edgecolors="none", rasterized=True, zorder=2)
axC2.plot(curves["Day"], curves["Pred_Shuffled"], color="#E74C3C", linewidth=1.8, zorder=3)
axC2.scatter(orig_ts["Days"], orig_ts["CO2"], s=6, color="#3498DB", alpha=0.25, edgecolors="none", rasterized=True, zorder=1)
axC2.set_title("Shuffled data  |  LS-MPR 'learned' function", pad=5)
axC2.set_xlabel("Day (shuffled)"); axC2.set_ylabel("CO$_2$ (g)")
axC2.text(0.95, 0.06, f"$R^2 = {r2_shuf:.3f}$\n(near-constant, no kinetics learned)", transform=axC2.transAxes,
         fontsize=7.5, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.85))
axC2.set_xlim(-5, 185)
axC2.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# (c) 内图例
legC = axC2.legend(handles=[
    Line2D([0],[0], marker="o", color="#34495E", lw=0, markersize=6, label="Original obs"),
    Line2D([0],[0], marker="o", color="#7F8C8D", lw=0, markersize=6, label="Shuffled obs"),
    Line2D([0],[0], color="#C0392B", linewidth=1.8, label="LS-MPR fit"),
], loc="upper left", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=7)
for t in legC.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.01,
         "Figure 2. 96% collapse under shuffle — statistical evidence. (a) LS-MPR R^2 collapses to ~0, Exp Decay is immune. "
         "(b) Shuffle R^2 distribution is indistinguishable from random guessing. "
         "(c) LS-MPR 'learns' a near-constant function — temporal proximity, not kinetics, drove the original R^2=0.995.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_02_Shuffle_Collapse.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_02_Shuffle_Collapse.svg")
print(f"Saved: {OUT_DIR / 'Fig_02_Shuffle_Collapse.png'}")
print(f"Saved: {OUT_DIR / 'Fig_02_Shuffle_Collapse.svg'}")
print("Done.")
plt.close(fig)