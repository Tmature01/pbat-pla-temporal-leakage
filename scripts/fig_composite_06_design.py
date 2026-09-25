"""
Figure 6 (复合): 问题在实验设计，不在方法
(a) 真实消融 (b) 合成消融 (c) 复杂度 (d) 学习曲线
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

abl  = pd.read_csv(str(FIGURE_DATA / "Fig_Ablation_Comparison.csv"))
comp = pd.read_csv(str(FIGURE_DATA / "Fig_Complexity_Generalization.csv"))
comp = comp.dropna(subset=["n_params"])

with open(str(FIGURE_DATA / "Fig_LearningCurve.csv"),"r") as f:
    lines = f.readlines()
col_names = ["TrainRatio"] + [l.strip().strip(",").strip('"') for l in lines[1:10]]
data_rows = []
for line in lines[10:]:
    parts = line.strip().split(",")
    if len(parts) >= 10: data_rows.append([float(x) for x in parts[:10]])
lc_ratios = np.array([r[0]*100 for r in data_rows])
lc_data = {col_names[i+1]: np.array([r[i+1] for r in data_rows]) for i in range(9)}

ridge_cols = ["Ridge (alpha=0.01)","Ridge (alpha=0.1)","Ridge (alpha=1.0)"]
svr_cols = ["SVR (C=1)","SVR (C=10)","SVR (C=100)"]
rf_cols = ["RF (100 trees)","RF (200 trees)","RF (500 trees)"]

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
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25,
                      left=0.06, right=0.96, top=0.94, bottom=0.07)

axA = fig.add_subplot(gs[0,0])
axB = fig.add_subplot(gs[0,1])
axC = fig.add_subplot(gs[1,0])
axD = fig.add_subplot(gs[1,1])

C_DAYS = "#C0392B"; C_ENV = "#3498DB"; C_SYN = "#E67E22"
features = abl["Feature"].values
x = np.arange(5); bar_w = 0.55

# ══════════════════════════════════════════════════════
# PANEL (a): 真实数据消融
# ══════════════════════════════════════════════════════
colors_a = [C_DAYS if f=="Days" else C_ENV for f in features]
axA.bar(x, abl["Real_Delta_R2"].values, bar_w, color=colors_a, edgecolor="white", linewidth=0.5, zorder=3)
for i,(f,v) in enumerate(zip(features, abl["Real_Delta_R2"].values)):
    if abs(v)>0.001:
        axA.text(i, v-0.03, f"{v:+.4f}", ha="center", fontsize=7.5, fontweight="bold", color="white")
    else:
        axA.text(i, v+0.02, "0.000", ha="center", fontsize=7, fontweight="bold", color=C_ENV)
axA.axhline(y=0, color="black", linewidth=0.5, zorder=1)
axA.set_title("(a) Real data: only Days matters", pad=6)
axA.set_ylabel("$\\Delta R^2$")
axA.set_xticks(x); axA.set_xticklabels(["Days","Temp","Humid","Ratio","CV"], rotation=20)
axA.set_ylim(-1.05, 0.08)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# PANEL (b): 合成数据消融
# ══════════════════════════════════════════════════════
colors_b = [C_DAYS if f=="Days" else C_SYN for f in features]
axB.bar(x, abl["Syn_Delta_R2"].values, bar_w, color=colors_b, edgecolor="white", linewidth=0.5, zorder=3)
for i,(f,v) in enumerate(zip(features, abl["Syn_Delta_R2"].values)):
    axB.text(i, v-0.02, f"{v:+.3f}", ha="center", fontsize=7.5, fontweight="bold", color="white" if abs(v)>0.15 else "#2C3E50")
axB.axhline(y=0, color="black", linewidth=0.5, zorder=1)
axB.set_title("(b) Synthetic data: all features contribute", pad=6)
axB.set_ylabel("$\\Delta R^2$")
axB.set_xticks(x); axB.set_xticklabels(["Days","Temp","Humid","Ratio","CV"], rotation=20)
axB.set_ylim(-1.05, 0.08)
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# PANEL (c): 复杂度 vs 泛化
# ══════════════════════════════════════════════════════
colors_cm = {"Persistence":"#2C3E50","Exp Decay":"#8E44AD","GPR-RBF":"#1ABC9C",
             "Lasso-MPR":"#2ECC71","SVR-RBF":"#E67E22","LS-MPR":"#E74C3C","Ridge-MPR":"#3498DB","RF":"#27AE60"}
persist_r2 = comp[comp["Method"]=="Persistence"]["R2_Temporal"].values[0]

axC.axhline(y=persist_r2, color="#2C3E50", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
axC.text(1.2, persist_r2-0.06, f"Persistence baseline (R^2={persist_r2:+.3f})", fontsize=7.5, fontweight="bold", color="#2C3E50", ha="left", va="top", alpha=0.7)
axC.axhline(y=0, color="black", linewidth=0.5, alpha=0.3, zorder=0)

for _, row in comp.iterrows():
    m = row["Method"]; rx = row["n_params"]; ry = row["R2_Temporal"]
    axC.scatter(rx, ry, s=100, color=colors_cm.get(m,"#333"), edgecolors="white", linewidth=1.0, zorder=4)
offsets = {"Persistence":(-45,-20),"Exp Decay":(40,-15),"GPR-RBF":(50,-15),"Lasso-MPR":(-15,-15),
           "SVR-RBF":(85,-18),"LS-MPR":(35,-15),"Ridge-MPR":(60,-30),"RF":(-100,-8)}
for _, row in comp.iterrows():
    m = row["Method"]; dx,dy = offsets.get(m,(15,10))
    axC.annotate(row["Label"], xy=(row["n_params"],row["R2_Temporal"]), xytext=(dx,dy),
                textcoords="offset points", fontsize=6.5, fontweight="bold", color=colors_cm.get(m,"#333"), ha="center",
                arrowprops=dict(arrowstyle="->", color=colors_cm.get(m,"#333"), lw=0.6),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=colors_cm.get(m,"#333"), alpha=0.85), zorder=5)

axC.text(120000, -6.0, f"Persistence baseline (R^2={persist_r2:+.3f})", fontsize=7.5, fontweight="bold", color="#2C3E50", ha="right", va="bottom", alpha=0.7)
axC.set_title("(c) Complexity does not improve\ntemporal generalization", pad=6)
axC.set_xlabel("Number of parameters (log scale)"); axC.set_ylabel("$R^2$ (temporal holdout)")
axC.set_xscale("log"); axC.set_xlim(0.8, 150000); axC.set_ylim(-6.5, 1.15)
axC.set_xticks([1,10,100,1000,10000,100000])
axC.set_xticklabels(["1","10","100","1k","10k","100k"])
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# PANEL (d): 学习曲线
# ══════════════════════════════════════════════════════
lc_colors = {ridge_cols[0]:"#6C8EBF",ridge_cols[1]:"#4A6FA5",ridge_cols[2]:"#2B4C7E",
             svr_cols[0]:"#F0A860",svr_cols[1]:"#E8853B",svr_cols[2]:"#C0601A",
             rf_cols[0]:"#5DAA8C",rf_cols[1]:"#3D8B6E",rf_cols[2]:"#1F6C4F"}
lc_styles = {ridge_cols[0]:"-",ridge_cols[1]:"--",ridge_cols[2]:":",
             svr_cols[0]:"-",svr_cols[1]:"--",svr_cols[2]:":",
             rf_cols[0]:"-",rf_cols[1]:"--",rf_cols[2]:":"}

axD.axhspan(-42, 0, facecolor="#FADBD8", alpha=0.10, linewidth=0, zorder=0)
axD.axhline(y=0, color="black", linewidth=0.6, linestyle="-", alpha=0.3, zorder=1)
for col in ridge_cols+svr_cols+rf_cols:
    axD.plot(lc_ratios, lc_data[col], color=lc_colors[col], linestyle=lc_styles.get(col,"-"),
             linewidth=1.3, marker="o", markersize=3, markerfacecolor="white", markeredgewidth=0.6, markeredgecolor=lc_colors[col], zorder=2)
for yv,clr in [(-0.058,"#2B4C7E"),(-0.203,"#C0601A"),(-0.312,"#1F6C4F")]:
    axD.axhline(y=yv, color=clr, linewidth=0.5, linestyle=(0,(4,6)), alpha=0.5, zorder=1)

axD.set_title("(d) Hyperparameter tuning cannot rescue", pad=6)
axD.set_xlabel("Training ratio (%)"); axD.set_ylabel("$R^2$ (time-series CV)")
axD.set_xlim(28, 92); axD.set_ylim(-42, 8)
axD.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# (d) 紧凑图例
leg_d = axD.legend(handles=[
    Line2D([0],[0], color="none", lw=0, label="Ridge/SVR/RF:"),
    Line2D([0],[0], color="#6C8EBF", lw=1.5, linestyle="-", label="alpha=0.01"),
    Line2D([0],[0], color="#4A6FA5", lw=1.5, linestyle="--", label="0.1"),
    Line2D([0],[0], color="#2B4C7E", lw=1.5, linestyle=":", label="1.0"),
    Line2D([0],[0], color="none", lw=0, label=""),
    Line2D([0],[0], color="#F0A860", lw=1.5, linestyle="-", label="C=1"),
    Line2D([0],[0], color="#E8853B", lw=1.5, linestyle="--", label="10"),
    Line2D([0],[0], color="#C0601A", lw=1.5, linestyle=":", label="100"),
    Line2D([0],[0], color="none", lw=0, label=""),
    Line2D([0],[0], color="#5DAA8C", lw=1.5, linestyle="-", label="100 trees"),
    Line2D([0],[0], color="#3D8B6E", lw=1.5, linestyle="--", label="200"),
    Line2D([0],[0], color="#1F6C4F", lw=1.5, linestyle=":", label="500"),
], loc="upper center", ncol=4, framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=5.8,
    bbox_to_anchor=(0.5, -0.18), borderpad=0.4, columnspacing=0.5)
for t in leg_d.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.01,
         "Figure 6. The problem is experimental design, not method. (a-b) Real data: only Days matters; "
         "Synthetic: all 5 independent features contribute. (c) Persistence (0 params) outperforms models with 90k+ params. "
         "(d) Hyperparameter tuning across 2 orders of magnitude cannot make any method positive.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_06_Design_Not_Method.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_06_Design_Not_Method.svg")
print(f"Saved: {OUT_DIR / 'Fig_06_Design_Not_Method.png'}")
print(f"Saved: {OUT_DIR / 'Fig_06_Design_Not_Method.svg'}")
print("Done.")
plt.close(fig)