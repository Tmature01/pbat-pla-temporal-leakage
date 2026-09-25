"""
Figure 3 (复合): 集体失效 — 所有方法、所有协议
(a) 七方法轨迹 (b) 协议矩阵 (c) 预测 vs 真实散点
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df_traj   = pd.read_csv(str(FIGURE_DATA / "Fig_Trajectories_All.csv"))
mat       = pd.read_csv(str(FIGURE_DATA / "Fig_ProtocolMatrix.csv"))
df_pred   = pd.read_csv(str(FIGURE_DATA / "Fig_PredVsTrue_Temporal.csv"))
met       = pd.read_csv(str(FIGURE_DATA / "Fig_PredVsTrue_Metrics.csv"))

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

fig = plt.figure(figsize=(17, 13))

# (a) 顶部全宽, (b)(c) 底部左右各半
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2],
                      hspace=0.30, left=0.04, right=0.97, top=0.96, bottom=0.05)
gs_top = gs[0].subgridspec(1, 1)
gs_bot = gs[1].subgridspec(1, 2, wspace=0.28)

axA = fig.add_subplot(gs_top[0])

# ══════════════════════════════════════════════════════
# PANEL (a): 七方法轨迹叠加
# ══════════════════════════════════════════════════════
train = df_traj[df_traj["Region"] == "Train"]
test  = df_traj[df_traj["Region"] == "Test"]
methods_m = ["LS_MPR", "Ridge", "Lasso", "SVR", "RF", "GPR"]
colors_m = {"LS_MPR":"#E74C3C","Ridge":"#E67E22","Lasso":"#2ECC71","SVR":"#9B59B6","RF":"#3498DB","GPR":"#1ABC9C"}

axA.axvspan(0, 119, facecolor="#EAECEE", alpha=0.40, linewidth=0, zorder=0)
axA.axvline(x=119, color="black", linewidth=1.0, linestyle=(0, (5, 4)), alpha=0.5, zorder=2)
axA.text(59, df_traj["Observed_CO2"].max()*0.97, "Training (Days 0-119)", ha="center", fontsize=8, fontweight="bold", color="#7F8C8D")
axA.text(149, df_traj["Observed_CO2"].max()*0.97, "Test (120-180)", ha="center", fontsize=8, fontweight="bold", color="#2C3E50")

axA.plot(df_traj["Day"], df_traj["Observed_CO2"], color="black", linewidth=1.8, zorder=3)
for m in methods_m:
    axA.plot(test["Day"], test[m], color=colors_m[m], linewidth=1.0, alpha=0.80, zorder=4)
axA.plot(test["Day"], test["Persistence"], color="#2C3E50", linewidth=1.3, linestyle="--", alpha=0.80, zorder=4)

# 图例 (右侧)
leg_els_a = [Line2D([0],[0], color="black", linewidth=2.0, label="Observed"),
             Line2D([0],[0], color="#2C3E50", linewidth=1.5, linestyle="--", label="Persistence")]
for m in methods_m:
    leg_els_a.append(Line2D([0],[0], color=colors_m[m], linewidth=1.8, label=m.replace("_","-")))
legA = axA.legend(handles=leg_els_a, loc="center left", framealpha=0.90, edgecolor="#BDC3C7",
                  fancybox=True, fontsize=6.8, ncol=1, bbox_to_anchor=(1.01, 0.5))
for t in legA.get_texts(): t.set_fontweight("bold")

axA.set_title("(a) All 7 methods diverge in the test region", pad=6)
axA.set_xlabel("Days"); axA.set_ylabel("CO$_2$ (g)")
axA.set_xlim(-3, 183)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# PANEL (b): 协议矩阵热图
# ══════════════════════════════════════════════════════
gs_bot_inner = gs_bot[0].subgridspec(1, 1)
axB = fig.add_subplot(gs_bot_inner[0])

methods_p = ["Persistence","LS-MPR","Ridge-MPR","Lasso-MPR","SVR-RBF",
             "RF","GPR-RBF","Exp Decay","Mean Predictor"]
protocols = ["Random Split","Temporal Holdout","LOCO (mean)","Temporal Shuffle"]
p_short = ["Random\nSplit","Temporal\nHoldout","LOCO\n(mean)","Temporal\nShuffle"]

r2_mat = np.full((9,4), np.nan)
rho_mat = np.full((9,4), np.nan)
for i,m in enumerate(methods_p):
    for j,p in enumerate(protocols):
        r2c = f"{p}_R2"; rhoc = f"{p}_rho"
        row = mat[mat["Method"]==m]
        if r2c in mat.columns and pd.notna(row[r2c].values[0]):
            r2_mat[i,j] = row[r2c].values[0]
        if rhoc in mat.columns and pd.notna(row[rhoc].values[0]):
            rho_mat[i,j] = row[rhoc].values[0]

cmap = LinearSegmentedColormap.from_list("sci", [
    (0.00,"#8B3A3A"),(0.18,"#C4776B"),(0.30,"#E8C4B8"),(0.42,"#F5EDE8"),
    (0.50,"#F7F7F7"),(0.58,"#E6EEF4"),(0.72,"#A8C8E0"),(0.85,"#5B8DB8"),(1.00,"#2C5F8A")])
norm = Normalize(vmin=-5, vmax=1.0)

for i in range(9):
    for j in range(4):
        r2 = r2_mat[i,j]; rho = rho_mat[i,j]
        x, y = j, 8-i
        if np.isnan(r2):
            axB.add_patch(plt.Rectangle((j-0.47, y-0.47), 0.94, 0.94, facecolor="#F2F3F5", edgecolor="#D5D8DC", linewidth=0.3))
            axB.text(x, y, "-", ha="center", va="center", fontsize=11, fontweight="bold", color="#B0B8C1")
            continue
        fc = cmap(norm(np.clip(r2, -5, 1.0)))
        axB.add_patch(plt.Rectangle((j-0.47, y-0.47), 0.94, 0.94, facecolor=fc, edgecolor="#F7F7F7", linewidth=0.6))
        r2s = f"{r2:+.3f}" if abs(r2)<10 else f"{r2:+.1f}"
        tc = "white" if abs(np.clip(r2,-5,1.0))>3.0 else "#2C3E50"
        axB.text(x, y+0.12, r2s, ha="center", va="center", fontsize=8, fontweight="bold", color=tc)
        if not np.isnan(rho):
            axB.text(x, y-0.22, f"rho={rho:+.2f}", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#7F8C8D")

axB.set_xlim(-0.5,3.5); axB.set_ylim(-0.5,8.5)
axB.set_xticks(range(4)); axB.set_xticklabels(p_short, fontsize=8)
axB.set_yticks(range(9)[::-1]); axB.set_yticklabels(methods_p, fontsize=8)
axB.tick_params(length=0); axB.xaxis.tick_top()
axB.set_title("(b) 9 methods x 4 protocols", pad=6)

cbar_ax = fig.add_axes([0.45, 0.30, 0.012, 0.15])
sm = ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("$R^2$", fontweight="bold", fontsize=8, labelpad=3)
cb.ax.tick_params(length=0); cb.set_ticks([-5,-2,0,0.5,1.0])

# ══════════════════════════════════════════════════════
# PANEL (c): 预测 vs 真实 2x2
# ══════════════════════════════════════════════════════
gs_c = gs_bot[1].subgridspec(2, 2, hspace=0.28, wspace=0.22)
models_c = [("Persistence_Pred","Persistence"),("LSMPR_Pred","LS-MPR"),
            ("GPR_Pred","GPR"),("ExpDecay_Pred","Exp Decay")]
c_colors = {"C1 (Pure PBAT)":"#E74C3C","C2 (Pure PLA)":"#3498DB","C3 (70/30 PBAT/PLA)":"#2ECC71"}
c_markers = {"C1 (Pure PBAT)":"o","C2 (Pure PLA)":"^","C3 (70/30 PBAT/PLA)":"s"}

lim_min, lim_max = 185, 310
for idx, (col, model) in enumerate(models_c):
    ax = fig.add_subplot(gs_c[idx//2, idx%2])
    r2 = met[met["Model"]==model]["R2"].values[0]
    rmse = met[met["Model"]==model]["RMSE"].values[0]; mae = met[met["Model"]==model]["MAE"].values[0]
    for cond in c_colors:
        sub = df_pred[df_pred["Condition"]==cond]
        ax.scatter(sub["CO2_True"], sub[col], s=12, color=c_colors[cond], marker=c_markers[cond], alpha=0.55, edgecolors="none", rasterized=True, zorder=3)
    ax.plot([lim_min,lim_max],[lim_min,lim_max], color="gray", linewidth=0.5, linestyle="--", alpha=0.4, zorder=1)
    ax.text(0.95, 0.06, f"$R^2$={r2:+.3f}\nRMSE={rmse:.1f}\nMAE={mae:.1f}", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", color="#2C3E50", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#BDC3C7", alpha=0.88))
    ax.set_title(model, pad=4, fontsize=9)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.set_xlim(lim_min,lim_max); ax.set_ylim(lim_min,lim_max)
    ax.set_aspect("equal")
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False, labelsize=6.5)

# (c) 图例
legC = fig.legend(handles=[
    Line2D([0],[0], marker="o", color="#E74C3C", lw=0, markersize=7, label="C1"),
    Line2D([0],[0], marker="^", color="#3498DB", lw=0, markersize=7, label="C2"),
    Line2D([0],[0], marker="s", color="#2ECC71", lw=0, markersize=7, label="C3"),
], loc="upper center", ncol=3, framealpha=0.90, edgecolor="#BDC3C7", fancybox=True, fontsize=8,
    bbox_to_anchor=(0.73, 0.95))
for t in legC.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.01,
         "Figure 3. Collective failure across all methods and protocols. (a) All 7 methods diverge in the temporal test region — "
         "a protocol-level phenomenon. (b) Column 1 (Random) = all converge; Columns 2-4 = massive divergence. "
         "(c) Only Persistence tracks y=x; all data-driven models systematically deviate.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_03_Collective_Failure.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_03_Collective_Failure.svg")
print(f"Saved: {OUT_DIR / 'Fig_03_Collective_Failure.png'}")
print(f"Saved: {OUT_DIR / 'Fig_03_Collective_Failure.svg'}")
print("Done.")
plt.close(fig)