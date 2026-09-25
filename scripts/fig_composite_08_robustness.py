"""
Figure 8 (复合): 稳健性与跨领域证据
(a) 种子敏感性 (b) 跨领域对比 (c) 反事实窗口
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

seed_df = pd.read_csv(str(FIGURE_DATA / "Fig_SeedSensitivity.csv"))
seed_summ = pd.read_csv(str(FIGURE_DATA / "Fig_SeedSensitivity_Summary.csv"))
raw_df = pd.read_csv(str(_REPO_ROOT / "data" / "pbat_pla_experimental_543.csv"), header=None, skiprows=1)
co2_cols = {"C1":6, "C2":17, "C3":26}
obs_full = {c: raw_df.iloc[:, col].values for c, col in co2_cols.items()}
days_full = raw_df.iloc[:, 0].values

TEMP = Path(str(FIGURE_DATA))
configs_af = [
    ("Fig_Antifactual_Growth.csv","Rapid growth",0,29),
    ("Fig_Antifactual_Growth+Decay.csv","Growth+decay",0,59),
    ("Fig_Antifactual_Mixed.csv","Mixed",0,89),
    ("Fig_Antifactual_Plateau.csv","Plateau",0,119),
]
test_dfs_af = {f: pd.read_csv(TEMP/f) for f,*_ in configs_af}

seeds = seed_df["Seed"].values
r2_rand = seed_df["R2_Random"].values
r2_temp = seed_df["R2_Temporal"].values
mean_r = seed_summ.loc[seed_summ["Metric"]=="Mean","Random_Split"].values[0]
std_r = seed_summ.loc[seed_summ["Metric"]=="Std","Random_Split"].values[0]
seed42 = seed_summ.loc[seed_summ["Metric"]=="Seed42_Value","Random_Split"].values[0]

cross_data = [
    ("This study\n(PBAT/PLA)","Polymer Sci.","R^2 inflation",96.5,True),
    ("Soil carbon\nspatial [21]","Soil Sci.","R^2 inflation",88.1,False),
    ("Sequential\nrecommend. [23]","RecSys/ML","Literature prev.",77.0,False),
    ("Material ML\nbenchmarks","Materials Sci.","Literature prev.",65.0,False),
    ("Electrochemical\npred. [22]","Electrochem.","RMSE inflation",35.0,False),
    ("LSTM water\nquality [32]","Environ. Eng.","RMSE inflation",20.5,False),
]

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

fig = plt.figure(figsize=(17, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.0],
                      hspace=0.32, wspace=0.26,
                      left=0.05, right=0.97, top=0.94, bottom=0.07)

axA = fig.add_subplot(gs[0,0])
axB = fig.add_subplot(gs[0,1])

# (c) 占下半部全宽: 4 个子图
gs_c = gs[1].subgridspec(1, 4, wspace=0.25)
axC_list = [fig.add_subplot(gs_c[i]) for i in range(4)]

# ══════════════════════════════════════════════════════
# PANEL (a): 种子敏感性 (双面板)
# ══════════════════════════════════════════════════════
gs_a = gs[0,0].subgridspec(2, 1, hspace=0.20)
axA1 = fig.add_subplot(gs_a[0])
axA2 = fig.add_subplot(gs_a[1])

axA1.axhspan(mean_r-std_r, mean_r+std_r, facecolor="#3498DB", alpha=0.12, linewidth=0)
axA1.axhline(y=mean_r, color="#3498DB", linewidth=1.0, linestyle="--", alpha=0.6)
for i in range(len(seeds)):
    is42 = seeds[i]==42
    axA1.scatter(seeds[i], r2_rand[i], s=50 if is42 else 18,
                color="#C0392B" if is42 else "#34495E", marker="*" if is42 else "o",
                edgecolors="white", linewidth=0.5, zorder=5 if is42 else 3)
axA1.text(42, seed42+0.002, f"Seed=42\n{seed42:.4f}", fontsize=6.5, fontweight="bold", color="#C0392B", ha="center")
axA1.set_title("Random split: 50 seeds", pad=4, fontsize=9)
axA1.set_ylabel("$R^2$"); axA1.set_xlim(-2,51); axA1.set_ylim(0.988, 0.998)
axA1.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA1.text(0.95, 0.08, f"Mean={mean_r:.4f}  SD={std_r:.4f}", transform=axA1.transAxes, fontsize=6.5, fontweight="bold", ha="right", color="#2C3E50",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#BDC3C7", alpha=0.85))
axA1.set_xticklabels([])

axA2.scatter(seeds, r2_temp, s=18, color="#34495E", edgecolors="white", linewidth=0.5, zorder=3)
axA2.axhline(y=r2_temp[0], color="#3498DB", linewidth=1.0, linestyle="--", alpha=0.6)
axA2.set_title("Temporal holdout: 50 seeds", pad=4, fontsize=9)
axA2.set_xlabel("Random seed"); axA2.set_ylabel("$R^2$")
axA2.set_xlim(-2,51); axA2.set_ylim(0.068, 0.076)
axA2.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA2.text(0.95, 0.08, "Deterministic (all = 0.0721)", transform=axA2.transAxes, fontsize=6.5, fontweight="bold", ha="right", color="#2C3E50",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#BDC3C7", alpha=0.85))

# ══════════════════════════════════════════════════════
# PANEL (b): 跨领域对比
# ══════════════════════════════════════════════════════
values_b = [d[3] for d in cross_data][::-1]
labels_b = [d[0] for d in cross_data][::-1]
mtypes_b = [d[2] for d in cross_data][::-1]
hls_b = [d[4] for d in cross_data][::-1]

y_b = np.arange(6)
colors_b = []
for h, mt in zip(hls_b, mtypes_b):
    if h: colors_b.append("#C0392B")
    elif "RMSE" in mt: colors_b.append("#A8C8E0")
    else: colors_b.append("#5B8DB8")

axB.barh(y_b, values_b, height=0.55, color=colors_b, edgecolor="white", linewidth=1.0, zorder=3)
for i,(v,h) in enumerate(zip(values_b, hls_b)):
    axB.text(v+1.0, i, f"{v:.1f}%", fontsize=9, fontweight="bold", color="#C0392B" if h else "#2C3E50", va="center")
axB.set_yticks(y_b); axB.set_yticklabels(labels_b, fontsize=7.5)
axB.set_xlabel("Performance inflation (%)"); axB.set_xlim(0, 130)
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=False, right=False)
axB.invert_yaxis()
axB.set_title("(b) Cross-discipline inflation", pad=6, fontsize=9)

legB = axB.legend(handles=[
    Patch(facecolor="#C0392B", label="This study (R^2)"),
    Patch(facecolor="#5B8DB8", label="Other (R^2/lit.)"),
    Patch(facecolor="#A8C8E0", label="Other (RMSE)"),
], loc="upper right", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=6.5, bbox_to_anchor=(0.98, 0.45))
for t in legB.get_texts(): t.set_fontweight("bold")

# ══════════════════════════════════════════════════════
# PANEL (c): 反事实四窗口
# ══════════════════════════════════════════════════════
COLOR_OBS = "#2C3E50"; COLOR_LS = "#C0392B"; COLOR_PERS = "#2471A3"
C_TRAIN_BG = "#E8F0FE"; C_TEST_BG = "#FEF9E7"

for idx, (ax, (fname, label, t_start, t_end)) in enumerate(zip(axC_list, configs_af)):
    df_test = test_dfs_af[fname]
    train_end = int(df_test["Train_End"].iloc[0])
    test_end = int(df_test["Day"].max())

    ax.axvspan(0, train_end, facecolor=C_TRAIN_BG, alpha=0.30, linewidth=0, zorder=0)
    ax.axvspan(train_end, test_end+1, facecolor=C_TEST_BG, alpha=0.22, linewidth=0, zorder=0)

    train_mask = days_full <= train_end
    ax.plot(days_full[train_mask], obs_full["C1"][train_mask], color=COLOR_OBS, linewidth=0.8, alpha=0.7, zorder=2)
    for cond in ["C1","C2","C3"]:
        cdf = df_test[df_test["Condition"]==cond]
        lw, al = (1.2,0.85) if cond=="C1" else (0.4,0.25)
        ax.plot(cdf["Day"], cdf["Observed_CO2"], color=COLOR_OBS, linewidth=lw, alpha=al, zorder=2)

    for cond in ["C1","C2","C3"]:
        cdf = df_test[df_test["Condition"]==cond]
        lw, al = (1.4,0.88) if cond=="C1" else (0.5,0.25)
        ax.plot(cdf["Day"], cdf["LSMPR_Predicted"], color=COLOR_LS, linewidth=lw, alpha=al, zorder=4)
        ax.plot(cdf["Day"], cdf["Persistence_Predicted"], color=COLOR_PERS, linewidth=lw, alpha=al, zorder=4)

    ax.axvline(x=train_end, color="black", linewidth=0.7, linestyle=(0,(5,3)), alpha=0.5, zorder=2)
    ax.set_title(label, pad=3, fontsize=8)
    ax.set_xlabel("Days"); ax.set_ylabel("CO$_2$ (g)")
    ax.set_xlim(-3, test_end+3)
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False, labelsize=6)

# (c) 图例
leg_c = fig.legend(handles=[
    Line2D([0],[0], color=COLOR_OBS, linewidth=1.5, label="Observed"),
    Line2D([0],[0], color=COLOR_LS, linewidth=1.5, label="LS-MPR"),
    Line2D([0],[0], color=COLOR_PERS, linewidth=1.5, label="Persistence"),
], loc="upper center", ncol=3, framealpha=0.90, edgecolor="#BDC3C7", fancybox=True, fontsize=8,
    bbox_to_anchor=(0.5, 0.42))
for t in leg_c.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.015,
         "Figure 8. Robustness and cross-disciplinary evidence. (a) Seed choice is irrelevant (SD=0.0011 across 50 seeds). "
         "(b) 6 studies across 5 disciplines — this is not an isolated finding. "
         "(c) Persistence advantage is phase-conditional: it fails in growth (panel 1) but excels in plateau (panel 4).",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_08_Robustness_CrossDiscipline.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_08_Robustness_CrossDiscipline.svg")
print(f"Saved: {OUT_DIR / 'Fig_08_Robustness_CrossDiscipline.png'}")
print(f"Saved: {OUT_DIR / 'Fig_08_Robustness_CrossDiscipline.svg'}")
print("Done.")
plt.close(fig)