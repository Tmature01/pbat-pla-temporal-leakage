"""
Figure V2: 真实 vs 合成消融对比 (画布不变 15x5.2, 文字x1.5, 无底注)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

abl = pd.read_csv(str(FIGURE_DATA / "Fig_Ablation_Comparison.csv"))
cond = pd.read_csv(str(FIGURE_DATA / "Fig_Ablation_ConditionNumber.csv"))

features = abl["Feature"].values
real_dr2 = abl["Real_Delta_R2"].values
syn_dr2  = abl["Syn_Delta_R2"].values

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial"],
    "font.size":13.5,"font.weight":"bold",
    "axes.labelsize":15,"axes.labelweight":"bold",
    "axes.titlesize":16.5,"axes.titleweight":"bold",
    "xtick.labelsize":13.5,"ytick.labelsize":12,
    "figure.dpi":300,"savefig.dpi":600,
    "savefig.bbox":"tight","savefig.pad_inches":0.08,
})

fig,(axA,axB,axC)=plt.subplots(1,3,figsize=(15,5.2))
fig.subplots_adjust(wspace=0.32,bottom=0.08,top=0.88,left=0.07,right=0.97)

C_DAYS="#C0392B";C_ENV="#3498DB";C_SYN="#E67E22"
x=np.arange(5);bar_w=0.55

# ══════════════════════════════════════════════════════
# (a) Real data
# ══════════════════════════════════════════════════════
colors_a = [C_DAYS if f=="Days" else C_ENV for f in features]
axA.bar(x,real_dr2,bar_w,color=colors_a,edgecolor="white",linewidth=0.6,zorder=3)

for i,(f,v) in enumerate(zip(features,real_dr2)):
    if abs(v)>0.001:
        axA.text(i,v-0.03,f"{v:+.4f}",ha="center",fontsize=12,fontweight="bold",
                color="white" if abs(v)>0.5 else C_DAYS)
    else:
        axA.text(i,v+0.02,"0.000",ha="center",fontsize=10.5,fontweight="bold",color=C_ENV)

axA.axhline(y=0,color="black",linewidth=0.6,zorder=1)
axA.set_title("(a) Real data ablation\n(VIF = inf)",pad=10)
axA.set_ylabel("$\\Delta R^2$")
axA.set_xticks(x);axA.set_xticklabels(["Days","Temp","Humid","Ratio","CV"],rotation=20)
axA.set_ylim(-1.05,0.08)
axA.tick_params(which="both",direction="in",bottom=True,top=False,left=True,right=False)

axA.annotate("4 features:\n$\\Delta R^2 = 0$\n(complete collinearity)",
            xy=(2.5,0),xytext=(3.3,-0.40),
            fontsize=11,fontweight="bold",color=C_ENV,ha="center",
            arrowprops=dict(arrowstyle="->",color=C_ENV,lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2",fc="white",ec=C_ENV,alpha=0.85))

# ══════════════════════════════════════════════════════
# (b) Synthetic data
# ══════════════════════════════════════════════════════
colors_b = [C_DAYS if f=="Days" else C_SYN for f in features]
axB.bar(x,syn_dr2,bar_w,color=colors_b,edgecolor="white",linewidth=0.6,zorder=3)

for i,(f,v) in enumerate(zip(features,syn_dr2)):
    axB.text(i,v-0.04,f"{v:+.3f}",ha="center",fontsize=11,fontweight="bold",color="#2C3E50")

axB.axhline(y=0,color="black",linewidth=0.6,zorder=1)
axB.set_title("(b) Synthetic data ablation\n(independent features)",pad=10)
axB.set_ylabel("$\\Delta R^2$")
axB.set_xticks(x);axB.set_xticklabels(["Days","Temp","Humid","Ratio","CV"],rotation=20)
axB.set_ylim(-1.05,0.08)
axB.tick_params(which="both",direction="in",bottom=True,top=False,left=True,right=False)

axB.annotate("All 5 features\ncontribute significantly",
            xy=(1.5,-0.28),xytext=(3.35,-0.42),
            fontsize=11,fontweight="bold",color=C_SYN,ha="center",
            arrowprops=dict(arrowstyle="->",color=C_SYN,lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2",fc="white",ec=C_SYN,alpha=0.85))

# ══════════════════════════════════════════════════════
# (c) Condition number
# ══════════════════════════════════════════════════════
labels_c = cond["Data"].values;log10_c=cond["Log10_Condition"].values
eff_r=cond["Effective_Rank"].values;full_r=cond["Full_Rank"].values

colors_c = ["#C0392B","#27AE60"]
axC.bar([0,1],log10_c,0.5,color=colors_c,edgecolor="white",linewidth=0.6,zorder=3)

for i,(lbl,lv,er,fr) in enumerate(zip(labels_c,log10_c,eff_r,full_r)):
    cn=cond["Condition_Number"].values[i]
    # 科学计数法转论文格式: 2.2e+32 → 2.2×10³²
    cn_exp = int(np.floor(np.log10(cn))) if cn > 0 else 0
    cn_mantissa = cn / 10**cn_exp
    cn_str = f"$\\kappa = {cn_mantissa:.1f} \\times 10^{{{cn_exp}}}$" if cn > 100 else f"$\\kappa = {cn:.1f}$"
    axC.text(i,lv+1.2,
            f"log$_{{{10}}}(\\kappa)$ = {lv:.1f}\n{cn_str}\nRank = {er}/{fr}",
            ha="center",fontsize=11,fontweight="bold",color=colors_c[i],
            bbox=dict(boxstyle="round,pad=0.2",fc="white",ec=colors_c[i],alpha=0.85))

axC.set_title("(c) Design matrix condition",pad=10)
axC.set_ylabel("Log$_{10}$ condition number")
axC.set_xticks([0,1]);axC.set_xticklabels(["Real\n(PBAT/PLA)","Synthetic\n(independent)"])
axC.set_ylim(0,38)
axC.tick_params(which="both",direction="in",bottom=True,top=False,left=True,right=False)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT_DIR / "fig_ablation_comparison_v2.png",dpi=600)
print(f"Saved: {OUT_DIR / 'fig_ablation_comparison_v2.png'}")
print("Done.")
plt.close(fig)