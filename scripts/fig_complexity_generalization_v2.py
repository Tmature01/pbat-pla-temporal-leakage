"""
Figure V2: 模型复杂度 vs 泛化性能 (画布不变 10.5x6.5, 文字x1.5, 无底注)
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

df = pd.read_csv(str(FIGURE_DATA / "Fig_Complexity_Generalization.csv"))
df = df.dropna(subset=["n_params"])
print(df[["Method","n_params","R2_Temporal"]].to_string(index=False))

colors = {
    "Persistence":"#2C3E50","Exp Decay":"#8E44AD","GPR-RBF":"#1ABC9C",
    "Lasso-MPR":"#2ECC71","SVR-RBF":"#E67E22","LS-MPR":"#E74C3C",
    "Ridge-MPR":"#3498DB","RF":"#27AE60",
}

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial"],
    "font.size":13.5,"font.weight":"bold",
    "axes.labelsize":16.5,"axes.labelweight":"bold",
    "axes.titlesize":18,"axes.titleweight":"bold",
    "xtick.labelsize":13.5,"ytick.labelsize":13.5,
    "figure.dpi":300,"savefig.dpi":600,
    "savefig.bbox":"tight","savefig.pad_inches":0.08,
})

fig, ax = plt.subplots(figsize=(10.5, 6.5))
fig.subplots_adjust(left=0.13,right=0.94,top=0.90,bottom=0.08)

persist_r2 = df[df["Method"]=="Persistence"]["R2_Temporal"].values[0]
ax.axhline(y=persist_r2,color="#2C3E50",linewidth=0.8,linestyle="--",alpha=0.45,zorder=0)
ax.text(120000,-5.5,f"Persistence baseline (R$^2$={persist_r2:+.3f})",
        fontsize=11,fontweight="bold",color="#2C3E50",ha="right",va="bottom",alpha=0.65)
ax.axhline(y=0,color="black",linewidth=0.5,alpha=0.25,zorder=0)

offsets = {
    "Persistence":(45,-18),"Exp Decay":(45,-15),"GPR-RBF":(55,-15),
    "Lasso-MPR":(-18,-15),"SVR-RBF":(90,-18),"LS-MPR":(40,-15),
    "Ridge-MPR":(65,-30),"RF":(-110,-8),
}

for _,row in df.iterrows():
    m=row["Method"]; x=row["n_params"]; y=row["R2_Temporal"]
    ax.scatter(x,y,s=120,color=colors[m],edgecolors="white",linewidth=1.2,zorder=4)
    dx,dy=offsets.get(m,(15,10))
    ax.annotate(row["Label"],xy=(x,y),xytext=(dx,dy),
                textcoords="offset points",fontsize=10,fontweight="bold",
                color=colors[m],ha="center",
                arrowprops=dict(arrowstyle="->",color=colors[m],lw=0.7),
                bbox=dict(boxstyle="round,pad=0.15",fc="white",ec=colors[m],alpha=0.85),zorder=5)

ax.set_xlabel("Number of parameters (log scale)")
ax.set_ylabel("$R^2$ (temporal holdout)")
ax.set_xscale("log"); ax.set_xlim(0.8,150000); ax.set_ylim(-6.5,1.15)
ax.tick_params(which="both",direction="in",bottom=True,top=False,left=True,right=False)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xticks([1,10,100,1000,10000,100000])
ax.set_xticklabels(["1","10","100","1k","10k","100k"])

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT_DIR / "fig_complexity_generalization_v2.png",dpi=600)
print(f"Saved: {OUT_DIR / 'fig_complexity_generalization_v2.png'}")
print("Done.")
plt.close(fig)