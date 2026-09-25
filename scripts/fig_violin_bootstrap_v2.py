"""
Figure V2: Bootstrap R² 分布对比 (画布不变 13x5.5, 文字x1.5, 无底注)
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

df = pd.read_csv(str(FIGURE_DATA / "Fig_Violin_Bootstrap.csv"))
methods = ["LS-MPR", "Ridge", "SVR", "RF"]
colors_m = {"LS-MPR":"#E74C3C","Ridge":"#3498DB","SVR":"#E67E22","RF":"#2ECC71"}

for m in methods:
    for p in ["Random","Temporal"]:
        sub = df[(df.Method==m)&(df.Protocol==p)]
        if len(sub)>0:
            print(f"{m:8s} {p:10s}: median={sub.R2.median():.4f}")

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial"],
    "font.size":13.5,"font.weight":"bold",
    "axes.labelsize":16.5,"axes.labelweight":"bold",
    "axes.titlesize":18,"axes.titleweight":"bold",
    "xtick.labelsize":15,"ytick.labelsize":13.5,
    "figure.dpi":300,"savefig.dpi":600,
    "savefig.bbox":"tight","savefig.pad_inches":0.08,
})

fig, (axL, axR) = plt.subplots(1,2,figsize=(13,5.5))
fig.subplots_adjust(wspace=0.14,bottom=0.06,top=0.90,left=0.09,right=0.96)

def plot_violins(ax, protocol, ylim, title):
    data_list = []; positions = []; method_list = []
    for i,m in enumerate(methods):
        sub = df[(df.Method==m)&(df.Protocol==protocol)]
        if len(sub)==0: continue
        data_list.append(sub["R2"].values)
        positions.append(i)
        method_list.append(m)

    vp = ax.violinplot(data_list,positions=positions,showmeans=False,showmedians=True,widths=0.60)
    for i,body in enumerate(vp["bodies"]):
        m=method_list[i]
        body.set_facecolor(colors_m[m]); body.set_alpha(0.35)
        body.set_edgecolor(colors_m[m]); body.set_linewidth(1.4)
    for part in ["cbars","cmins","cmaxes"]:
        if part in vp: vp[part].set_color("#7F8C8D"); vp[part].set_linewidth(1.0)
    if "cmedians" in vp: vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2.0)

    for i,(m,d) in enumerate(zip(method_list,data_list)):
        med = np.median(d)
        # 高位值下移避免出界
        if med > 0.98:
            med_text = med - 0.04
        elif med < -3.0:
            med_text = med + 0.3
        else:
            med_text = med
        # RF+1.00 左移
        xoff = 0.05 if med > 0.98 else 0.35
        ax.annotate(f"{med:+.2f}",xy=(i+xoff,med_text),fontsize=11,fontweight="bold",
                    color=colors_m[m],va="center",ha="left")

    if protocol == "Temporal":
        ax.axhline(y=-4.99,color="#2C3E50",linewidth=1.0,linestyle="--",alpha=0.45)
        ax.text(len(method_list)-1.3,-4.99,"Persistence R² = -4.99",fontsize=10.5,
                fontweight="bold",color="#2C3E50",va="bottom",ha="right",alpha=0.6)

    ax.set_title(title,pad=10)
    ax.set_ylabel("$R^2$")
    ax.set_xticks(range(len(method_list))); ax.set_xticklabels(method_list)
    ax.set_ylim(ylim)
    ax.axhline(y=0,color="black",linewidth=0.6,alpha=0.25,zorder=0)
    ax.tick_params(which="both",direction="in",bottom=True,top=False,left=True,right=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

plot_violins(axL,"Random",(-0.05,1.08),"(a) Random split bootstrap")
plot_violins(axR,"Temporal",(-5.5,1.05),"(b) Temporal holdout bootstrap")

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT_DIR / "fig_violin_bootstrap_v2.png",dpi=600)
print(f"Saved: {OUT_DIR / 'fig_violin_bootstrap_v2.png'}")
print("Done.")
plt.close(fig)