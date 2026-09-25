"""
Figure V2: 特征相关矩阵热图 + VIF (画布不变 7.5x6.2, 文字x1.5, 无底注)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

features = ["Days", "Temperature", "Humidity", "Ratio", "Compost\nVolume"]
n = len(features)

corr = np.array([
    [ 1.000,  0.174,  0.167,  0.095,  0.201],
    [ 0.174,  1.000, -0.786,  0.655, -0.684],
    [ 0.167, -0.786,  1.000, -0.549,  0.929],
    [ 0.095,  0.655, -0.549,  1.000, -0.643],
    [ 0.201, -0.684,  0.929, -0.643,  1.000],
])
vif_values = ["1.0", "∞", "∞", "∞", "∞"]

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial"],
    "font.size":13.5,"font.weight":"bold",
    "axes.labelsize":16.5,"axes.labelweight":"bold",
    "axes.titlesize":19.5,"axes.titleweight":"bold",
    "xtick.labelsize":14,"ytick.labelsize":14,
    "figure.dpi":300,"savefig.dpi":600,
    "savefig.bbox":"tight","savefig.pad_inches":0.15,
})

fig, ax = plt.subplots(figsize=(7.5, 6.2))
fig.subplots_adjust(left=0.12,bottom=0.06,top=0.90,right=0.88)

mask = np.tril(np.ones_like(corr,dtype=bool),k=-1)
corr_disp = np.ma.array(corr,mask=mask)
im = ax.imshow(corr_disp,cmap=plt.cm.RdBu_r,vmin=-1,vmax=1,aspect="equal")

for i in range(n):
    for j in range(n):
        if i <= j:
            val = corr[i,j]
            txt = "1.000" if i==j else f"{val:+.3f}"
            tc = "white" if abs(val)>0.65 else "black"
            ax.text(j,i,txt,ha="center",va="center",fontsize=15,fontweight="bold",color=tc)

ax.set_xticks(range(n));ax.set_xticklabels(features,fontsize=14)
ax.set_yticks(range(n));ax.set_yticklabels(features,fontsize=14,rotation=30,va="center")
ax.tick_params(which="both",length=0,bottom=False,left=False)

# VIF 列
vif_colors = ["#2C3E50","#C0392B","#C0392B","#C0392B","#C0392B"]
for i,(vif_val,vif_color) in enumerate(zip(vif_values,vif_colors)):
    ax.text(n+0.38,i,f"VIF = {vif_val}",ha="left",va="center",
            fontsize=15,fontweight="bold",color=vif_color)
ax.text(n+0.38,-0.65,"VIF",ha="left",va="center",fontsize=14,fontweight="bold",color="#7F8C8D")
ax.axvline(x=n-0.55,color="#BDC3C7",linewidth=1.4,zorder=5)

cbar = fig.colorbar(im,ax=ax,fraction=0.042,pad=0.02,ticks=[-1,-0.5,0,0.5,1])
cbar.set_label("Pearson r",fontweight="bold",fontsize=13.5,labelpad=-6)
cbar.ax.tick_params(length=0)
cbar.outline.set_visible(False)

ax.set_title("Feature Correlation Matrix & VIF",pad=18)

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT_DIR / "fig_correlation_vif_v2.png",dpi=600)
print(f"Saved: {OUT_DIR / 'fig_correlation_vif_v2.png'}")
print("Done.")
plt.close(fig)