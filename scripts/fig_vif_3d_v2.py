"""
Figure V2: VIF=inf 设计矩阵秩亏可视化 (画布不变 14x6, 文字x1.5, 删Why VIF=inf面板)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from matplotlib.patches import Polygon
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

pts = pd.read_csv(str(FIGURE_DATA / "Fig_VIF_3D_Points.csv"))
plane = pd.read_csv(str(FIGURE_DATA / "Fig_VIF_Plane_Grid.csv"))

C_COLORS = {"C1":"#E74C3C","C2":"#3498DB","C3":"#2ECC71"}

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial"],
    "font.size":13.5,"font.weight":"bold",
    "axes.labelsize":15,"axes.labelweight":"bold",
    "axes.titlesize":16.5,"axes.titleweight":"bold",
    "figure.dpi":300,"savefig.dpi":600,
    "savefig.bbox":"tight","savefig.pad_inches":0.08,
})

fig = plt.figure(figsize=(14, 6))
fig.subplots_adjust(wspace=0.35,bottom=0.06,top=0.90,left=0.05,right=0.97)

# ══════════════════════════════════════════════════════
# (a) 3D Scatter + Plane
# ══════════════════════════════════════════════════════
ax3d = fig.add_subplot(1,2,1,projection="3d")

for _,row in pts.iterrows():
    cond_name = row["Condition"].split("(")[0].strip()
    ax3d.scatter(row["T_raw"],row["H_raw"],row["Ratio_raw"],
                s=200,color=C_COLORS.get(cond_name,"#333"),
                edgecolors="white",linewidth=1.8,zorder=5,label=cond_name)
    ax3d.text(row["T_raw"]+0.5,row["H_raw"]+0.5,row["Ratio_raw"]+1.5,
             cond_name,fontsize=14,fontweight="bold",color=C_COLORS.get(cond_name,"#333"))

ax3d.plot_trisurf(plane["T"],plane["H"],plane["Ratio"],
                  color="#BDC3C7",alpha=0.30,linewidth=0,zorder=1)

A = np.column_stack([pts["T_raw"],pts["H_raw"],np.ones(3)])
coeff,_,_,_ = lstsq(A,pts["Ratio_raw"],rcond=None)
for i,row in pts.iterrows():
    z_plane = coeff[0]*row["T_raw"]+coeff[1]*row["H_raw"]+coeff[2]
    ax3d.plot([row["T_raw"],row["T_raw"]],[row["H_raw"],row["H_raw"]],
              [z_plane,row["Ratio_raw"]],color="gray",linewidth=0.8,linestyle=":",alpha=0.5,zorder=2)

ax3d.set_xlabel("Temperature (C)")
ax3d.set_ylabel("Humidity (%)")
ax3d.set_zlabel("Ratio (%)")
ax3d.set_title("(a) 3 conditions in environmental feature space",pad=10)
ax3d.view_init(elev=22,azim=-55)
leg = ax3d.legend(loc="upper right",framealpha=0.85,edgecolor="#BDC3C7",fancybox=True,fontsize=12)
for t in leg.get_texts(): t.set_fontweight("bold")

ax3d.text2D(0.04,0.98,"3 points span only\na 2D plane in 3D\n"
            "(in 4D: 3D hyperplane,\nneed >=4 for full rank)",
            transform=ax3d.transAxes,fontsize=11,fontweight="bold",
            color="#C0392B",va="top",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#C0392B",alpha=0.85))

# ══════════════════════════════════════════════════════
# (b) 2D Analogy (3 小图, 删掉第4个Why VIF=inf面板)
# ══════════════════════════════════════════════════════
gs = fig.add_gridspec(1,3,left=0.52,right=0.96,top=0.88,bottom=0.15,wspace=0.30)
analogy_axes = [fig.add_subplot(gs[i]) for i in range(3)]

# (i) 2 points -> 1D line
ax = analogy_axes[0]
ax.set_xlim(-0.5,5.5);ax.set_ylim(-0.5,5.5)
ax.scatter([1,4],[2,4],s=120,color="#E74C3C",edgecolors="white",linewidth=1.4,zorder=3)
ax.plot([-0.5,5.5],[1.33,5.33],color="#3498DB",linewidth=1.8,zorder=2)
ax.set_title("2 points → 1 line",fontsize=12,pad=6)
ax.text(2.5,-0.2,"solvable",ha="center",fontsize=11,fontweight="bold",color="#27AE60")
ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect("equal")

# (ii) 3 points -> 2D plane
ax = analogy_axes[1]
ax.set_xlim(-0.5,5.5);ax.set_ylim(-0.5,5.5)
pts_ii = np.array([[0.8,1.5],[3.5,1.0],[2.5,4.5]])
ax.scatter(pts_ii[:,0],pts_ii[:,1],s=120,color="#E74C3C",edgecolors="white",linewidth=1.4,zorder=3)
tri = Polygon(pts_ii,facecolor="#3498DB",alpha=0.25,edgecolor="#3498DB",linewidth=1.2)
ax.add_patch(tri)
ax.set_title("3 points → 1 plane",fontsize=12,pad=6)
ax.text(2.5,-0.2,"solvable",ha="center",fontsize=11,fontweight="bold",color="#27AE60")
ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect("equal")

# (iii) 3 points -> 4D space X
ax = analogy_axes[2]
ax.set_xlim(-0.5,5.5);ax.set_ylim(-0.5,5.5)
ax.scatter([1,3,2.5],[2,1.5,4],s=120,color="#E74C3C",edgecolors="white",linewidth=1.4,zorder=3)
ax.text(2.5,2.5,"X",fontsize=50,fontweight="bold",color="#C0392B",ha="center",va="center",alpha=0.7,zorder=4)
ax.set_title("3 points in 4D space",fontsize=12,pad=6)
ax.text(2.5,-0.2,"rank 3 < 4  unsolvable",ha="center",fontsize=11,fontweight="bold",color="#C0392B")
ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect("equal")

fig.text(0.73,0.10,"(b) Dimension analogy",ha="center",fontsize=15,fontweight="bold",color="#2C3E50")

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT_DIR / "fig_vif_3d_v2.png",dpi=600)
print(f"Saved: {OUT_DIR / 'fig_vif_3d_v2.png'}")
print("Done.")
plt.close(fig)