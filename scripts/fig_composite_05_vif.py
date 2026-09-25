"""
Figure 5 (复合): 为什么 VIF=inf — 设计矩阵缺陷
(a) 平行坐标图 (b) 相关矩阵+VIF (c) SVD 谱
"""

import pandas as pd
import numpy as np
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

svd_df = pd.read_csv(str(FIGURE_DATA / "Fig_SVD_Spectrum.csv"))

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

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.0],
                      wspace=0.30, left=0.06, right=0.96, top=0.90, bottom=0.12)

axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

# ══════════════════════════════════════════════════════
# PANEL (a): 平行坐标图
# ══════════════════════════════════════════════════════
features = ["Temperature\n(C)", "Humidity\n(%)", "Ratio\n(%)", "Compost\nVolume"]
raw = np.array([[50,70,70,0.90],[58,60,100,0.50],[58,70,0,0.75]])
norm_v = (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0) + 1e-10)
cond_names = ["C1 (PBAT)", "C2 (PLA)", "C3 (70/30)"]
colors_p = ["#C0392B", "#2471A3", "#27AE60"]
linestyles_p = ["-", "--", "-."]

x = np.arange(4)
for i in range(3):
    axA.plot(x, norm_v[i], color=colors_p[i], linewidth=2.2, linestyle=linestyles_p[i],
             marker="o", markersize=8, markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=colors_p[i], zorder=3)
    for j in range(4):
        val = raw[i,j]; txt = f"{val:.0f}" if j<3 else f"{val:.2f}"
        dy = 0.03 if i==1 else -0.03
        axA.text(j, norm_v[i,j]+dy, txt, fontsize=6, fontweight="bold", color=colors_p[i], ha="center",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec=colors_p[i], alpha=0.7, linewidth=0.4))

axA.annotate("C2 & C3 share\nsame T (58 C)", xy=(0.1, 0.97), xytext=(0.5, 0.35),
            fontsize=7, fontweight="bold", color="#2471A3",
            arrowprops=dict(arrowstyle="->", color="#2471A3", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#2471A3", alpha=0.85))
axA.annotate("3 lines in 4D:\nrank=3<4, VIF=inf", xy=(3.0, 0.5), xytext=(3.2, 0.75),
            fontsize=8, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C0392B", alpha=0.88))

for j in range(3):
    axA.axvline(x=j+0.5, color="#BDC3C7", linewidth=0.4, linestyle=":", alpha=0.5, zorder=0)

axA.set_xticks(x); axA.set_xticklabels(features, fontsize=9)
axA.set_ylabel("Normalized value")
axA.set_ylim(-0.08, 1.12); axA.set_xlim(-0.5, 3.5)
axA.set_title("(a) 3 conditions in 4D environmental space", pad=6)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

leg_els_a = [Line2D([0],[0], color=colors_p[i], linewidth=2.2, linestyle=linestyles_p[i], label=cond_names[i]) for i in range(3)]
legA = axA.legend(handles=leg_els_a, loc="upper left", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=8,
                  bbox_to_anchor=(-0.05, 1.0))
for t in legA.get_texts(): t.set_fontweight("bold")

# ══════════════════════════════════════════════════════
# PANEL (b): 相关矩阵 + VIF
# ══════════════════════════════════════════════════════
feat_labels = ["Days", "Temperature", "Humidity", "Ratio", "Compost\nVolume"]
n = 5
corr = np.array([
    [ 1.000,  0.174,  0.167,  0.095,  0.201],
    [ 0.174,  1.000, -0.786,  0.655, -0.684],
    [ 0.167, -0.786,  1.000, -0.549,  0.929],
    [ 0.095,  0.655, -0.549,  1.000, -0.643],
    [ 0.201, -0.684,  0.929, -0.643,  1.000],
])
vif_vals = ["1.0", "inf", "inf", "inf", "inf"]

mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
corr_disp = np.ma.array(corr, mask=mask)
im = axB.imshow(corr_disp, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="equal")

for i in range(n):
    for j in range(n):
        if i <= j:
            val = corr[i,j]
            txt = "1.000" if i==j else f"{val:+.3f}"
            tc = "white" if abs(val)>0.65 else "black"
            axB.text(j, i, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color=tc)

axB.set_xticks(range(n)); axB.set_xticklabels(feat_labels, fontsize=8)
axB.set_yticks(range(n)); axB.set_yticklabels(feat_labels, fontsize=8, rotation=30, va="center")
axB.tick_params(length=0)

vif_colors = ["#2C3E50","#C0392B","#C0392B","#C0392B","#C0392B"]
for i, (v, vc) in enumerate(zip(vif_vals, vif_colors)):
    axB.text(n+0.35, i, f"VIF={v}", ha="left", va="center", fontsize=9, fontweight="bold", color=vc)
axB.text(n+0.35, -0.65, "VIF", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#7F8C8D")
axB.axvline(x=n-0.55, color="#BDC3C7", linewidth=1.0, zorder=5)

cbar = fig.colorbar(im, ax=axB, fraction=0.04, pad=0.02, ticks=[-1,-0.5,0,0.5,1])
cbar.set_label("Pearson r", fontweight="bold", fontsize=8, labelpad=-4)
cbar.ax.tick_params(length=0); cbar.outline.set_visible(False)
axB.set_title("(b) Feature correlation matrix & VIF", pad=6)

# ══════════════════════════════════════════════════════
# PANEL (c): SVD 谱
# ══════════════════════════════════════════════════════
idx = svd_df["Index"].values
axC.semilogy(idx, svd_df["SingularValue_Raw"], "o-", color="#2C3E50", linewidth=1.5, markersize=5, markerfacecolor="white", markeredgewidth=1.2, label="Raw")
axC.semilogy(idx, svd_df["SingularValue_Zscore"], "s--", color="#E67E22", linewidth=1.5, markersize=5, markerfacecolor="white", markeredgewidth=1.2, label="Z-score")
axC.semilogy(idx, svd_df["SingularValue_ZscoreMinMax"], "^:", color="#3498DB", linewidth=1.5, markersize=5, markerfacecolor="white", markeredgewidth=1.2, label="ZS+MinMax")

axC.axvline(x=7.5, color="#C0392B", linewidth=1.2, linestyle=(0,(6,3)), alpha=0.7, zorder=2)
axC.axvspan(7.5, 21.5, facecolor="#FADBD8", alpha=0.10, linewidth=0, zorder=0)

axC.annotate("Effective rank = 7\n(93% information)", xy=(7, svd_df["SingularValue_Raw"].values[6]),
            xytext=(11, svd_df["SingularValue_Raw"].values[6]*5), fontsize=8, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8, connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C0392B", alpha=0.88))
axC.annotate("sigma_8 to sigma_21:\nnumerical noise", xy=(14, 1e-13), xytext=(16.5, 1e-8),
            fontsize=8, fontweight="bold", color="#7F8C8D",
            arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.88))

axC.set_title("(c) Singular value spectrum", pad=6)
axC.set_xlabel("Singular value index"); axC.set_ylabel("Singular value (log scale)")
axC.set_xlim(0.5, 21.5); axC.set_xticks([1,5,9,13,17,21])
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

legC = axC.legend(loc="lower left", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=7.5)
for t in legC.get_texts(): t.set_fontweight("bold")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "Figure 5. Why VIF = inf — design matrix deficiency. (a) 3 conditions span only a 3D hyperplane in 4D. "
         "(b) 4 environmental features have VIF=inf; only Days is identifiable. (c) Cliff at sigma_7->sigma_8: "
         "effective rank 7/21, confirming rank deficiency is algebraic, not numerical.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig_05_VIF_DesignDefect.png", dpi=600)
fig.savefig(OUT_DIR / "Fig_05_VIF_DesignDefect.svg")
print(f"Saved: {OUT_DIR / 'Fig_05_VIF_DesignDefect.png'}")
print(f"Saved: {OUT_DIR / 'Fig_05_VIF_DesignDefect.svg'}")
print("Done.")
plt.close(fig)