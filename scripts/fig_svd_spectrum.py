"""
Figure: 多项式设计矩阵奇异值谱 — 有效秩 vs 名义秩
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取数据 ──────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_SVD_Spectrum.csv"))
idx = df["Index"].values
sv_raw = df["SingularValue_Raw"].values
sv_zs  = df["SingularValue_Zscore"].values
sv_zm  = df["SingularValue_ZscoreMinMax"].values

eps_mach = np.finfo(np.float64).eps  # ~2.22e-16

print(f"Raw:      sigma1={sv_raw[0]:.3e}, sigma7={sv_raw[6]:.3e}, sigma8={sv_raw[7]:.3e}")
print(f"Z-score:   sigma1={sv_zs[0]:.3e}, sigma7={sv_zs[6]:.3e}, sigma8={sv_zs[7]:.3e}")
print(f"ZS+MinMax: sigma1={sv_zm[0]:.3e}, sigma7={sv_zm[6]:.3e}, sigma8={sv_zm[7]:.3e}")

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.12, right=0.94, top=0.92, bottom=0.14)

# ── 3 条奇异值曲线 ───────────────────────────────────
ax.semilogy(idx, sv_raw, "o-", color="#2C3E50", linewidth=1.5, markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, label="Raw", zorder=3)
ax.semilogy(idx, sv_zs,  "s--", color="#E67E22", linewidth=1.5, markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, label="Z-score", zorder=3)
ax.semilogy(idx, sv_zm,  "^:", color="#3498DB", linewidth=1.5, markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, label="Z-score + MinMax", zorder=3)

# ── 有效秩分界线 ─────────────────────────────────────
ax.axvline(x=7.5, color="#C0392B", linewidth=1.2, linestyle=(0, (6, 3)), alpha=0.7, zorder=2)
ax.axvspan(7.5, 21.5, facecolor="#FADBD8", alpha=0.12, linewidth=0, zorder=0)

# ── 标注 ────────────────────────────────────────────
ax.annotate("Effective rank = 7\n(93.0% information)",
            xy=(7, sv_raw[6]), xytext=(11, sv_raw[6] * 5),
            fontsize=8.5, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0,
                            connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C0392B", alpha=0.88))

ax.annotate("sigma_8 to sigma_21:\nnumerical noise",
            xy=(14, 1e-13), xytext=(16.5, 1e-8),
            fontsize=8.5, fontweight="bold", color="#7F8C8D",
            arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.88))

# ── 图例 ─────────────────────────────────────────────
leg = ax.legend(loc="lower left", framealpha=0.90, edgecolor="#BDC3C7",
                fancybox=True, fontsize=8.5)
for t in leg.get_texts():
    t.set_fontweight("bold")

# ── 坐标轴 ───────────────────────────────────────────
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value (log scale)")
ax.set_xlim(0.5, 21.5)
ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
ax.tick_params(which="both", direction="in", bottom=True, top=False,
               left=True, right=False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "5 features -> 21 polynomial terms, but only 7 carry information above numerical noise. "
         "Rank-deficient design: 3 conditions < 4 environmental features. VIF = inf is an algebraic inevitability.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_svd_spectrum.png", dpi=600)
fig.savefig(OUT_DIR / "fig_svd_spectrum.svg")
print(f"Saved: {OUT_DIR / 'fig_svd_spectrum.png'}")
print(f"Saved: {OUT_DIR / 'fig_svd_spectrum.svg'}")
print("Done.")
plt.close(fig)