"""
Figure V2: 数据泄漏概念 (画布不变 16x5.6, 文字×1.5, 无遮挡)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df  = pd.read_csv(str(FIGURE_DATA / "Fig_DataLeakage_Protocols.csv"))
kde = pd.read_csv(str(FIGURE_DATA / "Fig_DataLeakage_KDE.csv"))

# 画布不变, 文字 ×1.5
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 13.5, "font.weight": "bold",
    "axes.labelsize": 15, "axes.labelweight": "bold",
    "axes.titlesize": 16.5, "axes.titleweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5.6))
fig.subplots_adjust(wspace=0.30, bottom=0.10, top=0.86, left=0.06, right=0.98)

C_TR = "#2471A3"; C_TS = "#C0392B"; C_BG = "#BDC3C7"; C_BD = "#2C3E50"; C_GP = "#E67E22"

# ══════════════════════════════════════════════════════
# (a) Random Split
# ══════════════════════════════════════════════════════
axA.plot(df["Day"], df["CO2"], color=C_BG, linewidth=0.6, alpha=0.4, zorder=0)
axA.scatter(df[df["Random_Partition"]=="Train"]["Day"], df[df["Random_Partition"]=="Train"]["CO2"],
            s=14, color=C_TR, alpha=0.55, zorder=2, label="Train")
axA.scatter(df[df["Random_Partition"]=="Test"]["Day"], df[df["Random_Partition"]=="Test"]["CO2"],
            s=32, facecolors="none", edgecolors=C_TS, linewidth=1.3, zorder=3, label="Test")

# Test point 标注移到图内避开顶部
axA.annotate("Test point", xy=(120,237), xytext=(145,195),
            fontsize=10.5, fontweight="bold", color=C_TS,
            arrowprops=dict(arrowstyle="->", color=C_TS, lw=0.9),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_TS, alpha=0.85))

# 0-day gap — 上移避开轴标
axA.plot([114,130], [78,78], color=C_GP, linewidth=1.8, zorder=5)
axA.plot([114,114], [72,84], color=C_GP, linewidth=1.8, zorder=5)
axA.plot([130,130], [72,84], color=C_GP, linewidth=1.8, zorder=5)
axA.text(122, 68, "0-day gap\n(interpolation)", fontsize=10, fontweight="bold",
         color=C_GP, ha="center", va="top")

axA.set_title("(a) Random split — data leakage", pad=8)
axA.set_xlabel("Days"); axA.set_ylabel("CO$_2$ (g)")
axA.set_xlim(-5, 185)
leg = axA.legend(loc="lower right", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=11.5, handlelength=1.2)
for t in leg.get_texts(): t.set_fontweight("bold")
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ══════════════════════════════════════════════════════
# (b) Temporal Holdout
# ══════════════════════════════════════════════════════
axB.plot(df["Day"], df["CO2"], color=C_BG, linewidth=0.6, alpha=0.4, zorder=0)
axB.scatter(df[df["Temporal_Partition"]=="Train"]["Day"], df[df["Temporal_Partition"]=="Train"]["CO2"],
            s=14, color=C_TR, alpha=0.55, zorder=2, label="Train")
axB.scatter(df[df["Temporal_Partition"]=="Test"]["Day"], df[df["Temporal_Partition"]=="Test"]["CO2"],
            s=32, facecolors="none", edgecolors=C_TS, linewidth=1.3, zorder=3, label="Test")

axB.axvline(x=120, color=C_BD, linewidth=1.3, linestyle=(0,(6,3)), alpha=0.7, zorder=4)
axB.text(122, 195, "Train/Test\nboundary", ha="left", va="center",
         fontsize=12, fontweight="black", color=C_BD)

# 30+ day gap — 上移
axB.plot([120,160], [68,68], color=C_GP, linewidth=1.8, zorder=5)
axB.plot([120,120], [62,74], color=C_GP, linewidth=1.8, zorder=5)
axB.plot([160,160], [62,74], color=C_GP, linewidth=1.8, zorder=5)
axB.text(144, 58, "30+ day gap\n(extrapolation)", fontsize=10, fontweight="bold",
         color=C_GP, ha="center", va="top")

axB.set_title("(b) Temporal holdout — no leakage", pad=8)
axB.set_xlabel("Days"); axB.set_ylabel("CO$_2$ (g)")
axB.set_xlim(-5, 185)
leg = axB.legend(loc="lower right", framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=11.5, handlelength=1.2)
for t in leg.get_texts(): t.set_fontweight("bold")
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axB.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ══════════════════════════════════════════════════════
# (c) KDE
# ══════════════════════════════════════════════════════
axC.plot(kde["Day"], kde["Random_Train_Density"], color=C_TR,
         linewidth=2.0, label="Train (Random)", zorder=3)
axC.plot(kde["Day"], kde["Random_Test_Density"], color=C_TS,
         linewidth=2.0, linestyle="--", label="Test (Random)", zorder=3)
axC.plot(kde["Day"], kde["Temporal_Train_Density"], color=C_TR,
         linewidth=1.4, alpha=0.35, label="Train (Temporal)", zorder=2)
axC.plot(kde["Day"], kde["Temporal_Test_Density"], color=C_TS,
         linewidth=1.4, linestyle="--", alpha=0.35, label="Test (Temporal)", zorder=2)

# 标注分上下互斥位置
axC.annotate("Random: Train & Test\noverlap heavily (leakage)",
            xy=(90,0.78), xytext=(18,0.62),
            fontsize=10, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C0392B", alpha=0.85))

axC.annotate("Temporal: Train & Test\nfully separated",
            xy=(155,0.20), xytext=(110,0.30),
            fontsize=10, fontweight="bold", color="#2471A3",
            arrowprops=dict(arrowstyle="->", color="#2471A3", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2471A3", alpha=0.85))

axC.set_title("(c) Train/test temporal distribution", pad=8)
axC.set_xlabel("Days"); axC.set_ylabel("Normalized density")
# 图例到左下角
legC = axC.legend(loc="lower left", framealpha=0.85, edgecolor="#BDC3C7",
                   fancybox=True, fontsize=10, handlelength=1.2, ncol=1,
                   bbox_to_anchor=(0.0, 0.06), borderpad=0.5)
for t in legC.get_texts(): t.set_fontweight("bold")
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axC.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# 轴粗度
for ax in (axA, axB, axC):
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_data_leakage_v2.png", dpi=600)
print(f"Saved: {OUT_DIR / 'fig_data_leakage_v2.png'}")
print("Done.")
plt.close(fig)