"""
Figure: 时间数据泄漏概念示意图 (简洁版)
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

df  = pd.read_csv(str(FIGURE_DATA / "Fig_DataLeakage_Protocols.csv"))
kde = pd.read_csv(str(FIGURE_DATA / "Fig_DataLeakage_KDE.csv"))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "font.weight": "bold",
    "axes.labelsize": 10, "axes.labelweight": "bold",
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5.6))
fig.subplots_adjust(wspace=0.28, bottom=0.08, top=0.88, left=0.05, right=0.97)

C_TRAIN = "#2471A3"; C_TEST = "#C0392B"; C_BG = "#BDC3C7"; C_BOUND = "#2C3E50"; C_GAP = "#E67E22"

# ══════════════════════════════════════════════════════
# Panel (a): Random Split
# ══════════════════════════════════════════════════════
axA.plot(df["Day"], df["CO2"], color=C_BG, linewidth=0.6, alpha=0.5, zorder=0)

rand_train = df[df["Random_Partition"] == "Train"]
rand_test  = df[df["Random_Partition"] == "Test"]

axA.scatter(rand_train["Day"], rand_train["CO2"], s=14, color=C_TRAIN,
            edgecolors="none", alpha=0.60, zorder=2, label="Train")
axA.scatter(rand_test["Day"], rand_test["CO2"], s=28, facecolors="none",
            edgecolors=C_TEST, linewidth=1.2, zorder=3, label="Test")

axA.annotate("Test point", xy=(120, 237), xytext=(135, 240),
            fontsize=7.5, fontweight="bold", color=C_TEST,
            arrowprops=dict(arrowstyle="->", color=C_TEST, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_TEST, alpha=0.85))
axA.plot([118, 126], [45, 45], color=C_GAP, linewidth=1.5, zorder=5)
axA.plot([118, 118], [42, 48], color=C_GAP, linewidth=1.5, zorder=5)
axA.plot([126, 126], [42, 48], color=C_GAP, linewidth=1.5, zorder=5)
axA.text(122, 22, "0-day gap\n(interpolation)", fontsize=7, fontweight="bold",
         color=C_GAP, ha="center")

axA.set_title("(a) Random split — data leakage", pad=6)
axA.set_xlabel("Days"); axA.set_ylabel("CO$_2$ (g)")
axA.set_xlim(-5, 185)
leg = axA.legend(loc="lower right", framealpha=0.85, edgecolor="#BDC3C7",
                 fancybox=True, fontsize=8, handlelength=1.2)
for t in leg.get_texts(): t.set_fontweight("bold")
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axA.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ══════════════════════════════════════════════════════
# Panel (b): Temporal Holdout
# ══════════════════════════════════════════════════════
axB.plot(df["Day"], df["CO2"], color=C_BG, linewidth=0.6, alpha=0.5, zorder=0)

temp_train = df[df["Temporal_Partition"] == "Train"]
temp_test  = df[df["Temporal_Partition"] == "Test"]

axB.scatter(temp_train["Day"], temp_train["CO2"], s=14, color=C_TRAIN,
            edgecolors="none", alpha=0.60, zorder=2, label="Train")
axB.scatter(temp_test["Day"], temp_test["CO2"], s=28, facecolors="none",
            edgecolors=C_TEST, linewidth=1.2, zorder=3, label="Test")

axB.axvline(x=120, color=C_BOUND, linewidth=1.2, linestyle=(0, (6, 3)),
            alpha=0.7, zorder=4)
axB.text(122, 200, "Train/Test boundary",
         ha="left", va="bottom", fontsize=7.5, fontweight="bold", color=C_BOUND)

axB.plot([120, 155], [45, 45], color=C_GAP, linewidth=1.5, zorder=5)
axB.plot([120, 120], [42, 48], color=C_GAP, linewidth=1.5, zorder=5)
axB.plot([155, 155], [42, 48], color=C_GAP, linewidth=1.5, zorder=5)
axB.text(137.5, 28, "30+ day gap\n(extrapolation)", fontsize=7, fontweight="bold",
         color=C_GAP, ha="center")

axB.set_title("(b) Temporal holdout — no leakage", pad=6)
axB.set_xlabel("Days"); axB.set_ylabel("CO$_2$ (g)")
axB.set_xlim(-5, 185)
leg = axB.legend(loc="lower right", framealpha=0.85, edgecolor="#BDC3C7",
                 fancybox=True, fontsize=8, handlelength=1.2)
for t in leg.get_texts(): t.set_fontweight("bold")
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axB.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axB.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ══════════════════════════════════════════════════════
# Panel (c): KDE Density
# ══════════════════════════════════════════════════════
axC.plot(kde["Day"], kde["Random_Train_Density"], color=C_TRAIN,
         linewidth=1.8, label="Train (Random)", zorder=3)
axC.plot(kde["Day"], kde["Random_Test_Density"], color=C_TEST,
         linewidth=1.8, linestyle="--", label="Test (Random)", zorder=3)
axC.plot(kde["Day"], kde["Temporal_Train_Density"], color=C_TRAIN,
         linewidth=1.3, alpha=0.35, label="Train (Temporal)", zorder=2)
axC.plot(kde["Day"], kde["Temporal_Test_Density"], color=C_TEST,
         linewidth=1.3, linestyle="--", alpha=0.35, label="Test (Temporal)", zorder=2)

axC.annotate("Random: Train & Test\noverlap heavily (leakage)",
            xy=(90, 0.80), xytext=(50, 0.72),
            fontsize=7.5, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#C0392B", alpha=0.85))
axC.annotate("Temporal:\nTrain & Test\nfully separated",
            xy=(155, 0.30), xytext=(130, 0.50),
            fontsize=7.5, fontweight="bold", color="#2471A3",
            arrowprops=dict(arrowstyle="->", color="#2471A3", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#2471A3", alpha=0.85))

axC.set_title("(c) Train/test temporal distribution", pad=6)
axC.set_xlabel("Days"); axC.set_ylabel("Normalized density")
legC = axC.legend(loc="lower left", framealpha=0.85, edgecolor="#BDC3C7",
                   fancybox=True, fontsize=7, handlelength=1.2, ncol=1,
                   bbox_to_anchor=(0.0, 0.12))
for t in legC.get_texts(): t.set_fontweight("bold")
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axC.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axC.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_data_leakage.png", dpi=600)
fig.savefig(OUT_DIR / "fig_data_leakage.svg")
print(f"Saved: {OUT_DIR / 'fig_data_leakage.png'}")
print("Done.")
plt.close(fig)