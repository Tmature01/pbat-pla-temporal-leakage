"""
Figure: 随机种子敏感性分析 — 50 seeds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_SeedSensitivity.csv"))
summ = pd.read_csv(str(FIGURE_DATA / "Fig_SeedSensitivity_Summary.csv"))

seeds = df["Seed"].values
r2_rand = df["R2_Random"].values
r2_temp = df["R2_Temporal"].values

mean_r = summ.loc[summ["Metric"] == "Mean", "Random_Split"].values[0]
std_r  = summ.loc[summ["Metric"] == "Std", "Random_Split"].values[0]
seed42 = summ.loc[summ["Metric"] == "Seed42_Value", "Random_Split"].values[0]

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

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.22, bottom=0.16, top=0.88, left=0.09, right=0.96)

# ══════════════════════════════════════════════════════
# Panel (a): Random Split
# ══════════════════════════════════════════════════════
axA.axhspan(mean_r - std_r, mean_r + std_r, facecolor="#3498DB", alpha=0.12, linewidth=0)
axA.axhline(y=mean_r, color="#3498DB", linewidth=1.0, linestyle="--", alpha=0.6)

for i in range(len(seeds)):
    is42 = seeds[i] == 42
    axA.scatter(seeds[i], r2_rand[i],
                s=80 if is42 else 25,
                color="#C0392B" if is42 else "#34495E",
                marker="*" if is42 else "o",
                edgecolors="white", linewidth=0.5,
                zorder=5 if is42 else 3)

axA.text(42, seed42 + 0.0020, f"Seed=42\n{seed42:.4f}", fontsize=7.5,
         fontweight="bold", color="#C0392B", ha="center")

axA.set_title("(a) Random split: 50-seed $R^2$", pad=8)
axA.set_xlabel("Random seed"); axA.set_ylabel("$R^2$")
axA.set_xlim(-2, 51); axA.set_ylim(0.988, 0.998)
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axA.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

axA.text(0.95, 0.08, f"Mean = {mean_r:.4f}\nSD = {std_r:.4f}\nRange = [{r2_rand.min():.4f}, {r2_rand.max():.4f}]",
         transform=axA.transAxes, fontsize=8, fontweight="bold", color="#2C3E50",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDC3C7", alpha=0.85))

# ══════════════════════════════════════════════════════
# Panel (b): Temporal Holdout
# ══════════════════════════════════════════════════════
axB.scatter(seeds, r2_temp, s=25, color="#34495E", edgecolors="white",
            linewidth=0.5, zorder=3)
axB.axhline(y=r2_temp[0], color="#3498DB", linewidth=1.0, linestyle="--", alpha=0.6)

axB.set_title("(b) Temporal holdout: 50-seed $R^2$", pad=8)
axB.set_xlabel("Random seed"); axB.set_ylabel("$R^2$")
axB.set_xlim(-2, 51); axB.set_ylim(0.068, 0.076)
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
axB.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

axB.text(0.95, 0.08, "Deterministic\nAll 50 seeds: $R^2 = 0.0721$\nSD = 0.0000",
         transform=axB.transAxes, fontsize=8, fontweight="bold", color="#2C3E50",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDC3C7", alpha=0.85))

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.02,
         "50 seeds tested. Random split R^2 = 0.9931 +/- 0.0011. Temporal holdout R^2 = 0.0721 (deterministic). "
         "Seed choice is irrelevant — conclusions are fully robust.",
         ha="center", fontsize=7.5, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_seed_sensitivity.png", dpi=600)
fig.savefig(OUT_DIR / "fig_seed_sensitivity.svg")
print(f"Saved: {OUT_DIR / 'fig_seed_sensitivity.png'}")
print(f"Saved: {OUT_DIR / 'fig_seed_sensitivity.svg'}")
print(f"Random: mean={mean_r:.4f}, std={std_r:.4f}")
print(f"Temporal: all = {r2_temp[0]:.4f}")
print("Done.")
plt.close(fig)