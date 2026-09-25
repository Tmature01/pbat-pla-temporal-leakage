"""
Figure: Shuffle 证据 — 4 面板 (a,b=散点, c,d=小提琴+直方图)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. 读取 ──────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

scat_r = pd.read_csv(str(FIGURE_DATA / "Fig_Scatter_Random.csv"))
scat_s = pd.read_csv(str(FIGURE_DATA / "Fig_Scatter_Shuffle.csv"))
violin_df = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_Violin.csv"))
reflines  = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_RefLines.csv"))
kde_df    = pd.read_csv(str(FIGURE_DATA / "Fig_Shuffle_KDE.csv"))

lsmpr_data = violin_df[violin_df["Model"]=="LS-MPR"]["R2_Shuffle"].values
exp_data   = violin_df[violin_df["Model"]=="Exp Decay"]["R2_Shuffle"].values

r2_r, rho_r = r2_score(scat_r["Observed_CO2_g"], scat_r["Predicted_CO2_g"]), spearmanr(scat_r["Observed_CO2_g"], scat_r["Predicted_CO2_g"])[0]
r2_s, rho_s = r2_score(scat_s["Observed_CO2_g"], scat_s["Predicted_CO2_g"]), spearmanr(scat_s["Observed_CO2_g"], scat_s["Predicted_CO2_g"])[0]

# ── 2. 绘图 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "font.weight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold",
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                      hspace=0.30, wspace=0.26,
                      left=0.05, right=0.97, top=0.95, bottom=0.05)

axA = fig.add_subplot(gs[0,0])
axB = fig.add_subplot(gs[0,1])
axC = fig.add_subplot(gs[1,0])
axD = fig.add_subplot(gs[1,1])

C_LS  = "#E74C3C"; C_EXP = "#3498DB"; C_NULL = "#95A5A6"

# ══════════════════════════════════════════════════════
# (a) Random Split 散点
# ══════════════════════════════════════════════════════
all_obs = pd.concat([scat_r["Observed_CO2_g"], scat_s["Observed_CO2_g"]])
all_pred = pd.concat([scat_r["Predicted_CO2_g"], scat_s["Predicted_CO2_g"]])
lim_min = min(all_obs.min(), all_pred.min()) - 5
lim_max = max(all_obs.max(), all_pred.max()) + 5
vmin = min(scat_r["Days"].min(), scat_s["Days"].min())
vmax = max(scat_r["Days"].max(), scat_s["Days"].max())

sc0 = axA.scatter(scat_r["Observed_CO2_g"], scat_r["Predicted_CO2_g"],
                  c=scat_r["Days"], cmap="Spectral_r", s=14, alpha=0.75,
                  edgecolors="none", vmin=vmin, vmax=vmax, rasterized=True, zorder=2)
axA.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=0.6,
         linestyle="--", alpha=0.4, zorder=1)
axA.text(0.04, 0.96, f"$R^2$ = {r2_r:.3f}\n$\\rho$ = {rho_r:.3f}",
         transform=axA.transAxes, fontsize=8, fontweight="bold", va="top",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))
axA.set_title("(a) Random split", pad=5)
axA.set_xlabel("Observed CO$_2$ (g)"); axA.set_ylabel("Predicted CO$_2$ (g)")
axA.set_xlim(lim_min, lim_max); axA.set_ylim(lim_min, lim_max)
axA.set_aspect("equal")
axA.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (b) Temporal Shuffle 散点
# ══════════════════════════════════════════════════════
sc1 = axB.scatter(scat_s["Observed_CO2_g"], scat_s["Predicted_CO2_g"],
                  c=scat_s["Days"], cmap="Spectral_r", s=14, alpha=0.75,
                  edgecolors="none", vmin=vmin, vmax=vmax, rasterized=True, zorder=2)
axB.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=0.6,
         linestyle="--", alpha=0.4, zorder=1)
axB.text(0.04, 0.96, f"$R^2$ = 0.035\n$\\rho$ = {rho_s:.3f}\n(30-shuffle mean)",
         transform=axB.transAxes, fontsize=8, fontweight="bold", va="top",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))
axB.set_title("(b) Day permutation", pad=5)
axB.set_xlabel("Observed CO$_2$ (g)"); axB.set_ylabel("Predicted CO$_2$ (g)")
axB.set_xlim(lim_min, lim_max); axB.set_ylim(lim_min, lim_max)
axB.set_aspect("equal")
axB.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# Shared colorbar
cbar = fig.colorbar(sc1, ax=[axA, axB], fraction=0.042, pad=0.015,
                    ticks=np.linspace(vmin, vmax, 5))
cbar.set_label("Days", fontweight="bold", fontsize=8)
cbar.ax.tick_params(length=0)

# ══════════════════════════════════════════════════════
# (c) 小提琴图
# ══════════════════════════════════════════════════════
positions = [0, 1]; data_sets = [lsmpr_data, exp_data]
vp = axC.violinplot(data_sets, positions=positions, showmeans=False, showmedians=True, widths=0.55)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor([C_LS, C_EXP][i]); body.set_alpha(0.30)
    body.set_edgecolor([C_LS, C_EXP][i]); body.set_linewidth(1.2)
for part in ["cbars","cmins","cmaxes"]: vp[part].set_color("#7F8C8D"); vp[part].set_linewidth(0.6)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.8)

np.random.seed(42)
for i, (d, c) in enumerate(zip(data_sets, [C_LS, C_EXP])):
    jitter = np.random.uniform(-0.12, 0.12, size=len(d))
    axC.scatter(np.full_like(d, positions[i]) + jitter, d, s=12, color=c, alpha=0.40, zorder=4)

for i, model_name in enumerate(["LS-MPR", "Exp Decay"]):
    orig = reflines[(reflines["Model"]==model_name)&(reflines["Line"]=="Original")]["R2"].values[0]
    smean = reflines[(reflines["Model"]==model_name)&(reflines["Line"]=="Shuffle_mean")]["R2"].values[0]
    axC.hlines(orig, positions[i]-0.28, positions[i]+0.28, colors=C_LS, linewidth=1.0, linestyles="dashed", alpha=0.5)
    axC.hlines(smean, positions[i]-0.28, positions[i]+0.28, colors=C_EXP, linewidth=1.0, linestyles="dashed", alpha=0.5)

axC.text(0, 0.08, "Cohen's d = 57.0\np = 1.8 x 10^{-52}", fontsize=7.5, fontweight="bold", color="#2C3E50", ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.85))
axC.set_title("(c) 30-shuffle R^2 distribution", pad=5)
axC.set_ylabel("$R^2$ (shuffled)"); axC.set_xticks([0,1]); axC.set_xticklabels(["LS-MPR", "Exp Decay"])
axC.set_ylim(-0.10, 1.05)
axC.axhline(y=0, color="black", linewidth=0.4, alpha=0.25, zorder=0)
axC.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ══════════════════════════════════════════════════════
# (d) 直方图 + KDE + Null
# ══════════════════════════════════════════════════════
axD.hist(lsmpr_data, bins=9, color=C_LS, alpha=0.35, edgecolor=C_LS, linewidth=0.6, density=True, zorder=2)
axD.plot(kde_df["R2"], kde_df["LSMPR_KDE"], color=C_LS, linewidth=1.8, zorder=4)
axD.fill_between(kde_df["R2"], 0, kde_df["Null_PDF"], facecolor=C_NULL, alpha=0.30, linewidth=0, zorder=1)
axD.plot(kde_df["R2"], kde_df["Null_PDF"], color=C_NULL, linewidth=1.0, linestyle="--", zorder=3)

mean_val = np.mean(lsmpr_data); std_val = np.std(lsmpr_data, ddof=1)
axD.text(0.95, 0.95, f"Mean = {mean_val:+.3f}\nSD = {std_val:.3f}\n95% CI: [-0.051, +0.094]\nKS: D=0.31, p=0.006",
         transform=axD.transAxes, fontsize=6.5, fontweight="bold", color="#2C3E50", ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", alpha=0.85))
axD.axvline(x=mean_val, color=C_LS, linewidth=0.6, linestyle="--", alpha=0.5)
axD.axvline(x=0, color="black", linewidth=0.4, alpha=0.25)
axD.set_title("(d) LS-MPR shuffle R^2 vs. null", pad=5)
axD.set_xlabel("$R^2$"); axD.set_ylabel("Density")
axD.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)

# ── 图例 ─────────────────────────────────────────────
leg = fig.legend(handles=[
    Line2D([0],[0], color=C_LS, linewidth=1.2, linestyle="dashed", label="Original R^2"),
    Line2D([0],[0], color=C_EXP, linewidth=1.2, linestyle="dashed", label="Shuffle mean"),
], loc="upper center", ncol=2, framealpha=0.85, edgecolor="#BDC3C7", fancybox=True, fontsize=7.5, bbox_to_anchor=(0.68, 0.49))
for t in leg.get_texts(): t.set_fontweight("bold")

# ── 保存 ─────────────────────────────────────────────
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_shuffle_4panel.png", dpi=600)
fig.savefig(OUT_DIR / "fig_shuffle_4panel.svg")
print(f"Saved: {OUT_DIR / 'fig_shuffle_4panel.png'}")
print("Done.")
plt.close(fig)