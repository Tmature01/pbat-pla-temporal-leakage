"""
Figure V3: 反事实验证多窗口轨迹 (画布不变 14x10, 文字x1.5, 无遮挡)
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

RAW = str(_REPO_ROOT / "data" / "pbat_pla_experimental_543.csv")
raw = pd.read_csv(RAW, header=None, skiprows=1)
days_full = raw.iloc[:, 0].values
co2_cols = {"C1": 6, "C2": 17, "C3": 26}
obs_full = {c: raw.iloc[:, col].values for c, col in co2_cols.items()}

TEMP = Path(str(FIGURE_DATA))
configs = [
    ("Fig_Antifactual_Growth.csv",       "Rapid growth phase",    0, 29),
    ("Fig_Antifactual_Growth+Decay.csv", "Growth + decay phase",  0, 59),
    ("Fig_Antifactual_Mixed.csv",        "Mixed phase",           0, 89),
    ("Fig_Antifactual_Plateau.csv",      "Plateau phase",         0, 119),
]
test_dfs = {f: pd.read_csv(TEMP / f) for f,*_ in configs}

r2_labels = {
    "Fig_Antifactual_Growth.csv":       "R$^2$: Persist=-3.67 / LS-MPR=+0.02",
    "Fig_Antifactual_Growth+Decay.csv": "R$^2$: Persist=-1.81 / LS-MPR=-0.32",
    "Fig_Antifactual_Mixed.csv":        "R$^2$: Persist=+0.44 / LS-MPR=+0.85",
    "Fig_Antifactual_Plateau.csv":      "R$^2$: Persist=+0.95 / LS-MPR=-0.06",
}

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 17, "font.weight": "bold",
    "axes.labelsize": 19, "axes.labelweight": "bold",
    "axes.titlesize": 21, "axes.titleweight": "bold",
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

COLOR_OBS  = "#2C3E50"
COLOR_LS   = "#C0392B"
COLOR_PERS = "#2471A3"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.34, wspace=0.24, top=0.88, bottom=0.08, left=0.07, right=0.97)

for idx, (ax, (fname, label, t_start, t_end)) in enumerate(zip(axes.flat, configs)):
    df_test = test_dfs[fname]
    train_end = int(df_test["Train_End"].iloc[0])
    test_end  = int(df_test["Day"].max())

    ax.axvspan(0, train_end, facecolor="#E8F0FE", alpha=0.30, linewidth=0, zorder=0)
    ax.axvspan(train_end, test_end+1, facecolor="#FEF9E7", alpha=0.25, linewidth=0, zorder=0)

    train_mask = days_full <= train_end
    for cond in ["C1","C2","C3"]:
        ax.plot(days_full[train_mask], obs_full[cond][train_mask],
                color=COLOR_OBS, linewidth=0.8, alpha=0.5, zorder=1)

    for cond in ["C1","C2","C3"]:
        cdf = df_test[df_test["Condition"]==cond]
        lw, al = (1.3, 0.7) if cond=="C1" else (0.6, 0.35)
        ax.plot(cdf["Day"], cdf["Observed_CO2"], color=COLOR_OBS, linewidth=lw, alpha=al, zorder=1)

    for cond in ["C1","C2","C3"]:
        cdf = df_test[df_test["Condition"]==cond]
        lw, al, ls = (1.6, 0.85, "-") if cond=="C1" else (0.7, 0.35, "--")
        ax.plot(cdf["Day"], cdf["LSMPR_Predicted"], color=COLOR_LS, linewidth=lw, alpha=al, linestyle=ls, zorder=3)

    for cond in ["C1","C2","C3"]:
        cdf = df_test[df_test["Condition"]==cond]
        lw, al, ls = (1.6, 0.85, "-") if cond=="C1" else (0.7, 0.35, "--")
        ax.plot(cdf["Day"], cdf["Persistence_Predicted"], color=COLOR_PERS, linewidth=lw, alpha=al, linestyle=ls, zorder=3)

    ax.axvline(x=train_end, color="black", linewidth=1.1, linestyle=(0,(5,3)), alpha=0.6, zorder=2)

    # Train/Test 标签
    ax.text(train_end/2, obs_full["C1"].max()*0.95, "Train", ha="center", va="top",
            fontsize=17, fontweight="black", color="#5D6D7E", alpha=0.8)
    ax.text(train_end+(test_end-train_end)/2, obs_full["C1"].max()*0.95, "Test", ha="center", va="top",
            fontsize=17, fontweight="black", color="#D4AC0D", alpha=0.8)

    ax.set_title(f"({chr(97+idx)}) {label}", pad=8)
    ax.set_xlabel("Days"); ax.set_ylabel("CO$_2$ release (g)")
    ax.set_xlim(-2, test_end+2)
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=True, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.text(0.98, 0.06, r2_labels[fname], transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

# ── 图例 ─────────────────────────────────────────────
leg_els = [
    Line2D([0],[0], color=COLOR_OBS, linewidth=2.0, label="Observed"),
    Line2D([0],[0], color=COLOR_LS, linewidth=2.0, label="LS-MPR"),
    Line2D([0],[0], color=COLOR_PERS, linewidth=2.0, label="Persistence"),
]
leg = fig.legend(handles=leg_els, loc="upper center", ncol=3,
                 framealpha=0.90, edgecolor="gray", fancybox=True,
                 fontsize=18, bbox_to_anchor=(0.5, 1.04))
for t in leg.get_texts(): t.set_fontweight("bold")

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_antifactual_trajectories_v3.png", dpi=600)
print(f"Saved: {OUT_DIR / 'fig_antifactual_trajectories_v3.png'}")
print("Done.")
plt.close(fig)