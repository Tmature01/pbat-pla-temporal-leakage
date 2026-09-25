"""
Figure V2: 逐条件逐时段误差热图 (画布不变 13x4.5, 文字x1.5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(FIGURE_DATA / "Fig_ErrorHeatmap.csv"))
days = df["Day"].values
c1_err = df["C1_AbsError"].values
c2_err = df["C2_AbsError"].values
c3_err = df["C3_AbsError"].values
vmax = 68

for name, err in [("C1", c1_err), ("C2", c2_err), ("C3", c3_err)]:
    tm = err[days<=119].mean(); tem = err[days>=120].mean()
    print(f"{name}: train MAE={tm:.1f}, test MAE={tem:.1f}, x{tem/tm:.0f}")

cmap = LinearSegmentedColormap.from_list("error_cmap", [
    (0.00,"#FFFFFF"),(0.05,"#FFF5EB"),(0.15,"#FDEBD0"),(0.35,"#F5B041"),
    (0.60,"#E74C3C"),(0.85,"#922B21"),(1.00,"#4A0E0E")])

# ── 字体 x1.5, 画布不变 ──────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 13.5, "font.weight": "bold",
    "axes.labelsize": 16.5, "axes.labelweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

fig, axes = plt.subplots(3, 1, figsize=(13, 4.5))
fig.subplots_adjust(left=0.12, right=0.92, top=0.90, bottom=0.14, hspace=0.30)

cond_data = [("C1", c1_err), ("C2", c2_err), ("C3", c3_err)]
cond_labels = ["C1 (T=50, H=70, R=70)", "C2 (T=58, H=60, R=100)", "C3 (T=58, H=70, R=0)"]

for idx, (ax, (cname, err_data)) in enumerate(zip(axes, cond_data)):
    X_mesh = np.arange(0.5, 181.5, 1)
    Y_mesh = np.array([0, 1.6])
    err_2d = err_data.reshape(1, -1)
    im = ax.pcolormesh(X_mesh, Y_mesh, err_2d, cmap=cmap, vmin=0, vmax=vmax,
                        edgecolors="none", linewidth=0, rasterized=True)
    ax.axvline(x=119.5, color="black", linewidth=1.4, zorder=5)

    train_mae = err_data[days<=119].mean()
    test_mae  = err_data[days>=120].mean()
    ratio = test_mae / train_mae

    # MAE 标注: 热图条内部 (Y_mesh 加高到 1.6, 文字不溢出)
    ax.text(60, 0.8, f"Train MAE={train_mae:.1f}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#2C3E50")
    ax.text(150, 0.8, f"Test MAE={test_mae:.1f}  (x{ratio:.0f})", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#C0392B")

    ax.set_ylabel(cname, fontsize=14, labelpad=10, rotation=0, ha="right", va="center")
    ax.set_ylim(-0.1, 1.65)
    ax.set_xlim(0.5, 180.5)
    ax.set_yticks([])
    if idx == 2:
        ax.set_xlabel("Days")
        ax.set_xticks([0,30,60,90,120,150,180])
    else:
        ax.set_xticks([0,30,60,90,120,150,180])
        ax.set_xticklabels([])
    ax.tick_params(which="both", direction="in", bottom=True, top=False, left=False, right=False)

    ax.text(0.5, 0.88, cond_labels[idx], transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#7F8C8D", ha="left", va="top")

cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015,
                    ticks=[0, 10, 20, 30, 40, 50, 60, 68])
cbar.set_label("Absolute error (g)", fontweight="bold", fontsize=13.5, labelpad=5)
cbar.ax.tick_params(length=0)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_error_heatmap_v2.png", dpi=600)
fig.savefig(OUT_DIR / "fig_error_heatmap_v2.svg")
print(f"Saved: {OUT_DIR / 'fig_error_heatmap_v2.png'}")
print("Done.")
plt.close(fig)