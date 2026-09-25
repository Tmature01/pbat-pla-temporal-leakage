"""
Figure: 跨领域 R²/RMSE 通胀对比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from pathlib import Path

# ── 1. 数据 ──────────────────────────────────────────
data = [
    ("This study\n(PBAT/PLA)",      "Polymer Sci.",   "R^2 inflation",          96.5, True),
    ("Soil carbon\nspatial [21]",   "Soil Sci.",      "R^2 inflation",          88.1, False),
    ("Sequential\nrecommend. [23]", "RecSys/ML",      "Literature prevalence",  77.0, False),
    ("Material ML\nbenchmarks",     "Materials Sci.", "Literature prevalence",  65.0, False),
    ("Electrochemical\npred. [22]", "Electrochem.",   "RMSE inflation",         35.0, False),
    ("LSTM water\nquality [32]",    "Environ. Eng.",  "RMSE inflation",         20.5, False),
]

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

fig, ax = plt.subplots(figsize=(10, 4.8))
fig.subplots_adjust(left=0.24, right=0.92, top=0.90, bottom=0.16)

values = [d[3] for d in data][::-1]
labels = [d[0] for d in data][::-1]
fields = [d[1] for d in data][::-1]
mtypes = [d[2] for d in data][::-1]
hls    = [d[4] for d in data][::-1]

y = np.arange(6)
colors = []
for h, mt in zip(hls, mtypes):
    if h:
        colors.append("#C0392B")
    elif "RMSE" in mt:
        colors.append("#A8C8E0")
    else:
        colors.append("#5B8DB8")

ax.barh(y, values, height=0.55, color=colors, edgecolor="white", linewidth=1.0, zorder=3)

for i, (v, h) in enumerate(zip(values, hls)):
    ax.text(v + 1.0, i, f"{v:.1f}%", fontsize=10, fontweight="bold",
            color="#C0392B" if h else "#2C3E50", va="center")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Performance inflation (%)")
ax.set_xlim(0, 130)
ax.tick_params(which="both", direction="in", bottom=True, top=False, left=False, right=False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.invert_yaxis()

for i, (fld, mt) in enumerate(zip(fields, mtypes)):
    ax.text(127, i, f"{fld}  ({mt})", fontsize=6.2, fontweight="bold",
            color="#7F8C8D", va="center", ha="right")

ax.set_title("Cross-discipline performance inflation from ignoring\n"
             "temporal/spatial structure", pad=10)

leg_els = [
    Patch(facecolor="#C0392B", label="This study (R^2 inflation)"),
    Patch(facecolor="#5B8DB8", label="Other studies (R^2/literature)"),
    Patch(facecolor="#A8C8E0", label="Other studies (RMSE)"),
]
leg = ax.legend(handles=leg_els, loc="upper left", ncol=1,
                framealpha=0.85, edgecolor="#BDC3C7", fancybox=True,
                fontsize=7.5, bbox_to_anchor=(0.41, 0.965))
for t in leg.get_texts(): t.set_fontweight("bold")

fig.text(0.5, 0.025,
         "6 studies across 5 disciplines. This study's 96.5% is the most extreme case. "
         "R^2 and RMSE metrics not directly comparable — shown for cross-domain context only.",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_cross_discipline.png", dpi=600)
fig.savefig(OUT_DIR / "fig_cross_discipline.svg")
print(f"Saved: {OUT_DIR / 'fig_cross_discipline.png'}")
print(f"Saved: {OUT_DIR / 'fig_cross_discipline.svg'}")
print("Done.")
plt.close(fig)