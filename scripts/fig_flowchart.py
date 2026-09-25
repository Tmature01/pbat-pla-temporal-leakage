"""
Figure: 方法学总览流程图 — 5 阶段
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "font.weight": "bold",
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18); ax.set_ylim(0, 12)
ax.axis("off")

C_STAGE = {"1": "#2C3E50", "2": "#2471A3", "3": "#E74C3C",
           "4": "#8E44AD", "5": "#16A085"}
C_BG = {"1": "#EBEDEF", "2": "#EAF2F8", "3": "#FDEDEC",
        "4": "#F4ECF7", "5": "#E8F8F5"}

def node(ax, x, y, w, h, txt, c, style="box"):
    if style == "rounded":
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                             linewidth=1.5, facecolor=c, edgecolor="white", alpha=0.9)
    elif style == "dashed":
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                             linewidth=1.5, facecolor=c, edgecolor="white",
                             linestyle="dashed", alpha=0.9)
    else:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                             linewidth=1.2, facecolor=c, edgecolor="white", alpha=0.9)
    ax.add_patch(box)
    tc = "white" if c != "#27AE60" else "#1E5B3A"
    ax.text(x, y, txt, ha="center", va="center", fontsize=6.0,
            fontweight="bold", color=tc)

def arrow(ax, x1, y1, x2, y2, lbl=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.2))
    if lbl:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.12, lbl, ha="center", fontsize=5.5,
                fontweight="bold", color="#7F8C8D")

# ── 5 个阶段背景 ─────────────────────────────────────
stages = [
    ("STAGE 1: Experimental Design", 11.2, 10.6, 11.8, "#EBEDEF", "#2C3E50"),
    ("STAGE 2: Data Preprocessing", 9.6, 8.8, 10.4, "#EAF2F8", "#2471A3"),
    ("STAGE 3: Model Construction", 7.2, 6.3, 8.1, "#FDEDEC", "#E74C3C"),
    ("STAGE 4: Evaluation Protocols", 4.6, 4.0, 5.2, "#F4ECF7", "#8E44AD"),
    ("STAGE 5: Diagnostic Toolkit", 2.0, 1.2, 2.8, "#E8F8F5", "#16A085"),
]
for _, yc, y0, y1, bg, sc in stages:
    ax.add_patch(plt.Rectangle((0.3, y0), 17.4, y1-y0,
                 facecolor=bg, edgecolor=sc, linewidth=1.0, alpha=0.4, zorder=0))

for lbl, yc, _, _, _, sc in stages:
    ax.text(0.5, yc + 0.55, lbl, fontsize=7.5, fontweight="bold", color=sc, va="top")

# ══════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════
node(ax, 3.5, 11.2, 4.0, 0.60, "3 Conditions (C1, C2, C3)\nT, H, Ratio, CV varied", "#2C3E50", "rounded")
node(ax, 8.5, 11.2, 3.0, 0.60, "181 Days Each\n(Day 0-180, daily)", "#34495E", "box")
node(ax, 13.0, 11.2, 3.2, 0.60, "3 Targets: CO2, Residual,\nTensile (543 obs total)", "#2C3E50", "rounded")
arrow(ax, 5.5, 11.2, 7.0, 11.2)
arrow(ax, 10.0, 11.2, 11.4, 11.2)
arrow(ax, 9.0, 10.8, 9.0, 10.3, "Raw data")

# ══════════════════════════════════════════════════════
# STAGE 2
# ══════════════════════════════════════════════════════
s2 = [(3.0, "Missing\n(mean)"), (5.5, "3-sigma\nOutlier"),
      (8.0, "Z-score\nStandardize"), (10.5, "MinMax\n[0,1]"),
      (13.5, "Poly(d=2)\n5->21")]
for x, t in s2:
    node(ax, x, 9.6, 1.8, 0.55, t, "#2471A3", "box")
for i in range(len(s2)-1):
    arrow(ax, s2[i][0]+0.9, 9.6, s2[i+1][0]-0.9, 9.6)
arrow(ax, 13.5, 9.0, 13.5, 8.3, "21 terms")

# ══════════════════════════════════════════════════════
# STAGE 3
# ══════════════════════════════════════════════════════
node(ax, 3.5, 7.7, 3.5, 0.55, "LS-MPR (Focal)\nOLS, d=2, 21 coeff", "#E74C3C", "rounded")
node(ax, 14.0, 7.7, 4.5, 0.55, "Baselines: Persistence(0p), Mean,\nExp Decay(2p, mechanistic)", "#27AE60", "dashed")
s3 = [(3.0,"Ridge"),(5.5,"Lasso"),(8.0,"SVR"),(10.5,"RF"),(13.0,"GPR"),(15.5,"GA")]
for x, t in s3:
    node(ax, x, 6.8, 1.8, 0.50, t, "#E67E22", "box")
arrow(ax, 9.0, 6.2, 9.0, 5.3)

# ══════════════════════════════════════════════════════
# STAGE 4
# ══════════════════════════════════════════════════════
s4 = [(3.0,"Random Split\nInterpolation"),
      (6.5,"Temporal Holdout\nExtrapolation"),
      (10.0,"LOCO\nCross-condition"),
      (13.5,"Temp. Shuffle\nLeakage test")]
for x, t in s4:
    node(ax, x, 4.6, 2.5, 0.55, t, "#8E44AD", "box")
arrow(ax, 9.0, 3.9, 9.0, 3.1)

# ══════════════════════════════════════════════════════
# STAGE 5
# ══════════════════════════════════════════════════════
s5 = [(2.0,"1: VIF /\nCorrelation"),(5.0,"2: Temporal\nDecomp"),
      (8.0,"3: Counter-\nfactual"),(11.0,"4: Feature\nAblation"),
      (14.0,"5: SVD\nSpectrum"),(16.8,"6: ACF of\nResiduals")]
for x, t in s5:
    node(ax, x, 2.0, 2.2, 0.55, t, "#16A085", "box")

# ── 底部注释 ─────────────────────────────────────────
fig.text(0.5, 0.005, "3 conditions x 181 days -> 5 preprocessing steps -> "
         "7 models + 3 baselines -> 4 evaluation protocols -> 6 diagnostic tools",
         ha="center", fontsize=7, fontweight="bold", color="#7F8C8D")

# ── 保存 ─────────────────────────────────────────────
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "fig_flowchart.png", dpi=600)
fig.savefig(OUT_DIR / "fig_flowchart.svg")
print(f"Saved: {OUT_DIR / 'fig_flowchart.png'}")
print(f"Saved: {OUT_DIR / 'fig_flowchart.svg'}")
print("Done.")
plt.close(fig)