"""
Graphical Abstract for:
"Evaluation Protocols Shape Perceived Model Performance in Polymer Degradation Testing"
Target: Polymer Testing (Elsevier)
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

# ═══════════════════════════════════════════════════════
# SCI journal color palette (color-blind friendly, muted academic tones)
# ═══════════════════════════════════════════════════════
C_RANDOM    = '#2C3E50'   # dark slate — random split
C_TEMPORAL  = '#E74C3C'   # muted red — temporal holdout
C_PERSIST   = '#27AE60'   # green — persistence predictor
C_LSMPR     = '#3498DB'   # steel blue — LS-MPR
C_MECH      = '#8E44AD'   # purple — mechanistic model
C_CRISIS    = '#C0392B'   # deep red — crisis point
C_LIGHT     = '#ECF0F1'   # pale gray background
C_TEXT      = '#2C3E50'   # text color
C_ACCENT    = '#F39C12'   # amber accent
C_GREY      = '#95A5A6'   # annotation grey
C_WHITE     = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 9,
    'font.weight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ── Load data for degradation curve ──
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

backup = str(_REPO_ROOT / "data" / "pbat_pla_experimental_543.csv")
df = pd.read_csv(backup).dropna(subset=['Days','CO2_Release'])
c1 = df[(df['Temperature']==58) & (df['Humidity']==60) &
        (df['Ratio']==100) & (df['Compost_Volume']==0.5)].sort_values('Days')

# ═══════════════════════════════════════════════════════
# FIGURE: 3-panel horizontal Graphical Abstract
# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor(C_WHITE)

# ── Layout: 4 zones ──
# Left (x=0.02-0.38): Degradation curve story
# Center-left (x=0.40-0.58): R² bar comparison
# Center-right (x=0.60-0.78): Protocol matrix mini
# Right (x=0.80-0.98): Key message
# Bottom: Take-home message bar

# ═══════════════════════════════════════════════════════
# ZONE 1 (LEFT): Degradation curve with train/test split
# ═══════════════════════════════════════════════════════
ax1 = fig.add_axes([0.03, 0.18, 0.30, 0.68])
ax1.set_facecolor(C_WHITE)

days = c1['Days'].values
co2 = c1['CO2_Release'].values

# Shaded train region (Days 0-120)
ax1.axvspan(0, 120, facecolor='#EBF5FB', alpha=0.6, zorder=0)
ax1.axvspan(120, 180, facecolor='#FDEDEC', alpha=0.4, zorder=0)

# Degradation curve
ax1.plot(days, co2, color='#2C3E50', linewidth=2.5, zorder=3)

# Phase annotations
ax1.annotate('Growth\nPhase', xy=(45, 60), fontsize=7.5, fontweight='bold',
            color='#2471A3', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2471A3', alpha=0.85))
ax1.annotate('Deceleration', xy=(115, 210), fontsize=7, fontweight='bold',
            color='#7F8C8D', ha='center')
ax1.annotate('Plateau\nPhase', xy=(155, 245), fontsize=7.5, fontweight='bold',
            color='#C0392B', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#C0392B', alpha=0.85))

# Train/Test boundary
ax1.axvline(x=120, color='#E74C3C', linewidth=2.0, linestyle='--', zorder=2)
ax1.text(121, co2.max()*0.97, 'Train/Test\nBoundary', fontsize=6.5,
        fontweight='bold', color='#E74C3C', ha='left', va='top')

# Labels
ax1.set_xlabel('Days')
ax1.set_ylabel('CO$_2$ (g)')
ax1.set_xlim(-5, 190)
ax1.set_ylim(-10, 280)
ax1.tick_params(which='both', direction='in', labelsize=8)
ax1.text(0.02, 0.96, '(a) Degradation kinetics', transform=ax1.transAxes,
        fontsize=9, fontweight='bold', color=C_TEXT, va='top')

# ═══════════════════════════════════════════════════════
# ZONE 2 (CENTER-LEFT): R² bar comparison
# ═══════════════════════════════════════════════════════
ax2 = fig.add_axes([0.37, 0.18, 0.18, 0.68])
ax2.set_facecolor(C_WHITE)

# Data: R² values
methods = ['LS-MPR', 'Ridge', 'Lasso', 'SVR', 'RF', 'GPR', 'Exp\nDecay', 'Persist-\nence']
r2_random = [0.994, 0.993, 0.993, 0.997, 0.999, 1.000, 0.985, np.nan]
r2_temporal = [0.072, 0.424, 0.195, 0.890, 0.952, -5.596, -0.673, 0.955]

y_pos = np.arange(len(methods))
height = 0.35

# Random split bars (left side, negative values for left-facing)
bars_r = ax2.barh(y_pos + height/2, [min(r, 1.0) if not np.isnan(r) else 0 for r in r2_random],
                  height=height, color=C_RANDOM, alpha=0.85, edgecolor='white', linewidth=0.5,
                  label='Random Split', zorder=2)
# Temporal holdout bars
bars_t = ax2.barh(y_pos - height/2, [max(r, -1.5) for r in r2_temporal],
                  height=height, color=C_TEMPORAL, alpha=0.85, edgecolor='white', linewidth=0.5,
                  label='Temporal Holdout', zorder=2)

# GPR catastrophic bar (truncated, with annotation)
# R² = -5.596 is way off scale

ax2.axvline(x=0, color='black', linewidth=0.8, zorder=1)
ax2.axvline(x=1.0, color='#BDC3C7', linewidth=0.5, linestyle=':', alpha=0.5)

# Highlight Persistence
ax2.annotate('Best\n(0 params!)', xy=(1.02, y_pos[-1] - height/2),
            fontsize=7, fontweight='bold', color=C_PERSIST, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=C_PERSIST, alpha=0.9))

ax2.set_yticks(y_pos)
ax2.set_yticklabels(methods, fontsize=7)
ax2.set_xlabel('R$^2$', fontsize=9)
ax2.set_xlim(-1.5, 1.2)
ax2.tick_params(which='both', direction='in', labelsize=7)
ax2.text(0.5, 1.02, '(b) R$^2$ under two protocols', transform=ax2.transAxes,
        fontsize=8, fontweight='bold', color=C_TEXT, ha='center', va='bottom')

# Legend
leg = ax2.legend(loc='lower right', fontsize=6.5, framealpha=0.9, edgecolor='#BDC3C7')
for t in leg.get_texts():
    t.set_fontweight('bold')

# ═══════════════════════════════════════════════════════
# ZONE 3 (CENTER-RIGHT): The "96% leakage" visualization
# ═══════════════════════════════════════════════════════
ax3 = fig.add_axes([0.58, 0.18, 0.18, 0.68])
ax3.set_facecolor(C_WHITE)
ax3.axis('off')

# Big number: 96%
ax3.text(0.5, 0.78, '~96%', fontsize=48, fontweight='bold', color=C_CRISIS,
        ha='center', va='center')
ax3.text(0.5, 0.65, 'of apparent R$^2$', fontsize=10, fontweight='bold', color=C_TEXT,
        ha='center', va='center')
ax3.text(0.5, 0.58, 'comes from temporal\ndata leakage', fontsize=9, fontweight='bold',
        color=C_GREY, ha='center', va='center')

# Arrow: 0.994 → 0.035
ax3.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.52),
            arrowprops=dict(arrowstyle='->', color=C_CRISIS, lw=2.5))
ax3.text(0.5, 0.40, 'R$^2$ = 0.994 $\\rightarrow$ 0.035', fontsize=9,
        fontweight='bold', color=C_CRISIS, ha='center', va='center')
ax3.text(0.5, 0.34, 'p < 10$^{-50}$, d = 57.0', fontsize=7.5,
        fontweight='bold', color=C_GREY, ha='center', va='center')

# Key insight box
box = FancyBboxPatch((0.05, 0.02), 0.90, 0.28, boxstyle='round,pad=0.1',
                     facecolor='#FDEDEC', edgecolor=C_CRISIS, linewidth=1.5, alpha=0.6)
ax3.add_patch(box)
ax3.text(0.5, 0.16, 'Temporal adjacency — not\ndegradation kinetics — drives\napparent model performance',
        fontsize=8, fontweight='bold', color=C_CRISIS, ha='center', va='center',
        transform=ax3.transAxes)

# ═══════════════════════════════════════════════════════
# ZONE 4 (RIGHT): Key message + protocol comparison
# ═══════════════════════════════════════════════════════
ax4 = fig.add_axes([0.80, 0.18, 0.18, 0.68])
ax4.set_facecolor(C_WHITE)
ax4.axis('off')

# Protocol comparison mini
protocols = [
    ('Random\nSplit', C_RANDOM, 'Interpolation\n(leakage)', '✗'),
    ('Temporal\nHoldout', C_TEMPORAL, 'Extrapolation\n(real test)', '✓'),
    ('Temporal\nShuffle', C_MECH, 'Leakage\nquantification', '✓'),
    ('LOCO', C_ACCENT, 'Cross-condition\ngeneralization', '✓'),
]

for i, (name, color, desc, check) in enumerate(protocols):
    y_base = 0.82 - i * 0.18

    # Protocol name box
    box = FancyBboxPatch((0.05, y_base - 0.06), 0.90, 0.14,
                         boxstyle='round,pad=0.08', facecolor=color, edgecolor='white',
                         linewidth=1.0, alpha=0.15)
    ax4.add_patch(box)
    ax4.text(0.5, y_base + 0.01, name, fontsize=7, fontweight='bold', color=color,
            ha='center', va='center')
    ax4.text(0.5, y_base - 0.06, desc, fontsize=5.8, fontweight='bold', color=C_GREY,
            ha='center', va='center')

ax4.text(0.5, 0.02, 'Evaluation protocol —\nnot model architecture —\nis the dominant factor',
        fontsize=7.5, fontweight='bold', color=C_TEXT, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FEF9E7', ec=C_ACCENT, alpha=0.8))

# ═══════════════════════════════════════════════════════
# BOTTOM: Take-home message bar
# ═══════════════════════════════════════════════════════
fig.text(0.5, 0.04,
        'Random split masks model failure in time-structured polymer degradation data. '
        'A zero-parameter persistence predictor outperforms all ML methods under temporal holdout. '
        'Evaluation protocol choice is the dominant determinant of apparent model performance.',
        ha='center', fontsize=8.5, fontweight='bold', color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.3', fc=C_LIGHT, ec='#BDC3C7', alpha=0.8))

# ═══════════════════════════════════════════════════════
# TOP: Title
# ═══════════════════════════════════════════════════════
fig.text(0.5, 0.95,
        'Evaluation Protocols Shape Perceived Model Performance\nin Polymer Degradation Testing',
        ha='center', fontsize=13, fontweight='bold', color=C_TEXT)

# ── SAVE ──
OUT_DIR = Path(str(_REPO_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(OUT_DIR / 'fig_graphical_abstract.png', dpi=600, facecolor=C_WHITE)
fig.savefig(OUT_DIR / 'fig_graphical_abstract.svg', facecolor=C_WHITE)
print(f'Saved: {OUT_DIR / "fig_graphical_abstract.png"}')
print(f'Saved: {OUT_DIR / "fig_graphical_abstract.svg"}')
print('Done.')
plt.close(fig)