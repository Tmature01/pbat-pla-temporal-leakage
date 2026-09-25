"""
Generate all 11 figures per Nature standards for V10 manuscript.
Based on: Claude_Code_绘图指令_Nature标准.md
Data source: V10_Data_Inventory.md
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from fig_style import *

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")


FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def save(fig, name):
    for ext in ['.pdf', '.png']:
        p = os.path.join(FIG_DIR, name + ext)
        fig.savefig(p)
    print(f'  [OK] {name}')
    plt.close(fig)

# ================================================================
# FIG A: Feature Importance — Real vs Synthetic
# ================================================================
def figA():
    features = ['Days', 'Temperature', 'Humidity', 'Ratio', 'Compost Vol.']
    # Real data: ΔR² when dropped (negative = model gets worse = feature IS important)
    delta_real = [-33.042, +0.060, +0.072, -0.006, +0.013]
    # Synthetic PBAT/TPS Residual
    delta_tps = [-0.660, -0.130, -0.176, -0.446, +0.011]
    # Importance = -ΔR² (positive = important)
    imp_real = [-d for d in delta_real]
    imp_tps  = [-d for d in delta_tps]

    fig = plt.figure(figsize=(W2, 70/25.4))
    # Panel (a): Real data
    ax1 = fig.add_subplot(121)
    colors = [WONG['blue'] if f == 'Days' else '#888888' for f in features]
    bars = ax1.barh(range(len(features)), imp_real, color=colors, height=0.55,
                    edgecolor=['navy' if f=='Days' else '#666666' for f in features],
                    linewidth=[1.5 if f=='Days' else 0.5 for f in features])
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features)
    ax1.set_xlabel('Importance (−ΔR²)')
    ax1.set_title('(a) Real data — Temporal holdout', fontweight='bold', fontsize=8, loc='left')
    # Broken axis: split at ~1.5
    ax1.set_xlim(-1, 1.5)
    # Annotate Days value separately
    ax1.text(0.95, 0.95, f'Days: {imp_real[0]:.1f}', transform=ax1.transAxes,
             fontsize=7, fontweight='bold', color=WONG['blue'], ha='right', va='top')
    for bar, val in zip(bars, imp_real):
        if abs(val) < 2:
            ax1.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', fontsize=6.5)
    clean_spines(ax1)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.invert_yaxis()

    # Panel (b): Synthetic PBAT/TPS
    ax2 = fig.add_subplot(122)
    bars = ax2.barh(range(len(features)), imp_tps, color=colors, height=0.55,
                    edgecolor=['navy' if f=='Days' else '#666666' for f in features],
                    linewidth=[1.5 if f=='Days' else 0.5 for f in features])
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features)
    ax2.set_xlabel('Importance (−ΔR²)')
    ax2.set_title('(b) Synthetic data — Random split', fontweight='bold', fontsize=8, loc='left')
    for bar, val in zip(bars, imp_tps):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', fontsize=6.5)
    clean_spines(ax2)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.invert_yaxis()

    fig.suptitle('Figure A. Feature importance: real (rank-deficient) vs synthetic (full-rank) data.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figA_feature_importance')

# ================================================================
# FIG B: Symbolic Regression Accuracy-Parsimony
# ================================================================
def figB():
    models = ['Quadratic', 'Logistic', 'Exponential', 'M-Menten', 'Power Law']
    n_params = [3, 3, 2, 2, 2]
    r2 = [0.995, 0.992, 0.988, 0.977, 0.929]
    rmse = [5.12, 6.34, 7.89, 10.85, 18.94]
    interpretable = [False, True, True, True, False]

    fig, ax = plt.subplots(figsize=(W1, 80/25.4))

    sizes = np.interp(rmse, [5, 19], [50, 200])
    colors = [WONG['green'] if i else '#888888' for i in interpretable]

    for i, (m, n, r, s, c, inter) in enumerate(zip(models, n_params, r2, sizes, colors, interpretable)):
        ax.scatter(n, r, s=s, c=c, edgecolors='white', linewidth=1.2, zorder=5)
        offset = 0.06 if m != 'Quadratic' else -0.06
        voff = 0.0015 if m != 'Logistic' else -0.0025
        ax.annotate(m, (n + offset, r + voff), fontsize=6.5, color=c, fontweight='bold')
        if m == 'Exponential':
            ax.annotate('', xy=(n, r), xytext=(n+0.25, r+0.008),
                       arrowprops=dict(arrowstyle='->', color=WONG['red'], lw=1.5))
            ax.text(n + 0.28, r + 0.008, 'Selected', fontsize=6, color=WONG['red'],
                   fontweight='bold', va='center')

    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('R² (mean across 3 conditions)')
    ax.set_xticks([2, 3])
    ax.set_xlim(1.6, 3.4)
    ax.set_ylim(0.91, 1.0)
    ax.grid(alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=WONG['green'], label='Mechanistically interpretable'),
           Patch(facecolor='#888888', label='Empirical (non-interpretable)')]
    ax.legend(handles=leg, loc='lower left', fontsize=6.5, frameon=False)

    # Annotation box
    ax.text(0.98, 0.93, 'd=2 balances\naccuracy & parsimony', transform=ax.transAxes,
            fontsize=6.5, color='gray', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    clean_spines(ax)
    fig.suptitle('Figure B. Symbolic degradation forms: accuracy-parsimony trade-off.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figB_symbolic_regression')

# ================================================================
# FIG C: Pareto Frontier — Computational Cost vs Accuracy
# ================================================================
def figC():
    methods = ['LS-MPR', 'Ridge', 'Lasso', 'RF', 'SVR', 'GPR-Poly', 'GA']
    train_time = [0.003, 0.2, 0.3, 0.16, 2.3, 6.0, 35.0]
    r2_temporal = [-0.058, 0.013, -0.052, -0.312, -0.203, 0.162, -1.447]

    fig, ax = plt.subplots(figsize=(W1, 85/25.4))

    for i, (m, t, r2t) in enumerate(zip(methods, train_time, r2_temporal)):
        if m == 'LS-MPR':
            c, ec, lw, s = WONG['blue'], 'navy', 1.5, 70
        elif m == 'GPR-Poly':
            c, ec, lw, s = WONG['orange'], WONG['orange'], 1.0, 60
        else:
            c, ec, lw, s = '#AAAAAA', '#888888', 0.5, 50
        ax.scatter(t, r2t, s=s, c=c, edgecolors=ec, linewidths=lw, zorder=5)
        # Label
        off_y = 0.08 if m not in ['GA', 'RF'] else -0.12
        off_x = 1.3 if m == 'LS-MPR' else 1.1
        ax.annotate(m, (t * off_x, r2t + off_y), fontsize=6.5, color=c if m in ['LS-MPR','GPR-Poly'] else '#555555',
                   fontweight='bold' if m == 'LS-MPR' else 'normal')

    ax.set_xscale('log')
    ax.set_xlabel('Training time (s)')
    ax.set_ylabel('Temporal holdout R²')
    ax.axhline(y=0, color=WONG['red'], linewidth=0.8, linestyle='--', alpha=0.6, label='Mean predictor baseline')
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=6.5, frameon=False, loc='lower right')
    clean_spines(ax)
    fig.suptitle('Figure C. Computational cost vs. forward-prediction accuracy.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figC_pareto_frontier')

# ================================================================
# FIG D: Kinetic Parameters a vs b (6 materials)
# ================================================================
def figD():
    materials = ['PBAT/PLA', 'PBAT1', 'PBAT2', 'PLA', 'PBAT+PGA', 'PBAT+Talc']
    a_vals = [124.1, 181.2, 184.9, 173.5, 127.3, 178.6]
    b_vals = [0.021, 0.018, 0.017, 0.019, 0.020, 0.015]
    r2_fit = [0.988, 0.988, 0.987, 0.992, 0.991, 0.990]
    # Category colors
    cat_colors = {'blend': WONG['blue'], 'pure': WONG['red'], 'additive': WONG['pink']}
    categories = ['blend', 'pure', 'pure', 'pure', 'blend', 'additive']
    markers = {'blend': 'o', 'pure': 's', 'additive': 'D'}

    fig, ax = plt.subplots(figsize=(W1, 80/25.4))

    for m, a, b, r2, cat in zip(materials, a_vals, b_vals, r2_fit, categories):
        sz = np.interp(r2, [0.985, 0.995], [60, 120])
        ax.scatter(b, a, s=sz, c=cat_colors[cat], marker=markers[cat],
                  edgecolors='white', linewidth=1, zorder=5)
        # Label placement
        yo = 3 if m not in ['PBAT2', 'PBAT1'] else -5
        xo = 0.0002 if m != 'PBAT+Talc' else -0.001
        ax.annotate(m, (b + xo, a + yo), fontsize=6.5, ha='center')

    ax.set_xlabel('b — degradation rate constant (day$^{-1}$)')
    ax.set_ylabel('a — ultimate CO$_2$ capacity (g)')
    # Legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=WONG['blue'], label='Blend (PBAT/PLA, PBAT+PGA)'),
           Patch(facecolor=WONG['red'], label='Pure (PBAT1, PBAT2, PLA)'),
           Patch(facecolor=WONG['pink'], label='Additive (PBAT+Talc)')]
    ax.legend(handles=leg, fontsize=6, frameon=False, loc='lower right')
    ax.text(0.98, 0.03, 'Point size $\\propto$ R$^2$', transform=ax.transAxes, fontsize=6, color='gray', ha='right')
    # Arrow annotation
    ax.annotate('Faster + Higher capacity →', xy=(0.019, 180), xytext=(0.020, 188),
               fontsize=6.5, color='gray', ha='center',
               arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    clean_spines(ax)
    ax.tick_params(direction='in')
    fig.suptitle('Figure D. First-order kinetic parameters across six PBAT-based materials.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figD_kinetics_ab')

# ================================================================
# FIG E: Full Method × Dataset R² Heatmap
# ================================================================
def figE():
    import pandas as pd
    rows = [
        ('PBAT/PLA', 'Residual', 0.9938, 0.9938, 0.9937, 0.9928, 0.9974, 0.9991),
        ('PBAT/PLA', 'CO₂',     0.9908, 0.9908, 0.9908, 0.9496, 0.9970, 0.9972),
        ('PBAT/PLA', 'Tensile', 0.9780, 0.9780, 0.9780, 0.9616, 0.9977, 0.9990),
        ('PBAT/TPS', 'Residual', 0.8825, 0.8833, 0.8841, 0.1468, 0.9731, 0.8218),
        ('PBAT/TPS', 'CO₂',     0.8825, 0.8833, 0.8841, 0.0750, 0.9739, 0.8183),
        ('PBAT/TPS', 'Tensile', 0.9410, 0.9425, 0.9475, 0.5555, 0.9471, 0.9166),
        ('PBAT/PBS', 'Residual', 0.9633, 0.9632, 0.9632, -0.4512, 0.9353, 0.8715),
        ('PBAT/PBS', 'CO₂',     0.9577, 0.9580, 0.9583, 0.3467, 0.9465, 0.8586),
        ('PBAT/PBS', 'Tensile', 0.9570, 0.9570, 0.9576, 0.7213, 0.9357, 0.9279),
    ]
    methods = ['LS-MPR', 'Ridge', 'Lasso', 'GA', 'SVR', 'RF']
    matrix = np.array([[r[2+i] for i in range(6)] for r in rows])
    y_labels = [f'{r[0]} | {r[1]}'.replace('CO₂', 'CO$_2$') for r in rows]

    fig, ax = plt.subplots(figsize=(W2, 90/25.4))

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1.0)
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', norm=norm)

    for i in range(9):
        for j in range(6):
            val = matrix[i, j]
            txt = f'{val:.3f}' if val > -1 else f'{val:.2f}'
            tc = 'white' if val < 0.3 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color=tc, fontweight='bold')

    ax.set_xticks(range(6))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_yticks(range(9))
    ax.set_yticklabels(y_labels, fontsize=6.5)

    # Dataset group separators
    for y_sep in [2.5, 5.5]:
        ax.axhline(y=y_sep, color='black', linewidth=1.2)

    # Add dataset labels on the right
    for y_pos, label in [(1, 'PBAT/PLA'), (4, 'PBAT/TPS'), (7, 'PBAT/PBS')]:
        ax.text(6.5, y_pos, label, fontsize=7, fontweight='bold', va='center', rotation=270)

    # Red triangle markers for negative values (GA on TPS/PBS)
    for i in range(9):
        for j in range(6):
            if matrix[i, j] < 0.2 and matrix[i, j] > -1:
                ax.plot(j + 0.3, i - 0.3, marker='v', color=WONG['red'], markersize=5, zorder=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.85)
    cbar.set_label('Random-split R²', fontsize=7)
    clean_spines(ax)
    fig.suptitle('Figure E. Seven-method R² heatmap across datasets and targets (random split).',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figE_heatmap_full')


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('Generating Figures A–E (Nature standard)...')
    figA()
    figB()
    figC()
    figD()
    figE()
    print('Done.')
