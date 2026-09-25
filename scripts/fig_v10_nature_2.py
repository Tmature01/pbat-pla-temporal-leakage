"""
Figures F–K per Nature standards for V10 manuscript.
Based on: Claude_Code_绘图指令_Nature标准.md
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import seaborn as sns
from fig_style import *

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")


FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def save(fig, name):
    for ext in ['.pdf', '.png']:
        fig.savefig(os.path.join(FIG_DIR, name + ext))
    print(f'  [OK] {name}')
    plt.close(fig)

# ================================================================
# FIG F: Feature Ablation Waterfall (Temporal Holdout)
# ================================================================
def figF():
    baseline_r2 = -0.058
    features = ['Days', 'Humidity', 'Temperature', 'Compost Vol.', 'Ratio']
    r2_after = [-33.100, +0.014, +0.002, -0.045, -0.064]
    delta = [r - baseline_r2 for r in r2_after]
    # Sort by |delta| descending
    idx = np.argsort([abs(d) for d in delta])[::-1]
    features = [features[i] for i in idx]
    delta = [delta[i] for i in idx]
    r2_after = [r2_after[i] for i in idx]

    fig = plt.figure(figsize=(W1, 95/25.4))
    gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.08)

    # Top axes: Days (large negative)
    ax_top = fig.add_subplot(gs[0])
    # Bottom axes: other features
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    colors = []
    for d in delta:
        if d < -1: colors.append(WONG['red'])
        elif d < 0: colors.append('#F4A582')
        else: colors.append(WONG['green'])

    for ax in [ax_top, ax_bot]:
        x = np.arange(len(features))
        bars = ax.bar(x, delta, color=colors, edgecolor='white', width=0.6)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        for bar, val in zip(bars, delta):
            y_pos = bar.get_height()
            voff = 0.15 if y_pos >= 0 else -0.25
            ax.text(bar.get_x() + bar.get_width()/2, y_pos + voff,
                    f'{val:+.3f}' if abs(val) < 1 else f'{val:+.1f}',
                    ha='center', fontsize=6.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=15, ha='right')
        clean_spines(ax)

    # Set different y-limits
    ax_top.set_ylim(-34.5, -32.5)
    ax_bot.set_ylim(-0.15, 0.15)

    # Break axis indicators
    d_val = 0.015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=0.8)
    ax_top.plot((-d_val, +d_val), (-d_val*3, +d_val*3), **kwargs)
    ax_top.plot((-d_val, +d_val), (1-d_val*3, 1+d_val*3), **kwargs)
    kwargs2 = dict(transform=ax_bot.transAxes, color='k', clip_on=False, linewidth=0.8)
    ax_bot.plot((-d_val, +d_val), (-d_val*3, +d_val*3), **kwargs2)
    ax_bot.plot((-d_val, +d_val), (1-d_val*3, 1+d_val*3), **kwargs2)

    ax_bot.set_xlabel('Feature removed')
    fig.text(-0.02, 0.5, 'ΔR² (change when feature dropped)', va='center', rotation=90, fontsize=8)
    ax_top.set_title('(a) Feature ablation — Temporal holdout', fontweight='bold', fontsize=8, loc='left')

    # Annotation for Days
    ax_top.annotate('Dominant feature\n(ΔR² = −33.0)', xy=(0, -33.042), xytext=(0.3, -33.2),
                   fontsize=7, color=WONG['red'], fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=WONG['red'], lw=1.2))

    fig.suptitle('Figure F. Feature ablation waterfall: Days dominates under temporal holdout.',
                 fontsize=8, y=1.03)
    save(fig, 'figF_waterfall_ablation')

# ================================================================
# FIG G: Counterfactual Validation — 4 Window R²
# ================================================================
def figG():
    windows = ['0–29/30–90\n(Growth)', '0–59/60–120\n(Growth+Decay)',
               '0–89/90–150\n(Mixed)', '0–119/120–180\n(Plateau)']
    r2_persist = [-3.674, -1.807, +0.437, +0.951]
    r2_lsmpr   = [+0.022, -0.322, +0.849, -0.058]
    skill_score = [+0.79, +0.53, +0.73, -20.6]

    fig = plt.figure(figsize=(W1, 110/25.4))
    gs = GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.15)

    # Top: Grouped bar chart
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(4)
    w = 0.35
    b1 = ax1.bar(x - w/2, r2_persist, w, color='#AAAAAA', edgecolor='white', label='Persistence')
    b2 = ax1.bar(x + w/2, r2_lsmpr, w, color=WONG['blue'], edgecolor='white', label='LS-MPR')
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax1.set_ylabel('Test R²')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['']*4)
    ax1.legend(fontsize=6.5, frameon=False, loc='upper left')

    # Value labels on bars
    for bars, vals in [(b1, r2_persist), (b2, r2_lsmpr)]:
        for bar, val in zip(bars, vals):
            ypos = bar.get_height()
            voff = 0.07 if ypos >= 0 else -0.25
            ax1.text(bar.get_x() + bar.get_width()/2, ypos + voff,
                    f'{val:+.3f}' if abs(val) < 1 else f'{val:+.2f}',
                    ha='center', fontsize=6, rotation=90)

    # Background shading
    ax1.axhspan(-5, 0, alpha=0.05, color='red')
    ax1.axhspan(0, 1.2, alpha=0.05, color='green')
    ax1.set_title('(a) Model R² across test windows', fontweight='bold', fontsize=8, loc='left')
    clean_spines(ax1)

    # Bottom: Skill Score
    ax2 = fig.add_subplot(gs[1])
    # Split axis for extreme value
    ax2.plot(x[:3], skill_score[:3], 'o-', color=WONG['orange'], linewidth=1.5, markersize=6)
    ax2.set_ylabel('Skill Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows, fontsize=6.5)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_ylim(-0.5, 1.5)

    # Annotation for plateau
    ax2.annotate(f'Plateau SS=−20.6\n(off scale, persistence dominates)',
                xy=(3, 1.2), fontsize=6.5, color=WONG['red'],
                ha='center', fontweight='bold')

    ax2.set_title('(b) Skill Score (SS > 0 = LS-MPR beats persistence)', fontweight='bold', fontsize=8, loc='left')
    clean_spines(ax2)

    fig.suptitle('Figure G. Anti-factual validation: persistence succeeds only at equilibrium.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figG_counterfactual')

# ================================================================
# FIG H: LOCO 3-fold × 3-target R² Heatmap
# ================================================================
def figH():
    r2_matrix = np.array([
        [-1.365, +0.201, -7.366],
        [-0.004, +0.383, -5.270],
        [-0.075, +0.125, -1.562],
    ])
    targets = ['CO$_2$ release', 'Residual mass', 'Tensile strength']
    folds = ['C1\n(T50/H70/R70)', 'C2\n(T58/H60/R100)', 'C3\n(T58/H70/R0)']

    fig, ax = plt.subplots(figsize=(W1, 75/25.4))
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=1)
    im = ax.imshow(r2_matrix, aspect='auto', cmap='coolwarm', norm=norm)

    for i in range(3):
        for j in range(3):
            val = r2_matrix[i, j]
            ax.text(j, i, f'{val:.3f}' if val > -10 else f'{val:.2f}',
                   ha='center', va='center', fontsize=7.5,
                   color='white' if val < 0 else 'black', fontweight='bold')

    # Red border on C3 column
    for i in range(3):
        ax.add_patch(plt.Rectangle((1.5, i-0.5), 1, 1, fill=False,
                                   edgecolor=WONG['red'], linewidth=2))

    ax.set_xticks(range(3))
    ax.set_xticklabels(folds, fontsize=6.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(targets, fontsize=7)
    ax.set_xlabel('Left-out condition (LOCO fold)', fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('Temporal R²', fontsize=7)

    ax.set_title('C3 (pure PLA, no-ratio condition) causes systematic failure',
                 fontsize=7.5, fontweight='bold', loc='center')
    ax.text(0.5, -0.18, 'C3 = compost ratio fixed at 0% (unseen PBAT-free condition)',
            transform=ax.transAxes, fontsize=6.5, color='gray', ha='center')
    clean_spines(ax)
    fig.suptitle('Figure H. LOCO cross-validation: fold-specific failure modes.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figH_loco_heatmap')

# ================================================================
# FIG I: Cross-Material vs Single-Material R²
# ================================================================
def figI():
    materials = ['PBAT/PLA\n(train)', 'PBAT+PGA', 'PBAT+Talc', 'PBAT1', 'PBAT2', 'PLA']
    r2_cross  = [0.9938, 0.9818, 0.4853, 0.4238, 0.3566, 0.5261]
    r2_single = [0.9938, 0.9916, 0.9954, 0.9954, 0.9956, 0.9955]
    co2_d180  = [123.3, 124.7, 173.5, 177.9, 181.4, 170.0]
    # Category
    cat = ['train', 'similar', 'dissimilar', 'dissimilar', 'dissimilar', 'dissimilar']
    colors = [WONG['blue'], WONG['green'], WONG['red'], WONG['red'], WONG['red'], WONG['red']]
    markers = ['*', 'o', '^', '^', '^', '^']
    sizes = [180, 80, 80, 80, 80, 80]

    fig, ax = plt.subplots(figsize=(W1, 85/25.4))

    for i, (m, rc, rs, c180, c, mk, sz) in enumerate(
        zip(materials, r2_cross, r2_single, co2_d180, colors, markers, sizes)):
        s = np.interp(c180, [120, 185], [40, 120])
        ax.scatter(rs, rc, s=s, c=c, marker=mk, edgecolors='white', linewidth=1, zorder=5)
        # Label
        offx = 0.0005 if m != 'PBAT/PLA\n(train)' else 0
        offy = 0.03 if m not in ['PBAT2', 'PBAT/Talc'] else -0.04
        ax.annotate(m.replace('\n',' '), (rs + offx, rc + offy), fontsize=6, ha='center', color=c,
                   fontweight='bold' if m == 'PBAT/PLA\n(train)' else 'normal')

    # y=x line
    ax.plot([0.98, 1.001], [0.98, 1.001], 'k--', linewidth=0.8, alpha=0.5, label='Perfect generalization')
    # y=0.5 threshold
    ax.axhline(y=0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.7, label='Practical threshold')
    ax.axhspan(-0.1, 0.5, alpha=0.06, color='red')

    ax.set_xlabel('Single-material R² (Days-only, d=2)')
    ax.set_ylabel('Cross-material R² (trained on PBAT/PLA)')
    ax.set_xlim(0.98, 1.001)
    ax.set_ylim(0.30, 1.01)

    # Legend
    leg = [plt.Line2D([0],[0], marker='*', color='w', markerfacecolor=WONG['blue'], markersize=10, label='PBAT/PLA (train)'),
           plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=WONG['green'], markersize=8, label='PBAT+PGA (similar)'),
           plt.Line2D([0],[0], marker='^', color='w', markerfacecolor=WONG['red'], markersize=8, label='Dissimilar structure')]
    ax.legend(handles=leg, fontsize=6, frameon=False, loc='lower left')
    ax.text(0.98, 0.5, 'Size $\\propto$ CO$_2$ capacity', transform=ax.transAxes, fontsize=6, color='gray', ha='right')
    clean_spines(ax)

    fig.suptitle('Figure I. Cross-material generalization: single-material fit ≠ cross-material prediction.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figI_cross_material')

# ================================================================
# FIG J: GPR Kernel Ablation Horizontal Bar
# ================================================================
def figJ():
    kernels = ['Dot+RBF+WhiteNoise\n(Full)', 'WhiteNoise only', 'RBF only', 'Dot only']
    r2_vals = [+0.162, -0.015, -0.104, -0.321]
    rmse_vals = [20.71, 22.21, 23.45, 26.65]

    fig, ax = plt.subplots(figsize=(W1, 75/25.4))

    colors = [WONG['blue'] if k == kernels[0] else '#BBBBBB' for k in kernels]
    edge_colors = ['navy' if k == kernels[0] else '#888888' for k in kernels]
    lws = [1.5 if k == kernels[0] else 0.5 for k in kernels]

    y_pos = range(len(kernels))
    bars = ax.barh(y_pos, r2_vals, color=colors, edgecolor=edge_colors, linewidth=lws, height=0.5)

    for bar, val, rmse in zip(bars, r2_vals, rmse_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'R²={val:+.3f}  RMSE={rmse:.1f}',
                va='center', fontsize=6.5, color='#333333')

    ax.axvline(x=0, color='black', linewidth=0.7, linestyle='--')
    ax.axvspan(-0.5, 0, alpha=0.06, color='red')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(kernels, fontsize=6.5)
    ax.set_xlabel('Temporal holdout R² (CO$_2$)')
    ax.invert_yaxis()

    # Highlight full kernel row
    ax.axhspan(-0.4, 0.4, alpha=0.08, color=WONG['blue'])

    ax.text(0.98, 0.05, 'Composite synergy:\neach component alone fails',
            transform=ax.transAxes, fontsize=6.5, color='gray', ha='right')

    clean_spines(ax)
    ax.set_title('GPR kernel ablation', fontweight='bold', fontsize=8, loc='left')
    fig.suptitle('Figure J. GPR kernel ablation: RBF kernel is the source of marginal advantage.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figJ_kernel_ablation')

# ================================================================
# FIG K: Material Degradation — CO₂ + Degradation Rate Dual Axis
# ================================================================
def figK():
    materials = ['PBAT/PLA\n(80:20)', 'PBAT1', 'PBAT2', 'PLA', 'PBAT+PGA', 'PBAT+Talc']
    co2 = [123.34, 177.90, 181.40, 170.04, 124.70, 173.52]
    co2_err = [15.2, 0, 0, 0, 0, 0]
    deg_rate = [84.5, 85.9, 86.3, 89.9, 81.2, 85.0]
    deg_err = [3.1, 0, 0, 0, 0, 0]

    fig, ax1 = plt.subplots(figsize=(W1, 80/25.4))

    x = np.arange(6)
    wong_6 = [WONG['blue'], WONG['red'], WONG['red'], WONG['red'],
              WONG['green'], WONG['pink']]

    # CO₂ bars (left y-axis)
    bars = ax1.bar(x, co2, 0.55, color=wong_6, edgecolor='white', linewidth=0.8)
    # Error bar only for PBAT/PLA
    ax1.errorbar(0, co2[0], yerr=co2_err[0], color='black', capsize=3, linewidth=0.8)
    ax1.set_ylabel('CO$_2$ accumulated at Day 180 (g)', color=WONG['blue'])
    ax1.set_ylim(0, 215)
    ax1.tick_params(axis='y', labelcolor=WONG['blue'])

    # Degradation rate (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(x, deg_rate, 'D-', color='black', linewidth=1.2, markersize=5, zorder=10)
    ax2.errorbar(0, deg_rate[0], yerr=deg_err[0], color='black', capsize=3, linewidth=0.8)
    ax2.set_ylabel('Degradation rate at Day 180 (%)', color='black')
    ax2.set_ylim(78, 93)

    ax1.set_xticks(x)
    ax1.set_xticklabels(materials, rotation=12, ha='right', fontsize=6.5)
    ax1.set_xlabel('')

    # Annotations
    ax1.annotate('Highest CO$_2$', xy=(2, co2[2]), xytext=(3, 200),
                fontsize=6.5, color=WONG['red'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=WONG['red'], lw=1))

    # Legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=WONG['blue'], label='CO$_2$ accumulated (g)'),
           plt.Line2D([0],[0], marker='D', color='black', linewidth=1.2, label='Degradation rate (%)')]
    ax1.legend(handles=leg, fontsize=6.5, frameon=False, loc='upper left')

    clean_spines(ax1)
    fig.suptitle('Figure K. Material comparison: CO$_2$ accumulation capacity and degradation extent.',
                 fontsize=8, y=1.03)
    plt.tight_layout()
    save(fig, 'figK_material_comparison')


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('Generating Figures F–K (Nature standard)...')
    figF()
    figG()
    figH()
    figI()
    figJ()
    figK()
    print('Done.')
