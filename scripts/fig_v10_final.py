"""
V10 figures — Nature standard, no overlapping labels, clean layout.
All 11 figures regenerated with proper spacing and label positioning.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

# ====== Nature style ======
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

W1 = 89/25.4; W2 = 183/25.4  # Nature widths (inches)
C = {'blue':'#0072B2','orange':'#E69F00','green':'#009E73','red':'#D55E00',
     'pink':'#CC79A7','sky':'#56B4E9','yellow':'#F0E442','grey':'#888888',
     'navy':'#000080'}

from pathlib import Path as _RepoPath

import sys as _SysShim
if hasattr(_SysShim.stdout, "reconfigure"):
    _SysShim.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

FIG_DIR = str(_REPO_ROOT / "figures" / "A-K")

def clean(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)

def save(fig, name):
    for ext in ['.png','.pdf']:
        fig.savefig(os.path.join(FIG_DIR, name+ext), dpi=300)
    plt.close(fig)
    print(f'  {name}')

# ================================================================
# FIG A: Feature Importance — Horizontal grouped bars, clean
# ================================================================
def figA():
    features = ['Days','Temperature','Humidity','Ratio','Compost Vol.']
    imp_real = [33.042, 0.060, 0.072, 0.006, 0.013]  # importance = -ΔR² (positive)
    imp_synth = [0.660, 0.130, 0.176, 0.446, 0.011]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.4))

    colors = [C['blue'] if f == 'Days' else C['grey'] for f in features]
    for ax, imp, title in [(ax1, imp_real, 'Real data — Temporal holdout'),
                            (ax2, imp_synth, 'Synthetic data — Random split')]:
        y = range(len(features))
        bars = ax.barh(y, imp, height=0.55, color=colors, edgecolor='white', lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels(features, fontsize=7)
        ax.set_xlabel('Importance (−ΔR²)', fontsize=7.5)
        ax.set_title(title, fontsize=7.5, fontweight='bold', loc='left', pad=6)
        ax.invert_yaxis()
        clean(ax)
        ax.axvline(x=0, color='black', lw=0.4)
        # Annotate bars — place values to the right of each bar
        for bar, val in zip(bars, imp):
            w = bar.get_width()
            ax.text(w + max(imp)*0.02, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}' if val>1 else f'{val:.3f}',
                    va='center', fontsize=6.5, color='#333')

    # "Days off-scale" note for real data
    ax1.text(0.95, 0.08, 'Days = 33.0 (off-scale)', transform=ax1.transAxes,
             fontsize=6.5, color=C['blue'], ha='right', fontweight='bold')
    ax1.set_xlim(0, max(imp_real)*1.25)

    fig.tight_layout(pad=1.2)
    save(fig, 'figA_feature_importance')


# ================================================================
# FIG B: Symbolic regression — clean scatter, no overlap
# ================================================================
def figB():
    models = ['Quadratic','Logistic','Exponential','M-Menten','Power Law']
    n_param = [3, 3, 2, 2, 2]
    r2 = [0.995, 0.992, 0.988, 0.977, 0.929]
    rmse = [5.12, 6.34, 7.89, 10.85, 18.94]
    interp = [False, True, True, True, False]

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    sizes = np.interp(rmse, [5,19], [60,180])
    for i,(m,n,r,s,intrp) in enumerate(zip(models,n_param,r2,sizes,interp)):
        c = C['green'] if intrp else C['grey']
        ax.scatter(n, r, s=s, c=c, edgecolors='white', lw=1, zorder=5)

        # Manual offset per point to avoid overlap
        offsets = {'Quadratic': (0.08, 0.001), 'Logistic': (0.08, -0.0025),
                   'Exponential': (0.09, 0.0005), 'M-Menten': (0.08, 0.001),
                   'Power Law': (0.08, -0.001)}
        ox, oy = offsets[m]
        ax.annotate(m, (n+ox, r+oy), fontsize=6, color=c, fontweight='bold', va='center')

    # Highlight Exponential
    ax.annotate('Selected', xy=(2, 0.988), xytext=(2.35, 0.995), fontsize=6.5,
               color=C['red'], fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->', color=C['red'], lw=1.2))

    ax.set_xlabel('Number of parameters'); ax.set_ylabel('R²')
    ax.set_xticks([2,3]); ax.set_xlim(1.5, 3.5); ax.set_ylim(0.915, 1.002)
    ax.grid(alpha=0.25, linestyle='--')
    clean(ax)
    leg = [Patch(fc=C['green'], label='Interpretable'), Patch(fc=C['grey'], label='Empirical')]
    ax.legend(handles=leg, fontsize=6, frameon=False, loc='lower right')

    fig.tight_layout(pad=0.8)
    save(fig, 'figB_symbolic_regression')


# ================================================================
# FIG C: Pareto frontier — remove extreme negative GA, focus
# ================================================================
def figC():
    methods = ['LS-MPR','Ridge','Lasso','RF','SVR','GPR-Poly','GA']
    times = [0.003, 0.2, 0.3, 0.16, 2.3, 6.0, 35.0]
    r2_temporal = [-0.058, 0.013, -0.052, -0.312, -0.203, 0.162, -1.447]

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for i,(m,t,r2t) in enumerate(zip(methods, times, r2_temporal)):
        if m == 'LS-MPR': c, s, ec, lw = C['blue'], 80, C['navy'], 1.5
        elif m == 'GPR-Poly': c, s, ec, lw = C['orange'], 60, C['orange'], 1
        elif m == 'GA': c, s, ec, lw = C['grey'], 50, C['grey'], 0.5
        else: c, s, ec, lw = '#CCCCCC', 45, '#AAAAAA', 0.5

        ax.scatter(t, r2t, s=s, c=c, edgecolors=ec, lw=lw, zorder=5)

        # Label offsets tuned per point
        off = {'LS-MPR':(1.5, 0.06), 'Ridge':(1.2, 0.05), 'Lasso':(1.2, -0.07),
               'RF':(1.2, 0.05), 'SVR':(1.2, -0.06), 'GPR-Poly':(1.2, 0.04),
               'GA':(1.2, 0.06)}
        ax.annotate(m, (t*off[m][0], r2t+off[m][1]), fontsize=6,
                   color='#333', ha='left')

    ax.set_xscale('log')
    ax.set_xlabel('Training time (s)'); ax.set_ylabel('Temporal holdout R²')
    ax.axhline(y=0, color=C['red'], lw=0.7, ls='--', alpha=0.5)
    ax.set_ylim(-1.6, 0.35)
    ax.grid(alpha=0.2, linestyle='--')
    clean(ax)

    # Arrow pointing to LS-MPR
    ax.annotate('0.003 s / R²=-0.06', xy=(0.003, -0.058), xytext=(0.03, 0.22),
               fontsize=7, color=C['blue'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=C['blue'], lw=1.3))

    fig.tight_layout(pad=0.8)
    save(fig, 'figC_pareto_frontier')


# ================================================================
# FIG D: Kinetic parameters — clean scatter
# ================================================================
def figD():
    materials = ['PBAT/PLA','PBAT1','PBAT2','PLA','PBAT+PGA','PBAT+Talc']
    a_vals = [124.1, 181.2, 184.9, 173.5, 127.3, 178.6]
    b_vals = [0.021, 0.018, 0.017, 0.019, 0.020, 0.015]
    r2 = [0.988, 0.988, 0.987, 0.992, 0.991, 0.990]
    cats = ['blend','pure','pure','pure','blend','additive']
    cat_c = {'blend':C['blue'],'pure':C['red'],'additive':C['pink']}
    cat_m = {'blend':'o','pure':'s','additive':'D'}

    fig, ax = plt.subplots(figsize=(W1, 2.6))

    # Manual label offsets to avoid overlap
    lbl_offsets = {
        'PBAT/PLA': (0.0003, 3), 'PBAT1': (0.0003, -4),
        'PBAT2': (0.0003, 3), 'PLA': (0.0003, -4),
        'PBAT+PGA': (0.0003, 3), 'PBAT+Talc': (-0.0015, 3),
    }

    for m, a, b, r, cat in zip(materials, a_vals, b_vals, r2, cats):
        sz = np.interp(r, [0.985,0.995], [55,130])
        ax.scatter(b, a, s=sz, c=cat_c[cat], marker=cat_m[cat],
                  edgecolors='white', lw=1, zorder=5)
        ox, oy = lbl_offsets[m]
        ax.annotate(m, (b+ox, a+oy), fontsize=6, ha='center', color='#333')

    ax.set_xlabel('b — rate constant (day$^{-1}$)')
    ax.set_ylabel('a — ultimate CO$_2$ capacity (g)')
    ax.tick_params(direction='in')
    clean(ax)
    leg = [Patch(fc=C['blue'], label='Blends'), Patch(fc=C['red'], label='Pure'),
           Patch(fc=C['pink'], label='+Talc')]
    ax.legend(handles=leg, fontsize=6.5, frameon=False, loc='lower right')

    fig.tight_layout(pad=0.8)
    save(fig, 'figD_kinetics_ab')


# ================================================================
# FIG E: R² Heatmap — clean, well-spaced
# ================================================================
def figE():
    rows = [
        ('PBAT/PLA','Residual',0.9938,0.9938,0.9937,0.9928,0.9974,0.9991),
        ('PBAT/PLA','CO$_2$',  0.9908,0.9908,0.9908,0.9496,0.9970,0.9972),
        ('PBAT/PLA','Tensile', 0.9780,0.9780,0.9780,0.9616,0.9977,0.9990),
        ('PBAT/TPS','Residual',0.8825,0.8833,0.8841,0.1468,0.9731,0.8218),
        ('PBAT/TPS','CO$_2$',  0.8825,0.8833,0.8841,0.0750,0.9739,0.8183),
        ('PBAT/TPS','Tensile', 0.9410,0.9425,0.9475,0.5555,0.9471,0.9166),
        ('PBAT/PBS','Residual',0.9633,0.9632,0.9632,-0.4512,0.9353,0.8715),
        ('PBAT/PBS','CO$_2$',  0.9577,0.9580,0.9583,0.3467,0.9465,0.8586),
        ('PBAT/PBS','Tensile', 0.9570,0.9570,0.9576,0.7213,0.9357,0.9279),
    ]
    methods = ['LS-MPR','Ridge','Lasso','GA','SVR','RF']
    mat = np.array([[r[2+i] for i in range(6)] for r in rows])
    ylabels = [f'{r[0]}  |  {r[1]}' for r in rows]

    fig, ax = plt.subplots(figsize=(W2, 3.0))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.25, vmax=1.0)
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', norm=norm)

    for i in range(9):
        for j in range(6):
            val = mat[i,j]
            tc = 'white' if val < 0.35 else 'black'
            ax.text(j, i, f'{val:.3f}' if val>-1 else f'{val:.2f}',
                   ha='center', va='center', fontsize=6.5, fontweight='bold', color=tc)

    # Red border for GA failures (negative R² on TPS/PBS)
    for i in range(3,9):
        if mat[i,3] < 0.5:
            ax.add_patch(plt.Rectangle((2.5,i-0.5), 1, 1, fill=False,
                                        edgecolor=C['red'], lw=1.5, ls='--'))

    ax.set_xticks(range(6)); ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_yticks(range(9)); ax.set_yticklabels(ylabels, fontsize=6.5)

    # Group separators
    for y in [2.5, 5.5]:
        ax.axhline(y=y, color='black', lw=1)

    # Dataset labels
    for y,d in [(1,'PBAT/PLA'),(4,'PBAT/TPS (synth.)'),(7,'PBAT/PBS (synth.)')]:
        ax.text(-1.1, y, d, fontsize=7, fontweight='bold', va='center', ha='right')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label('R² (random split)', fontsize=7)
    clean(ax)
    fig.tight_layout(pad=0.8)
    save(fig, 'figE_heatmap_full')


# ================================================================
# FIG F: Feature Ablation Waterfall — broken axis
# ================================================================
def figF():
    baseline = -0.058
    features = ['Days','Humidity','Temperature','Compost Vol.','Ratio']
    delta = [-33.042, +0.072, +0.060, +0.013, -0.006]  # sorted by |Δ|

    fig = plt.figure(figsize=(W1, 3.0))
    gs = GridSpec(2, 1, height_ratios=[0.8, 3], hspace=0.06)
    ax_top = fig.add_subplot(gs[0]); ax_bot = fig.add_subplot(gs[1])

    colors = []
    for d in delta:
        if d < -1: colors.append(C['red'])
        elif d < 0: colors.append('#F4A582')
        else: colors.append(C['green'])

    x = np.arange(len(features))
    for ax in [ax_top, ax_bot]:
        bars = ax.bar(x, delta, color=colors, edgecolor='white', width=0.55)
        ax.axhline(y=0, color='black', lw=0.5, ls='--')
        for bar, val in zip(bars, delta):
            yp = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, yp + (0.15 if yp>=0 else -0.3),
                   f'{val:+.3f}' if abs(val)<1 else f'{val:+.1f}',
                   ha='center', fontsize=6.5, fontweight='bold')
        ax.set_xticks(x); clean(ax)

    ax_top.set_ylim(-34.0, -32.0)
    ax_bot.set_ylim(-0.15, 0.15)
    ax_top.set_xticklabels([])
    ax_bot.set_xticklabels(features, rotation=12, ha='right')
    ax_bot.set_xlabel('Feature removed')

    # Break marks
    for ax in [ax_top, ax_bot]:
        d_ = 0.015
        kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=0.7)
        ax.plot((-d_,+d_), (-d_*3,+d_*3), **kw)
        ax.plot((-d_,+d_), (1-d_*3,1+d_*3), **kw)

    ax_top.set_ylabel('ΔR²'); ax_top.yaxis.set_label_coords(-0.12, 0.5)
    fig.tight_layout(pad=0.8)
    save(fig, 'figF_waterfall_ablation')


# ================================================================
# FIG G: Counterfactual — grouped bars + Skill Score line
# ================================================================
def figG():
    windows = ['Growth','Growth+\nDecay','Mixed','Plateau']
    r2_p = [-3.674, -1.807, 0.437, 0.951]
    r2_l = [0.022, -0.322, 0.849, -0.058]
    ss = [0.79, 0.53, 0.73, -20.6]

    fig = plt.figure(figsize=(W1, 3.5))
    gs = GridSpec(2, 1, height_ratios=[1.8, 1], hspace=0.18)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

    # Top: bars
    x = np.arange(4); w = 0.3
    b1 = ax1.bar(x-w/2, r2_p, w, color='#AAAAAA', edgecolor='white', label='Persistence')
    b2 = ax1.bar(x+w/2, r2_l, w, color=C['blue'], edgecolor='white', label='LS-MPR')
    ax1.axhline(y=0, color='black', lw=0.5, ls='--')
    ax1.set_ylabel('Test R²'); ax1.set_xticks(x); ax1.set_xticklabels([])
    ax1.legend(fontsize=6.5, frameon=False, loc='upper left')
    ax1.axhspan(-4.5, 0, alpha=0.04, color='red'); ax1.axhspan(0, 1.2, alpha=0.04, color='green')
    clean(ax1)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height(); va = 0.06 if h>=0 else -0.25
            ax1.text(bar.get_x()+bar.get_width()/2, h+va, f'{h:+.3f}' if abs(h)<1 else f'{h:+.2f}',
                    ha='center', fontsize=5.5, rotation=90, va='bottom' if h>=0 else 'top')

    # Bottom: Skill Score (limit y to avoid extreme -20.6)
    ax2.plot(x, ss, 'o-', color=C['orange'], lw=1.5, markersize=6)
    ax2.axhline(y=0, color='black', lw=0.5, ls='--')
    ax2.set_ylabel('Skill Score'); ax2.set_xticks(x)
    ax2.set_xticklabels(windows, fontsize=6.5)
    ax2.set_ylim(-1.5, 1.5)
    clean(ax2)
    # Annotate plateau SS as off-scale
    ax2.annotate('Plateau SS=−20.6\n(persistence dominates)', xy=(3, 1.2), fontsize=6,
                ha='center', color=C['red'], fontweight='bold')

    for i, s in enumerate(ss):
        if s > -10:
            ax2.text(i, s+0.12, f'{s:+.2f}', ha='center', fontsize=6, fontweight='bold')

    fig.tight_layout(pad=0.8)
    save(fig, 'figG_counterfactual')


# ================================================================
# FIG H: LOCO Heatmap
# ================================================================
def figH():
    mat = np.array([[-1.365, 0.201, -7.366], [-0.004, 0.383, -5.270], [-0.075, 0.125, -1.562]])
    targets = ['CO$_2$ release','Residual mass','Tensile strength']
    folds = ['C1\n(T50/H70/R70)','C2\n(T58/H60/R100)','C3\n(T58/H70/R0)']

    fig, ax = plt.subplots(figsize=(W1, 2.4))
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=1)
    im = ax.imshow(mat, aspect='auto', cmap='coolwarm', norm=norm)

    for i in range(3):
        for j in range(3):
            val = mat[i,j]
            ax.text(j, i, f'{val:.3f}' if val>-10 else f'{val:.2f}',
                   ha='center', va='center', fontsize=8,
                   color='white' if val<-1 else 'black', fontweight='bold')

    # Red border on C3
    for i in range(3):
        ax.add_patch(plt.Rectangle((1.5,i-0.5), 1, 1, fill=False,
                                    edgecolor=C['red'], lw=2))

    ax.set_xticks(range(3)); ax.set_xticklabels(folds, fontsize=6.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(targets, fontsize=7)
    ax.set_xlabel('Left-out condition', fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, shrink=0.9)
    cbar.set_label('Temporal R²', fontsize=6.5)
    clean(ax)
    fig.tight_layout(pad=0.8)
    save(fig, 'figH_loco_heatmap')


# ================================================================
# FIG I: Cross-material generalization
# ================================================================
def figI():
    materials = ['PBAT/PLA(t)','PBAT+PGA','PBAT+Talc','PBAT1','PBAT2','PLA']
    r2c = [0.9938, 0.9818, 0.4853, 0.4238, 0.3566, 0.5261]
    r2s = [0.9938, 0.9916, 0.9954, 0.9954, 0.9956, 0.9955]
    d180 = [123.3, 124.7, 173.5, 177.9, 181.4, 170.0]
    cats = ['train','similar','dissim','dissim','dissim','dissim']
    colors = [C['blue'],C['green'],C['red'],C['red'],C['red'],C['red']]
    markers = ['*','o','^','^','^','^']
    sizes_ = [160, 80, 80, 80, 80, 80]
    # Label offsets
    offs = [(-0.001,0.03),(-0.001,0.03),(0.0005,-0.04),(0.0005,-0.04),
            (0.0005,0.03),(0.0005,-0.04)]

    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for i in range(6):
        s = np.interp(d180[i],[120,185],[40,120])
        ax.scatter(r2s[i], r2c[i], s=s, c=colors[i], marker=markers[i],
                  edgecolors='white', lw=1, zorder=5)
        ox, oy = offs[i]
        ax.annotate(materials[i], (r2s[i]+ox, r2c[i]+oy), fontsize=6,
                   ha='center', color=colors[i], fontweight='bold')

    ax.plot([0.98,1.001],[0.98,1.001],'k--',lw=0.7,alpha=0.4)
    ax.axhline(y=0.5, color='grey', lw=0.7, ls=':', alpha=0.6)
    ax.axhspan(-0.1,0.5, alpha=0.04, color='red')
    ax.set_xlabel('Single-material R² (Days-only, d=2)')
    ax.set_ylabel('Cross-material R² (trained on PBAT/PLA)')
    ax.set_xlim(0.979,1.0012); ax.set_ylim(0.28,1.02)
    clean(ax)
    leg = [plt.Line2D([0],[0],marker='*',c=C['blue'],ms=10,lw=0,label='Train'),
           plt.Line2D([0],[0],marker='o',c=C['green'],ms=7,lw=0,label='Similar'),
           plt.Line2D([0],[0],marker='^',c=C['red'],ms=7,lw=0,label='Dissimilar')]
    ax.legend(handles=leg, fontsize=6, frameon=False, loc='lower left')
    fig.tight_layout(pad=0.8)
    save(fig, 'figI_cross_material')


# ================================================================
# FIG J: GPR Kernel Ablation
# ================================================================
def figJ():
    kernels = ['Dot+RBF+WhiteNoise\n(Full composite)','WhiteNoise only','RBF only','Dot only']
    r2v = [0.162, -0.015, -0.104, -0.321]
    rmse = [20.71, 22.21, 23.45, 26.65]

    fig, ax = plt.subplots(figsize=(W1, 2.3))
    y = range(len(kernels))
    colors = [C['blue'],'#CCCCCC','#CCCCCC','#CCCCCC']
    ecs = [C['navy'],'#AAAAAA','#AAAAAA','#AAAAAA']
    bars = ax.barh(y, r2v, height=0.45, color=colors, edgecolor=ecs, lw=[1.2,0.5,0.5,0.5])

    for bar, val, rm in zip(bars, r2v, rmse):
        ax.text(val + 0.015, bar.get_y()+bar.get_height()/2,
               f'R²={val:+.3f}  (RMSE={rm:.1f})',
               va='center', fontsize=6.5, color='#333')

    ax.axvline(x=0, color='black', lw=0.7, ls='--')
    ax.axvspan(-0.5, 0, alpha=0.04, color='red')
    ax.set_yticks(y); ax.set_yticklabels(kernels, fontsize=7)
    ax.set_xlabel('Temporal holdout R$^2$ (CO$_2$)')
    ax.invert_yaxis(); clean(ax)

    # Highlight full kernel row
    ax.axhspan(-0.4, 0.4, alpha=0.06, color=C['blue'])
    ax.text(0.98, 0.06, 'RBF kernel is the\ndominant component', transform=ax.transAxes,
           fontsize=6, color='#666', ha='right')

    fig.tight_layout(pad=0.8)
    save(fig, 'figJ_kernel_ablation')


# ================================================================
# FIG K: Material comparison — dual axis
# ================================================================
def figK():
    materials = ['PBAT/PLA\n(80:20)','PBAT1','PBAT2','PLA','PBAT+PGA','PBAT+Talc']
    co2 = [123.34,177.90,181.40,170.04,124.70,173.52]
    co2_err = [15.2,0,0,0,0,0]
    deg = [84.5,85.9,86.3,89.9,81.2,85.0]
    deg_err = [3.1,0,0,0,0,0]
    colors6 = [C['blue'],C['red'],C['red'],C['red'],C['green'],C['pink']]

    fig, ax1 = plt.subplots(figsize=(W1, 2.6))
    x = np.arange(6)
    ax1.bar(x, co2, 0.5, color=colors6, edgecolor='white', lw=0.6)
    ax1.errorbar(0, co2[0], yerr=co2_err[0], color='black', capsize=3, lw=0.7)
    ax1.set_ylabel('CO$_2$ at Day 180 (g)', color=C['blue'])
    ax1.set_ylim(0, 220)
    ax1.tick_params(axis='y', labelcolor=C['blue'])
    ax1.set_xticks(x); ax1.set_xticklabels(materials, rotation=12, ha='right', fontsize=6.5)

    ax2 = ax1.twinx()
    ax2.plot(x, deg, 'D-', color='black', lw=1.2, markersize=5, zorder=10)
    ax2.errorbar(0, deg[0], yerr=deg_err[0], color='black', capsize=3, lw=0.7)
    ax2.set_ylabel('Degradation (%)', color='black')
    ax2.set_ylim(78, 93)
    clean(ax1)
    # Legend
    leg = [Patch(fc=C['blue'], label='CO$_2$ (g)'),
           plt.Line2D([0],[0],marker='D',c='black',lw=1.2,label='Degradation (%)')]
    ax1.legend(handles=leg, fontsize=6.5, frameon=False, loc='upper left')

    # Highlight highest CO₂
    ax1.annotate('Highest', xy=(2,co2[2]), xytext=(3.5,205), fontsize=6.5,
                color=C['red'], fontweight='bold',
                arrowprops=dict(arrowstyle='->',color=C['red'],lw=1))

    fig.tight_layout(pad=0.8)
    save(fig, 'figK_material_comparison')


# ================================================================
if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'Output: {FIG_DIR}\n')
    for f in [figA, figB, figC, figD, figE, figF, figG, figH, figI, figJ, figK]:
        f()
    print(f'\nDone — 11 figures saved.')
