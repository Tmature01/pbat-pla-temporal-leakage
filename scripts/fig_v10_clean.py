"""
V10 figures — CLEAN, no overlap, properly scaled.
One-at-a-time, carefully tuned.
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch
# Use DejaVu Sans (system default, handles special chars well)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'lines.linewidth': 1.3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})
W1 = 89/25.4; W2 = 183/25.4
BLUE='#0072B2'; RED='#D55E00'; GREEN='#009E73'; ORANGE='#E69F00'
PINK='#CC79A7'; GREY='#888888'; SKY='#56B4E9'; NAVY='#000080'
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

FIG = str(_REPO_ROOT / "figures" / "A-K")
os.makedirs(FIG, exist_ok=True)

def cls(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

def save(fig, name):
    fig.savefig(os.path.join(FIG, name+'.png'), dpi=300)
    fig.savefig(os.path.join(FIG, name+'.pdf'))
    plt.close(fig)

# ================================================================
# FIG A — Feature importance: Real vs Synthetic
# Key fix: Days value UP TO 2 (broken axis), shows real bar height
# ================================================================
def figA():
    feats = ['Days','Temperature','Humidity','Ratio','Compost Vol.']
    # Real temporal holdout: importance = -ΔR²
    imp_r = [33.0, 0.06, 0.07, 0.01, 0.01]
    # Synthetic random split
    imp_s = [0.66, 0.13, 0.18, 0.45, 0.01]

    fig = plt.figure(figsize=(W2, 2.6))
    # Left: Real data — broken axis
    gs_l = fig.add_gridspec(2, 1, left=0.05, right=0.48, hspace=0.07, height_ratios=[1, 2.5])
    ax_top = fig.add_subplot(gs_l[0]); ax_bot = fig.add_subplot(gs_l[1])

    cols = [BLUE if f=='Days' else GREY for f in feats]
    y = range(len(feats))
    for ax in [ax_top, ax_bot]:
        ax.barh(y, imp_r, height=0.5, color=cols, edgecolor='white', lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(x=0, color='black', lw=0.4); cls(ax); ax.invert_yaxis()
    ax_top.set_ylim(-0.5, 0.5); ax_top.set_xlim(0, 1.5)
    ax_bot.set_ylim(4.5, -0.5); ax_bot.set_xlim(0, 1.5)
    ax_top.set_xticklabels([]); ax_top.set_title(''); ax_bot.set_xlabel('Importance (−ΔR²)', fontsize=8)
    ax_top.set_title('Real data — Temporal holdout', fontsize=9, fontweight='bold', pad=4, loc='left')
    # Days annotation
    ax_top.text(0.5, 0.5, 'Days = 33.0 →', transform=ax_top.transAxes, fontsize=8,
               color=BLUE, fontweight='bold', ha='center', va='center')
    # Break marks
    for ax in [ax_top, ax_bot]:
        d = 0.02
        kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=0.8)
        ax.plot((-d,d), (1-d*3,1+d*3), **kw)
        ax.plot((-d,d), (-d*3,+d*3), **kw)
    # bar labels
    for i, v in enumerate(imp_r):
        if v < 2:
            ax_bot.text(v+0.03, i, f'{v:.2f}', va='center', fontsize=7)
    ax_top.text(0.03, 0, f'{imp_r[0]:.1f}', va='center', fontsize=7, color=BLUE, fontweight='bold')

    # Right: Synthetic
    ax_r = fig.add_subplot(122)
    ax_r.barh(y, imp_s, height=0.55, color=cols, edgecolor='white', lw=0.5)
    ax_r.set_yticks(y); ax_r.set_yticklabels(feats, fontsize=8)
    ax_r.set_xlabel('Importance (−ΔR²)', fontsize=8); ax_r.invert_yaxis(); cls(ax_r)
    ax_r.set_title('Synthetic data — Random split', fontsize=9, fontweight='bold', pad=4, loc='left')
    for i, v in enumerate(imp_s):
        ax_r.text(v+0.015, i, f'{v:.2f}', va='center', fontsize=7)
    ax_r.set_xlim(0, 0.85)

    save(fig, 'figA_feature_importance')

# ================================================================
# FIG B — Symbolic regression: clean scatter, no overlaps
# ================================================================
def figB():
    models = ['Quadratic','Logistic','Exponential','M-Menten','Power Law']
    n_p = [3, 3, 2, 2, 2]
    r2 = [0.995, 0.992, 0.988, 0.977, 0.929]
    interpretable = [False, True, True, True, False]

    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for i, (m, n, r, intrp) in enumerate(zip(models, n_p, r2, interpretable)):
        c = GREEN if intrp else GREY
        ax.scatter(n, r, s=120, c=c, edgecolors='white', lw=1.5, zorder=5)
        # Label: alternate left/right based on position
        if n == 2:
            ox, ha = 0.12, 'left'
        else:
            ox, ha = 0.12, 'left'
        ax.annotate(m, (n+ox, r), fontsize=7.5, color=c, fontweight='bold', va='center', ha=ha)

    # Circle Exponential
    ax.annotate('← Selected', xy=(2, 0.988), xytext=(2.3, 0.998), fontsize=7.5,
               color=RED, fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    ax.set_xlabel('Number of parameters'); ax.set_ylabel('R²')
    ax.set_xticks([2,3]); ax.set_xlim(1.5, 3.5); ax.set_ylim(0.91, 1.005)
    ax.grid(alpha=0.2, linestyle='--'); cls(ax)
    ax.legend(handles=[Patch(fc=GREEN, label='Interpretable'), Patch(fc=GREY, label='Empirical')],
             fontsize=7, frameon=False, loc='lower right')
    save(fig, 'figB_symbolic_regression')

# ================================================================
# FIG C — Pareto frontier: scale y to exclude GA extreme, annotate it
# ================================================================
def figC():
    methods = ['LS-MPR','Ridge','Lasso','RF','SVR','GPR-Poly','GA']
    times = [0.003, 0.2, 0.3, 0.16, 2.3, 6.0, 35.0]
    r2t = [-0.058, 0.013, -0.052, -0.312, -0.203, 0.162, -1.447]

    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for m, t, r2 in zip(methods, times, r2t):
        if m == 'LS-MPR': c, s, lw, ec = BLUE, 100, 1.5, NAVY
        elif m == 'GPR-Poly': c, s, lw, ec = ORANGE, 70, 1, ORANGE
        elif m == 'GA': c, s, lw, ec = '#CCCCCC', 60, 0.5, '#AAAAAA'
        else: c, s, lw, ec = '#DDDDDD', 50, 0.5, '#BBBBBB'

        ax.scatter(t, max(r2, -0.6), s=s, c=c, edgecolors=ec, lw=lw, zorder=5)

        if m == 'LS-MPR':
            ax.annotate('LS-MPR', (t*2.5, r2+0.04), fontsize=8, color=BLUE, fontweight='bold')
        elif m == 'GPR-Poly':
            ax.annotate('GPR-Poly', (t*1.5, r2+0.04), fontsize=7, color=ORANGE)
        elif m == 'GA':
            ax.annotate(f'GA (R²={r2t[6]:.2f}, off-scale)', (t*1.3, -0.55), fontsize=7, color=GREY, ha='left')
        else:
            ax.annotate(m, (t*1.3, r2+0.03), fontsize=6.5, color='#777')

    ax.set_xscale('log'); ax.set_xlabel('Training time (s)')
    ax.set_ylabel('Temporal holdout R²'); ax.set_ylim(-0.65, 0.3)
    ax.axhline(y=0, color=RED, lw=0.7, ls='--', alpha=0.5)
    ax.grid(alpha=0.2, linestyle='--'); cls(ax)

    # Pareto frontier annotation
    ax.annotate('Best efficiency', xy=(0.003, -0.058), xytext=(0.02, 0.20),
               fontsize=8, color=BLUE, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))

    save(fig, 'figC_pareto_frontier')

# ================================================================
# FIG D — Kinetic parameters a vs b
# ================================================================
def figD():
    mats = ['PBAT/PLA','PBAT1','PBAT2','PLA','PBAT+PGA','PBAT+Talc']
    a_vals = [124.1, 181.2, 184.9, 173.5, 127.3, 178.6]
    b_vals = [0.021, 0.018, 0.017, 0.019, 0.020, 0.015]
    r2_fit = [0.988, 0.988, 0.987, 0.992, 0.991, 0.990]
    # Categories
    cat_c = {'blend': BLUE, 'pure': RED, 'talc': PINK}
    cat_m = {'blend': 'o', 'pure': 's', 'talc': 'D'}
    cat_list = ['blend','pure','pure','pure','blend','talc']

    # Label offsets tuned to avoid ALL overlap
    offsets = {
        'PBAT/PLA':  ( 0.0004,  4.0),
        'PBAT1':     ( 0.0004, -5.0),
        'PBAT2':     ( 0.0004,  4.5),
        'PLA':       ( 0.0004, -5.0),
        'PBAT+PGA':  ( 0.0004,  4.0),
        'PBAT+Talc': (-0.0020,  3.0),
    }

    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for m, a, b, r2, cat in zip(mats, a_vals, b_vals, r2_fit, cat_list):
        sz = np.interp(r2, [0.985, 0.995], [50, 140])
        ax.scatter(b, a, s=sz, c=cat_c[cat], marker=cat_m[cat],
                  edgecolors='white', lw=1.2, zorder=5)
        ox, oy = offsets[m]
        ax.annotate(m, (b+ox, a+oy), fontsize=7, ha='center', color='#222')

    ax.set_xlabel('b — degradation rate constant (day$^{-1}$)')
    ax.set_ylabel('a — ultimate CO$_2$ capacity (g)')
    ax.tick_params(direction='in'); cls(ax)
    ax.legend(handles=[
        Patch(fc=BLUE, label='Blends'),
        Patch(fc=RED, label='Pure'),
        Patch(fc=PINK, label='+Talc')],
        fontsize=7, frameon=False, loc='lower right')
    save(fig, 'figD_kinetics_ab')

# ================================================================
# FIG E — Full R² heatmap
# ================================================================
def figE():
    rows = [
        ('PBAT/PLA','Residual',0.9938,0.9938,0.9937,0.9928,0.9974,0.9991),
        ('PBAT/PLA','CO$_2$', 0.9908,0.9908,0.9908,0.9496,0.9970,0.9972),
        ('PBAT/PLA','Tensile',0.9780,0.9780,0.9780,0.9616,0.9977,0.9990),
        ('PBAT/TPS','Residual',0.8825,0.8833,0.8841,0.1468,0.9731,0.8218),
        ('PBAT/TPS','CO$_2$', 0.8825,0.8833,0.8841,0.0750,0.9739,0.8183),
        ('PBAT/TPS','Tensile',0.9410,0.9425,0.9475,0.5555,0.9471,0.9166),
        ('PBAT/PBS','Residual',0.9633,0.9632,0.9632,-0.4512,0.9353,0.8715),
        ('PBAT/PBS','CO$_2$', 0.9577,0.9580,0.9583,0.3467,0.9465,0.8586),
        ('PBAT/PBS','Tensile',0.9570,0.9570,0.9576,0.7213,0.9357,0.9279),
    ]
    methods = ['LS-MPR','Ridge','Lasso','GA','SVR','RF']
    mat = np.array([[r[2+i] for i in range(6)] for r in rows])

    fig, ax = plt.subplots(figsize=(W2, 3.3))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.25, vmax=1.0)
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', norm=norm)

    for i in range(9):
        for j in range(6):
            v = mat[i,j]
            ax.text(j, i, f'{v:.3f}' if v>-1 else f'{v:.2f}',
                   ha='center', va='center', fontsize=7,
                   color='white' if v<0.35 else 'black', fontweight='bold')

    # Red dash boxes for GA failures
    for i in range(9):
        if mat[i,3] < 0.6:
            ax.add_patch(plt.Rectangle((2.5,i-0.5), 1, 1, fill=False,
                                        edgecolor=RED, lw=1.5, ls='--'))

    ax.set_xticks(range(6)); ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(9))
    ax.set_yticklabels([f'{r[0]} | {r[1]}' for r in rows], fontsize=7)
    # Group separators
    for y in [2.5, 5.5]: ax.axhline(y=y, color='black', lw=1.2)
    # Dataset labels on left
    for y, d in [(1,'PBAT/PLA'),(4,'PBAT/TPS'),(7,'PBAT/PBS')]:
        ax.text(-1.2, y, d, fontsize=8, fontweight='bold', va='center', ha='right')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.82)
    cbar.set_label('R² (random split)', fontsize=8)
    save(fig, 'figE_heatmap_full')

# ================================================================
# FIG F — Feature ablation waterfall (broken axis)
# ================================================================
def figF():
    baseline = -0.058
    features = ['Days','Humidity','Temperature','Compost Vol.','Ratio']
    delta = [-33.042, +0.072, +0.060, +0.013, -0.006]

    fig = plt.figure(figsize=(W1, 3.3))
    gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.06)
    ax_t = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1])

    cols = [RED if d<-1 else ('#F4A582' if d<0 else GREEN) for d in delta]
    x = np.arange(len(features))

    for ax in [ax_t, ax_b]:
        bars = ax.bar(x, delta, color=cols, edgecolor='white', width=0.5)
        ax.axhline(y=0, color='black', lw=0.5, ls='--')
        ax.set_xticks(x); cls(ax)
        for bar, val in zip(bars, delta):
            yp = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, yp+(0.15 if yp>=0 else -0.3),
                   f'{val:+.3f}' if abs(val)<1 else f'{val:+.1f}',
                   ha='center', fontsize=7, fontweight='bold')

    ax_t.set_ylim(-34.5, -32.0); ax_t.set_xticklabels([])
    ax_b.set_ylim(-0.20, 0.20); ax_b.set_xticklabels(features, rotation=12, ha='right', fontsize=8)
    ax_b.set_xlabel('Feature removed')

    # Break marks
    for ax in [ax_t, ax_b]:
        d = 0.02
        kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=0.8)
        ax.plot((-d,d), (1-d*3,1+d*3), **kw)
        ax.plot((-d,d), (-d*3,+d*3), **kw)

    ax_t.set_ylabel('ΔR²'); ax_t.yaxis.set_label_coords(-0.14, 0.5)
    save(fig, 'figF_waterfall_ablation')

# ================================================================
# FIG G — Counterfactual validation
# ================================================================
def figG():
    windows = ['Growth','Growth+\nDecay','Mixed','Plateau']
    r2_p = [-3.674, -1.807, 0.437, 0.951]
    r2_l = [0.022, -0.322, 0.849, -0.058]
    ss = [0.79, 0.53, 0.73, -20.6]

    fig = plt.figure(figsize=(W1, 3.8))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.18)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

    # Bars
    x = np.arange(4); w = 0.32
    b1 = ax1.bar(x-w/2, r2_p, w, color='#BBBBBB', edgecolor='white', label='Persistence (0 params)')
    b2 = ax1.bar(x+w/2, r2_l, w, color=BLUE, edgecolor='white', label='LS-MPR (21 params)')
    ax1.axhline(y=0, color='black', lw=0.5, ls='--'); ax1.set_ylabel('Test R²')
    ax1.set_xticks(x); ax1.set_xticklabels([]); ax1.legend(fontsize=7, frameon=False)
    ax1.axhspan(-4.5, 0, alpha=0.04, color='red'); ax1.axhspan(0, 1.2, alpha=0.04, color='green')
    cls(ax1)

    # Bar labels
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, h+(0.07 if h>=0 else -0.3),
                    f'{h:+.3f}' if abs(h)<1 else f'{h:+.2f}',
                    ha='center', fontsize=6, rotation=90, va='bottom' if h>=0 else 'top')

    # Skill Score
    ax2.plot(x, [max(s,-10) for s in ss], 'o-', color=ORANGE, lw=1.5, markersize=7)
    ax2.axhline(y=0, color='black', lw=0.5, ls='--')
    ax2.set_ylabel('Skill Score'); ax2.set_xticks(x)
    ax2.set_xticklabels(windows, fontsize=7.5); ax2.set_ylim(-0.8, 1.5)
    cls(ax2)
    for i, s in enumerate(ss):
        if s > -10:
            ax2.text(i, s+0.1, f'{s:+.2f}', ha='center', fontsize=7, fontweight='bold')
    ax2.annotate('Plateau SS = −20.6\n(off-scale, persistence dominates)',
                xy=(3, 1.2), fontsize=7, ha='center', color=RED, fontweight='bold')

    save(fig, 'figG_counterfactual')

# ================================================================
# FIG H — LOCO heatmap
# ================================================================
def figH():
    mat = np.array([[-1.365, 0.201, -7.366], [-0.004, 0.383, -5.270], [-0.075, 0.125, -1.562]])
    targets = ['CO₂ release','Residual mass','Tensile strength']
    folds = ['C1\n(T50/H70/R70)','C2\n(T58/H60/R100)','C3\n(T58/H70/R0)']

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=1)
    im = ax.imshow(mat, aspect='auto', cmap='coolwarm', norm=norm)

    for i in range(3):
        for j in range(3):
            v = mat[i,j]
            ax.text(j, i, f'{v:.3f}' if v>-10 else f'{v:.2f}',
                   ha='center', va='center', fontsize=9,
                   color='white' if v<-1 else 'black', fontweight='bold')
        ax.add_patch(plt.Rectangle((1.5,i-0.5), 1, 1, fill=False,
                                    edgecolor=RED, lw=2.5))

    ax.set_xticks(range(3)); ax.set_xticklabels(folds, fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(targets, fontsize=8)
    ax.set_xlabel('Left-out condition', fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04, shrink=0.85)
    cbar.set_label('Temporal R²', fontsize=7.5)
    cls(ax)
    save(fig, 'figH_loco_heatmap')

# ================================================================
# FIG I — Cross-material generalization
# ================================================================
def figI():
    mats = ['PBAT/PLA (train)','PBAT+PGA','PBAT+Talc','PBAT1','PBAT2','PLA']
    r2c = [0.9938, 0.9818, 0.4853, 0.4238, 0.3566, 0.5261]
    r2s = [0.9938, 0.9916, 0.9954, 0.9954, 0.9956, 0.9955]
    cats = ['train','sim','dis','dis','dis','dis']
    cols = [BLUE, GREEN, RED, RED, RED, RED]
    mkrs = ['*','o','^','^','^','^']
    # offsets: (dx in R² units, dy in R² units)
    offs = [(-0.0008,0.04),(-0.0008,0.04),(0.0008,-0.05),(0.0008,-0.05),(0.0008,0.04),(0.0008,-0.05)]

    fig, ax = plt.subplots(figsize=(W1, 3.0))
    for i in range(6):
        ax.scatter(r2s[i], r2c[i], s=np.interp([123,124,174,178,181,170][i],[120,185],[50,140]),
                  c=cols[i], marker=mkrs[i], edgecolors='white', lw=1.2, zorder=5)
        ox, oy = offs[i]
        ax.annotate(mats[i], (r2s[i]+ox, r2c[i]+oy), fontsize=7, ha='center',
                   color=cols[i], fontweight='bold')

    ax.plot([0.978,1.002],[0.978,1.002],'k--',lw=0.6,alpha=0.35, label='Perfect generalization')
    ax.axhline(y=0.5, color='grey', lw=0.7, ls=':', alpha=0.5)
    ax.axhspan(-0.1, 0.5, alpha=0.04, color='red')
    ax.set_xlabel('Single-material R² (Days-only, d=2)'); ax.set_xlim(0.978, 1.002)
    ax.set_ylabel('Cross-material R² (trained on PBAT/PLA)'); ax.set_ylim(0.25, 1.03)
    cls(ax)
    ax.legend(handles=[
        plt.Line2D([0],[0],marker='*',c=BLUE,ms=12,lw=0,label='Train'),
        plt.Line2D([0],[0],marker='o',c=GREEN,ms=8,lw=0,label='Similar structure'),
        plt.Line2D([0],[0],marker='^',c=RED,ms=8,lw=0,label='Dissimilar')],
        fontsize=7, frameon=False, loc='lower left')
    save(fig, 'figI_cross_material')

# ================================================================
# FIG J — Kernel ablation
# ================================================================
def figJ():
    kernels = ['Dot+RBF+WhiteNoise\n(Full)','WhiteNoise only','RBF only','Dot only']
    r2v = [0.162, -0.015, -0.104, -0.321]
    rmse = [20.71, 22.21, 23.45, 26.65]

    fig, ax = plt.subplots(figsize=(W1, 2.5))
    y = range(len(kernels))
    cols = [BLUE, '#CCCCCC', '#CCCCCC', '#CCCCCC']
    ecs = [NAVY, '#AAAAAA', '#AAAAAA', '#AAAAAA']
    bars = ax.barh(y, r2v, height=0.5, color=cols, edgecolor=ecs, lw=[1.5,0.5,0.5,0.5])

    for bar, v, rm in zip(bars, r2v, rmse):
        ax.text(v + (0.02 if v>=0 else -0.02), bar.get_y()+bar.get_height()/2,
               f'R²={v:+.3f}  (RMSE={rm:.1f} g)',
               va='center', fontsize=7.5, ha='left' if v>=0 else 'right', color='#333')

    ax.axvline(x=0, color='black', lw=0.7, ls='--')
    ax.axvspan(-0.5, 0, alpha=0.04, color='red')
    ax.set_yticks(y); ax.set_yticklabels(kernels, fontsize=8)
    ax.set_xlabel('Temporal holdout R² (CO$_2$)', fontsize=8); ax.invert_yaxis(); cls(ax)
    ax.axhspan(-0.4, 0.4, alpha=0.06, color=BLUE)
    ax.text(0.98, 0.06, 'RBF is the dominant\ncomponent of Full',
           transform=ax.transAxes, fontsize=7, color='#555', ha='right')
    save(fig, 'figJ_kernel_ablation')

# ================================================================
# FIG K — Material comparison dual-axis
# ================================================================
def figK():
    mats = ['PBAT/PLA\n(80:20)','PBAT1','PBAT2','PLA','PBAT+PGA','PBAT+Talc']
    co2 = [123.34, 177.90, 181.40, 170.04, 124.70, 173.52]
    deg = [84.5, 85.9, 86.3, 89.9, 81.2, 85.0]
    cols6 = [BLUE, RED, RED, RED, GREEN, PINK]

    fig, ax1 = plt.subplots(figsize=(W1, 2.8))
    x = np.arange(6)
    ax1.bar(x, co2, 0.5, color=cols6, edgecolor='white', lw=0.5)
    ax1.errorbar(0, co2[0], yerr=15.2, color='black', capsize=4, lw=0.8)
    ax1.set_ylabel('CO$_2$ at Day 180 (g)', color=BLUE, fontsize=8)
    ax1.set_ylim(0, 230); ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.set_xticks(x); ax1.set_xticklabels(mats, rotation=10, ha='right', fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(x, deg, 'D-', color='black', lw=1.3, markersize=6, zorder=10)
    ax2.errorbar(0, deg[0], yerr=3.1, color='black', capsize=4, lw=0.8)
    ax2.set_ylabel('Degradation at Day 180 (%)', fontsize=8); ax2.set_ylim(76, 94)
    cls(ax1)
    ax1.legend(handles=[
        Patch(fc=BLUE, label='CO$_2$ accumulated (g)'),
        plt.Line2D([0],[0],marker='D',c='black',lw=1.3,label='Degradation rate (%)')],
        fontsize=7, frameon=False, loc='upper left')
    ax1.annotate('Highest CO$_2$', xy=(2,co2[2]), xytext=(3.8,215), fontsize=7.5,
                color=RED, fontweight='bold', arrowprops=dict(arrowstyle='->',color=RED,lw=1.2))
    save(fig, 'figK_material_comparison')


if __name__ == '__main__':
    print(f'Output: {FIG}\n')
    for f in [figA, figB, figC, figD, figE, figF, figG, figH, figI, figJ, figK]:
        try:
            f()
        except Exception as e:
            print(f'  FAILED {f.__name__}: {e}')
    print('\nDone.')