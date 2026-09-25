"""
V10 figures — scientific-figure-pro house style, clean, no-overlap.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

# === House style (scientific-figure-pro principles) ===
BLUE='#0F4D92'; BLUE2='#3775BA'; RED='#B64342'
GREEN='#8BCF8B'; TEAL='#42949E'; GREY='#CFCECE'; VIOLET='#9A4D8E'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans','Helvetica','Arial','sans-serif'],
    'font.size': 7,
    'axes.labelsize': 7, 'axes.titlesize': 8,
    'axes.linewidth': 1.2, 'axes.spines.right': False, 'axes.spines.top': False,
    'legend.frameon': False, 'legend.fontsize': 6.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.transparent': False,
})

W1, W2 = 89/25.4, 183/25.4
from pathlib import Path as _RepoPath
_REPO_ROOT = _RepoPath(__file__).resolve().parent.parent
FIGURE_DATA = _REPO_ROOT / "figure_data"
FIGURE_DATA.mkdir(parents=True, exist_ok=True)

OUT = Path(str(_REPO_ROOT / "figures" / "A-K"))
OUT.mkdir(parents=True, exist_ok=True)

def finalize(fig, name, dpi=300):
    base = OUT / name
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext, edpi in [('pdf',None),('png',dpi)]:
        kw = {'format': ext, 'bbox_inches': 'tight', 'pad_inches': 0.05}
        if edpi: kw['dpi'] = edpi
        fig.savefig(str(base.with_suffix(f'.{ext}')), **kw)
    plt.close(fig)

# ================================================================
# FIG A: Feature importance — grouped horizontal bars
# ================================================================
def figA():
    feats = ['Days','Temperature','Humidity','Ratio','Compost Vol.']
    imp_r = [33.0, 0.06, 0.07, 0.01, 0.01]   # Real
    imp_s = [0.66, 0.13, 0.18, 0.45, 0.01]   # Synthetic

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    for ax, imp, title in [(axes[0], imp_r, 'Real — Temporal holdout'),
                            (axes[1], imp_s, 'Synthetic — Random split')]:
        cols = [BLUE if f=='Days' else GREY for f in feats]
        y = range(len(feats))
        ax.barh(y, imp, height=0.55, color=cols, edgecolor='white', lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=7)
        ax.invert_yaxis()
        for i, v in enumerate(imp):
            ax.text(v+max(imp)*0.025, i, f'{v:.1f}' if v>1 else f'{v:.2f}',
                    va='center', fontsize=6.5)
        ax.set_xlabel('Importance (−ΔR²)', fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
    axes[0].set_xlim(0, 1.8); axes[0].text(0.95, 0.1, 'Days=33.0 →', transform=axes[0].transAxes,
        fontsize=7, color=BLUE, fontweight='bold', ha='right', va='bottom')
    axes[1].set_xlim(0, 0.85)
    finalize(fig, 'figA_feature_importance', dpi=300)

# ================================================================
# FIG B: Symbolic regression — scatter with manual labels
# ================================================================
def figB():
    models = ['Quadratic','Logistic','Exponential','M-Menten','Power Law']
    n_p, r2 = [3,3,2,2,2], [0.995,0.992,0.988,0.977,0.929]
    interp = [False, True, True, True, False]

    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for m, n, r, intrp in zip(models, n_p, r2, interp):
        c = GREEN if intrp else GREY
        ax.scatter(n, r, s=110, c=c, edgecolors='white', lw=1.5, zorder=5)
        ax.annotate(m, (n+0.1, r+(0.003 if m!='Logistic' else -0.004)),
                   fontsize=7, color=c, fontweight='bold', va='center')

    ax.annotate('← Selected', xy=(2,0.988), xytext=(2.35,0.998), fontsize=7.5,
               color=RED, fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->',color=RED,lw=1.5))
    ax.set_xlabel('Number of parameters'); ax.set_ylabel('R²')
    ax.set_xticks([2,3]); ax.set_xlim(1.5,3.5); ax.set_ylim(0.91,1.005)
    ax.grid(alpha=0.2, linestyle='--')
    ax.legend(handles=[Patch(fc=GREEN,label='Interpretable'),Patch(fc=GREY,label='Empirical')],
             fontsize=7, frameon=False, loc='lower right')
    finalize(fig, 'figB_symbolic_regression', dpi=300)

# ================================================================
# FIG C: Pareto frontier
# ================================================================
def figC():
    methods = ['LS-MPR','Ridge','Lasso','RF','SVR','GPR-Poly','GA']
    times  = [0.003,0.2,0.3,0.16,2.3,6.0,35.0]
    r2t    = [-0.058,0.013,-0.052,-0.312,-0.203,0.162,-1.447]

    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for m,t,r in zip(methods, times, r2t):
        if m=='LS-MPR':     c,s,ec,lw = BLUE,90,'#0A3A6B',1.5
        elif m=='GPR-Poly': c,s,ec,lw = BLUE2,65,'#1A5290',1
        elif m=='GA':       c,s,ec,lw = '#DDDDDD',55,'#BBBBBB',0.5
        else:               c,s,ec,lw = '#E8E8E8',45,'#CCCCCC',0.5

        y_plot = r if r > -0.6 else -0.55
        ax.scatter(t, y_plot, s=s, c=c, edgecolors=ec, lw=lw, zorder=5)

        if m == 'LS-MPR':
            ax.annotate('LS-MPR', (t*2.5, r+0.04), fontsize=8, color=BLUE, fontweight='bold')
        elif m == 'GPR-Poly':
            ax.annotate('GPR-Poly', (t*1.5, r+0.04), fontsize=7, color=BLUE2)
        elif m == 'GA':
            ax.annotate(f'GA (R²={r2t[6]:.2f})', (t*1.3, -0.53), fontsize=6.5, color=GREY, ha='left')
        elif m in ('Ridge','Lasso'):
            ax.annotate(m, (t*1.3, r+0.03 if m=='Ridge' else r-0.08), fontsize=6.5, color='#777')
        else:
            ax.annotate(m, (t*1.3, r+0.03), fontsize=6.5, color='#777')

    ax.set_xscale('log'); ax.set_xlabel('Training time (s)'); ax.set_ylabel('Temporal holdout R²')
    ax.set_ylim(-0.65, 0.3); ax.axhline(y=0, color=RED, lw=0.7, ls='--', alpha=0.5)
    ax.grid(alpha=0.2, linestyle='--')
    ax.annotate('Best efficiency', xy=(0.003,-0.058), xytext=(0.02,0.20),
               fontsize=8, color=BLUE, fontweight='bold',
               arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.5))
    finalize(fig, 'figC_pareto_frontier', dpi=300)

# ================================================================
# FIG D: Kinetic parameters
# ================================================================
def figD():
    mats = ['PBAT/PLA','PBAT1','PBAT2','PLA','PBAT+PGA','PBAT+Talc']
    a_v = [124.1,181.2,184.9,173.5,127.3,178.6]
    b_v = [0.021,0.018,0.017,0.019,0.020,0.015]
    r2f = [0.988,0.988,0.987,0.992,0.991,0.990]
    cats = ['blend','pure','pure','pure','blend','talc']
    cat_c = {'blend':BLUE,'pure':RED,'talc':VIOLET}
    cat_m = {'blend':'o','pure':'s','talc':'D'}
    offs = {'PBAT/PLA':(0.0004,4),'PBAT1':(0.0004,-5),'PBAT2':(0.0004,4.5),
            'PLA':(0.0004,-5),'PBAT+PGA':(0.0004,4),'PBAT+Talc':(-0.002,3)}

    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for m,a,b,r2,cat in zip(mats, a_v, b_v, r2f, cats):
        ax.scatter(b, a, s=np.interp(r2,[0.985,0.995],[50,140]),
                  c=cat_c[cat], marker=cat_m[cat], edgecolors='white', lw=1.2, zorder=5)
        ox,oy = offs[m]; ax.annotate(m, (b+ox,a+oy), fontsize=7, ha='center', color='#222')

    ax.set_xlabel('b — rate constant (day$^{-1}$)'); ax.set_ylabel('a — ultimate CO$_2$ capacity (g)')
    ax.tick_params(direction='in')
    ax.legend(handles=[Patch(fc=BLUE,label='Blends'),Patch(fc=RED,label='Pure'),Patch(fc=VIOLET,label='+Talc')],
             fontsize=7, frameon=False, loc='lower right')
    finalize(fig, 'figD_kinetics_ab', dpi=300)

# ================================================================
# FIG E: Heatmap
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
            v = mat[i,j]; tc = 'white' if v < 0.35 else 'black'
            ax.text(j, i, f'{v:.3f}' if v>-1 else f'{v:.2f}',
                   ha='center', va='center', fontsize=6.5, fontweight='bold', color=tc)
        if mat[i,3] < 0.6:
            ax.add_patch(plt.Rectangle((2.5,i-0.5), 1, 1, fill=False, edgecolor=RED, lw=1.5, ls='--'))

    ax.set_xticks(range(6)); ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(9))
    ax.set_yticklabels([f'{r[0]} | {r[1]}' for r in rows], fontsize=6.5)
    for y in [2.5,5.5]: ax.axhline(y=y, color='black', lw=1.2)
    for y,d in [(1,'PBAT/PLA'),(4,'PBAT/TPS'),(7,'PBAT/PBS')]:
        ax.text(-1.2, y, d, fontsize=8, fontweight='bold', va='center', ha='right')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.82)
    cbar.set_label('R² (random split)', fontsize=8)
    finalize(fig, 'figE_heatmap_full', dpi=300)

# ================================================================
# FIG F: Waterfall
# ================================================================
def figF():
    features = ['Days','Humidity','Temperature','Compost Vol.','Ratio']
    delta = [-33.042, +0.072, +0.060, +0.013, -0.006]
    cols = [RED if d<-1 else ('#F4A582' if d<0 else GREEN) for d in delta]

    fig = plt.figure(figsize=(W1, 3.3))
    gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.06)
    ax_t, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x = np.arange(len(features))

    for ax in [ax_t, ax_b]:
        bars = ax.bar(x, delta, color=cols, edgecolor='white', width=0.5)
        ax.axhline(y=0, color='black', lw=0.5, ls='--'); ax.set_xticks(x)
        for bar, val in zip(bars, delta):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(0.15 if val>=0 else -0.3),
                   f'{val:+.3f}' if abs(val)<1 else f'{val:+.1f}',
                   ha='center', fontsize=7, fontweight='bold')

    ax_t.set_ylim(-34.5,-32.0); ax_t.set_xticklabels([])
    ax_b.set_ylim(-0.20,0.20); ax_b.set_xticklabels(features, rotation=12, ha='right', fontsize=8)
    ax_b.set_xlabel('Feature removed')
    for ax in [ax_t, ax_b]:
        d=0.02; kw=dict(transform=ax.transAxes, color='k', clip_on=False, lw=0.8)
        ax.plot((-d,d),(1-d*3,1+d*3),**kw); ax.plot((-d,d),(-d*3,+d*3),**kw)
    ax_t.set_ylabel('ΔR²'); ax_t.yaxis.set_label_coords(-0.14, 0.5)
    finalize(fig, 'figF_waterfall_ablation', dpi=300)

# ================================================================
# FIG G: Counterfactual
# ================================================================
def figG():
    windows = ['Growth','Growth+\nDecay','Mixed','Plateau']
    r2_p = [-3.674, -1.807, 0.437, 0.951]
    r2_l = [0.022, -0.322, 0.849, -0.058]
    ss = [0.79, 0.53, 0.73, -20.6]

    fig = plt.figure(figsize=(W1, 3.8))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.18)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    x, w = np.arange(4), 0.32
    b1 = ax1.bar(x-w/2, r2_p, w, color=GREY, edgecolor='white', label='Persistence (0 params)')
    b2 = ax1.bar(x+w/2, r2_l, w, color=BLUE, edgecolor='white', label='LS-MPR (21 params)')
    ax1.axhline(y=0, color='black', lw=0.5, ls='--'); ax1.set_ylabel('Test R²')
    ax1.set_xticks(x); ax1.set_xticklabels([]); ax1.legend(fontsize=7, frameon=False)
    ax1.axhspan(-4.5,0,alpha=0.04,color='red'); ax1.axhspan(0,1.2,alpha=0.04,color='green')
    for bars in [b1,b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, h+(0.07 if h>=0 else -0.3),
                    f'{h:+.3f}' if abs(h)<1 else f'{h:+.2f}',
                    ha='center', fontsize=6, rotation=90, va='bottom' if h>=0 else 'top')

    ax2.plot(x, [max(s,-10) for s in ss], 'o-', color=BLUE2, lw=1.5, markersize=7)
    ax2.axhline(y=0, color='black', lw=0.5, ls='--')
    ax2.set_ylabel('Skill Score'); ax2.set_xticks(x); ax2.set_xticklabels(windows, fontsize=7.5)
    ax2.set_ylim(-0.8, 1.5)
    for i, s in enumerate(ss):
        if s > -10: ax2.text(i, s+0.1, f'{s:+.2f}', ha='center', fontsize=7, fontweight='bold')
    ax2.annotate('Plateau SS = −20.6\n(persistence dominates)', xy=(3,1.2), fontsize=7, ha='center', color=RED, fontweight='bold')
    finalize(fig, 'figG_counterfactual', dpi=300)

# ================================================================
# FIG H: LOCO heatmap
# ================================================================
def figH():
    mat = np.array([[-1.365,0.201,-7.366],[-0.004,0.383,-5.270],[-0.075,0.125,-1.562]])
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=1)
    im = ax.imshow(mat, aspect='auto', cmap='coolwarm', norm=norm)
    for i in range(3):
        for j in range(3):
            v = mat[i,j]
            ax.text(j, i, f'{v:.3f}' if v>-10 else f'{v:.2f}',
                   ha='center', va='center', fontsize=9, color='white' if v<-1 else 'black', fontweight='bold')
        ax.add_patch(plt.Rectangle((1.5,i-0.5), 1, 1, fill=False, edgecolor=RED, lw=2.5))
    ax.set_xticks(range(3)); ax.set_xticklabels(['C1\n(T50/H70/R70)','C2\n(T58/H60/R100)','C3\n(T58/H70/R0)'], fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(['CO₂ release','Residual mass','Tensile strength'], fontsize=8)
    ax.set_xlabel('Left-out condition', fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04, shrink=0.85)
    cbar.set_label('Temporal R²', fontsize=7.5)
    finalize(fig, 'figH_loco_heatmap', dpi=300)

# ================================================================
# FIG I: Cross-material
# ================================================================
def figI():
    mats = ['PBAT/PLA (train)','PBAT+PGA','PBAT+Talc','PBAT1','PBAT2','PLA']
    r2c = [0.9938,0.9818,0.4853,0.4238,0.3566,0.5261]
    r2s = [0.9938,0.9916,0.9954,0.9954,0.9956,0.9955]
    cols = [BLUE,GREEN,RED,RED,RED,RED]; mkrs = ['*','o','^','^','^','^']
    offs = [(-0.0008,0.04),(-0.0008,0.04),(0.0008,-0.05),(0.0008,-0.05),(0.0008,0.04),(0.0008,-0.05)]
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    for i in range(6):
        ax.scatter(r2s[i], r2c[i], s=np.interp([123,124,174,178,181,170][i],[120,185],[50,140]),
                  c=cols[i], marker=mkrs[i], edgecolors='white', lw=1.2, zorder=5)
        ox,oy = offs[i]; ax.annotate(mats[i], (r2s[i]+ox, r2c[i]+oy), fontsize=7, ha='center', color=cols[i], fontweight='bold')
    ax.plot([0.978,1.002],[0.978,1.002],'k--',lw=0.6,alpha=0.35)
    ax.axhline(y=0.5, color='grey', lw=0.7, ls=':', alpha=0.5)
    ax.axhspan(-0.1,0.5, alpha=0.04, color='red')
    ax.set_xlabel('Single-material R² (Days-only, d=2)'); ax.set_xlim(0.978,1.002)
    ax.set_ylabel('Cross-material R² (trained on PBAT/PLA)'); ax.set_ylim(0.25,1.03)
    ax.legend(handles=[
        plt.Line2D([0],[0],marker='*',c=BLUE,ms=12,lw=0,label='Train'),
        plt.Line2D([0],[0],marker='o',c=GREEN,ms=8,lw=0,label='Similar'),
        plt.Line2D([0],[0],marker='^',c=RED,ms=8,lw=0,label='Dissimilar')],
        fontsize=7, frameon=False, loc='lower left')
    finalize(fig, 'figI_cross_material', dpi=300)

# ================================================================
# FIG J: Kernel ablation
# ================================================================
def figJ():
    kernels = ['Dot+RBF+WhiteNoise\n(Full)','WhiteNoise only','RBF only','Dot only']
    r2v = [0.162, -0.015, -0.104, -0.321]
    rmse = [20.71, 22.21, 23.45, 26.65]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    y = range(len(kernels))
    cols = [BLUE, GREY, GREY, GREY]; ecs = ['#0A3A6B','#AAAAAA','#AAAAAA','#AAAAAA']
    bars = ax.barh(y, r2v, height=0.5, color=cols, edgecolor=ecs, lw=[1.5,0.5,0.5,0.5])
    for bar,v,rm in zip(bars,r2v,rmse):
        ax.text(v+(0.02 if v>=0 else -0.02), bar.get_y()+bar.get_height()/2,
               f'R²={v:+.3f}  (RMSE={rm:.1f})', va='center', fontsize=7,
               ha='left' if v>=0 else 'right', color='#333')
    ax.axvline(x=0, color='black', lw=0.7, ls='--')
    ax.axvspan(-0.5,0,alpha=0.04,color='red')
    ax.set_yticks(y); ax.set_yticklabels(kernels, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel('Temporal holdout R² (CO$_2$)')
    ax.axhspan(-0.4,0.4,alpha=0.06,color=BLUE)
    ax.text(0.98,0.06,'RBF is dominant\ncomponent',transform=ax.transAxes,fontsize=6.5,color='#555',ha='right')
    finalize(fig, 'figJ_kernel_ablation', dpi=300)

# ================================================================
# FIG K: Material comparison
# ================================================================
def figK():
    mats = ['PBAT/PLA\n(80:20)','PBAT1','PBAT2','PLA','PBAT+PGA','PBAT+Talc']
    co2 = [123.34,177.90,181.40,170.04,124.70,173.52]
    deg = [84.5,85.9,86.3,89.9,81.2,85.0]
    cols6 = [BLUE,RED,RED,RED,GREEN,VIOLET]
    fig, ax1 = plt.subplots(figsize=(W1, 2.8))
    x = np.arange(6)
    ax1.bar(x, co2, 0.5, color=cols6, edgecolor='white', lw=0.5)
    ax1.errorbar(0, co2[0], yerr=15.2, color='black', capsize=4, lw=0.8)
    ax1.set_ylabel('CO$_2$ at Day 180 (g)', color=BLUE); ax1.set_ylim(0,230)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.set_xticks(x); ax1.set_xticklabels(mats, rotation=10, ha='right', fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(x, deg, 'D-', color='black', lw=1.5, markersize=6, zorder=10)
    ax2.errorbar(0, deg[0], yerr=3.1, color='black', capsize=4, lw=0.8)
    ax2.set_ylabel('Degradation at Day 180 (%)'); ax2.set_ylim(76,94)

    ax1.legend(handles=[
        Patch(fc=BLUE, label='CO$_2$ (g)'),
        plt.Line2D([0],[0],marker='D',c='black',lw=1.5,label='Degradation (%)')],
        fontsize=7, frameon=False, loc='upper left')
    ax1.annotate('Highest CO$_2$', xy=(2,co2[2]), xytext=(3.8,215), fontsize=7.5,
                color=RED, fontweight='bold', arrowprops=dict(arrowstyle='->',color=RED,lw=1.2))
    finalize(fig, 'figK_material_comparison', dpi=300)

# ================================================================
if __name__ == '__main__':
    print(f'Output: {OUT}')
    for f in [figA,figB,figC,figD,figE,figF,figG,figH,figI,figJ,figK]:
        try:
            f(); print(f'  {f.__name__} OK')
        except Exception as e:
            print(f'  {f.__name__} FAILED: {e}')
    print('Done.')