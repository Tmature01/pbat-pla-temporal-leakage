"""
Shared Nature-standard style configuration for all V10 figures.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    'font.family':       'Arial',
    'font.size':         8,
    'axes.titlesize':    8,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.minor.size':  2,
    'ytick.minor.size':  2,
    'lines.linewidth':   1.2,
    'patch.linewidth':   0.8,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
})

# Wong colorblind-friendly 8 colors (Nature preferred)
WONG = {
    'black':  '#000000',
    'orange': '#E69F00',
    'sky':    '#56B4E9',
    'green':  '#009E73',
    'yellow': '#F0E442',
    'blue':   '#0072B2',
    'red':    '#D55E00',
    'pink':   '#CC79A7',
}

# Nature size conventions (inches)
W1 = 89 / 25.4   # Single column: 89 mm
W2 = 183 / 25.4  # Double column: 183 mm

# Global spine cleanup helper
def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='top', ha='left')
