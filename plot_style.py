"""Unified IEEE-paper figure style for matplotlib.
Imported by every plotting script to ensure consistent look.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# IEEE single-column ~ 3.5 in, double-column ~ 7.16 in
COL = 3.5
DCOL = 7.16

# Custom colour palette: cool, distinguishable, print-friendly
PALETTE = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#A2142F', '#000000']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'stix',
        'font.size': 8.5,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4.5,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.5,
        'grid.linestyle': ':',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '0.5',
        'legend.fancybox': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })

def style_axes(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.tick_params(width=0.6)
