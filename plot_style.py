"""Unified IEEE-conference figure style for matplotlib.

Tuned to address common reviewer complaints about Matlab/Python default plots:
larger fonts, bolder strokes, softer minor grid, refined colour palette,
high-DPI raster export.
"""
import matplotlib as mpl

COL = 3.5      # IEEE single-column width (in)
DCOL = 7.16    # IEEE double-column width (in)

# Refined palette: high contrast, colour-blind safe ordering, print-friendly.
PALETTE = ['#1F4E79', '#C0392B', '#2E7D32', '#6A3D9A', '#E07B00', '#000000']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
LINESTYLES = ['-', '--', '-.', (0, (3, 1, 1, 1)), ':', (0, (5, 1))]


def apply():
    mpl.rcParams.update({
        # Typography ------------------------------------------------------
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10.0,
        'axes.labelsize': 10.5,
        'axes.titlesize': 10.5,
        'axes.labelweight': 'normal',
        'legend.fontsize': 9.0,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        # Lines & markers -------------------------------------------------
        'lines.linewidth': 1.6,
        'lines.markersize': 5.5,
        'lines.markeredgewidth': 0.8,
        'patch.linewidth': 0.8,
        # Axes ------------------------------------------------------------
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#222222',
        'axes.labelcolor': '#111111',
        'axes.titleweight': 'normal',
        'axes.prop_cycle': mpl.cycler(color=PALETTE),
        # Grid: visible but unobtrusive ----------------------------------
        'axes.grid': True,
        'axes.grid.which': 'major',
        'grid.linewidth': 0.45,
        'grid.alpha': 0.35,
        'grid.color': '#888888',
        'grid.linestyle': '-',
        # Ticks -----------------------------------------------------------
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.top': True,
        'ytick.right': True,
        # Legend ----------------------------------------------------------
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#444444',
        'legend.fancybox': False,
        'legend.borderpad': 0.45,
        'legend.handlelength': 2.2,
        'legend.handletextpad': 0.55,
        'legend.columnspacing': 1.2,
        'legend.labelspacing': 0.35,
        # Output ----------------------------------------------------------
        'figure.dpi': 120,
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
    })


def style_axes(ax):
    """Cosmetic touch-ups applied after each plot is drawn."""
    for s in ax.spines.values():
        s.set_linewidth(1.0)
        s.set_color('#222222')
    ax.tick_params(which='major', width=0.9, length=4.0)
    ax.tick_params(which='minor', width=0.6, length=2.0)
    ax.set_axisbelow(True)
