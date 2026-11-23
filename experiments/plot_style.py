# same look for every figure (colourblind-safe colours, always used in this order)
import matplotlib.pyplot as plt
from cycler import cycler

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

DARK, DARKGREY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BG, GRAY = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#c3c2b7"
# blue ramp for sequential stuff (light -> dark)
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]


def setup():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.titlelocation": "left", "axes.titlecolor": DARK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": DARKGREY,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": DARKGREY, "ytick.labelcolor": DARKGREY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 8,
        "axes.prop_cycle": cycler(color=SERIES),
        "savefig.dpi": 160,
    })


def log2_axis(ax, axis="x"):
    # log2 scale but with plain numbers (2, 4, 8 ...) instead of 2^1, 2^2 ...
    if axis == "x":
        ax.set_xscale("log", base=2)
        lo, hi = ax.get_xlim()
    else:
        ax.set_yscale("log", base=2)
        lo, hi = ax.get_ylim()
    ticks = [2 ** i for i in range(0, 12) if lo <= 2 ** i <= hi]
    if axis == "x":
        ax.set_xticks(ticks, [str(t) for t in ticks])
    else:
        ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def ref_line(ax, y=None, x=None, label=None):
    # thin reference line (law, zero, bound...), dashed because it's a threshold, not data
    if y is not None:
        ax.axhline(y, color=DARKGREY, lw=1, ls="--", zorder=1)
    if x is not None:
        ax.axvline(x, color=DARKGREY, lw=1, ls="--", zorder=1)
