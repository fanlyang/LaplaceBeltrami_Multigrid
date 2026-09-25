"""Figure style: one palette, one set of marks, applied to every plot.

Colour is assigned by the job it does, not by taste:

* **Categorical** (identity: which smoother) uses a fixed four-slot order that
  was validated all-pairs -- including colour-vision deficiency -- against the
  light chart surface. Four is the cap: a fifth slot cannot clear the floors on
  an all-pairs chart, so a fifth series is faceted or folded away rather than
  given a generated hue. ``validate_palette.js`` was run on the exact hex list in
  ``CATEGORICAL`` and every hard gate passes; the one WARN (aqua below 3:1
  contrast) is met with the relief the rule requires -- every curve is directly
  labelled and every number also appears in a CSV table under ``results/``.
* **Sequential** (magnitude: kappa_epsilon) is a single blue hue, light to dark.
* **Diverging** (polarity: a signed error field) is blue <-> red about a neutral
  grey midpoint, equal steps per arm. Never a rainbow.

The same series always gets the same colour on every figure, and the order is
never cycled.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Validated categorical order (blue, orange, aqua, violet). All-pairs CVD
#: separation 9.2 (deutan), worst normal-vision 16.3, both above the floors.
CATEGORICAL: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

#: Fixed method -> colour. Colour follows the entity, so a figure that omits a
#: method does not repaint the ones that remain.
#:
#: There are five methods but only four slots, and that is a constraint rather
#: than an oversight: four is the largest set that clears the all-pairs
#: colour-vision floors, and a fifth generated hue would fail them. The two
#: DeepONet variants therefore share the fourth hue and are told apart by dash
#: pattern and marker -- composite encoding, which is the sanctioned way to go
#: past the slot cap. Hue means "which family of method"; the dash means "which
#: variant". Every curve is also directly labelled, so identity never rests on
#: colour alone.
METHOD_COLOR: Dict[str, str] = {
    "jacobi": CATEGORICAL[0],
    "gs": CATEGORICAL[1],
    "sgs_ssor": CATEGORICAL[2],
    "deeponet_plain": CATEGORICAL[3],
    "deeponet_skip": CATEGORICAL[3],
    "deeponet_skip_fourier": CATEGORICAL[3],
    "coarse_only": INK_MUTED,
}

#: Human labels used on axes and in legends. The labels must be distinct: three
#: variants all called "DeepONet" would make a table unreadable and would put
#: identical direct labels on different curves.
METHOD_LABEL: Dict[str, str] = {
    "jacobi": "damped Jacobi",
    "gs": "Gauss-Seidel",
    "sgs_ssor": "symmetric GS / SSOR",
    "deeponet_plain": "DeepONet (plain MLP)",
    "deeponet_skip": "DeepONet (trig trunk, + Jacobi skip)",
    "deeponet_skip_fourier": "DeepONet (fourier trunk, + Jacobi skip)",
    "coarse_only": "coarse correction only",
}

#: Short labels for direct labelling at the end of a curve, where the full label
#: would run off the axes.
METHOD_SHORT: Dict[str, str] = {
    "jacobi": "Jacobi",
    "gs": "Gauss-Seidel",
    "sgs_ssor": "SGS / SSOR",
    "deeponet_plain": "DON plain",
    "deeponet_skip": "DON +skip",
    "deeponet_skip_fourier": "DON +skip 4x4",
    "coarse_only": "coarse only",
}

#: Dash pattern and marker. Together with the hue this is what keeps the three
#: DeepONet variants apart, and it also carries identity through a greyscale
#: print. The third pattern is written out explicitly rather than borrowed,
#: because every named dash style is already spoken for.
DASH_PLAIN = ":"                              # plain MLP, no skip
DASH_SKIP = "-"                               # + Jacobi skip, trig trunk
DASH_SKIP_FOURIER = (0, (5, 1, 1, 1))         # + Jacobi skip, fourier trunk

METHOD_STYLE: Dict[str, Dict[str, object]] = {
    "jacobi": {"ls": "-", "marker": "o"},
    "gs": {"ls": "--", "marker": "s"},
    "sgs_ssor": {"ls": "-.", "marker": "^"},
    "deeponet_plain": {"ls": DASH_PLAIN, "marker": "D"},
    "deeponet_skip": {"ls": DASH_SKIP, "marker": "v"},
    "deeponet_skip_fourier": {"ls": DASH_SKIP_FOURIER, "marker": "P"},
    "coarse_only": {"ls": ":", "marker": None},
}

#: The architecture variant a method name encodes, for figures whose channel for
#: "which variant" is the dash pattern rather than the hue.
VARIANT_DASH: Dict[str, object] = {
    "plain": DASH_PLAIN,
    "skip": DASH_SKIP,
    "skip_fourier": DASH_SKIP_FOURIER,
}


def sequential_cmap() -> LinearSegmentedColormap:
    """Single-hue blue ramp, light (near zero) to dark (large)."""
    steps = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    return LinearSegmentedColormap.from_list("hcsm_blue", steps)


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue <-> red about a neutral grey midpoint, equal steps per arm."""
    neg = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb"]
    pos = ["#f7d3d3", "#e88b8b", "#d24b4b", "#8f1f1f"]
    mid = "#f0efec"
    return LinearSegmentedColormap.from_list("hcsm_div", neg + [mid] + pos)


def diverging_norm(V, symmetric: bool = True):
    """A norm whose neutral grey sits at zero and whose arms have equal range."""
    v = abs(V).max() if symmetric else None
    if v is None or v == 0:
        v = 1.0
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "lines.solid_capstyle": "round",
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    })


def style_axes(ax) -> None:
    """Recessive chrome: drop the top/right spines, keep the grid as a hairline."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=3, width=0.8)


def label_line(ax, x, y, text, color, dx=6, dy=0, fontsize=8, va="center",
               ha="left"):
    """Direct label at the end of a curve, in the curve's own colour.

    Text wears the series colour only here, where it *is* the mark's identity
    label; every other number on the figure stays in ink tokens.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=fontsize, va=va, ha=ha,
                fontweight="semibold")
