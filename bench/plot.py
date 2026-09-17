#!/usr/bin/env python3
"""Scaling figures from a benchmark CSV.

    ./bench/plot.py bench/results-d3.csv [outdir]

Writes scaling.pdf and scaling.png: a 2x2 panel of log-log scaling plots, one
panel per quantity that separates the methods.

The point of the figure is the *slopes*. Because every method solves the same
discretisation, the quantities that differ are:

  * how many iterations it needs, and whether that grows with the mesh;
  * how long the linear algebra takes (assembly is excluded: it is the same
    O(N) work for every method and only dilutes the comparison);
  * how much memory it holds, counted two ways -- what the solver stores as
    deal.II objects, and what the process actually uses. The third and fourth
    panels are meant to be read against each other: the direct solver's
    algebraic footprint is unremarkable and its process footprint is not, and
    the difference between the panels is its fill-in.

Exponents in the legend are fitted on the largest half of the points, where
the asymptotics have taken over; a fit through every point would be dragged
down by the small problems, where a single level makes the coarse solve exact
and there is nothing to iterate.
"""

import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-5 of the reference palette, assigned in fixed order.
# Validated for the adjacent-pair gates in light mode; three of the five sit
# below 3:1 on the light surface, which is why the full table lives in the
# README alongside the legend.
SERIES = [
    ("mg-solver", "MG standalone", "#2a78d6"),
    ("cg-mg", "CG + MG", "#eb6834"),
    ("cg-jacobi", "CG + Jacobi", "#1baf7a"),
    ("cg-ssor", "CG + SSOR", "#eda100"),
    ("direct-umfpack", "Direct (UMFPACK)", "#e87ba4"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# key, y-label, whether to put direct labels at the right-hand end
PANELS = [
    ("iterations", "Iterations to $10^{-10}$", True),
    ("t_la", "Time in the linear algebra (s)", True),
    ("algebraic_mb", "Memory held as solver objects (MB)", False),
    ("rss_growth_mb", "Process memory above baseline (MB)", False),
]

# In the memory panels several series coincide exactly, because what a method
# stores depends only on which matrices it keeps, not on how it iterates. One
# annotation each, rather than five overlapping end-labels.
COINCIDENT = {
    "algebraic_mb": [
        ("mg-solver", "MG standalone, CG + MG: active matrix plus every level"),
        ("cg-jacobi", "CG + Jacobi, CG + SSOR, Direct: active matrix only"),
    ],
}


def load(path):
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = {
                k: (v if k in ("method", "converged") else float(v))
                for k, v in row.items()
            }
            row.setdefault("converged", "yes")
            rows[row["method"]].append(row)
    for rs in rows.values():
        rs.sort(key=lambda d: d["dofs"])
        for r in rs:
            r["t_la"] = r["t_pc_setup"] + r["t_solve"]
    return rows


def slope(points):
    pts = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    if den == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den


def tail_slope(pts):
    """Slope on the largest half of the points, matching analyse.py."""
    usable = [(x, y) for x, y in pts if x > 0 and y > 0]
    tail = usable[len(usable) // 2 :] if len(usable) >= 4 else usable
    return slope(tail)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(path) or "."
    rows = load(path)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=SURFACE)
    axes = axes.ravel()

    handles = []
    for ax, (key, ylabel, label_ends) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)

        for method, label, color in SERIES:
            rs = rows.get(method)
            if not rs:
                continue

            def usable(want_ok):
                return [
                    (r["dofs"], r[key])
                    for r in rs
                    if r["dofs"] > 0
                    and r[key] > 0
                    and (r["converged"] == "yes") == want_ok
                ]

            ok = usable(True)
            capped = usable(False)

            if not ok:
                continue
            px, py = zip(*ok)

            (line,) = ax.plot(
                px,
                py,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=label,
                zorder=3,
            )

            # A run that exhausted its iteration budget is a lower bound, not
            # a measurement: drawn hollow, and excluded from the fit and from
            # the line, so it cannot bend the exponent it did not reach.
            if capped:
                cx, cy = zip(*capped)
                ax.plot(
                    cx,
                    cy,
                    linewidth=0,
                    marker="o",
                    markersize=4.5,
                    markerfacecolor=SURFACE,
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    zorder=4,
                )

            if ax is axes[0]:
                b = tail_slope(ok)
                line.set_label(
                    label if b is None else f"{label}   $N^{{{b:.2f}}}$"
                )
                handles.append(line)

            # Selective direct labels: only the two multigrid series, since
            # they are the subject; and only where they do not collide.
            if label_ends and method in ("cg-mg", "mg-solver"):
                ax.annotate(
                    label,
                    xy=(px[-1], py[-1]),
                    xytext=(7, 0),
                    textcoords="offset points",
                    color=INK_2,
                    fontsize=8.5,
                    va="center",
                    ha="left",
                    zorder=4,
                )

        for method, note in COINCIDENT.get(key, []):
            rs = rows.get(method)
            if not rs:
                continue
            ax.annotate(
                note,
                xy=(rs[-1]["dofs"], rs[-1][key]),
                xytext=(7, 0),
                textcoords="offset points",
                color=INK_2,
                fontsize=8.0,
                va="center",
                ha="left",
                zorder=4,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("unknowns $N$", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8.5, which="both")
        # Room on the right for the labels.
        ax.margins(x=0.55 if (label_ends or key in COINCIDENT) else 0.05)

    handles = [h for h in handles if h.get_label() and not h.get_label().startswith("_")]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
    )
    fig.suptitle(
        "Scaling of five solvers for the Laplace–Beltrami problem on a torus",
        color=INK,
        fontsize=12,
        y=1.075,
    )
    fig.text(
        0.5,
        1.028,
        "exponents fitted on the largest half of the converged points;"
        "  hollow markers ran out of their iteration budget",
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"scaling.{ext}")
        fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
