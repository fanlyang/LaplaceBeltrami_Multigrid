#!/usr/bin/env python3
"""Turn a benchmark CSV from run_sweep.sh into the tables the write-up needs.

    ./bench/analyse.py bench/results-d3.csv

Prints, per method:

  * iterations, time and memory at every problem size, and
  * the observed scaling exponent of time and memory against the number of
    unknowns, fitted as log(y) = a + b log(N) on the largest few points.

The exponents are the point of the exercise. An optimal method has b = 1 for
both time and memory; a method whose iteration count grows with the mesh
shows a larger b for time, and a direct factorisation shows one larger still
for memory, because its fill-in grows faster than the matrix itself.
"""

import csv
import math
import sys
from collections import defaultdict


def load(path):
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = {
                k: (v if k in ("method", "converged") else float(v))
                for k, v in row.items()
            }
            # CSVs written before the convergence column existed: every run
            # in them converged, or it would not have been written at all.
            row.setdefault("converged", "yes")
            rows[row["method"]].append(row)
    for r in rows.values():
        r.sort(key=lambda d: int(d["dofs"]))
        for r_ in r:
            # The linear algebra proper: building the preconditioner or the
            # factorisation, plus applying it. Assembly, error evaluation and
            # file output are the same work for every method and only dilute
            # the comparison, so the exponents are quoted on this column.
            r_["t_la"] = r_["t_pc_setup"] + r_["t_solve"]
    return rows


def slope(points):
    """Least-squares slope of log(y) vs log(x); None if fewer than 2 usable."""
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


def fmt(v, w=12, p=4):
    if v is None:
        return " " * (w - 3) + "n/a"
    if v == 0:
        return f"{0.0:>{w}.{p}g}"
    if abs(v) < 1e-3 or abs(v) >= 1e5:
        return f"{v:>{w}.{p-1}e}"
    return f"{v:>{w}.{p}f}"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    rows = load(sys.argv[1])
    order = ["cg-jacobi", "cg-ssor", "mg-solver", "cg-mg", "direct-umfpack"]
    methods = [m for m in order if m in rows] + [
        m for m in sorted(rows) if m not in order
    ]

    for m in methods:
        rs = rows[m]
        print(f"\n=== {m} " + "=" * max(0, 60 - len(m)))
        print(
            f"{'dofs':>9} {'cells':>8} {'it':>6} {'setup':>10} {'pc-setup':>10} "
            f"{'solve':>10} {'lin.alg':>10} {'total':>10} {'alg MB':>9} "
            f"{'grow MB':>9} {'resid':>10}"
        )
        for r in rs:
            # A star marks a run that ran out of its iteration budget, so an
            # iteration count of 1000 means "at least 1000", not "converged".
            star = "" if r["converged"] == "yes" else "*"
            print(
                f"{int(r['dofs']):>9} {int(r['cells']):>8} "
                f"{str(int(r['iterations'])) + star:>6} "
                f"{fmt(r['t_setup'], 10)} {fmt(r['t_pc_setup'], 10)} "
                f"{fmt(r['t_solve'], 10)} {fmt(r['t_la'], 10)} "
                f"{fmt(r['t_total'], 10)} "
                f"{fmt(r['algebraic_mb'], 9)} {fmt(r['rss_growth_mb'], 9)} "
                f"{fmt(r['residual'], 10)}"
            )

        # Scaling on the largest half of the points, where the asymptotics
        # rule. A run that hit its iteration budget is a lower bound rather
        # than a measurement -- including it would *understate* the growth it
        # failed to complete -- so those points are left out.
        conv = [r for r in rs if r["converged"] == "yes"]
        tail = conv[len(conv) // 2 :] if len(conv) >= 4 else conv
        pts = [(r["dofs"], r) for r in tail]
        b_la = slope([(n, r["t_la"]) for n, r in pts])
        b_time = slope([(n, r["t_total"]) for n, r in pts])
        b_rss = slope([(n, r["rss_growth_mb"]) for n, r in pts])
        b_alg = slope([(n, r["algebraic_mb"]) for n, r in pts])
        b_it = slope([(n, r["iterations"]) for n, r in pts])

        parts = [
            f"{label} ~ N^{b:.2f}"
            for label, b in (
                ("iters", b_it),
                ("linear algebra", b_la),
                ("total time", b_time),
                ("RSS growth", b_rss),
                ("algebraic mem", b_alg),
            )
            if b is not None
        ]
        print(f"  scaling vs N (last {len(tail)} points): " + ",  ".join(parts))

    return 0


if __name__ == "__main__":
    sys.exit(main())
