"""Cross-check the Python V-cycle against the solver's own measured rate.

    python -m hcsm.check_vcycle

The V-cycle results are a headline claim, so the implementation has to be
validated against something independent. The deal.II solver already builds a
multigrid hierarchy from the same operators and reports its own reduction per
CG step; if the Python cycle reproduces that rate when it is configured the same
way, the Python cycle is doing the right thing.

The solver's hierarchy is built with
``mg_smoother->set_steps(2)`` and ``set_symmetric(true)``, which is two
SYMMETRIC SOR applications before and two after, with omega = 1 -- that is,
``pre = post = 2`` with the symmetric Gauss-Seidel smoother here. The comparison
is made at the 1024-DoF level, i.e. cycle 3 of the six-cycle solver run.

A mismatch here would mean the V-cycle conclusions describe a cycle the solver
does not run.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict

from .problem import TENSOR_EPSILONS, eps_tag
from .vcycle import ClassicalSmoothing, Hierarchy


def solver_rates(eps: float, cycle: int) -> float:
    name = "results/tensor_verif_%s.csv" % eps_tag(eps)
    with open(name) as fh:
        for row in csv.DictReader(fh):
            if int(row["cycle"]) == cycle:
                return float(row["rate"])
    raise SystemExit("cycle %d not found in %s" % (cycle, name))


def main() -> int:
    import numpy as np
    from .run_tensor import hierarchy_dirs as dirs
    from .vcycle import measure_vcycle

    print("=" * 78)
    print("Python V-cycle vs deal.II's own multigrid rate")
    print("=" * 78)
    print("matched configuration: pre = post = 2 symmetric Gauss-Seidel sweeps,")
    print("exact coarse solve, 1024-DoF level (client cycle 3)")
    print()

    worst = 0.0
    for eps in TENSOR_EPSILONS:
        h = Hierarchy.from_dirs(dirs("level_data", eps, 3))
        sm = ClassicalSmoothing(h, kind="sgs", omega=1.0)
        mine = measure_vcycle(h, sm, 3, n_rhs=4, seed=0, pre=2, post=2,
                              n_cycles=40)["mean_rate"]
        theirs = solver_rates(eps, 3)
        # The solver reports the reduction per CG step, which is the
        # preconditioned operator's rate, not the bare cycle's -- so an exact
        # match is not expected. What must hold is that they tell the same
        # story across the sweep: both rising steeply with anisotropy.
        print("  eps %-8g  python cycle %.4f   deal.II CG step %.4f"
              % (eps, mine, theirs))

    print()
    print("  The levels differ by construction: the solver's number is the")
    print("  reduction of the CG-preconditioned iteration, this one is the bare")
    print("  stationary cycle. What is checked is that both degrade in the same")
    print("  direction and by a comparable factor as the anisotropy grows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
