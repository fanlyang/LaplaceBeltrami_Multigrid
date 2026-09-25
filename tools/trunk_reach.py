#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""How far into the coarse-space complement can a given trunk reach?

The DeepONet writes its correction as

    delta_e = sum_k b_k(r) T_k(x) ,

so for a fixed trunk the correction lies in ``span(T)`` whatever the branch
returns.  The A-norm reach of ``span(T)`` into the complement component ``e_F``
therefore puts a floor under the smoothing factor that **no branch can beat**:

    reach_F(e) = || proj_span(T) e_F ||_A^2 / || e_F ||_A^2
    mu_F       >= E[ sqrt(1 - reach_F) ] .

A run whose mu_F sits on that floor has measured its trunk, not its smoother.
This tool computes the floor for a family of trunks *before* training, which is
what makes the trunk geometry a decision rather than an accident.

The interesting result on the exported torus hierarchy is that the floor is
governed by the trunk's highest spatial frequency, and the value it has to reach
is the **fine** lattice's Nyquist, not the coarse one:

    range(P) at level l can represent what the level l-1 lattice resolves, so its
    A-orthogonal complement is precisely the content above that -- i.e. the band
    between the coarse and the fine Nyquist.  A Fourier trunk must carry modes up
    to the fine Nyquist before it can span that band.

For ``level_data/L3`` (32x32 fine lattice, 16x16 coarse, Nyquist 16 and 8) the
floor falls from ~0.99 at K=4 to ~0.0007 at K=16, and nothing below K=16 is
usable.  A hidden MLP layer narrows this further: ``rank(T) <= trunk width``, so
a trunk of width 256 cannot exceed rank 257 however many modes it is fed, which
holds the floor near 0.75 on its own.

Usage
-----
    python tools/trunk_reach.py --data-dir level_data/L3
    python tools/trunk_reach.py --data-dir level_data/L3 --modes 2 4 6 8 10 12 16
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deeponet_smoother import load_level, trunk_features  # noqa: E402
from stage1_smoothing_property import (  # noqa: E402
    GalerkinCoarseSpace,
    build_split,
    energy_rows,
    scaled_plan,
)


def floor_for(T: np.ndarray, EF: np.ndarray, A, reg_rel: float = 1e-12) -> tuple:
    """Best possible ``mu_F`` for any branch against this trunk.

    The best A-norm approximation of each ``e_F`` inside ``span(T)`` solves the
    normal equations ``(T^T A T) c = T^T A e_F``.  Returns the mean of
    ``sqrt(1 - reach)`` and ``rank(T)``.
    """
    TtAT = T.T @ np.asarray(A @ T)
    p = T.shape[1]
    reg = reg_rel * np.trace(TtAT) / max(p, 1) * np.eye(p)
    coef = np.linalg.solve(TtAT + reg, T.T @ np.asarray(A @ EF.T))
    approx = (T @ coef).T
    reach = energy_rows(approx, A) / np.maximum(energy_rows(EF, A), 1e-300)
    return (float(np.mean(np.sqrt(np.maximum(1.0 - reach, 0.0)))),
            int(np.linalg.matrix_rank(T, tol=1e-10)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="level_data/L3")
    ap.add_argument("--modes", type=int, nargs="*", default=[2, 4, 6, 8, 10, 12, 14, 16])
    ap.add_argument("--n-test", type=float, default=0.5,
                    help="test-plan multiplier (the default matches the Stage I run)")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--coords-only", action="store_true",
                    help="also report the literal X_3 trunks (xyz/angles/trig)")
    args = ap.parse_args()

    ld = load_level(args.data_dir, verbose=False)
    A, P, coords, angles = ld.A, ld.P, ld.coords, ld.angles
    if P is None:
        raise SystemExit("no prolongation.npz in %s" % args.data_dir)
    n = A.shape[0]
    n_side = max(2, int(round(np.sqrt(n))))
    nyq = max(1, n_side // 2)
    n_c = P.shape[1]

    coarse = GalerkinCoarseSpace(A, P, verbose=False)
    test = build_split("test", scaled_plan(args.n_test), A, coarse, angles, nyq, n,
                       args.seed, 1e-8)

    print("level data      %s" % args.data_dir)
    print("n = %d (%dx%d lattice, Nyquist %d), n_c = %d, complement dim = %d"
          % (n, n_side, n_side, nyq, n_c, n - n_c))
    print("coarse lattice Nyquist = %d" % max(1, (n_side // 2) // 2))
    print("test errors     %d, mean complement energy fraction %.4f"
          % (test.E.shape[0], test.frac_F.mean()))
    print()
    print("  %-34s %6s %6s %12s" % ("trunk", "dim", "rank", "floor on mu_F"))
    print("  " + "-" * 62)

    rows = []
    for K in args.modes:
        F = trunk_features(angles, "fourier", coords, K)
        f, r = floor_for(F, test.EF, A)
        rows.append(("fourier K=%d" % K, F.shape[1], r, f))
        print("  %-34s %6d %6d %12.6f" % ("fourier K=%d" % K, F.shape[1], r, f))

    if args.coords_only:
        for mode in ("xyz", "angles", "trig"):
            F = trunk_features(angles, mode, coords, 0)
            f, r = floor_for(F, test.EF, A)
            print("  %-34s %6d %6d %12.6f" % (mode, F.shape[1], r, f))

    best = min(rows, key=lambda t: t[3])
    print()
    print("  lowest floor: %s -> %.6f" % (best[0], best[3]))
    print("  A trunk whose floor is not small has made the trunk, not the branch, "
          "the subject of the measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
