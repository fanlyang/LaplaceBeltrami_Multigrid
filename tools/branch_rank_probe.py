#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""How much of the complement can a *linear* map of rank <= k reach?

The DeepONet's branch is an MLP that narrows to ``width`` units before widening
again.  Its Jacobian factors through ``R^width`` at every point, so for the
*linear* target this stage actually has (see the note on linearity in
STAGE1_SMOOTHING_PROPERTY.md) a width-``w`` branch is -- to first order -- a
rank-``w`` map.  The complement is ``n - n_c = 768`` dimensional here, so a
branch narrower than that cannot express the exact complement correction.

This tool computes the reference curve that says how much is lost: the **best**
linear map ``r -> delta_e`` of rank at most ``k``, fitted to the same training
pairs in the same A-inner product, scored on the same test errors.  It is the
reduced-rank regression

    min over rank(M) <= k of  || M X - Y ||_F^2 ,   X = R^T,  Y = G E^T,

whose solution is ``M_k = Y V_k S_k^{-1} U_k^T`` from the SVD ``X = U S V^T``,
followed by ``W = G^{-1} M_k`` with ``G^T G = A``.

This is a property of the *data*, not of any training run, so it needs no
network and takes seconds.  Where the trained network's ``mu_F`` sits relative to
this curve is what separates "the network did not learn it" from "a branch this
narrow cannot represent it".

Usage
-----
    python tools/branch_rank_probe.py --data-dir level_data/L3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deeponet_smoother import load_level  # noqa: E402
from stage1_smoothing_property import (  # noqa: E402
    GalerkinCoarseSpace,
    build_split,
    energy_rows,
    rho_of,
    scaled_plan,
)


def reduced_rank_map(R: np.ndarray, E: np.ndarray, G: np.ndarray, A,
                     k: int, w: np.ndarray | None = None) -> np.ndarray:
    """Best rank-``k`` linear map ``r -> delta_e`` in the A-inner product.

    ``w`` are the per-sample weights ``1/||e_i||_A``; supplying them fits the
    *relative* energy the network is trained on and the table is scored with,
    rather than the absolute one.  Without them the reference would be solving an
    easier problem than the network and the comparison would flatter the network.
    """
    if w is not None:
        R = R * w[:, None]
        E = E * w[:, None]
    X = R.T                                   # (n, N) = R^T
    Y = G @ E.T                               # (n, N) = G E^T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, int((S > 1e-12 * S[0]).sum()), S.size)
    Uk, Sk, Vk = U[:, :k], S[:k], Vt[:k].T
    M = (Y @ Vk) * (1.0 / Sk) @ Uk.T          # (n, n), rank <= k
    return np.linalg.solve(G, M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="level_data/L3")
    ap.add_argument("--ranks", type=int, nargs="*",
                    default=[64, 128, 192, 256, 384, 512, 640, 768, 1024])
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--ftol", type=float, default=1e-8)
    ap.add_argument("--ssor-omega", type=float, default=1.0)
    args = ap.parse_args()

    ld = load_level(args.data_dir, verbose=False)
    A, P, coords, angles = ld.A, ld.P, ld.coords, ld.angles
    n = A.shape[0]
    n_side = max(2, int(round(np.sqrt(n))))
    nyq = max(1, n_side // 2)
    coarse = GalerkinCoarseSpace(A, P, verbose=False)
    comp_dim = n - P.shape[1]

    tr = build_split("train", scaled_plan(1.0), A, coarse, angles, nyq, n,
                     args.seed, args.ftol)
    te = build_split("test", scaled_plan(0.5), A, coarse, angles, nyq, n,
                     args.seed, args.ftol)

    G = np.linalg.cholesky(np.asarray(A.todense())).T         # G^T G = A

    # Same weighting as the training loss: relative, not absolute, A-energy.
    w = 1.0 / np.maximum(np.sqrt(energy_rows(tr.EF, A)), 1e-12)

    print("level data   %s" % args.data_dir)
    print("n = %d, n_c = %d, complement dim = %d" % (n, P.shape[1], comp_dim))
    print("train %d, test %d errors" % (tr.E.shape[0], te.E.shape[0]))

    # The training residuals span at most ker(P^T); report that rank, because it
    # caps what ANY linear map fitted to this data can do.
    s = np.linalg.svd(tr.RF, compute_uv=False)
    eff = int((s > 1e-10 * s[0]).sum())
    print("rank of the training residual set = %d  (complement dim %d)"
          % (eff, comp_dim))
    print()

    print("  %5s  %9s  %9s  %9s" % ("rank k", "mu_F", "mu_C", "mu_C/mu_F"))
    print("  " + "-" * 40)
    for k in args.ranks:
        W = reduced_rank_map(tr.RF, tr.EF, G, A, k, w)
        mF = float(rho_of(te.EF, te.EF - te.RF @ W.T, A).mean())
        mC = float(rho_of(te.EC, te.EC - te.RC @ W.T, A).mean())
        # mu_F -> 0 is exact here (the map is recovered), so the ratio is
        # reported as infinite rather than as a huge finite number.
        ratio = ("inf" if mF < 1e-12 else "%9.2f" % (mC / mF))
        print("  %5d  %9.4f  %9.4f  %9s" % (k, mF, mC, ratio))

    print()
    print("  A trained branch of width w sits near the k = w row.  If its mu_F is")
    print("  at that row rather than below it, the width -- not the optimiser -- is")
    print("  what the run measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
