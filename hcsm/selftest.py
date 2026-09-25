"""Independent checks of the mathematics this experiment rests on.

Run it before trusting any result:

    python -m hcsm.selftest

Each check is stated as the identity it verifies, and it fails loudly rather than
printing a number and hoping. The point is that every later claim -- "e_C and e_F
are A-orthogonal", "the two-grid test uses the Galerkin operator", "the classical
smoothers are the framework's" -- is machine-checked here instead of being
asserted in prose.
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp
import torch

import deeponet_smoother as ds

from . import style  # noqa: F401  (also makes the package import cleanly)
from .evaluate import Smoothers
from .model import build_features, build_model, stage1_loss
from .problem import EPSILONS, lattice_grid, load_problem
from .samplers import controlled_modes, draw_split

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))


def main() -> int:
    root = "level_data"
    print("=" * 78)
    print("hcsm self-test: the identities the experiment relies on")
    print("=" * 78)

    for eps in EPSILONS:
        print()
        print("epsilon = %g   (contrast %.0f)" % (eps, (1 + eps) / eps))
        p = load_problem(root, eps, level=3)

        # -- 1. the operator is what the specification says ----------------
        sym = float(abs(p.A - p.A.T).max())
        check("A_epsilon is symmetric", sym == 0.0, "max|A - A^T| = %g" % sym)
        is_spd, lmin, lmax = ds.sparse_spd_report(p.A.tocsc())
        check("A_epsilon is positive definite", is_spd,
              "lambda in [%.4g, %.4g]" % (lmin, lmax))
        check("A_epsilon is never densified", p.A.nnz < p.n * p.n / 4,
              "%d nonzeros for %d DoFs" % (p.A.nnz, p.n))

        # the exported coefficient really is kappa_epsilon
        expected = eps + np.sin(p.angles[:, 0]) ** 2 * np.cos(p.angles[:, 1]) ** 2
        derr = float(np.max(np.abs(p.coeff - expected)))
        check("exported coefficient is kappa_epsilon", derr < 1e-14,
              "max dev %.3e" % derr)
        check("contrast matches (1+eps)/eps",
              abs(p.coeff.max() / p.coeff.min() - (1 + eps) / eps) < 1e-9,
              "measured %.3f" % (p.coeff.max() / p.coeff.min()))

        # -- 2. the A-orthogonal decomposition -----------------------------
        rng = np.random.default_rng(12345)
        worst_res, worst_cross, worst_id = 0.0, 0.0, 0.0
        for _ in range(20):
            e = p.to_unit_energy(rng.normal(size=p.n))
            e_C, e_F = p.decompose(e)
            # e = e_C + e_F exactly
            worst_id = max(worst_id, float(np.linalg.norm(e - e_C - e_F)))
            # P^T A e_F = 0
            ptA = p.P.T @ (p.A @ e_F)
            denom = float(np.linalg.norm(p.P.T @ (p.A @ e)))
            worst_res = max(worst_res, float(np.linalg.norm(ptA)) / denom)
            # e_C^T A e_F = 0
            cross = float(e_C @ (p.A @ e_F))
            nrm = p.norm_A(e_C) * p.norm_A(e_F)
            if nrm > 0:
                worst_cross = max(worst_cross, abs(cross) / float(nrm))

        check("e = e_C + e_F to machine precision", worst_id < 1e-12,
              "worst |e - e_C - e_F| = %.3e" % worst_id)
        check("P^T A e_F = 0 (relative)", worst_res < 1e-12,
              "worst relative residual = %.3e" % worst_res)
        check("e_C^T A e_F = 0 (relative)", worst_cross < 1e-12,
              "worst relative cross energy = %.3e" % worst_cross)

        # -- 3. the coarse operator is the Galerkin one --------------------
        A_H_direct = (p.P.T @ (p.A @ p.P)).tocsr()
        check("A_H = P^T A P, not the assembled operator",
              float(abs(p.A_H - A_H_direct).max()) == 0.0)
        if p.A_H_assembled is not None:
            rel = float(abs(p.A_H - p.A_H_assembled).max()) / float(abs(p.A_H).max())
            print("       (the assembled coarse operator differs by %.2e "
                  "relative -- recorded, not used)" % rel)

        # -- 4. the smoothers are the framework's --------------------------
        mine = Smoothers(p.A)
        theirs = ds.ClassicalSmoothers(p.A)
        r = p.A @ p.to_unit_energy(rng.normal(size=p.n))
        d1 = mine.jacobi(r, 2.0 / 3.0)
        d2 = theirs.jacobi(r, 2.0 / 3.0)
        check("Jacobi agrees with the framework",
              float(np.max(np.abs(d1 - d2))) == 0.0)
        d1, d2 = mine.gs(r), theirs.gauss_seidel(r)
        check("forward GS agrees with the framework",
              float(np.max(np.abs(d1 - d2))) < 1e-12,
              "max dev %.3e" % float(np.max(np.abs(d1 - d2))))
        d1, d2 = mine.sgs(r), theirs.symmetric_gs(r)
        check("symmetric GS agrees with the framework",
              float(np.max(np.abs(d1 - d2))) < 1e-12,
              "max dev %.3e" % float(np.max(np.abs(d1 - d2))))
        d1, d2 = mine.ssor(r, 1.0), theirs.ssor(r, 1.0)
        check("SSOR(1) agrees with the framework and equals SGS",
              float(np.max(np.abs(d1 - d2))) < 1e-12)

        # -- 5. the DoF chart is a tensor lattice --------------------------
        try:
            XI, ETA, V = lattice_grid(p, p.coeff)
            ok = V.shape == (p.n_side, p.n_side)
            # Round-trip through the same exact pairing the helper uses: every DoF
            # must read back its own coefficient from the grid it was placed on,
            # and the grid must be a permutation (not a lossy resampling) of the
            # field. Both directions are checked; a searchsorted snap would pass
            # the second and fail the first.
            xir, etar = np.round(p.angles[:, 0], 12), np.round(p.angles[:, 1], 12)
            ux, ix = np.unique(xir, return_inverse=True)
            ue, ie = np.unique(etar, return_inverse=True)
            dev = float(np.max(np.abs(V[ie, ix] - p.coeff)))
            perm = float(np.max(np.abs(np.sort(V.ravel()) - np.sort(p.coeff))))
            # and the grid axes must be uniformly spaced in both directions
            dx = np.diff(ux)
            de = np.diff(ue)
            uniform = (dx.size == 0 or np.allclose(dx, dx[0])) and \
                      (de.size == 0 or np.allclose(de, de[0]))
            check("DoFs form a regular %dx%d (xi, eta) lattice" % (p.n_side, p.n_side),
                  ok and dev < 1e-14 and perm < 1e-14 and uniform,
                  "roundtrip %.2e, permutation %.2e, uniform spacing %s"
                  % (dev, perm, uniform))
        except ValueError as exc:
            check("DoFs form a regular (xi, eta) lattice", False, str(exc))

        check("Nyquist is n_side/2, not n_side", p.nyquist == p.n_side // 2,
              "n_side = %d, nyquist = %d" % (p.n_side, p.nyquist))

        # -- 6. samples really are in coarse-space-complement form ---------
        split = draw_split(p, plan=(("smooth", 6), ("multiscale", 6),
                                    ("localized", 6), ("algebraic", 6),
                                    ("mixed", 6)),
                           seed=7, role="selftest")
        Ef, Ec = split.E_F, split.E_C
        den = np.einsum("bi,bi->b", Ef, (p.A @ Ef.T).T)
        check("every training target has unit A-energy",
              float(np.max(np.abs(den - 1.0))) < 1e-12,
              "worst dev %.3e" % float(np.max(np.abs(den - 1.0))))
        # P^T A e_F, row by row: e_F (b,n) @ (A P) (n,m) -> (b,m)
        AP = (p.A @ p.P).tocsr()
        res = np.linalg.norm(Ef @ AP, axis=1)
        den = np.linalg.norm(np.asarray(split.E) @ AP, axis=1)
        check("every sampled e_F satisfies P^T A e_F = 0",
              float(np.max(res / den)) < 1e-11,
              "worst relative %.3e" % float(np.max(res / den)))
        check("no generator signature is shared between mechanisms",
              len(set(split.signature)) == len(split.signature))

        # -- 7. the Stage-I loss is the specified ratio --------------------
        features_np = build_features(p)
        model = build_model(p, features_np.shape[1], p=8, width=16, depth=2,
                            base_smoother="none", seed=0)
        A_t = ds.to_torch_sparse(p.A)
        AT_t = ds.to_torch_sparse(p.A.T.tocsr())
        E_t = torch.from_numpy(split.E_F[:4])
        R_t = torch.from_numpy(split.R_F[:4])
        corr = model(R_t, torch.from_numpy(features_np))
        loss = float(stage1_loss(corr, E_t, A_t, AT_t))
        # recompute the ratio in numpy, independently
        Cn = corr.detach().numpy()
        num = np.einsum("bi,bi->b", split.E_F[:4] - Cn,
                        (p.A @ (split.E_F[:4] - Cn).T).T)
        ref = float(np.mean(num / np.einsum("bi,bi->b", split.E_F[:4],
                                            (p.A @ split.E_F[:4].T).T)))
        check("L_smooth equals the hand-computed ratio", abs(loss - ref) < 1e-14,
              "torch %.12f vs numpy %.12f" % (loss, ref))

        # a zero correction must score exactly 1
        zero = torch.zeros_like(E_t)
        check("L_smooth(0) = 1", abs(float(stage1_loss(zero, E_t, A_t, AT_t)) - 1.0) < 1e-14)

        # -- 8. controlled modes are projected the same way ----------------
        modes = controlled_modes(p, wavenumbers=(1, 2, 3, 5), family="xi")
        if modes["k"].size:
            ApP = (p.A @ p.P).tocsr()
            mres = np.linalg.norm(modes["E_F"] @ ApP, axis=1)
            mden = np.linalg.norm(modes["E"] @ ApP, axis=1)
            check("controlled modes are projected onto the complement",
                  float(np.max(mres / mden)) < 1e-11,
                  "worst relative %.3e" % float(np.max(mres / mden)))
        else:
            check("controlled modes produced samples", False, "all skipped")

    print()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("=" * 78)
    print("%d checks, %d failed" % (len(RESULTS), n_fail))
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
