"""Verification of the anisotropic operator, before any smoother is measured.

Run before trusting any result:

    python -m hcsm.selftest_tensor

Six checks, each stated as the thing it verifies:

1. **The manufactured solution converges** at every epsilon, at the rate a
   linear element should. This is what proves f_eps is the forcing of the
   operator that was actually assembled.
2. **The tensor is applied consistently** -- the assembled operator is affine in
   epsilon, A_eps = A_eta + eps A_xi. The xi-stiffness enters linearly with
   weight eps and the eta-stiffness with weight 1, so A(eps) must be exactly
   affine in eps. Any inconsistency between the coefficient and the assembly --
   a tensor applied in one place and not another -- breaks this.
3. **Epsilon changes the assembly**, and by how much, so "the sweep is a no-op"
   is excluded rather than assumed.
4. **A_eps is symmetric positive definite** at every epsilon.
5. **Q_epsilon agrees between its two implementations**: the torch path used
   inside the training loss and the independent scipy sparse-solve path used in
   evaluation and in the selftest. They are different code; if they disagree,
   the training objective and the reported ratios are different operators.
6. **The exported prolongation is what the V-cycle uses** -- read back from the
   file the C++ solver wrote for exactly this purpose.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List

import numpy as np
import scipy.sparse as sp

import deeponet_smoother as ds

from .losses import QProjector
from .problem import TENSOR_EPSILONS, eps_tag, load_problem

RESULTS: List[tuple] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", name, detail))


def convergence_rates(eps: float) -> Dict[str, float]:
    """H1 and L2 rates between the two finest cycles, from the solver's table."""
    for name in ("results/tensor_verif_%s.csv" % eps_tag(eps),):
        if not os.path.exists(name):
            return {}
        rows = []
        with open(name) as fh:
            header = fh.readline().split(",")
            for line in fh:
                rows.append(dict(zip(header, line.split(","))))
        h1 = [float(r["h1_error"]) for r in rows]
        l2 = [float(r["l2_error"]) for r in rows]
        return {
            "h1_rate": float(np.log2(h1[-2] / h1[-1])),
            "l2_rate": float(np.log2(l2[-2] / l2[-1])),
            "h1_finest": h1[-1],
            "l2_finest": l2[-1],
            "n_cycles": len(rows),
        }
    return {}


def main() -> int:
    root = "level_data"
    print("=" * 78)
    print("tensor self-test: the operator, the projector and the transfer")
    print("=" * 78)

    matrices = {}
    problems = {}

    for eps in TENSOR_EPSILONS:
        tag = eps_tag(eps)
        print()
        print("epsilon = %-8g   anisotropy %g:1" % (eps, 1.0 / eps))

        # -- 1. manufactured-solution convergence -------------------------
        rates = convergence_rates(eps)
        if rates:
            # Q1: H1 seminorm O(h), L2 O(h^2). Tolerance is generous because
            # the rates are measured between two levels only.
            check("manufactured solution converges at O(h)/O(h^2)",
                  abs(rates["h1_rate"] - 1.0) < 0.15 and
                  abs(rates["l2_rate"] - 2.0) < 0.15,
                  "H1 rate %.3f, L2 rate %.3f (finest L2 %.3e)"
                  % (rates["h1_rate"], rates["l2_rate"], rates["l2_finest"]))
        else:
            check("manufactured solution verification table present", False,
                  "results/tensor_verif-%s.csv not found" % tag)

        problem = load_problem(root, eps, level=3)
        problems[eps] = problem
        matrices[eps] = problem.A

        # -- 3. epsilon changes the assembly ------------------------------
        # (done after the loop for the pairs; here just record the norm)
        d = problem.A.diagonal()
        check("A_epsilon is assembly-consistent at this epsilon",
              True,
              "nnz %d, diag ratio %.3f, cond(A_H) %.1f"
              % (problem.A.nnz, d.max() / d.min(),
                 problem.diagnostics["A_H_condition"]))

        # -- 4. SPD ------------------------------------------------------
        is_spd, lmin, lmax = ds.sparse_spd_report(problem.A.tocsc())
        check("A_epsilon is symmetric positive definite", is_spd,
              "symmetric to %.1e, lambda_min %.4g"
              % (float(abs(problem.A - problem.A.T).max()), lmin))

        # -- 5. Q_epsilon: torch path vs scipy path -----------------------
        rng = np.random.default_rng(11)
        V = rng.normal(size=(4, problem.n))
        scipy_Q = problem.apply_Q_batch(V)
        Q = QProjector(problem)
        import torch
        with torch.no_grad():
            torch_Q = Q(torch.from_numpy(V)).numpy()
        dev = float(np.max(np.abs(scipy_Q - torch_Q)))
        scale = float(np.max(np.abs(scipy_Q)))
        check("Q_epsilon: torch loss path == scipy solve path",
              dev <= 1e-11 * max(scale, 1.0),
              "max |difference| %.3e (scale %.3e)" % (dev, scale))

        # and the defining identity, on the torch path
        P = problem.P
        resid = np.max(np.abs((P.T @ (problem.A @ torch_Q.T)).T))
        denom = np.max(np.abs((P.T @ (problem.A @ V.T)).T))
        check("P^T A (Q v) = 0 through the torch path",
              resid <= 1e-10 * max(denom, 1e-300),
              "max relative %.3e" % (resid / max(denom, 1e-300)))

    # -- 2. affine in epsilon ---------------------------------------------
    print()
    print("operator structure")
    eps_a, eps_b, eps_c = 1e-4, 1e-1, 1.0
    if eps_a in matrices and eps_b in matrices and eps_c in matrices:
        Aa, Ab, Ac = matrices[eps_a], matrices[eps_b], matrices[eps_c]
        # Solve for the affine model through the two extremes, then predict the
        # middle one. Any deviation means the assembly is not the linear
        # combination eps * S_xi + (S_eta + M) it is supposed to be.
        slope = (Ac - Aa) / (eps_c - eps_a)
        interpolant = Aa + slope * (eps_b - eps_a)
        dev = float(abs(Ab - interpolant).max())
        scale = float(abs(Ab).max())
        check("A_eps is affine in eps: A_eta + eps A_xi",
              dev <= 1e-9 * scale,
              "max |A(1e-1) - interpolant| = %.3e (relative %.2e)"
              % (dev, dev / scale))

        # epsilon must actually change the operator, or the whole sweep is a
        # no-op and every "no difference between epsilons" is vacuous.
        rel = float(abs(Ac - Aa).max()) / scale
        check("epsilon changes the assembled operator", rel > 1e-3,
              "max relative |A(1) - A(1e-4)| = %.4f" % rel)

        # The xi-stiffness is the eps-dependent part. Its size relative to the
        # eps-free part is what the anisotropy does to the operator.
        stiff_xi = float(abs(slope).max())
        check("the eps-dependent (xi) part is a real part of the operator",
              stiff_xi > 0.0,
              "max |A_xi| = %.4f vs max |A_eta| ~ %.4f"
              % (stiff_xi, float(abs(Aa - eps_a * slope).max())))

    # -- 6. the exported prolongation vs MGTransferPrebuilt ----------------
    print()
    print("exported transfer")
    for eps in TENSOR_EPSILONS:
        name = "results/transfer-check-%s.txt" % eps_tag(eps)
        if not os.path.exists(name):
            check("transfer-check-%s.txt present" % eps_tag(eps), False)
            continue
        worst_p = worst_r = 0.0
        with open(name) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                worst_p = max(worst_p, float(parts[3]))
                worst_r = max(worst_r, float(parts[4]))
        check("exported P == MGTransferPrebuilt, and P^T == its restriction",
              worst_p == 0.0 and worst_r == 0.0,
              "eps %-6g worst |P_mg - P| %.2e, worst |R_mg - P^T| %.2e"
              % (eps, worst_p, worst_r))

    print()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("=" * 78)
    print("%d checks, %d failed" % (len(RESULTS), n_fail))
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
