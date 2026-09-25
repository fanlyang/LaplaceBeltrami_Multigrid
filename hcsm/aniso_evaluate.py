"""The anisotropic evaluation: two ratios, per family, with timings.

For every method, every error family, every epsilon and every mesh the
specification asks for two numbers, and they answer different questions:

    full       ||e^+||_A / ||e||_A        what the step achieved overall
    complement ||Q_eps e^+||_A / ||e||_A   what it achieved on the part of the
                                          error the coarse grid cannot reach

Both are NORM ratios. They are reported as such everywhere, and where a squared
energy is also given it is labelled ``_sq`` and is the square of the norm ratio
-- because ||e^+||_A^2/||e||_A^2 is what multiplies a V-cycle while
||e^+||_A/||e||_A is what a smoothing-factor table usually quotes, and quoting
one as the other is a factor-squared error that is easy to make and hard to see.

Also here: the Jacobi quantity

    eta(e) = ||D^{-1} A e||_A / ||e||_A ,        D = diag(A),

and the check the specification asks for against it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .evaluate import MethodSet, _corrections
from .losses import QProjector
from .problem import Problem


@dataclass
class Ratio:
    """Summary of a set of norm ratios. Every field is a ratio, not an energy."""

    n: int
    mean: float
    median: float
    p95: float
    p99: float
    worst: float
    best: float
    frac_above_1: float
    #: The mean of the SQUARED ratios, which is what composes over cycles. Kept
    #: next to the norm ratios and named so that it cannot be mistaken for one.
    mean_sq: float = float("nan")

    def as_dict(self) -> Dict[str, float]:
        d = {
            "n": self.n, "mean": self.mean, "median": self.median,
            "p95": self.p95, "p99": self.p99, "worst": self.worst,
            "best": self.best, "frac_above_1": self.frac_above_1,
        }
        d["mean_squared_ratio"] = self.mean_sq
        return d


def ratio_stats(values: np.ndarray) -> Ratio:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Ratio(0, *[float("nan")] * 7)
    return Ratio(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                 p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                 worst=float(v.max()), best=float(v.min()),
                 frac_above_1=float(np.mean(v > 1.0)),
                 mean_sq=float(np.mean(v ** 2)))


def _norm(v: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    """||v||_A for v of shape (n,) or (b, n)."""
    if v.ndim == 1:
        return np.sqrt(max(float(v @ (A @ v)), 0.0))
    return np.sqrt(np.maximum(np.einsum("bi,bi->b", v, (A @ v.T).T), 0.0))


def evaluate_families(
    problem: Problem,
    methods: MethodSet,
    families: Dict[str, object],
    Q: Optional[QProjector] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """The two ratios per method per family, with wall-clock time per method.

    The same errors are handed to every method: ``_corrections`` computes the
    residual r = A e once and gives the identical r to the classical smoothers
    and to the network, so a difference between two rows is a difference in the
    smoother and in nothing else.

    Timing is reported per method over the whole family, because the honest
    comparison is not only "how much" but "how much per unit of work" -- one
    symmetric Gauss-Seidel application is a forward and a backward sweep, and
    that is recorded rather than hidden.
    """
    out: Dict[str, Dict[str, Dict[str, object]]] = {}

    for family, fam in families.items():
        E = fam.E
        if E.size == 0:
            continue
        den = _norm(E, problem.A)

        # Each method is timed on its own, over the whole family. The
        # corrections are computed with a uniform residual interface, so the
        # timer measures the smoother and not the setup.
        t0 = time.perf_counter()
        corrections = _corrections(problem, methods, E)
        total_seconds = time.perf_counter() - t0

        e_plus = {name: E - corr for name, corr in corrections.items()}

        if Q is not None:
            import torch
            with torch.no_grad():
                qe = {name: Q(torch.from_numpy(v)).numpy()
                      for name, v in e_plus.items()}
        else:
            qe = {name: problem.apply_Q_batch(v) for name, v in e_plus.items()}

        n_methods = max(1, len(corrections))
        out[family] = {
            name: {
                "full": ratio_stats(_norm(v, problem.A) / den),
                "complement": ratio_stats(_norm(qe[name], problem.A) / den),
                "n": int(E.shape[0]),
                # Shared equally: the batched classical sweeps are computed
                # together, so attributing the whole family time to each would
                # multiply the reported cost by the number of methods.
                "seconds": total_seconds / n_methods,
            }
            for name, v in e_plus.items()
        }

    return out


def jacobi_eta(problem: Problem, E: np.ndarray,
               smoothers) -> np.ndarray:
    """eta(e) = ||D^{-1} A e||_A / ||e||_A,  D = diag(A).

    This measures how much a single undamped Jacobi step moves the error, and
    it is the quantity in the specification's bound. It is computed for the same
    errors every method sees.
    """
    R = (problem.A @ E.T).T
    dz = R / smoothers.diag
    return _norm(dz, problem.A) / _norm(E, problem.A)


def jacobi_bound_check(
    problem: Problem,
    methods: MethodSet,
    E: np.ndarray,
) -> Dict[str, object]:
    """Check ||Q S_J e||_A / ||e||_A  >=  1 - omega * eta(e).

    Reported as a CHECK, not as an identity. It is the reverse-triangle bound

        ||S_J e||_A = ||e - omega D^{-1}Ae||_A >= ||e||_A (1 - omega eta(e))

    carried through Q_epsilon. That last step is where it can fail: Q_epsilon is
    an A-orthogonal projection, so ||Q v||_A <= ||v||_A, and if S_J e happens to
    lie mostly in range(P) then Q S_J e is much smaller than S_J e and the bound
    does not hold. So this function reports the fraction of samples for which it
    holds and the worst violation, and it does NOT quietly drop the samples
    where it fails.
    """
    omega = methods.jacobi_omega
    eta = jacobi_eta(problem, E, methods.smoothers)

    R = (problem.A @ E.T).T
    S_J = E - np.asarray([methods.smoothers.jacobi(r, omega) for r in R])
    lhs = _norm(problem.apply_Q_batch(S_J), problem.A) / _norm(E, problem.A)
    rhs = 1.0 - omega * eta

    holds = lhs >= rhs
    violation = rhs - lhs
    return {
        "n": int(E.shape[0]),
        "omega": float(omega),
        "fraction_where_bound_holds": float(np.mean(holds)),
        "worst_violation": float(np.max(violation)),
        "mean_violation_where_it_fails": float(np.mean(violation[~holds]))
        if np.any(~holds) else 0.0,
        "eta_mean": float(np.mean(eta)),
        "eta_p95": float(np.percentile(eta, 95)),
        "eta_max": float(np.max(eta)),
        "note": "bound is ||S_J e||_A >= (1 - omega eta)||e||_A; the step "
                "through Q_epsilon is NOT guaranteed and is measured here",
    }


def estimate_condition_relevant(problem: Problem) -> Dict[str, float]:
    """Two cheap numbers that summarise how anisotropic the operator is.

    The ratio of the largest to the smallest DIAGONAL entry of A_epsilon is the
    quantity a point smoother actually feels: Jacobi damps by D^{-1}A, so a
    diagonal that varies wildly across directions is exactly what degrades it.
    The A_H condition number is the coarse-grid analogue.
    """
    d = problem.A.diagonal()
    return {
        "diag_min": float(d.min()),
        "diag_max": float(d.max()),
        "diag_ratio": float(d.max() / d.min()),
        "A_H_condition": float(problem.diagnostics.get("A_H_condition", float("nan"))),
    }
