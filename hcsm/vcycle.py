"""The actual V-cycle on the exported hierarchy, and a compatible outer solver.

Everything else in this experiment measures a SINGLE smoothing step on a
manufactured error. That is not the same as multigrid performance, and the
specification is explicit that the two must be reported separately. This module
runs the real thing:

    x <- x + M^{-1}(b - A x),   with M^{-1} one V-cycle,

on the exported per-level operators and prolongations, from general initial
errors and right-hand sides -- not from a manufactured error, and not from a
zero right-hand side.

**Correction form, not error form.** The cycle solves A_l x = r for its
argument, so the coarse level receives the RESTRICTED RESIDUAL. Written in
"error propagation" form it is easy to hand the coarse level P^T r as if it were
an error, which silently omits the coarse solve; the cycle then diverges and the
result looks like a smoother failure rather than a coding error.

**A flexible outer solver.** A learned smoother is in general NONLINEAR in the
residual -- the scale-equivariant wrapper is homogeneous but the tanh branch is
not linear -- and a V-cycle built from a nonlinear smoother is a nonlinear
preconditioner. Ordinary PCG assumes a fixed linear operator and its convergence
theory does not apply; the stationary iteration can still be run, and where an
accelerated method is wanted the correct one is flexible CG (Notay's FCG), which
recomputes the preconditioned direction every iteration and remains valid for a
varying preconditioner. Both are provided; the learned smoother is reported with
FCG, the classical smoothers with ordinary PCG and with the stationary iteration,
so the comparison never rests on an assumption the learned smoother breaks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .evaluate import Smoothers


@dataclass
class Hierarchy:
    """The exported multigrid hierarchy, re-read from level_data."""

    A: List[sp.csr_matrix]          # A[0] is coarsest
    P: List[Optional[sp.csr_matrix]]  # P[l] maps level l-1 -> l, None for l = 0

    @property
    def n_levels(self) -> int:
        return len(self.A)

    @staticmethod
    def from_dirs(dirs: Sequence[str]) -> "Hierarchy":
        """Load level_data directories ordered COARSE to FINE.

        Each directory holds ``level_matrix.npz`` for that level and, for every
        level above the coarsest, ``prolongation.npz`` mapping the level below
        into it. The directories must therefore be given coarsest first.
        """
        A, P = [], []
        for i, d in enumerate(dirs):
            A.append(sp.load_npz("%s/level_matrix.npz" % d).tocsr())
            if i == 0:
                P.append(None)
            else:
                P.append(sp.load_npz("%s/prolongation.npz" % d).tocsr())
                if P[-1].shape != (A[-1].shape[0], A[-2].shape[0]):
                    raise SystemExit(
                        "prolongation at level %d has shape %r but connects "
                        "%d and %d DoFs"
                        % (i, P[-1].shape, A[-2].shape[0], A[-1].shape[0]))
        return Hierarchy(A=A, P=P)


class ClassicalSmoothing:
    """One V-cycle smoothing step per level, classical."""

    def __init__(self, hierarchy: Hierarchy, kind: str = "sgs", omega: float = 2.0 / 3.0):
        self.h = hierarchy
        self.kind = kind
        self.omega = omega
        self.sm = [Smoothers(A) for A in hierarchy.A]

    def apply(self, level: int, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """One smoothing step on `level`: x += S(r - A x)."""
        s = self.sm[level]
        res = r - self.h.A[level] @ x
        if self.kind == "jacobi":
            return x + s.jacobi(res, self.omega)
        if self.kind == "gs":
            return x + s.gs(res)
        return x + s.sgs(res)


class LearnedSmoothing:
    """The same interface, backed by a trained network on ONE level.

    The model is bound to a single DoF count and ordering (the branch consumes a
    fixed-length residual), so it can only smooth the level it was trained on.
    Every other level falls back to a classical sweep, and which levels those
    are is reported rather than glossed over.
    """

    def __init__(self, hierarchy: Hierarchy, model, features, learned_level: int,
                 fallback_kind: str = "sgs", fallback_omega: float = 2.0 / 3.0):
        self.h = hierarchy
        self.learned_level = learned_level
        self.fallback = ClassicalSmoothing(hierarchy, fallback_kind, fallback_omega)
        self.sm = self.fallback.sm

        import torch
        from .model import to_torch_sparse

        self.torch = torch
        self.model = model
        self.A_t = to_torch_sparse(hierarchy.A[learned_level])
        self.X = torch.from_numpy(np.asarray(features, dtype=np.float64))
        self.levels_learned = 0
        self.levels_classical = 0

    def apply(self, level: int, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        if level != self.learned_level:
            self.levels_classical += 1
            return self.fallback.apply(level, x, r)

        self.levels_learned += 1
        t = self.torch
        res = r - self.h.A[level] @ x
        with t.no_grad():
            Rt = t.from_numpy(res[None, :])
            delta = self.model(Rt, self.X, None)[0].numpy()
        return x + delta


def vcycle(
    hierarchy: Hierarchy,
    smoother,
    level: int,
    r: np.ndarray,
    pre: int = 1,
    post: int = 1,
    coarse_solver: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """One V-cycle solving A_level x = r, returned as a correction.

    Correction form throughout: the coarse grid gets P^T applied to the
    RESIDUAL, and the coarse solution is prolonged and added.
    """
    if level == 0:
        if coarse_solver is None:
            return spla.spsolve(hierarchy.A[0].tocsc(), r)
        return coarse_solver(r)

    x = np.zeros_like(r)
    for _ in range(pre):
        x = smoother.apply(level, x, r)

    coarse_rhs = hierarchy.P[level].T @ (r - hierarchy.A[level] @ x)
    coarse_correction = vcycle(hierarchy, smoother, level - 1, coarse_rhs,
                               pre, post, coarse_solver)
    x = x + hierarchy.P[level] @ coarse_correction

    for _ in range(post):
        x = smoother.apply(level, x, r)
    return x


def measure_vcycle(
    hierarchy: Hierarchy,
    smoother,
    level: int,
    n_rhs: int = 4,
    seed: int = 0,
    pre: int = 1,
    post: int = 1,
    n_cycles: int = 30,
    tolerance: float = 1e-10,
) -> Dict[str, object]:
    """Stationary defect correction: x <- x + M^{-1}(b - A x), no acceleration.

    This measures the V-cycle itself, which is what the specification asks for:
    it is the pure contraction factor of the cycle, with no Krylov acceleration
    to hide a bad smoother or to rescue a good one.
    """
    A = hierarchy.A[level]
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n_rhs, n))
    b_norm = np.linalg.norm(b, axis=1)

    rates, iters = [], []
    t0 = time.perf_counter()
    for k in range(n_rhs):
        x = np.zeros(n)
        res = b[k] - A @ x
        hist = [np.linalg.norm(res) / b_norm[k]]
        for _ in range(n_cycles):
            x = x + vcycle(hierarchy, smoother, level, b[k] - A @ x, pre, post)
            res = b[k] - A @ x
            hist.append(np.linalg.norm(res) / b_norm[k])
            if hist[-1] <= tolerance:
                break
        iters.append(len(hist) - 1)
        # Asymptotic rate from the last few residuals, not the first few: the
        # first cycles still contain the transient.
        tail = hist[-6:]
        rates.append(float(np.exp(np.polyfit(range(len(tail)),
                                             np.log(np.maximum(tail, 1e-300)), 1)[0])))
    seconds = time.perf_counter() - t0

    return {
        "n_rhs": n_rhs,
        "pre": pre, "post": post,
        "mean_rate": float(np.mean(rates)),
        "worst_rate": float(np.max(rates)),
        "best_rate": float(np.min(rates)),
        "mean_iterations": float(np.mean(iters)),
        "seconds": seconds,
        "seconds_per_rhs": seconds / n_rhs,
        "note": "stationary defect correction, no Krylov acceleration",
    }


def solve_with_preconditioner(
    hierarchy: Hierarchy,
    smoother,
    level: int,
    b: np.ndarray,
    tol: float = 1e-10,
    maxiter: int = 500,
    flexible: bool = True,
) -> Dict[str, object]:
    """Outer solve: FCG when the preconditioner may be nonlinear, PCG otherwise.

    FCG (Notay) accepts a preconditioner that changes from iteration to
    iteration, which a learned V-cycle does. It reduces to ordinary PCG when the
    preconditioner is fixed, so it is a safe default; ``flexible=False`` runs
    ordinary PCG, which is only valid for the classical smoothers.
    """
    A = hierarchy.A[level]
    n = A.shape[0]
    Aop = spla.aslinearoperator(A.tocsr())

    def M(v):
        return vcycle(hierarchy, smoother, level, np.asarray(v).ravel())

    Mop = spla.aslinearoperator(spla.LinearOperator((n, n), matvec=M))

    iters: List[int] = []
    t0 = time.perf_counter()

    def callback(_):
        iters.append(1)

    if flexible:
        # Flexible CG: the search direction is rebuilt from the current
        # preconditioned residual every iteration, so a varying M stays valid.
        x = np.zeros(n)
        r = b.copy()
        z = M(r)
        p = z.copy()
        rz = float(r @ z)
        b_norm = np.linalg.norm(b)
        for _ in range(maxiter):
            if np.linalg.norm(r) / b_norm <= tol:
                break
            Ap = A @ p
            denom = float(p @ Ap)
            if denom == 0.0:
                break
            alpha = rz / denom
            x = x + alpha * p
            r = r - alpha * Ap
            z = M(r)
            rz_new = float(r @ z)
            if rz == 0.0:
                break
            beta = rz_new / rz
            rz = rz_new
            p = z + beta * p
            iters.append(1)
        converged = np.linalg.norm(r) / b_norm <= tol
    else:
        try:
            x, info = spla.cg(Aop, b, rtol=tol, maxiter=maxiter, M=Mop,
                              callback=callback)
            converged = info == 0
        except TypeError:            # older scipy spells it `tol`
            x, info = spla.cg(Aop, b, tol=tol, maxiter=maxiter, M=Mop,
                              callback=callback)
            converged = info == 0
        r = b - A @ x

    seconds = time.perf_counter() - t0
    return {
        "iterations": len(iters),
        "converged": bool(converged),
        "residual": float(np.linalg.norm(r) / np.linalg.norm(b)),
        "seconds": seconds,
        "flexible": bool(flexible),
    }
