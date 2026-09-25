"""Every measurement the experiment reports.

The uniform interface is a *smoother correction*: for an error e, every method
produces a delta such that its smoothing step is ``S(e) = e - delta``. Each
method is then scored by the one quantity that decides whether it is a smoother,

    rho(e) = ||S(e)||_A / ||e||_A ,

evaluated on exactly the same errors for every method.

Two conventions are fixed here once, rather than being left to each call site:

* **One smoothing step** in the main comparison. The classical sweeps are one
  Jacobi / Gauss-Seidel / SSOR application; the DeepONet is one application of
  B_theta. A repeated-step variant is available and is reported separately.
* **Unit A-energy errors.** Every error handed to a method is canonicalised to
  ||e||_A = 1. The classical ratios are invariant under this anyway; the
  DeepONet's is not, because its branch is a tanh network, so canonicalising is
  what makes the comparison like-for-like.

Nothing here inspects a *mean* on its own. Each table carries the median, the
standard deviation, the 95th percentile, the maximum and the fraction of samples
the step made worse, because a smoother that helps most errors while amplifying a
few is a different object from one that helps all of them, and the mean cannot
tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

import deeponet_smoother as ds

from .model import build_features, to_torch_sparse
from .problem import Problem, area_weights
from .samplers import Split, controlled_modes

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def rho_stats(rho: np.ndarray) -> Dict[str, float]:
    """The six numbers every reduction is reported with.

    ``frac_gt_1`` is the fraction of samples the step made *worse*. It is the
    number a mean hides, and for a smoother it is the one that decides whether a
    method is usable inside a V-cycle.
    """
    r = np.asarray(rho, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "median", "std", "p95", "max", "frac_gt_1",
                 "frac_lt_1", "mean_energy")}
    return {
        "n": int(r.size),
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "std": float(r.std(ddof=1)) if r.size > 1 else 0.0,
        "p95": float(np.percentile(r, 95)),
        "max": float(r.max()),
        "frac_gt_1": float(np.mean(r > 1.0)),
        "frac_lt_1": float(np.mean(r < 1.0)),
        # The mean of rho^2 is what actually multiplies a V-cycle: an energy
        # reduction is a squared norm, so this is the number that composes.
        "mean_energy": float(np.mean(r ** 2)),
    }


# ---------------------------------------------------------------------------
# The smoothers
# ---------------------------------------------------------------------------


class Smoothers:
    """One application of each classical smoother, as a correction.

    Definitions are the existing framework's (``deeponet_smoother.ClassicalSmoothers``);
    this subclass only precomputes the triangular factors, because the parent
    rebuilds ``(D + L)`` on every call, which at 250 samples x 4 epsilons x 3
    methods is measurable. ``tools/selftest.py`` pins the two implementations to
    each other.
    """

    def __init__(self, A: sp.csr_matrix):
        self._base = ds.ClassicalSmoothers(A)
        self.A = self._base.A
        self.n = self._base.n
        self.diag = self._base.diag
        self.L = self._base.L
        self.U = self._base.U
        self.D = self._base.D
        self._DpL = (self.D + self.L).tocsc()
        self._DpU = (self.D + self.U).tocsc()
        self._ssor_cache: Dict[float, Tuple[sp.csc_matrix, sp.csc_matrix]] = {}

    # -- corrections -------------------------------------------------------
    def jacobi(self, r: np.ndarray, omega: float) -> np.ndarray:
        return omega * r / self.diag

    def jacobi_batch(self, R: np.ndarray, omega: float) -> np.ndarray:
        """Row-wise damped Jacobi. A diagonal scaling is the one smoother that
        batches exactly, and that is part of the cost comparison, not hidden."""
        return omega * R / self.diag

    def gs(self, r: np.ndarray) -> np.ndarray:
        return spla.spsolve_triangular(self._DpL, r, lower=True)

    def sgs(self, r: np.ndarray) -> np.ndarray:
        y = spla.spsolve_triangular(self._DpL, r, lower=True)
        return spla.spsolve_triangular(self._DpU, self.diag * y, lower=False)

    def ssor(self, r: np.ndarray, omega: float = 1.0) -> np.ndarray:
        """At omega = 1 this is exactly symmetric Gauss-Seidel."""
        if omega == 1.0:
            return self.sgs(r)
        if omega not in self._ssor_cache:
            self._ssor_cache[omega] = (
                (self.D + omega * self.L).tocsc(),
                (self.D + omega * self.U).tocsc(),
            )
        lo, up = self._ssor_cache[omega]
        y = spla.spsolve_triangular(lo, r, lower=True)
        z = spla.spsolve_triangular(up, self.diag * y, lower=False)
        return omega * (2.0 - omega) * z

    # -- batched helpers ---------------------------------------------------
    def rho(self, E: np.ndarray, correction: Callable[[np.ndarray], np.ndarray],
            problem: Problem) -> np.ndarray:
        """||E - delta||_A / ||E||_A for each row of E."""
        out = np.empty(E.shape[0], dtype=np.float64)
        for i, e in enumerate(E):
            d = correction(problem.A @ e)
            out[i] = np.sqrt(max(problem.energy(e - d), 0.0)) / np.sqrt(
                max(problem.energy(e), 1e-300))
        return out

    def rho_direct(self, corrected: np.ndarray, E: np.ndarray,
                   problem: Problem) -> np.ndarray:
        num = np.sqrt(np.maximum(problem.energy(corrected), 0.0))
        den = np.sqrt(np.maximum(problem.energy(E), 1e-300))
        return num / den


class DeepONetSmoother:
    """A trained checkpoint, wrapped as a correction map.

    Two entry points, named for what they take, because the smoother consumes a
    *residual* while most callers hold an *error*:

        from_residual(r)  ->  B_theta(r, X)
        from_error(e)     ->  B_theta(A_epsilon e, X)

    An earlier version had a single ``correction`` that took an error, and the
    two-grid driver -- whose uniform interface is residual-based, because that is
    what a classical smoother needs -- called it with a residual. It silently
    computed ``A(A e)`` for the DeepONet and ``A e`` for everything else, which
    would have made the two-grid table compare two different smoother steps. The
    two names exist so that mistake cannot be made again by accident.
    """

    def __init__(self, model: torch.nn.Module, problem: Problem,
                 features: np.ndarray, dtype: torch.dtype = torch.float64):
        self.model = model
        self.problem = problem
        self.A = to_torch_sparse(problem.A, dtype)
        self.X = torch.from_numpy(np.asarray(features, dtype=np.float64)).to(dtype)
        self.dtype = dtype

    @torch.no_grad()
    def from_residual(self, R: np.ndarray) -> np.ndarray:
        """B_theta(r, X) for residuals r of shape (n,) or (b, n)."""
        self.model.eval()
        one = np.asarray(R).ndim == 1
        Rt = torch.from_numpy(np.atleast_2d(np.asarray(R, dtype=np.float64))).to(self.dtype)
        out = self.model(Rt, self.X, None).cpu().numpy()
        return out[0] if one else out

    def from_error(self, E: np.ndarray) -> np.ndarray:
        """B_theta(A_epsilon e, X) for errors e of shape (n,) or (b, n)."""
        one = np.asarray(E).ndim == 1
        E2 = np.atleast_2d(np.asarray(E, dtype=np.float64))
        R = (self.problem.A @ E2.T).T
        out = self.from_residual(R)
        return out[0] if one else out

    def rho(self, E: np.ndarray) -> np.ndarray:
        corr = self.from_error(E)
        return np.sqrt(np.maximum(self.problem.energy(E - corr), 0.0)) / \
            np.sqrt(np.maximum(self.problem.energy(E), 1e-300))


# ---------------------------------------------------------------------------
# Section I / J -- rho on the coarse-space-complement and coarse components
# ---------------------------------------------------------------------------


@dataclass
class MethodSet:
    """Everything that can be scored on a common set of errors."""

    smoothers: Smoothers
    deeponets: Dict[str, DeepONetSmoother]
    jacobi_omega: float
    ssor_omega: float

    def names(self) -> List[str]:
        return (["jacobi", "gs", "sgs_ssor"]
                + ["deeponet_" + k for k in self.deeponets])


def tune_classical(
    problem: Problem,
    smoothers: Smoothers,
    E_val: np.ndarray,
    jacobi_grid: Sequence[float] = (0.4, 0.5, 0.6, 2.0 / 3.0, 0.7, 0.8, 0.9, 1.0),
    ssor_grid: Sequence[float] = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9),
) -> Dict[str, object]:
    """Choose the damping parameters on the VALIDATION errors, never on test.

    A damped Jacobi needs an omega and so does an SSOR. Sweeping them on the test
    set would hand the classical methods a tuned advantage the network never got,
    so the sweep happens on held-out validation errors and the chosen value is
    then frozen for everything the experiment reports. The oracle values -- the
    best omega on the test set itself -- are computed too, but only to be printed
    as a bound, never to be reported as a method's result.
    """
    diag = smoothers.diag
    R = problem.A @ E_val.T
    R = R.T

    def sweep(grid, apply_corr):
        table = {}
        for w in grid:
            corr = np.asarray([apply_corr(r, w) for r in R])
            rho = smoothers.rho_direct(E_val - corr, E_val, problem)
            table[float(w)] = float(np.mean(rho))
        best = min(table.items(), key=lambda kv: kv[1])
        return best[0], table

    j_best, j_table = sweep(jacobi_grid,
                            lambda r, w: w * r / diag)
    s_best, s_table = sweep(ssor_grid,
                            lambda r, w: smoothers.ssor(r, w))
    return {
        "jacobi_omega": j_best,
        "jacobi_table": j_table,
        "ssor_omega": s_best,
        "ssor_table": s_table,
    }


def oracle_classical(
    problem: Problem,
    smoothers: Smoothers,
    E_test: np.ndarray,
    jacobi_grid: Sequence[float] = (0.4, 0.5, 0.6, 2.0 / 3.0, 0.7, 0.8, 0.9, 1.0),
    ssor_grid: Sequence[float] = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9),
) -> Dict[str, object]:
    """The best achievable classical reduction on the TEST set. A bound only."""
    return tune_classical(problem, smoothers, E_test, jacobi_grid, ssor_grid)


def evaluate_methods(
    problem: Problem,
    methods: MethodSet,
    E: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """rho statistics for every method on one set of errors.

    ``methods.names()`` fixes the method list, so the same set is scored by
    exactly the same methods for e_F, for e_C and for the controlled modes.
    """
    out: Dict[str, Dict[str, float]] = {}

    out["jacobi"] = rho_stats(methods.smoothers.rho(
        E, lambda r: methods.smoothers.jacobi(r, methods.jacobi_omega), problem))
    out["gs"] = rho_stats(methods.smoothers.rho(
        E, methods.smoothers.gs, problem))
    out["sgs_ssor"] = rho_stats(methods.smoothers.rho(
        E, lambda r: methods.smoothers.ssor(r, methods.ssor_omega), problem))
    for name, dn in methods.deeponets.items():
        out["deeponet_" + name] = rho_stats(dn.rho(E))
    return out


def per_sample_rho(
    problem: Problem,
    methods: MethodSet,
    E: np.ndarray,
) -> Dict[str, np.ndarray]:
    """The raw per-sample ratios, so a hard-error analysis can select on them."""
    out: Dict[str, np.ndarray] = {}
    out["jacobi"] = methods.smoothers.rho(
        E, lambda r: methods.smoothers.jacobi(r, methods.jacobi_omega), problem)
    out["gs"] = methods.smoothers.rho(E, methods.smoothers.gs, problem)
    out["sgs_ssor"] = methods.smoothers.rho(
        E, lambda r: methods.smoothers.ssor(r, methods.ssor_omega), problem)
    for name, dn in methods.deeponets.items():
        out["deeponet_" + name] = dn.rho(E)
    return out


def repeated_steps(
    problem: Problem,
    methods: MethodSet,
    E: np.ndarray,
    steps: int = 3,
) -> Dict[str, Dict[str, float]]:
    """The reduction after ``steps`` applications of each smoother.

    Reported because one step and several steps are different claims. A method
    that is only marginally better after one step can still be clearly better
    after three, and the other way round; the specification fixes one step for the
    main comparison, so the multi-step number is carried alongside rather than
    instead.
    """
    out: Dict[str, Dict[str, float]] = {}

    def iterate(corr_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        cur = E.copy()
        for _ in range(steps):
            cur = cur - corr_fn(cur)
        return np.sqrt(np.maximum(problem.energy(cur), 0.0)) / np.sqrt(
            np.maximum(problem.energy(E), 1e-300))

    out["jacobi"] = rho_stats(iterate(
        lambda V: np.asarray([methods.smoothers.jacobi(problem.A @ v,
                                                       methods.jacobi_omega)
                              for v in V])))
    out["gs"] = rho_stats(iterate(
        lambda V: np.asarray([methods.smoothers.gs(problem.A @ v) for v in V])))
    out["sgs_ssor"] = rho_stats(iterate(
        lambda V: np.asarray([methods.smoothers.ssor(problem.A @ v,
                                                     methods.ssor_omega)
                              for v in V])))
    for name, dn in methods.deeponets.items():
        out["deeponet_" + name] = rho_stats(iterate(
            lambda V, dn=dn: dn.from_error(V)))
    return out


# ---------------------------------------------------------------------------
# Section K -- controlled modes
# ---------------------------------------------------------------------------


def controlled_mode_scan(
    problem: Problem,
    methods: MethodSet,
    families: Sequence[str] = ("xi", "eta"),
    wavenumbers: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, object]]:
    """k vs rho_k for sin(k xi) and sin(k eta), for every method.

    Each mode is first projected onto the coarse-space complement with the same
    Galerkin solve as every other sample, so what is measured is the reduction of
    the part of the mode the coarse grid does *not* already contain.
    """
    out: Dict[str, Dict[str, object]] = {}
    for fam in families:
        modes = controlled_modes(problem, wavenumbers=wavenumbers, family=fam)
        if modes["k"].size == 0:
            out[fam] = {"k": np.zeros(0), "skipped_k": modes["skipped_k"]}
            continue
        E_F = modes["E_F"]
        per = per_sample_rho(problem, methods, E_F)
        rec: Dict[str, object] = {
            "k": modes["k"],
            "skipped_k": modes["skipped_k"],
            "ef_ratio": modes["ef_ratio"],
        }
        for name, rho in per.items():
            rec[name] = rho
        out[fam] = rec
    return out


# ---------------------------------------------------------------------------
# Section L -- the empirically hard errors
# ---------------------------------------------------------------------------


def hard_errors(
    problem: Problem,
    methods: MethodSet,
    split: Split,
    n_worst: int = 20,
    reference: str = "jacobi",
) -> Dict[str, object]:
    """The test errors this reference smoother handles worst, and what the others do.

    The reference is damped Jacobi because that is the method the chain under
    test starts from. Its worst samples are selected *empirically* from the test
    set -- no assumption is made in advance about which errors are hard for it,
    and the last thing this function does is look for a spatial pattern, not
    assume one.

    For every selected error it keeps the raw objects the specification asks for:
    e_F, r_F = A e_F, and the smoothed errors S_J(e_F), S_GS(e_F), S_theta(e_F),
    together with the three ratios. Everything is returned as arrays so the
    caller can write it to an NPZ and plot it.
    """
    E_F = split.E_F
    rho = per_sample_rho(problem, methods, E_F)
    rho_ref = rho[reference]

    order = np.argsort(-rho_ref)[:n_worst]
    idx = np.sort(order)

    selected: Dict[str, object] = {
        "index": idx,
        "rho_jacobi": rho_ref[idx],
        "mechanism": [split.mechanism[i] for i in idx],
        "E_F": E_F[idx],
        "R_F": split.R_F[idx],
    }

    A = problem.A
    sJ = np.asarray([E_F[i] - methods.smoothers.jacobi(A @ E_F[i],
                                                       methods.jacobi_omega)
                     for i in idx])
    sG = np.asarray([E_F[i] - methods.smoothers.gs(A @ E_F[i]) for i in idx])
    selected["S_jacobi"] = sJ
    selected["S_gs"] = sG

    for name, dn in methods.deeponets.items():
        selected["rho_deeponet_" + name] = rho["deeponet_" + name][idx]
        selected["S_deeponet_" + name] = E_F[idx] - dn.from_error(E_F[idx])

    selected["rho_gs"] = rho["gs"][idx]
    selected["rho_sgs_ssor"] = rho["sgs_ssor"][idx]

    # Does the hard set look different from the test set as a whole? Reported as
    # a comparison, not as a claim.
    selected["hard_mean_rho_jacobi"] = float(np.mean(rho_ref[idx]))
    selected["all_mean_rho_jacobi"] = float(np.mean(rho_ref))
    selected["hard_frac_gt_1"] = float(np.mean(rho_ref[idx] > 1.0))
    selected["all_frac_gt_1"] = float(np.mean(rho_ref > 1.0))

    # Spatial diagnostic: where does the error energy actually sit, relative to
    # the low-kappa regions? We do NOT assume the hard errors live there -- the
    # correlation is measured and reported whichever way it comes out, including
    # when it is negative.
    #
    # The energy is weighted by the surface area element, not counted per DoF:
    # on this torus the area element h_xi = R + r cos(eta) varies by a factor of
    # three between the inner and outer equator, so a DoF count would bias the
    # answer systematically. See hcsm.problem.area_weights.
    w = area_weights(problem)
    kappa = problem.coeff
    low = kappa <= float(np.percentile(kappa, 25))

    e2_all = np.mean(E_F ** 2, axis=0) * w
    e2_hard = np.mean(selected["E_F"] ** 2, axis=0) * w
    frac_all = float(e2_all[low].sum() / e2_all.sum()) if e2_all.sum() > 0 else float("nan")
    frac_hard = float(e2_hard[low].sum() / e2_hard.sum()) if e2_hard.sum() > 0 else float("nan")

    selected["low_kappa_area_fraction"] = float((w * low).sum() / w.sum())
    selected["error_energy_in_low_kappa_all"] = frac_all
    selected["error_energy_in_low_kappa_hard"] = frac_hard
    # 1.0 means the hard errors are spread exactly like the rest; > 1 means they
    # are concentrated in the low-kappa region; < 1 means the opposite.
    selected["low_kappa_enrichment"] = (frac_hard / frac_all) if frac_all > 0 else float("nan")
    selected["energy_weighting"] = "surface area element h_xi(eta), not a DoF count"

    return selected


def cross_evaluate_hard(
    problem: Problem,
    methods: MethodSet,
    split: Split,
    hard_index: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Score every method on exactly the selected hard errors.

    The point of the selection is that it is *shared*: a comparison on the errors
    Jacobi handles worst is only meaningful if every method sees the identical
    list.
    """
    return evaluate_methods(problem, methods, split.E_F[hard_index])


# ---------------------------------------------------------------------------
# Section M -- the ideal Galerkin two-grid method
# ---------------------------------------------------------------------------


def _corrections(problem: Problem, methods: MethodSet,
                 E: np.ndarray) -> Dict[str, np.ndarray]:
    """One smoothing correction per method, all from the same residuals.

    Every method is handed the residual ``r = A_epsilon e``. That is the point of
    collecting them here: the classical methods and the network then differ only
    in what they do with r, never in what they were given.
    """
    R = (problem.A @ E.T).T
    sm = methods.smoothers
    out = {
        "jacobi": sm.jacobi_batch(R, methods.jacobi_omega),
        "gs": np.asarray([sm.gs(r) for r in R]),
        "sgs_ssor": np.asarray([sm.ssor(r, methods.ssor_omega) for r in R]),
    }
    for name, dn in methods.deeponets.items():
        out["deeponet_" + name] = dn.from_residual(R)
    return out


def two_grid(
    problem: Problem,
    methods: MethodSet,
    E: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """rho_TG = ||C_G S(e)||_A / ||e||_A, one smoothing step plus an exact coarse solve.

    C_G is applied by solving, never by forming ``I - P (P^T A P)^{-1} P^T A``.
    The same test errors are used for every method, so a difference between two
    rows is a difference in the smoother and in nothing else.
    """
    out: Dict[str, Dict[str, float]] = {}
    den = np.sqrt(np.maximum(problem.energy(E), 1e-300))

    for label, corr in _corrections(problem, methods, E).items():
        rho = np.empty(E.shape[0], dtype=np.float64)
        for i in range(E.shape[0]):
            e_tg = problem.coarse_correction(E[i] - corr[i])
            rho[i] = np.sqrt(max(problem.energy(e_tg), 0.0)) / den[i]
        out[label] = rho_stats(rho)

    # The coarse correction alone, as a reference floor: if a smoother adds
    # nothing to this row, its two-grid number is the coarse grid's, not its own.
    rho = np.empty(E.shape[0], dtype=np.float64)
    for i in range(E.shape[0]):
        e_tg = problem.coarse_correction(E[i])
        rho[i] = np.sqrt(max(problem.energy(e_tg), 0.0)) / den[i]
    out["coarse_only"] = rho_stats(rho)
    return out


def two_grid_on_ef(
    problem: Problem,
    methods: MethodSet,
    split: Split,
) -> Dict[str, Dict[str, float]]:
    """The same two-grid step, but run on the e_F samples the model was trained on.

    This isolates the smoother: on an error that has no coarse-space component,
    the coarse correction can only act on whatever the smoother leaves in
    range(P), so the two-grid reduction is governed almost entirely by the
    smoothing step. It is reported next to the general case, not instead of it.
    """
    return two_grid(problem, methods, split.E_F)
