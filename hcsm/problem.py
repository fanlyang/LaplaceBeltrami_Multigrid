"""The problem family, the coarse operator, and the A-orthogonal decomposition.

Everything here is per-epsilon. The mesh, the finite element, the mapping, the
DoF ordering and the prolongation are *identical* for all four epsilons -- only
the coefficient changes -- so a difference between two epsilons is a difference
made by the coefficient and by nothing else.

What is deliberately NOT here
-----------------------------
* No matrix inverse is ever formed. The coarse solve is a sparse LU
  factorisation of ``A_H = P^T A P``, applied to a right-hand side. The
  projector ``Pi_C = P (P^T A P)^{-1} P^T A`` is written down in the
  specification and is *never materialised*.
* ``A`` is never densified, and neither is ``A_H``.
* The independently assembled coarse operator (``A-level-2`` of the exported
  hierarchy) is loaded only to *measure* how far it is from ``P^T A P`` on this
  curved surface. It is not used by any experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# The existing framework supplies the verified loader and the sparse
# positive-definiteness test. Reusing them keeps one implementation of each
# check instead of two that can drift apart.
import deeponet_smoother as ds

# ---------------------------------------------------------------------------
# The coefficient family
# ---------------------------------------------------------------------------

#: The SCALAR family's epsilons -- the first experiment's sweep.
EPSILONS: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)

#: The TENSOR family's epsilons: 1 is the isotropic control (D = I), then the
#: anisotropy 1/eps grows to 10000:1. Unlike the scalar family, where eps only
#: shifted a smooth coefficient, here eps controls a genuine directional
#: weighting and eps -> 0 makes the xi coupling vanish.
TENSOR_EPSILONS: Tuple[float, ...] = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)

#: The two coefficient families the solver supports. "tensor" is
#: D_eps = eps e_xi (x) e_xi + e_eta (x) e_eta; "scalar" is the first
#: experiment's kappa_eps = eps + sin^2 cos^2.
COEFFICIENTS: Tuple[str, ...] = ("tensor", "scalar")

#: Torus radii. Must match TorusGeometry in src/main.cpp.
R_MAJOR = 2.0
R_MINOR = 1.0


def eps_tag(eps: float) -> str:
    """A filesystem-safe, sortable name for an epsilon: 1e-1 -> '1e-01'."""
    return "%.0e" % eps


def contrast(eps: float) -> float:
    """kappa_max / kappa_min = (1 + epsilon) / epsilon."""
    return (1.0 + eps) / eps


def kappa_epsilon(angles: np.ndarray, eps: float) -> np.ndarray:
    """kappa_epsilon(xi, eta) = epsilon + sin^2(xi) cos^2(eta).

    Evaluated at the physical support points of the level DoFs, using the same
    angle chart as the C++ solver. ``angles`` is (n, 2). This is the SCALAR
    family's coefficient.
    """
    xi, eta = angles[:, 0], angles[:, 1]
    return eps + np.sin(xi) ** 2 * np.cos(eta) ** 2


def anisotropy(eps: float, coefficient: str = "tensor") -> float:
    """How many times stronger the eta direction is than the xi direction.

    For the tensor family D_eps = eps e_xi(x)e_xi + e_eta(x)e_eta the weights
    are eps and 1, so the ratio is exactly 1/eps. For the scalar family there
    is no direction, and the analogous quantity is the pointwise contrast.
    """
    if coefficient == "tensor":
        return 1.0 / eps
    return (1.0 + eps) / eps


def expected_coefficient(angles: np.ndarray, eps: float,
                         coefficient: str) -> np.ndarray:
    """The field the solver should have written into coeff-<tag>.txt.

    For the tensor family this is a CONSTANT: D_eps is eps in the xi direction
    and 1 in the eta direction at every point, so there is no position-dependent
    scalar to sample and the exporter writes eps itself. Inventing a scalar
    proxy for the tensor here would make the verification check a quantity the
    solver never used.
    """
    if coefficient == "tensor":
        return np.full(angles.shape[0], float(eps))
    return kappa_epsilon(angles, eps)


# ---------------------------------------------------------------------------
# The level problem
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """One epsilon: A_epsilon on the fine level, its prolongation, its DoF chart.

    Attributes
    ----------
    eps          the coefficient offset
    A            A_epsilon, CSR, n x n, symmetric positive definite
    coords       physical support points of the DoFs, (n, 3)
    angles       the same points in the torus chart, (n, 2) = (xi, eta)
    coeff        the exported coefficient field, (n,). For the SCALAR family
                 this is kappa_epsilon at the support points; for the TENSOR
                 family the coefficient is position-independent and this is
                 the constant eps, which is what the solver exports.
    P            prolongation from the next coarser level, n x m
    A_H          P^T A P, the Galerkin coarse operator used by Stage I
    coefficient  which family this problem was assembled with
    """

    eps: float
    A: sp.csr_matrix
    coords: np.ndarray
    angles: np.ndarray
    coeff: np.ndarray
    P: sp.csr_matrix
    A_H: sp.csr_matrix
    level: int = 0
    A_H_assembled: Optional[sp.csr_matrix] = None
    coefficient: str = "tensor"
    diagnostics: Dict[str, float] = field(default_factory=dict)
    _coarse_solve: Callable[[np.ndarray], np.ndarray] = field(
        default=None, repr=False
    )

    # -- shapes ------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.P.shape[1]

    @property
    def n_side(self) -> int:
        return max(2, int(round(np.sqrt(self.n))))

    @property
    def nyquist(self) -> int:
        """Largest wavenumber the DoF lattice can represent: n_side / 2.

        Not n_side. Capping generators at a fraction of n_side instead of n_side/2
        aliases every mode above the true Nyquist.
        """
        return max(1, self.n_side // 2)

    # -- energy ------------------------------------------------------------
    def energy(self, v: np.ndarray) -> np.ndarray:
        """||v||_A^2 = v^T A v, sample-wise for v of shape (n,) or (b, n)."""
        if v.ndim == 1:
            return float(v @ (self.A @ v))
        return np.einsum("bi,bi->b", v, (self.A @ v.T).T)

    def norm_A(self, v: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.energy(v), 0.0))

    def to_unit_energy(self, v: np.ndarray) -> Optional[np.ndarray]:
        """Scale v to ||v||_A = 1, or return None if that is not well defined.

        Every method in this experiment is compared through the *ratio*
        ||S(v)||_A / ||v||_A. For the classical smoothers that ratio is exactly
        scale-invariant; for the DeepONet it is not, because the branch is a tanh
        network. Canonicalising every error to unit A-energy therefore does two
        things at once: it makes the reported ratios comparable across methods,
        and it puts the network's input on the same scale it was trained on.
        """
        e = self.energy(v)
        if not np.isfinite(e) or e <= 0.0:
            return None
        return v / np.sqrt(e)

    # -- the A-orthogonal decomposition -----------------------------------
    def decompose(self, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split e = e_C + e_F with e_C in range(P) and e_F A-orthogonal to it.

            b_H = P^T A e ,   A_H z = b_H ,   e_C = P z ,   e_F = e - e_C

        The coarse solve is one sparse triangular pair; no inverse is formed.
        ``A_H = P^T A P`` makes the two defining identities exact in exact
        arithmetic:

            P^T A e_F = P^T A e - (P^T A P) z = 0
            e_C^T A e_F = z^T P^T A e - z^T (P^T A P) z = 0
        """
        b_H = self.P.T @ (self.A @ e)
        z = self._coarse_solve(b_H)
        e_C = self.P @ z
        return e_C, e - e_C

    def apply_Q(self, v: np.ndarray) -> np.ndarray:
        """Q_epsilon v = v - P (P^T A_epsilon P)^{-1} P^T A_epsilon v.

        The complement projector of the specification, applied by a coarse
        SOLVE. No dense inverse and no projector matrix is ever formed: the
        factorisation of A_H is reused for every right-hand side. This is
        exactly the second component of ``decompose``, named for the symbol the
        experiment uses.

        Note the Galerkin diagnostic Q_epsilon and the actual V-cycle are
        different objects and are kept distinct: Q_epsilon uses the Galerkin
        coarse operator P^T A P, while the V-cycle the solver runs uses the
        independently REDISCRETIZED level operators, whose difference on this
        curved surface is measured and recorded in ``diagnostics``.
        """
        return self.decompose(v)[1]

    def apply_Q_batch(self, V: np.ndarray) -> np.ndarray:
        """apply_Q for a batch of vectors, shape (b, n)."""
        out = np.empty_like(V, dtype=np.float64)
        for i in range(V.shape[0]):
            out[i] = self.apply_Q(V[i])
        return out

    def decomposition_errors(self, e: np.ndarray) -> Dict[str, float]:
        """Normalised residuals of the two identities the decomposition claims.

        Both are reported per sample and aggregated by the caller. They are
        *relative*: a raw ``||P^T A e_F||`` of 1e-12 means nothing without the
        scale of ``||P^T A e||`` next to it.
        """
        e_C, e_F = self.decompose(e)

        ptA_eF = self.P.T @ (self.A @ e_F)
        ptA_e = self.P.T @ (self.A @ e)
        n_ptA_e = float(np.linalg.norm(ptA_e))
        orth_residual = float(np.linalg.norm(ptA_eF)) / (n_ptA_e if n_ptA_e > 0 else 1.0)

        # e_C^T A e_F is a single number; normalise it by the two energies so it
        # reads as a cosine of the angle between the components.
        cross = float(e_C @ (self.A @ e_F))
        nC, nF = self.norm_A(e_C), self.norm_A(e_F)
        denom = float(nC) * float(nF)
        orth_cross = abs(cross) / denom if denom > 0 else 0.0

        return {
            "coarse_residual": orth_residual,
            "cross_energy": orth_cross,
            "ef_over_e": float(nF) / float(self.norm_A(e)) if self.norm_A(e) > 0 else 0.0,
            "abs_cross": cross,
        }

    def coarse_correction(self, v: np.ndarray) -> np.ndarray:
        """Apply the exact Galerkin coarse-grid correction C_G = I - Pi_C to v.

        This is the second half of a two-grid step: restrict, solve on the
        coarse space, prolong and subtract. It is the operator C_G of the
        specification, evaluated by solving rather than by forming an inverse.
        """
        r_H = self.P.T @ (self.A @ v)
        z = self._coarse_solve(r_H)
        return v - self.P @ z


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def level_dir(root: str, eps: float, level: int = 3,
              coefficient: str = "tensor") -> str:
    """Where the converted data for one (family, epsilon, level) lives."""
    return os.path.join(root, "%s-eps-%s-L%d"
                        % (coefficient, eps_tag(eps), level))


def load_problem(
    root: str,
    eps: float,
    level: int = 3,
    verbose: bool = False,
    coefficient: str = "tensor",
) -> Problem:
    """Load one epsilon's level problem and verify it before trusting it.

    ``ds.load_level`` already checks shape, symmetry, finite entries and
    positive definiteness (sparsely). On top of that this function verifies that
    the exported coefficient is what the requested family should produce -- a
    mismatch would silently make the two-grid and energy numbers refer to a
    different problem than the one being reported.
    """
    if coefficient not in COEFFICIENTS:
        raise SystemExit("unknown coefficient family %r; expected one of %s"
                         % (coefficient, ", ".join(COEFFICIENTS)))
    d = level_dir(root, eps, level, coefficient)
    ld = ds.load_level(d, verbose=verbose)

    A = ld.A.tocsr()
    A.sum_duplicates()
    A.sort_indices()

    if ld.P is None:
        raise SystemExit(
            "%s has no prolongation; the Galerkin coarse operator P^T A P "
            "cannot be built and Stage I cannot run" % d
        )
    P = ld.P.tocsr()
    P.sum_duplicates()
    P.sort_indices()

    # -- the coefficient is what this family should produce ----------------
    if ld.coeff is not None:
        expected = expected_coefficient(ld.angles, eps, coefficient)
        err = float(np.max(np.abs(ld.coeff - expected)))
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if err > 1e-10 * scale:
            raise SystemExit(
                "the exported coefficient is not %s at epsilon=%.0e: max "
                "|coeff - expected| = %.3e (scale %.3e)"
                % (coefficient, eps, err, scale))
        coeff = ld.coeff
    else:
        coeff = expected_coefficient(ld.angles, eps, coefficient)

    # For the tensor family the coefficient is position-independent by
    # construction. Check that the file really is constant rather than merely
    # close to the constant eps: a spatially varying field here would mean the
    # solver assembled something other than D_eps.
    if coefficient == "tensor":
        spread = float(np.max(coeff) - np.min(coeff))
        if spread > 1e-12 * max(float(np.max(np.abs(coeff))), 1.0):
            raise SystemExit(
                "the tensor coefficient should be constant, but the exported "
                "field varies by %.3e -- the assembly is not the uniform "
                "directional tensor D_eps" % spread)

    # -- Galerkin coarse operator -----------------------------------------
    A_H = (P.T @ (A @ P)).tocsr()
    A_H.sum_duplicates()

    is_spd, lam_min, lam_max = ds.sparse_spd_report(A_H.tocsc())
    if not is_spd:
        raise SystemExit(
            "P^T A P is not positive definite for epsilon=%.0e; the coarse "
            "solve would not be well posed" % eps
        )

    # A single factorisation, reused for every right-hand side. That is a
    # factorisation, not an inverse: no dense matrix is ever formed from it.
    lu = spla.factorized(A_H.tocsc())

    diag: Dict[str, float] = {
        "eps": float(eps),
        "coefficient": 0.0 if coefficient == "tensor" else 1.0,
        "coefficient_name": coefficient,
        # "How extreme is the coefficient": for the tensor family the ratio of
        # the eta weight to the xi weight (1/eps), for the scalar family the
        # pointwise contrast (1+eps)/eps. One number, one meaning per family --
        # a field called "contrast" carried over from the scalar family would be
        # meaningless for a tensor.
        "anisotropy": float(anisotropy(eps, coefficient)),
        "n_dof": float(A.shape[0]),
        "n_coarse": float(A_H.shape[0]),
        "n_side": float(max(2, int(round(np.sqrt(A.shape[0]))))),
        "nyquist": float(max(1, int(round(np.sqrt(A.shape[0]))) // 2)),
        "A_nnz": float(A.nnz),
        "A_symmetry_linf": float(abs(A - A.T).max()) if A.nnz else 0.0,
        "A_H_lambda_min": float(lam_min) if lam_min is not None else float("nan"),
        "A_H_lambda_max": float(lam_max) if lam_max is not None else float("nan"),
        "A_H_condition": float(lam_max / lam_min)
        if (lam_min is not None and lam_max is not None and lam_min > 0)
        else float("nan"),
        "coeff_min": float(coeff.min()),
        "coeff_max": float(coeff.max()),
        "coeff_contrast_measured": float(coeff.max() / coeff.min()),
        "mass_matrix_available": 0.0,
    }

    # -- a measurement only: how far the assembled coarse operator is ------
    # This is *not* used by Stage I. It is recorded because on a curved surface
    # P^T A P and the assembled A_{l-1} are not the same operator, and the size
    # of the gap explains why the specification pins Stage I to the Galerkin one.
    A_H_assembled = None
    coarse_path = os.path.join(d, "coarse_matrix.npz")
    if os.path.exists(coarse_path):
        A_c = sp.load_npz(coarse_path).tocsr()
        if A_c.shape == A_H.shape:
            A_H_assembled = A_c
            num = float(abs(A_H - A_c).max())
            den = float(abs(A_H).max())
            diag["galerkin_vs_assembled_rel"] = num / den if den > 0 else float("nan")

    return Problem(
        eps=eps,
        A=A,
        coords=np.asarray(ld.coords, dtype=np.float64),
        angles=np.asarray(ld.angles, dtype=np.float64),
        coeff=np.asarray(coeff, dtype=np.float64),
        P=P,
        A_H=A_H,
        level=level,
        coefficient=coefficient,
        A_H_assembled=A_H_assembled,
        diagnostics=diag,
        _coarse_solve=lu,
    )


def lattice_grid(problem: Problem, values: np.ndarray,
                 tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a per-DoF field onto the (xi, eta) DoF lattice.

    The DoFs of a uniformly refined torus mesh form a tensor-product lattice in
    the parameter domain, so a field living on them can be drawn as an image
    rather than as a cloud of points. That is an assumption, so it is *checked*:
    each angle is snapped to its nearest unique value and the residual must be
    below ``tol``. Returns ``(XI, ETA, V)`` with ``V`` of shape (n_side, n_side)
    in the order (eta index, xi index).

    A failed check raises rather than silently producing a scrambled picture.
    """
    xi, eta = problem.angles[:, 0], problem.angles[:, 1]

    # Round first and take the inverse index from the *rounded* values. Snapping
    # with searchsorted on the unrounded value is subtly wrong: np.round may
    # round either way, so a value that rounded down lands one slot past its own
    # lattice line and picks up a whole grid spacing of error. Pairing the unique
    # rounded values with return_inverse is exact by construction.
    xir, etar = np.round(xi, 12), np.round(eta, 12)
    if np.max(np.abs(xir - xi)) > tol or np.max(np.abs(etar - eta)) > tol:
        raise ValueError("the DoF angles are not on a regular lattice to %g" % tol)

    ux, ix = np.unique(xir, return_inverse=True)
    ue, ie = np.unique(etar, return_inverse=True)
    if ux.size * ue.size != problem.n:
        raise ValueError(
            "the DoF angles are not a %d x %d tensor lattice (%d x %d unique "
            "values for %d DoFs)" % (ux.size, ue.size, ux.size, ue.size, problem.n)
        )
    if np.bincount(ix).min() != ue.size or np.bincount(ie).min() != ux.size:
        raise ValueError("the DoF angles are not a full tensor product lattice")

    V = np.empty((ue.size, ux.size), dtype=np.float64)
    V[ie, ix] = np.asarray(values, dtype=np.float64)
    XI, ETA = np.meshgrid(ux, ue)
    return XI, ETA, V


def torus_surface(problem: Problem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The (X, Y, Z) coordinates of the DoF lattice on the torus, shaped like the grid.

    Built from the *physical* support points (not re-evaluated from the chart), so
    a picture drawn from it is the mesh the solver actually used.
    """
    XI, ETA, _ = lattice_grid(problem, np.zeros(problem.n))
    X = (R_MAJOR + R_MINOR * np.cos(ETA)) * np.cos(XI)
    Y = R_MINOR * np.sin(ETA)
    Z = (R_MAJOR + R_MINOR * np.cos(ETA)) * np.sin(XI)
    return X, Y, Z


def area_weights(problem: Problem) -> np.ndarray:
    """The surface measure of each DoF, up to the constant parameter cell area.

    The DoFs sit on a uniform lattice in (xi, eta), but the torus is not
    uniformly stretched: the area element is ``h_xi deta dxi`` with
    ``h_xi = R + r cos(eta)``, which ranges from R-r = 1 at the inner equator to
    R+r = 3 at the outer one. Counting DoFs therefore under-weights the outer
    equator by up to a factor of three, and any statement of the form "this much
    of the error energy sits in that region" would inherit that bias. The
    constant ``dxi deta`` is common to every DoF and cancels in every ratio this
    is used for, so it is dropped.
    """
    return R_MAJOR + R_MINOR * np.cos(problem.angles[:, 1])


def constant_error(problem: Problem, value: float = 1.0) -> Optional[np.ndarray]:
    """A1 -- the constant field, canonicalised to unit A-energy.

    On a closed surface the constant is the one field a point smoother and a
    coarse space treat completely differently, so it is a useful fixed probe.
    """
    v = np.full(problem.n, value, dtype=np.float64)
    return problem.to_unit_energy(v)
