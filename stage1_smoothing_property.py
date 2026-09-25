#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Stage I -- does the DeepONet learn the *smoothing property*?

The question
------------
A smoother exists to reduce the part of the algebraic error that a coarse-grid
correction cannot reach.  On one fixed multigrid level, with the Galerkin coarse
operator

    A_c = P^T A P ,   Pi_C = P (P^T A P)^{-1} P^T A ,   Pi_F = I - Pi_C ,

every error splits A-orthogonally as ``e = e_C + e_F``, with ``e_C = Pi_C e`` in
``range(P)`` and ``e_F = Pi_F e`` in its A-orthogonal complement.  Both are exact
projections, so

    e_C^T A e_F = 0     and     P^T A e_F = 0

hold to machine precision.  This program *asserts* both rather than assuming them.

Stage I trains a DeepONet ``B_theta`` on **the complement component only**,

    L_smooth(theta) = (1/N) sum_i || e_F,i - B_theta(A e_F,i, X_3) ||_A^2
                            / || e_F,i ||_A^2 ,

and then asks whether that training produced a *selective* operator: one that
reduces ``e_F`` but leaves ``e_C`` alone.  Nothing in the loss mentions ``e_C``.
The reported pair is

    mu_F = E[ ||S_theta(e_F)||_A / ||e_F||_A ] ,
    mu_C = E[ ||S_theta(e_C)||_A / ||e_C||_A ] ,

each with the distribution behind it (median, sd, 95th percentile, maximum, and
the share of amplified samples ``P(rho > 1)``).  Selectivity is ``mu_F << mu_C``.

Why the experiment is well posed here
-------------------------------------
``P^T`` is surjective, so ``dim ker(P^T) = n - n_c``; ``range(A P)`` has dimension
``n_c``; and ``P^T A P z = 0`` forces ``z = 0`` because the Galerkin operator is
SPD.  Hence

    R^n = range(A P)  (+)  ker(P^T)     (direct, dimensions n_c + (n - n_c) = n).

Now ``r_F = A e_F`` satisfies ``P^T r_F = P^T A e_F = 0``, so **every training
input lies in ker(P^T)**, while every coarse residual ``r_C = A e_C`` lies in
``range(A P)``.  The two input sets occupy complementary subspaces meeting only
at the origin.  Stage I is therefore a genuine extrapolation question -- the
network must extend a function it learned on ``ker(P^T)`` to a subspace it has
never seen a sample from -- and not a memorisation question.

Both rank ceilings are measured, not assumed
--------------------------------------------
The DeepONet output is ``delta_e = sum_k b_k(r) T_k(x)``.  That gives the model
**two** rank ceilings, and either can silently become the thing under measurement:

* **the trunk.**  For a fixed trunk the correction lies in ``span(T)`` whatever
  the branch returns, so a small ``mu_F`` can be a statement about the trunk
  rather than about smoothing.  ``--trunk-freeze fourier-orth`` (the default)
  freezes ``T`` to an orthonormal basis of the coordinate features, which spans
  ``R^n`` here, making the floor on ``mu_F`` exactly zero and the trunk provably
  irrelevant.  ``tools/trunk_reach.py`` computes that floor for any trunk.
* **the branch.**  The branch narrows to ``width`` units before widening again,
  so its Jacobian factors through ``R^width``; the complement is ``n - n_c``
  dimensional.  A branch narrower than that cannot represent the exact
  correction, whatever the optimiser does.  ``tools/branch_rank_probe.py``
  computes the best linear map of rank ``<= k`` on the same data -- the reference
  curve a width-``k`` branch is measured against.

This is not a hypothetical hazard.  At ``--width 256`` with ``n - n_c = 768`` the
network lands within 1% of the best rank-256 *linear* map, i.e. it measured its
own width; at ``--width 768`` the same architecture is the best single-step
method in the table.  Both runs are committed, and the report says so.

Scope -- what this program is not
---------------------------------
* One smoothing step is measured.  No V-cycle is run and no claim is made about
  the complete multigrid solver.
* No coarse-grid correction appears in the loss or in the evaluation; the step
  deliberately stops after smoothing.
* The operator is bound to one DoF count and one DoF ordering; ``n`` is baked
  into the first branch layer.
* Every norm is ``v^T A v`` on coefficient vectors.  ``M_h`` is not exported, so
  no continuous ``L^2(Gamma)`` norm is available and none is claimed.
* The network does not learn a discretisation-independent operator, and no
  cross-mesh generalisation is claimed or measured.

Provenance
----------
``A_l``, ``P_l`` and the DoF coordinates come from the deal.II surface
(Laplace-Beltrami) solver in ``src/main.cpp``, which writes them with
``export_level_data``; ``tools/convert_level_data.py`` converts the dumps into
the ``.npz``/``.npy`` this program reads.  Everything else needed to run Stage I
is in this file.

Usage
-----
    python stage1_smoothing_property.py --selftest
    python stage1_smoothing_property.py --data-dir level_data/L3 --out-dir results/stage1_L3
    python stage1_smoothing_property.py --replot --out-dir results/stage1_L3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ===========================================================================
# PART A -- machinery: geometry, level data, error generators, smoothers, model
# ===========================================================================


# ---------------------------------------------------------------------------
# A1. Torus geometry -- must match TorusGeometry in src/main.cpp
# ---------------------------------------------------------------------------
R_MAJOR = 2.0
R_MINOR = 1.0


def coords_to_angles(X: np.ndarray) -> np.ndarray:
    """Torus support points (n,3) -> parameter angles (n,2) = (xi, eta).

    Inverts  x = (R + r cos eta) cos xi,  y = r sin eta,
             z = (R + r cos eta) sin xi
    exactly.  xi and eta are both 2*pi-periodic, so any trigonometric polynomial
    in them is a well-defined function on the torus with no seam.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    rho = np.sqrt(x * x + z * z)
    xi = np.arctan2(z, x)
    eta = np.arctan2(y, rho - R_MAJOR)
    return np.stack([xi, eta], axis=1)


def trunk_features(angles: np.ndarray, mode: str, coords: np.ndarray,
                   n_modes: int = 0) -> np.ndarray:
    """The trunk's input: a fixed feature map of the DoF coordinates.

    ``trig`` encodes the torus periodicity exactly (sin/cos of the two angles) so
    the trunk has no artificial jump at the xi = +-pi seam.  ``fourier`` adds
    sin/cos of all modes |m|,|n| <= n_modes, which is what gives the trunk the
    spatial frequencies needed to represent an *oscillatory* correction -- the
    complement of the coarse space is exactly the band above the coarse lattice's
    resolution, so a trunk without those frequencies cannot span it at all.  See
    STAGE1_SMOOTHING_PROPERTY.md section 4.  ``xyz`` and ``angles`` are the raw
    coordinates as literally specified in the brief.
    """
    if mode == "xyz":
        return np.ascontiguousarray(coords, dtype=np.float64)
    if mode == "angles":
        return np.ascontiguousarray(angles, dtype=np.float64)
    xi, eta = angles[:, 0], angles[:, 1]
    if mode == "trig":
        return np.stack([np.sin(xi), np.cos(xi), np.sin(eta), np.cos(eta)], axis=1)
    if mode == "fourier":
        K = max(1, int(n_modes))
        feats = []
        for m in range(-K, K + 1):
            for n in range(-K, K + 1):
                if m == 0 and n == 0:
                    continue
                feats.append(np.sin(m * xi + n * eta))
                feats.append(np.cos(m * xi + n * eta))
        return np.stack(feats, axis=1)
    raise ValueError("unknown trunk feature mode %r" % mode)


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed.  ``hash()`` is salted per process, so it cannot
    be used for anything that has to be reproducible across runs."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# A2. Level data: load one multigrid level and verify it before trusting it
# ---------------------------------------------------------------------------
@dataclass
class LevelData:
    A: sp.csr_matrix
    coords: np.ndarray
    P: Optional[sp.csr_matrix]
    angles: np.ndarray
    lam_min: Optional[float] = None
    lam_max: Optional[float] = None

    @property
    def n(self) -> int:
        return self.A.shape[0]


def sparse_spd_report(S: sp.csc_matrix) -> Tuple[bool, Optional[float], Optional[float]]:
    """Is a sparse symmetric matrix positive definite?  Tested WITHOUT densifying.

    A dense ``np.linalg.cholesky(S.toarray())`` is fine at 1024 DoFs (8 MB) and
    fatal at 16384 (2.1 GB plus an O(n^3) factorisation), so two sparse tests:
    definiteness from a SuperLU factorisation in symmetric mode with no pivoting
    (its U diagonal holds the pivots, positive exactly when S is SPD), and the
    extreme eigenvalues from sparse Lanczos for reporting.
    """
    try:
        lu = spla.splu(S.tocsc(), diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        is_spd = bool(np.all(lu.U.diagonal() > 0.0))
    except RuntimeError:
        is_spd = False
    lam_max = lam_min = None
    try:
        lam_max = float(spla.eigsh(S, k=1, which="LA", return_eigenvectors=False)[0])
    except Exception:
        pass
    try:
        lam_min = float(spla.eigsh(S, k=1, which="SA", return_eigenvectors=False)[0])
    except Exception:
        pass
    return is_spd, lam_min, lam_max


def load_level(data_dir: str, verbose: bool = False) -> LevelData:
    """Load a converted level directory and VERIFY it before trusting it.

    Shapes, symmetry, diagonal sign, definiteness and finiteness are checked.  A
    failed check is a hard error, not a warning: the point is to build on a matrix
    whose properties are known.

    ``lam_min``/``lam_max`` are the extreme eigenvalues of the *Jacobi-scaled*
    matrix ``D^{-1/2} A D^{-1/2}``, which has the same spectrum as ``D^{-1} A``.
    ``lam_max`` is therefore the damped-Jacobi stability limit: the iteration
    diverges for ``omega >= 2 / lam_max``.
    """
    a_path = os.path.join(data_dir, "level_matrix.npz")
    x_path = os.path.join(data_dir, "dof_coordinates.npy")
    if not os.path.exists(a_path):
        raise SystemExit("missing %s -- run tools/convert_level_data.py first" % a_path)
    if not os.path.exists(x_path):
        raise SystemExit("missing %s" % x_path)

    A = sp.load_npz(a_path).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    coords = np.load(x_path)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise SystemExit("dof_coordinates.npy must be (n,3), got %r" % (coords.shape,))
    if A.shape[0] != A.shape[1]:
        raise SystemExit("A must be square, got %r" % (A.shape,))
    if coords.shape[0] != A.shape[0]:
        raise SystemExit("A is %d x %d with %d coordinates -- the DoF ordering "
                         "cannot be matched" % (A.shape[0], A.shape[1], coords.shape[0]))
    if not np.all(np.isfinite(A.data)) or not np.all(np.isfinite(coords)):
        raise SystemExit("A or the coordinates contain non-finite entries")

    asym = abs(A - A.T)
    asym_max = float(asym.max()) if asym.nnz else 0.0
    if asym_max > 1e-9 * max(float(abs(A).max()), 1.0):
        raise SystemExit("max|A - A^T| = %.3e is too large for a symmetric operator" % asym_max)

    diag = A.diagonal()
    if np.any(diag <= 0.0):
        raise SystemExit("A has a non-positive diagonal entry (min %.3e)" % diag.min())

    d_is = 1.0 / np.sqrt(diag)
    S = (sp.diags(d_is) @ A @ sp.diags(d_is)).tocsc()
    S = ((S + S.T) * 0.5).tocsc()
    spd_ok, lam_min, lam_max = sparse_spd_report(S)
    if not spd_ok:
        raise SystemExit("A is not positive definite (smallest scaled eigenvalue %s)"
                         % ("%.3e" % lam_min if lam_min is not None else "unavailable"))

    P = None
    p_path = os.path.join(data_dir, "prolongation.npz")
    if os.path.exists(p_path):
        P = sp.load_npz(p_path).tocsr()
        if P.shape[0] != A.shape[0]:
            raise SystemExit("prolongation has %d rows, A has %d" % (P.shape[0], A.shape[0]))

    if verbose:
        log("  %s: n = %d, nnz = %d, P %s"
            % (data_dir, A.shape[0], A.nnz, "%r" % (P.shape,) if P is not None else "absent"))
    return LevelData(A=A, coords=coords, P=P, angles=coords_to_angles(coords),
                     lam_min=lam_min, lam_max=lam_max)


# ---------------------------------------------------------------------------
# A3. Sparse matrix-vector products in torch, with A never densified
# ---------------------------------------------------------------------------
def to_torch_sparse(M: sp.spmatrix, dtype=torch.float64) -> torch.Tensor:
    M = M.tocoo()
    idx = torch.from_numpy(np.vstack([M.row, M.col]).astype(np.int64))
    val = torch.from_numpy(np.asarray(M.data, dtype=np.float64)).to(dtype)
    return torch.sparse_coo_tensor(idx, val, size=M.shape, dtype=dtype).coalesce()


class SparseApply(torch.autograd.Function):
    """``y = M v`` for a constant sparse ``M``, with the transpose passed in.

    Hand-written rather than relying on torch's sparse autograd coverage: the
    backward pass of ``M v`` is ``M^T g``, and supplying ``M^T`` as a constant
    makes that exact for any sparsity pattern.  Only the dense operand (an error
    or a correction) needs a gradient; ``A`` never does.
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, M: torch.Tensor, MT: torch.Tensor) -> torch.Tensor:
        ctx.MT = MT
        return torch.sparse.mm(M, v)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return torch.sparse.mm(ctx.MT, g), None, None


def A_apply(V: torch.Tensor, A: torch.Tensor, AT: torch.Tensor) -> torch.Tensor:
    """``A V`` for ``V`` shaped (B, n) -- sample-major, one error per row.

    torch's sparse mm wants the dense operand as (n, B), so the transpose is done
    here once instead of at every call site.
    """
    return SparseApply.apply(V.t(), A, AT).t()


def energy_norm(v: torch.Tensor, A: torch.Tensor, AT: torch.Tensor) -> torch.Tensor:
    """``v^T A v`` for each row of ``v`` (B,n) -> (B,).  Sparse throughout."""
    return (v * A_apply(v, A, AT)).sum(dim=1)


# ---------------------------------------------------------------------------
# A4. Error generators
# ---------------------------------------------------------------------------
MECHANISMS = ("smooth", "multiscale", "localized", "algebraic")


def _modes_to_field(angles, modes, amps, phases):
    xi, eta = angles[:, 0], angles[:, 1]
    out = np.zeros_like(xi)
    for (m, n), a, ph in zip(modes, amps, phases):
        out += a * np.cos(m * xi + n * eta + ph)
    return out


def gen_smooth(rng, angles, nyq, n_dof):
    k = int(rng.integers(1, 3))
    kmax = max(1, min(2, nyq // 4))
    modes = [(int(rng.integers(-kmax, kmax + 1)), int(rng.integers(-kmax, kmax + 1)))
             for _ in range(k)]
    modes = [(m, n) for (m, n) in modes if (m, n) != (0, 0)] or [(1, 0)]
    amps = rng.normal(size=len(modes))
    phases = rng.uniform(0.0, 2 * np.pi, size=len(modes))
    vals = _modes_to_field(angles, modes, amps, phases)
    return vals, {"modes": modes, "amps": amps.tolist(), "phases": phases.tolist()}, \
        ("smooth", tuple(sorted(modes)))


def gen_multiscale(rng, angles, nyq, n_dof):
    """Superpose periodic modes over a *band* of wavenumbers with decaying
    amplitudes.

    Wavenumbers are capped strictly below the mesh Nyquist limit so the sampled
    field does not alias: sampling a mode above Nyquist onto the DoFs silently
    produces a different, smoother function, which would make the recorded
    generator parameters a lie.
    """
    kmax = max(2, int(0.75 * nyq))
    k = int(rng.integers(3, 7))
    decay = float(rng.uniform(0.4, 1.4))
    modes, amps = [], []
    for _ in range(k):
        m = int(rng.integers(-kmax, kmax + 1))
        n = int(rng.integers(-kmax, kmax + 1))
        if (m, n) == (0, 0):
            continue
        modes.append((m, n))
        amps.append((np.hypot(m, n) ** (-decay)) * float(rng.normal()))
    if not modes:
        modes, amps = [(1, 1)], [1.0]
    phases = rng.uniform(0.0, 2 * np.pi, size=len(modes))
    vals = _modes_to_field(angles, modes, amps, phases)
    return vals, {"modes": modes, "amps": [float(a) for a in amps],
                  "phases": phases.tolist(), "decay": decay,
                  "wavenumber_cap": kmax, "nyquist": int(nyq)}, \
        ("multiscale", tuple(sorted(modes)), round(decay, 3))


def gen_localized(rng, angles, nyq, n_dof):
    """Periodic Gaussian bumps: localised in space, hence broad in wavenumber.
    Distances use the periodic parameter square, so a bump straddling the
    xi = +-pi seam stays one connected bump instead of two half-bumps."""
    n_bumps = int(rng.integers(1, 3))
    cell = 2 * np.pi / (2 * nyq)
    xi, eta = angles[:, 0], angles[:, 1]
    vals = np.zeros_like(xi)
    bumps = []
    for _ in range(n_bumps):
        xi0 = float(rng.uniform(-np.pi, np.pi))
        eta0 = float(rng.uniform(-np.pi, np.pi))
        width = float(rng.uniform(1.0, 3.0)) * cell
        amp = float(rng.normal())
        dxi = np.pi - np.abs(np.pi - np.abs(xi - xi0))
        det = np.pi - np.abs(np.pi - np.abs(eta - eta0))
        vals += amp * np.exp(-(dxi ** 2 + det ** 2) / (2.0 * width ** 2))
        bumps.append({"xi": xi0, "eta": eta0, "width": width, "amp": amp})
    return vals, {"bumps": bumps}, \
        ("localized", tuple((round(b["xi"], 3), round(b["eta"], 3),
                             round(b["width"], 6)) for b in bumps))


def gen_algebraic(rng, angles, nyq, n_dof):
    """The classical algebraic test error: independent random coefficients in
    the FE basis.  The draw index is part of the signature so two splits can
    never share one draw."""
    vals = rng.normal(size=n_dof)
    return vals, {"kind": "iid standard-normal coefficients"}, \
        ("algebraic", "draw", int(rng.integers(0, 2 ** 31)))


def _call_gen(rng, mechanism, angles, nyq, n_dof):
    if mechanism == "algebraic":
        return gen_algebraic(rng, angles, nyq, n_dof)
    if mechanism == "smooth":
        return gen_smooth(rng, angles, nyq, n_dof)
    if mechanism == "multiscale":
        return gen_multiscale(rng, angles, nyq, n_dof)
    if mechanism == "localized":
        return gen_localized(rng, angles, nyq, n_dof)
    raise ValueError(mechanism)


def draw_sample(rng, mechanism, angles, nyq, n_dof):
    """One error sample.  Returns (values at DoFs, generator params, signature).

    ``mixed`` draws a Dirichlet weight per mechanism, normalises each component
    separately and superposes them, so a mixed sample is a genuine mixture rather
    than one mechanism scaled.  The RNG call ORDER inside each generator is part
    of the reproducibility contract: changing it changes every sample.
    """
    if mechanism == "mixed":
        w = rng.dirichlet(np.ones(4) * 0.7)
        parts = list(MECHANISMS)
        field = np.zeros(n_dof)
        sig_parts, params = [], {"weights": {}}
        for wi, part in zip(w, parts):
            params["weights"][part] = float(wi)
            if wi < 1e-6:
                continue
            vals, p, s = _call_gen(rng, part, angles, nyq, n_dof)
            nrm = float(np.linalg.norm(vals))
            if nrm > 0.0:
                field += wi * vals / nrm      # each component normalised, then mixed
            sig_parts.append(s)
            params[part] = p
        return field, params, ("mixed", tuple(sig_parts))
    vals, params, sig = _call_gen(rng, mechanism, angles, nyq, n_dof)
    return vals, params, sig


def normalise_and_residualise(vals, A):
    """Scale an error to unit A-energy.

    Explicitly guards the degenerate cases: a vanishing or non-finite draw
    returns None so the caller skips it and counts it, instead of emitting
    inf/nan silently.
    """
    energy = float(vals @ (A @ vals))
    if not np.isfinite(energy) or energy <= 0.0:
        return None
    return vals / np.sqrt(energy)


# ---------------------------------------------------------------------------
# A5. Classical smoothers -- one full sweep each, as a correction to the error
# ---------------------------------------------------------------------------
class ClassicalSmoothers:
    """One application of each classical smoother, expressed as a correction.

    Jacobi is a diagonal scaling and is batched; Gauss-Seidel and SSOR are
    sequential triangular solves and are applied one sample at a time.  With
    omega = 1, SSOR *is* symmetric Gauss-Seidel, so those two rows of the results
    table coincide exactly -- that is a property of the methods, not a bug.
    """

    def __init__(self, A: sp.csr_matrix):
        self.A = A.tocsr()
        self.n = A.shape[0]
        self.diag = A.diagonal()
        self.D = sp.diags(self.diag)
        self.L = sp.tril(self.A, k=-1, format="csc")
        self.U = sp.triu(self.A, k=1, format="csc")

    def jacobi(self, r, omega):
        """``omega * D^{-1} r``; batched, so ``r`` is (N, n) sample-major."""
        return omega * r / self.diag

    def gauss_seidel(self, r, omega=1.0):
        return spla.spsolve_triangular((self.D + omega * self.L).tocsc(), r, lower=True)

    def symmetric_gs(self, r):
        y = spla.spsolve_triangular((self.D + self.L).tocsc(), r, lower=True)
        return spla.spsolve_triangular((self.D + self.U).tocsc(), self.diag * y, lower=False)

    def ssor(self, r, omega=1.0):
        """With omega = 1 this is exactly symmetric Gauss-Seidel."""
        y = spla.spsolve_triangular((self.D + omega * self.L).tocsc(), r, lower=True)
        z = spla.spsolve_triangular((self.D + omega * self.U).tocsc(), self.diag * y, lower=False)
        return omega * (2.0 - omega) * z


# ---------------------------------------------------------------------------
# A6. The DeepONet
# ---------------------------------------------------------------------------
class Branch(nn.Module):
    """Encodes the residual.  ``bias=False`` throughout, so a zero residual maps
    to a zero branch output: an exact discrete solution stays a fixed point of the
    smoothing step by construction, not merely by training."""

    def __init__(self, in_dim, width, p, depth):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width, bias=False), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, p, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trunk(nn.Module):
    """Encodes the evaluation point -- the DoF support point."""

    def __init__(self, in_dim, width, p, depth):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrozenTrunk(nn.Module):
    """A trunk that is a fixed matrix ``T`` (n, p) and is never trained.

    Freezing is what makes the trunk *provably* non-binding rather than
    empirically so.  When ``T`` has orthonormal columns spanning ``R^n`` -- the
    orthonormalised Fourier basis up to the fine Nyquist -- then
    ``span(T) = R^n``, so the floor on ``mu_F`` is exactly zero and ``mu_F`` can
    only be a statement about the branch.  Orthonormality additionally keeps the
    parameterisation well conditioned: ``T^T T = I``, so the branch's coefficients
    are exactly the correction's own Fourier coefficients.
    """

    def __init__(self, T: np.ndarray):
        super().__init__()
        self.register_buffer("T", torch.from_numpy(np.ascontiguousarray(T, dtype=np.float64)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.T


class DeepONetSmoother(nn.Module):
    """``delta_e = sum_k branch_k(r) * T_k(x)``, evaluated at every DoF at once.

    ``n_dof`` is baked into the first branch layer, which is precisely why the
    model is bound to one DoF count and one DoF ordering.

    ``base_smoother="jacobi"`` adds a diagonal skip with a learnable coefficient,

        delta_e = omega * D^{-1} r + branch(r) . T(x) ,

    so the network only has to learn what a damped Jacobi step leaves behind.  The
    branch biases stay False, so ``delta_e`` vanishes at ``r = 0`` either way.
    """

    def __init__(self, n_dof, trunk_dim, p, width, depth,
                 base_smoother="none", diag=None, omega_init=2.0 / 3.0):
        super().__init__()
        self.p = p
        self.base_smoother = base_smoother
        self.branch = Branch(n_dof, width, p, depth)
        self.trunk = Trunk(trunk_dim, width, p, depth)
        if base_smoother == "jacobi":
            if diag is None:
                raise ValueError("base_smoother='jacobi' needs the matrix diagonal")
            self.register_buffer(
                "dinv", torch.from_numpy(1.0 / np.asarray(diag, dtype=np.float64)))
            self.omega = nn.Parameter(torch.tensor(float(omega_init), dtype=torch.float64))

    def forward(self, residual, features, coeff=None):
        out = self.branch(residual) @ self.trunk(features).t()
        if self.base_smoother == "jacobi":
            out = self.omega.to(out.dtype) * (residual * self.dinv.to(out.dtype)) + out
        return out


class Stage1Net(nn.Module):
    """``delta_e = ||r||_2 * B_theta(r / ||r||_2, X)``.

    A deliberate modelling choice, stated because it constrains the answer.  Every
    classical smoother in the comparison table is a *linear* operator, and for a
    linear smoother the map ``r -> delta_e`` is exactly homogeneous of degree one
    -- the exact solve ``delta_e = A^{-1} r`` being the extreme case.  Imposing
    homogeneity puts the network in the same class as the objects it is compared
    against and makes ``rho`` scale-free.  A non-homogeneous DeepONet has strictly
    more freedom, all of it on the coarse side, which is exactly where the
    question is; so the constraint is the conservative choice, not the flattering
    one.

    ``base_smoother="jacobi"`` preserves homogeneity: the skip acts on ``r``, the
    learned part on ``r/||r||`` and is then rescaled, so both terms are degree one.
    """

    def __init__(self, inner: DeepONetSmoother, features: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("features", features)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """``R`` (B, n) raw residuals ``A e`` -> (B, n) corrections."""
        nrm = torch.linalg.norm(R, dim=1, keepdim=True).clamp_min(1e-300)
        return nrm * self.inner(R / nrm, self.features)


# ===========================================================================
# PART B -- Stage I
# ===========================================================================

# ---------------------------------------------------------------------------
# B1. Palette -- the set already used by the repository's C++ benchmark plots
# (bench/plot.py on the branch this work grew from), kept identical so a figure
# from here can sit beside a figure from there.  Fixed slot per method, never
# keyed by rank and never cycled.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = {
    "jacobi": "#2a78d6",
    "damped_jacobi": "#eb6834",
    "gauss_seidel": "#1baf7a",
    "symmetric_gs": "#eda100",
    "ssor": "#e87ba4",
    "deeponet": "#7b5cd6",
}
LABEL = {
    "jacobi": "Jacobi",
    "damped_jacobi": "Damped Jacobi",
    "gauss_seidel": "Gauss-Seidel",
    "symmetric_gs": "Symmetric GS",
    "ssor": "SSOR",
    "deeponet": "DeepONet",
}
# The headline table, in the order the brief lists it.
TABLE_ORDER = ["jacobi", "damped_jacobi", "gauss_seidel", "symmetric_gs", "ssor", "deeponet"]
CLASSICAL = ["jacobi", "damped_jacobi", "gauss_seidel", "symmetric_gs", "ssor"]


def energy_rows(V: np.ndarray, A: sp.spmatrix) -> np.ndarray:
    """``v^T A v`` for each row of ``V`` (N, n) -> (N,).

    Every energy in this file goes through here.  The arrays are sample-major
    (a row is one error) while ``A @ V.T`` is DoF-major, and mixing the two
    layouts silently reduces over the wrong axis -- so the transpose lives in one
    place instead of at twenty call sites.
    """
    return np.einsum("si,si->s", V, np.asarray(A @ V.T).T)


# ---------------------------------------------------------------------------
# B2. The Galerkin coarse space and its A-orthogonal projection
# ---------------------------------------------------------------------------
class GalerkinCoarseSpace:
    """``A_c = P^T A P`` plus the exact A-orthogonal split ``e = e_C + e_F``.

    The inverse of ``P^T A P`` is never formed.  The coarse system is solved --
    what the brief prescribes, and the only sane thing to do: the explicit inverse
    is a dense ``n_c x n_c`` object, while the sparse factorisation is built once
    and reused for every sample.
    """

    def __init__(self, A: sp.csr_matrix, P: sp.csr_matrix, verbose: bool = True):
        self.A = A.tocsr()
        self.P = P.tocsr()
        self.n = A.shape[0]
        self.n_c = P.shape[1]
        if self.P.shape[0] != self.n:
            raise SystemExit("P has %d rows but A has %d -- P must map coarse -> fine"
                             % (self.P.shape[0], self.n))

        Ac = (self.P.T @ self.A @ self.P).tocsc()
        Ac = ((Ac + Ac.T) * 0.5).tocsc()          # remove the round-off asymmetry
        asym = abs(Ac - Ac.T)
        self.galerkin_asym = float(asym.max()) if asym.nnz else 0.0
        self.A_c = Ac
        self.lu = spla.splu(Ac)

        # lambda_min(A_c) > 0 is what makes the direct sum in the module docstring
        # *direct*: P^T A P z = 0 must force z = 0.  Verified, not assumed.
        self.A_c_eigmin = float(spla.eigsh(Ac, k=1, which="SA",
                                           return_eigenvectors=False)[0])
        if self.A_c_eigmin <= 0.0:
            raise SystemExit("Galerkin operator is not positive definite "
                             "(smallest eigenvalue %.3e)" % self.A_c_eigmin)
        if verbose:
            log("  coarse space: n_c = %d of n = %d, complement dim = %d"
                % (self.n_c, self.n, self.n - self.n_c))
            log("  A_c symmetric to %.2e, lambda_min(A_c) = %.6f"
                % (self.galerkin_asym, self.A_c_eigmin))

    def split(self, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``E`` is (N, n) sample-major.  Returns ``(E_C, E_F)``, ``E_C = Pi_C E``."""
        AE = np.asarray(self.A @ E.T)                  # (n, N)
        Z = self.lu.solve(np.asarray(self.P.T @ AE))   # (n_c, N)
        EC = np.asarray(self.P @ Z).T                  # (N, n)
        return EC, E - EC

    def verify_split(self, E: np.ndarray, EC: np.ndarray,
                     EF: np.ndarray) -> Dict[str, float]:
        """The identities the brief states, measured on real samples.

        Returned as relative residuals so they are comparable across sample scale;
        anything above a few 1e-12 means the split is wrong.
        """
        AEF = np.asarray(self.A @ EF.T).T              # (N, n)
        ec_aef = np.einsum("si,si->s", EC, AEF)
        orth = np.abs(ec_aef) / np.maximum(
            np.linalg.norm(EC, axis=1) * np.linalg.norm(AEF, axis=1), 1e-300)

        PtAEF = np.asarray(self.P.T @ AEF.T)           # (n_c, N)
        gorth = np.linalg.norm(PtAEF, axis=0) / np.maximum(
            np.linalg.norm(AEF, axis=1), 1e-300)

        eAe = energy_rows(E, self.A)
        ecAec = energy_rows(EC, self.A)
        efAef = energy_rows(EF, self.A)
        pyth = np.abs(eAe - ecAec - efAef) / np.maximum(np.abs(eAe), 1e-300)

        return {
            "max_rel_eC_A_eF": float(orth.max()),
            "max_rel_PtA_eF": float(gorth.max()),
            "max_rel_energy_pythagoras": float(pyth.max()),
        }


# ---------------------------------------------------------------------------
# B3. Errors, split into their coarse and complement components
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """One role's errors, both components, and the residuals of each.

    ``RF``/``RC`` are ``A e_F`` / ``A e_C`` in sample-major form (N, n), formed
    once: ``A`` is constant and ``A e_F`` is the entire input distribution of this
    stage.
    """
    role: str
    E: np.ndarray
    EC: np.ndarray
    EF: np.ndarray
    RC: np.ndarray
    RF: np.ndarray
    frac_F: np.ndarray
    mechanism: List[str]
    discarded: int = 0

    @property
    def n(self) -> int:
        return self.E.shape[1]


def build_split(role: str, plan: Sequence[Tuple[str, int]], A: sp.csr_matrix,
                coarse: GalerkinCoarseSpace, angles: np.ndarray, nyq: int,
                n_dof: int, master_seed: int, ftol: float) -> Split:
    """Draw errors, split them, drop the ones with no complement content.

    The brief discards samples with ``e_F^T A e_F ~ 0``.  The tolerance is
    *relative* to ``e^T A e`` rather than absolute, because every error here is
    normalised to unit A-energy, so an absolute threshold would mean different
    things for different generators.  Each mechanism gets its own RNG stream
    derived from (seed, role, mechanism, count), so the three roles are
    independent and cannot share a draw.
    """
    E_list, mech_list = [], []
    discarded = 0
    for mechanism, count in plan:
        rng = np.random.default_rng(stable_seed(master_seed, role, mechanism, count))
        got, attempts = 0, 0
        while got < count and attempts < count * 20:
            attempts += 1
            vals, _params, _sig = draw_sample(rng, mechanism, angles, nyq, n_dof)
            e = normalise_and_residualise(vals, A)
            if e is None:
                discarded += 1
                continue
            E_list.append(e)
            mech_list.append(mechanism)
            got += 1
        if got < count:
            raise SystemExit("could not draw %d usable %s samples for %s"
                             % (count, mechanism, role))

    E = np.array(E_list)
    EC, EF = coarse.split(E)
    frac_F = energy_rows(EF, A) / np.maximum(energy_rows(E, A), 1e-300)

    keep = frac_F >= ftol
    dropped = int((~keep).sum())
    E, EC, EF = E[keep], EC[keep], EF[keep]
    return Split(role=role, E=E, EC=EC, EF=EF,
                 RC=np.asarray(A @ EC.T).T, RF=np.asarray(A @ EF.T).T,
                 frac_F=frac_F[keep],
                 mechanism=[m for m, k in zip(mech_list, keep) if k],
                 discarded=discarded + dropped)


# ---------------------------------------------------------------------------
# B4. Training -- on the complement component only
# ---------------------------------------------------------------------------
def smooth_loss(model: Stage1Net, R: torch.Tensor, EF: torch.Tensor,
                den: torch.Tensor, A_t: torch.Tensor, AT_t: torch.Tensor,
                eps: float = 1e-30) -> torch.Tensor:
    """``L_smooth`` -- relative A-energy of ``e_F`` left after one learned step."""
    remaining = EF - model(R)
    return (energy_norm(remaining, A_t, AT_t) / den.clamp_min(eps)).mean()


def train(model: Stage1Net, tr: Split, va: Split, A_t, AT_t, epochs: int,
          batch: int, lr: float, weight_decay: float, patience: int, seed: int,
          verbose_every: int, log_fn=None) -> Dict[str, object]:
    """Train on ``e_F`` only.

    The validation set is also only ever seen through its complement component:
    ``e_C`` is not touched until evaluation.  ``log_fn`` defaults to stdout; main
    passes the tee that also writes ``run.log``.
    """
    say = log_fn if log_fn is not None else log
    torch.manual_seed(seed)

    R_tr = torch.from_numpy(np.ascontiguousarray(tr.RF))
    E_tr = torch.from_numpy(np.ascontiguousarray(tr.EF))
    den_tr = energy_norm(E_tr, A_t, AT_t).detach()

    R_va = torch.from_numpy(np.ascontiguousarray(va.RF))
    E_va = torch.from_numpy(np.ascontiguousarray(va.EF))
    den_va = energy_norm(E_va, A_t, AT_t).detach()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    gen = torch.Generator().manual_seed(seed)

    history = {"epoch": [], "train_loss": [], "val_mse": [], "lr": []}
    best = {"val_mse": float("inf"), "epoch": -1, "state": None}
    stale, n_tr = 0, R_tr.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=gen)
        total = 0.0
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            loss = smooth_loss(model, R_tr[idx], E_tr[idx], den_tr[idx], A_t, AT_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * idx.numel()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_mse = float(smooth_loss(model, R_va, E_va, den_va, A_t, AT_t))
        history["epoch"].append(epoch)
        history["train_loss"].append(total / max(n_tr, 1))
        history["val_mse"].append(val_mse)
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        if val_mse < best["val_mse"] - 1e-9:
            best = {"val_mse": val_mse, "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            say("    epoch %5d  train %.6f  val mean sq rho_F %.6f%s"
                % (epoch, total / max(n_tr, 1), val_mse, "  *" if stale == 0 else ""))
        if patience and stale >= patience:
            say("    early stop at epoch %d (no improvement for %d epochs)"
                % (epoch, patience))
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history,
            "best": {"val_mse": best["val_mse"],
                     "val_mu_F": float(np.sqrt(max(best["val_mse"], 0.0))),
                     "epoch": best["epoch"], "epochs_run": len(history["epoch"])}}


# ---------------------------------------------------------------------------
# B5. Evaluation
# ---------------------------------------------------------------------------
def reduction_stats(rho: np.ndarray) -> Dict[str, float]:
    rho = np.asarray(rho, dtype=np.float64)
    return {
        "mean": float(rho.mean()),
        "median": float(np.median(rho)),
        "std": float(rho.std(ddof=1)) if rho.size > 1 else 0.0,
        "p95": float(np.percentile(rho, 95)),
        "max": float(rho.max()),
        "frac_amplified": float((rho > 1.0).mean()),
    }


def rho_of(E: np.ndarray, Enew: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    """Per-sample ``||Enew||_A / ||E||_A``, both (N, n)."""
    return np.sqrt(np.maximum(energy_rows(Enew, A), 0.0)
                   / np.maximum(energy_rows(E, A), 1e-300))


def apply_classical(kind: str, E: np.ndarray, R: np.ndarray,
                    cs: ClassicalSmoothers, omega: float,
                    ssor_omega: float) -> np.ndarray:
    """``Enew = E - correction`` for one full sweep.  Both ``E`` and ``R`` are
    (N, n) sample-major, which is the layout ``ClassicalSmoothers`` expects --
    asserted here because ``jacobi`` divides by a length-n vector and under the
    transpose it would silently broadcast along the wrong axis."""
    if R.shape != E.shape:
        raise ValueError("R %r and E %r must be sample-major and equal" % (R.shape, E.shape))
    if kind == "jacobi":
        return E - cs.jacobi(R, 1.0)
    if kind == "damped_jacobi":
        return E - cs.jacobi(R, omega)
    if kind == "gauss_seidel":
        return E - np.array([cs.gauss_seidel(r) for r in R])
    if kind == "symmetric_gs":
        return E - np.array([cs.symmetric_gs(r) for r in R])
    if kind == "ssor":
        return E - np.array([cs.ssor(r, ssor_omega) for r in R])
    raise ValueError("unknown classical smoother %r" % kind)


def pick_jacobi_omega(cs: ClassicalSmoothers, split: Split, A: sp.csr_matrix,
                      lam_max: float, grid_n: int = 13) -> Dict[str, object]:
    """Choose the damping on *validation* ``mu_F``, then reuse it on test.

    The sweep is expressed relative to ``2 / lambda_max(D^{-1} A)``, the stability
    limit: beyond it damped Jacobi diverges and there is nothing to tune.
    Selecting on the complement component gives the classical method its best shot
    at the same target the network is trained on, so neither is handicapped.
    """
    limit = 2.0 / float(lam_max)
    table = {}
    for w in np.linspace(0.05, 1.0, grid_n) * limit:
        Enew = apply_classical("damped_jacobi", split.EF, split.RF, cs, float(w), 1.0)
        table[float(w)] = reduction_stats(rho_of(split.EF, Enew, A))["mean"]
    best_w = min(table.items(), key=lambda kv: kv[1])
    return {"omega": float(best_w[0]), "val_mu_F": float(best_w[1]),
            "stability_limit": limit, "sweep": table}


# ---------------------------------------------------------------------------
# B6. Diagnostics: a linear reference for the same data
# ---------------------------------------------------------------------------
def fit_ridge(tr: Split, va: Split, A: sp.csc_matrix,
              lam_grid: Sequence[float]) -> Dict[str, object]:
    """Best *linear* map ``r -> delta_e`` fitted to the same training pairs.

    Diagnostic only, never in the headline table.  Fitted in the same A-inner
    product the network is trained in: with ``G`` a square root of ``A``
    (Cholesky factor transposed), the objective is

        min_V  sum_i || V r_i - G e_i ||^2 + lam ||V||_F^2 ,   W = G^{-1} V ,

    whose normal equations are ``(R^T R + lam I) V^T = R^T (G E^T)^T``.

    The network minimises the *relative* energy,
    ``sum_i ||.||_A^2 / ||e_i||_A^2``, which is a weighted least squares with
    ``w_i = 1/||e_i||_A^2``.  The weights are applied here too, by scaling the
    data by ``sqrt(w_i)`` and running the regression as if unweighted; without
    them the reference would optimise a different objective from the one the
    table scores, and would understate what a linear map can do.
    """
    n = tr.RF.shape[1]
    G = np.linalg.cholesky(np.asarray(A.todense())).T       # G^T G = A
    w = 1.0 / np.maximum(np.sqrt(energy_rows(tr.EF, A)), 1e-12)
    Rs = tr.RF * w[:, None]
    Es = tr.EF * w[:, None]

    Zt = (G @ Es.T).T                                       # (N, n)
    RtR = Rs.T @ Rs
    RtZ = Rs.T @ Zt

    def predict(R: np.ndarray, W: np.ndarray) -> np.ndarray:
        """``delta_e`` for each row of ``R``.

        ``W`` acts on a column vector, so row-wise this is ``R @ W.T``.  Writing
        ``R @ W`` instead silently applies the transpose; both are plausible ridge
        fits, so the mistake is invisible in the loss curve and shows up only as
        an unexplained number in the table.
        """
        return R @ W.T

    best = None
    grid = {}
    for lam in lam_grid:
        Vt = np.linalg.solve(RtR + float(lam) * np.eye(n), RtZ)
        W = np.linalg.solve(G, Vt.T)
        mu = reduction_stats(rho_of(va.EF, va.EF - predict(va.RF, W), A))["mean"]
        grid[float(lam)] = float(mu)
        if best is None or mu < best[1]:
            best = (float(lam), float(mu), W)
    lam, mu_va, W = best

    # A ridge fit must reproduce its own training data.  If the transpose were
    # wrong this is where it would show, so it is checked rather than assumed.
    mu_tr = reduction_stats(rho_of(tr.EF, tr.EF - predict(tr.RF, W), A))["mean"]
    return {"lambda": lam, "val_mu_F": mu_va, "train_mu_F": float(mu_tr),
            "grid": grid, "W": W, "predict": predict}


def trunk_floor(T: np.ndarray, EF: np.ndarray, A: sp.spmatrix,
                reg_rel: float = 1e-12) -> Tuple[float, int]:
    """Floor on ``mu_F`` for ANY branch against the trunk ``T``.

    The best A-norm approximation of each ``e_F`` inside ``span(T)`` solves
    ``(T^T A T) c = T^T A e_F``, so ``mu_F >= E[ sqrt(1 - reach) ]``.  Returns
    that floor and ``rank(T)``.  With ``span(T) = R^n`` the floor is zero and the
    trunk is provably not what is being measured.
    """
    TtAT = T.T @ np.asarray(A @ T)
    p = T.shape[1]
    coef = np.linalg.solve(
        TtAT + reg_rel * (np.trace(TtAT) / max(p, 1)) * np.eye(p),
        T.T @ np.asarray(A @ EF.T))
    approx = (T @ coef).T
    reach = energy_rows(approx, A) / np.maximum(energy_rows(EF, A), 1e-300)
    return float(np.mean(np.sqrt(np.maximum(1.0 - reach, 0.0)))), \
        int(np.linalg.matrix_rank(T, tol=1e-10))


# ---------------------------------------------------------------------------
# B7. Figures
# ---------------------------------------------------------------------------
def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def plot_loss_curves(history: Dict[str, list], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _style_axis(ax)
    ax.plot(history["epoch"], history["train_loss"], color=SERIES["deeponet"],
            linewidth=1.6, label="train $L_{smooth}$")
    ax.plot(history["epoch"], history["val_mse"], color=SERIES["damped_jacobi"],
            linewidth=1.6, label="validation $E[\\rho_F^2]$")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("relative $A$-energy left", color=INK_2, fontsize=9)
    ax.set_title("Stage I training, complement component only",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8.5)
    fig.savefig(outpath, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(rho: Dict[str, Dict[str, np.ndarray]], outpath: str) -> None:
    keys = ["deeponet", "damped_jacobi", "gauss_seidel"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), facecolor=SURFACE)
    for ax, comp, sub in zip(axes, ("F", "C"),
                             ("Coarse-space complement $e_F$", "Coarse component $e_C$")):
        _style_axis(ax)
        for k in keys:
            ax.hist(rho[k][comp], bins=40, range=(0.0, 1.4), histtype="step",
                    linewidth=1.6, color=SERIES[k], density=True, label=LABEL[k])
        ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xlabel(r"$\rho=\Vert Se\Vert_A/\Vert e\Vert_A$", color=INK_2, fontsize=9)
        ax.set_ylabel("density", color=INK_2, fontsize=9)
        ax.set_title(sub, color=INK, fontsize=10.5, loc="left")
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=8.5)
    fig.savefig(outpath, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_selectivity(stats: Dict[str, Dict[str, Dict[str, float]]], outpath: str) -> None:
    """The headline figure: every method as a point in the (mu_F, mu_C) plane.

    Selectivity is ``mu_C > mu_F`` -- the complement reduced more than the coarse
    component -- so the selective half-plane is the one **above** the diagonal,
    and the diagonal ``mu_F = mu_C`` is the non-selective locus.  Both regions are
    labelled, because the figure is easy to read backwards.
    """
    fig, ax = plt.subplots(figsize=(7.0, 6.2), facecolor=SURFACE)
    _style_axis(ax)
    lim = max(1.05, max(max(s["F"]["mean"], s["C"]["mean"])
                        for s in stats.values()) * 1.10)
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate("non-selective  $\\mu_F=\\mu_C$", (lim * 0.52, lim * 0.455),
                color=MUTED, fontsize=8.5, ha="left", va="top", rotation=39,
                rotation_mode="anchor")
    ax.annotate("selective\n$\\mu_C>\\mu_F$", (lim * 0.045, lim * 0.90),
                color=MUTED, fontsize=8.5, ha="left", va="top")

    # Methods whose reduction factors coincide are labelled once, together: SSOR
    # at omega = 1 *is* symmetric Gauss-Seidel, so the two points are the same
    # point and two labels would print on top of each other.
    groups: Dict[Tuple[float, float], List[str]] = {}
    colors: Dict[Tuple[float, float], str] = {}
    for k in TABLE_ORDER:
        s = stats[k]
        key = (round(s["F"]["mean"], 4), round(s["C"]["mean"], 4))
        groups.setdefault(key, []).append(LABEL[k])
        colors.setdefault(key, SERIES[k])

    for (xf, yc), names in groups.items():
        ax.scatter([xf], [yc], s=72, color=colors[(xf, yc)],
                   edgecolor=SURFACE, linewidth=1.3, zorder=3)
        ax.annotate(" / ".join(names), (xf, yc), textcoords="offset points",
                    xytext=(9, 7), color=INK_2, fontsize=8.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\mu_F$  (complement component)", color=INK_2, fontsize=9)
    ax.set_ylabel(r"$\mu_C$  (coarse component)", color=INK_2, fontsize=9)
    ax.set_title("Smoothing selectivity on level 3", color=INK, fontsize=10.5, loc="left")
    fig.savefig(outpath, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_per_mechanism(per_mech: Dict[str, Dict[str, float]], outpath: str) -> None:
    keys = ["deeponet", "damped_jacobi", "gauss_seidel"]
    names = list(MECHANISMS) + ["mixed"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2), facecolor=SURFACE)
    _style_axis(ax)
    x = np.arange(len(names))
    width = 0.26
    for j, k in enumerate(keys):
        ax.bar(x + (j - 1) * width, [per_mech[k][m] for m in names], width,
               color=SERIES[k], edgecolor=SURFACE, linewidth=0.8, label=LABEL[k])
    ax.set_xticks(x)
    ax.set_xticklabels(names, color=INK_2, fontsize=9)
    ax.set_ylabel(r"$\mu_F$", color=INK_2, fontsize=9)
    ax.set_title(r"Complement reduction $\mu_F$ by error mechanism",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8.5)
    fig.savefig(outpath, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def replot(out_dir: str) -> int:
    """Redraw the figures that need nothing but the run's own recorded numbers.

    ``metrics.json`` holds every mean (so the whole selectivity map and the
    per-mechanism bars) and ``history.csv`` holds the loss trace, so a change to
    how a figure is *drawn* does not require re-training.  ``reduction_histograms.png``
    is the exception: it needs the per-sample ``rho`` arrays, which are derived
    during evaluation and deliberately not written to disk.
    """
    m_path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(m_path):
        raise SystemExit("no metrics.json in %s -- nothing to replot" % out_dir)
    with open(m_path, encoding="utf-8") as fh:
        metrics = json.load(fh)
    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    rows = dict(metrics["headline_table"])
    if metrics.get("ridge_table"):
        rows["ridge"] = metrics["ridge_table"]
    rows["exact_solve"] = metrics["reference_exact_solve"]
    stats = {k: {"F": {"mean": r["mu_F"]}, "C": {"mean": r["mu_C"]}}
             for k, r in rows.items()}
    plot_selectivity(stats, os.path.join(plots, "selectivity_map.png"))
    plot_per_mechanism(metrics["per_mechanism_mu_F"],
                       os.path.join(plots, "per_mechanism.png"))

    h_path = os.path.join(out_dir, "history.csv")
    if os.path.exists(h_path):
        hist = {"epoch": [], "train_loss": [], "val_mse": [], "lr": []}
        with open(h_path, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                if not line.strip():
                    continue
                e, tr_, va, lr = line.split(",")
                hist["epoch"].append(int(e))
                hist["train_loss"].append(float(tr_))
                hist["val_mse"].append(float(va))
                hist["lr"].append(float(lr))
        plot_loss_curves(hist, os.path.join(plots, "loss_curves.png"))

    log("replotted %s -> selectivity_map.png, per_mechanism.png%s"
        % (out_dir, ", loss_curves.png" if os.path.exists(h_path) else ""))
    return 0


# ---------------------------------------------------------------------------
# B8. Self-test -- no training, no data generation
# ---------------------------------------------------------------------------
def selftest(data_dir: str) -> int:
    """Check the projector identities and every smoother against its dense form.

    Each classical smoother is compared against the dense matrix that *defines*
    it, because a triangular-solve implementation can be subtly transposed or
    un-scaled and still produce plausible reduction factors.
    """
    ld = load_level(data_dir, verbose=False)
    A, P = ld.A, ld.P
    if P is None:
        raise SystemExit("no prolongation.npz in %s -- Stage I is defined by the "
                         "Galerkin coarse space" % data_dir)
    coarse = GalerkinCoarseSpace(A, P)
    rng = np.random.default_rng(0)
    n = A.shape[0]

    E = rng.standard_normal((6, n))
    EC, EF = coarse.split(E)
    log("  projector identities (relative residuals):")
    for k, v in coarse.verify_split(E, EC, EF).items():
        log("    %-28s %.3e" % (k, v))

    Ad = np.asarray(A.todense())
    rank_P = int(np.linalg.matrix_rank(P.todense()))
    dim_ker = n - rank_P
    log("    rank(P) = %d, dim ker(P^T) = %d (expected %d)   %s"
        % (rank_P, dim_ker, n - P.shape[1],
           "ok" if dim_ker == n - P.shape[1] else "FAILED"))
    if dim_ker != n - P.shape[1]:
        raise SystemExit("FAILED: ker(P^T) has the wrong dimension")

    lam = np.linalg.eigvals(np.linalg.solve(np.diag(np.diag(Ad)), Ad)).real
    log("    lambda_max(D^-1 A) = %.6f, load_level reports %.6f"
        % (lam.max(), ld.lam_max))
    if abs(lam.max() - ld.lam_max) > 1e-6 * max(1.0, ld.lam_max):
        raise SystemExit("FAILED: Jacobi-scaled eigenvalue range disagrees")
    if lam.max() > 2.0:
        log("    NOTE: lambda_max(D^-1 A) = %.3f > 2, so UNDAMPED Jacobi diverges "
            "here; the stability limit is omega < %.4f" % (lam.max(), 2.0 / lam.max()))

    # Dense reference operators, built from the definitions in the brief.
    D = np.diag(np.diag(Ad))
    L = np.tril(Ad, -1)
    U = np.triu(Ad, 1)
    I = np.eye(n)
    S_ref = {
        "jacobi": I - np.linalg.solve(D, Ad),
        "damped_jacobi": I - 0.4 * np.linalg.solve(D, Ad),
        "gauss_seidel": I - np.linalg.solve(D + L, Ad),
    }
    S_ref["symmetric_gs"] = (I - np.linalg.solve(D + U, Ad)) @ (I - np.linalg.solve(D + L, Ad))
    S_ref["ssor"] = S_ref["symmetric_gs"]      # omega = 1 is symmetric GS by definition

    cs = ClassicalSmoothers(A)
    e = E[0]
    r = np.asarray(A @ e)
    for kind in CLASSICAL:
        Enew = apply_classical(kind, e[None, :], r[None, :], cs, 0.4, 1.0)[0]
        ref = S_ref[kind] @ e
        err = np.linalg.norm(Enew - ref) / max(np.linalg.norm(ref), 1e-300)
        log("    %-16s vs its dense definition: rel. error %.3e   %s"
            % (kind, err, "ok" if err < 1e-10 else "FAILED"))
        if err > 1e-10:
            raise SystemExit("FAILED: %s does not match its dense definition" % kind)

    log("  all self-tests passed")
    return 0


# ---------------------------------------------------------------------------
# B9. Main
# ---------------------------------------------------------------------------
ROLE_MULT = {"train": 1.0, "val": 0.25, "test": 0.5}
BASE_PLAN = [("mixed", 1024), ("smooth", 256), ("multiscale", 256),
             ("localized", 256), ("algebraic", 256)]


def scaled_plan(mult: float) -> List[Tuple[str, int]]:
    return [(m, max(1, int(round(c * mult)))) for m, c in BASE_PLAN]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage I: does the DeepONet learn preferential reduction of the "
                    "Galerkin coarse-space complement?")
    ap.add_argument("--data-dir", default="level_data/L3")
    ap.add_argument("--out-dir", default="results/stage1_L3")
    ap.add_argument("--level", type=int, default=3,
                    help="recorded in the report only; --data-dir decides what is loaded")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--p", type=int, default=1024,
                    help="branch/trunk width; bounds the rank of the correction. "
                         "n makes the trunk-span floor vacuous")
    ap.add_argument("--width", type=int, default=768,
                    help="branch hidden width. This is a RESULT, not a tuning knob: "
                         "the branch's Jacobian factors through it, so a width below "
                         "the complement dimension caps mu_F no matter what the "
                         "optimiser does -- see tools/branch_rank_probe.py")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--trunk-features", default="fourier",
                    choices=["xyz", "angles", "trig", "fourier"])
    ap.add_argument("--trunk-modes", type=int, default=16)
    ap.add_argument("--trunk-width", type=int, default=-1,
                    help="trunk hidden width; -1 reuses --width.  rank(T) is bounded "
                         "by the trunk width, so a narrow trunk can bind mu_F on its own")
    ap.add_argument("--trunk-depth", type=int, default=0,
                    help="trunk hidden depth; 0 makes the trunk a single linear map of "
                         "the features")
    ap.add_argument("--trunk-freeze", default="fourier-orth",
                    choices=["none", "fourier-orth"],
                    help="'fourier-orth' freezes the trunk to an orthonormal basis of "
                         "the features, so span(T) = R^n and the trunk-span floor is "
                         "exactly zero. 'none' trains the trunk")
    ap.add_argument("--base-smoother", default="none", choices=["none", "jacobi"],
                    help="'jacobi' adds a learnable damped-Jacobi skip")
    ap.add_argument("--ftol", type=float, default=1e-8,
                    help="relative complement energy below which a sample is dropped")
    ap.add_argument("--ssor-omega", type=float, default=1.0)
    ap.add_argument("--ridge-grid", type=float, nargs="*",
                    default=[1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0])
    ap.add_argument("--no-ridge", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inspect-only", action="store_true",
                    help="load, split, report the geometry, train nothing")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figures that depend only on metrics.json and "
                         "history.csv (selectivity, per-mechanism, loss curves), "
                         "without training. reduction_histograms.png needs the "
                         "per-sample arrays and so needs a run")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.data_dir)
    if args.replot:
        return replot(args.out_dir)
    if args.tag:
        args.out_dir = args.out_dir + "_" + args.tag
    if args.threads:
        torch.set_num_threads(args.threads)

    # float64 throughout (the energies being compared differ in the 4th decimal),
    # and the sparse invariant check off, since it only warns once per construction.
    dtype = torch.float64
    torch.sparse.check_sparse_tensor_invariants.disable()

    # Seed BEFORE anything is constructed.  torch seeds its global RNG
    # nondeterministically at process start, so seeding only inside train() --
    # which runs after the model is built -- leaves the initial weights varying
    # from run to run.  Two runs of this file with identical arguments then
    # disagree from the first epoch, and nothing in metrics.json would say why.
    torch.manual_seed(args.seed)

    t_start = time.perf_counter()
    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)
    run_log = open(os.path.join(args.out_dir, "run.log"), "w", encoding="utf-8")

    def log2(msg: str = "") -> None:
        log(msg)
        run_log.write(msg + "\n")
        run_log.flush()

    log2("Stage I -- learning the smoothing property (level %d)" % args.level)
    log2("=" * 74)
    log2("  data-dir %s" % args.data_dir)
    log2("  scope: ONE smoothing step; complement-only training; no coarse-grid "
         "correction in the loss")

    ld = load_level(args.data_dir, verbose=False)
    A, P, coords, angles = ld.A, ld.P, ld.coords, ld.angles
    if P is None:
        raise SystemExit("need prolongation.npz in %s" % args.data_dir)
    n = A.shape[0]
    n_side = max(2, int(round(np.sqrt(n))))
    nyq = max(1, n_side // 2)
    log2("  n = %d DoFs (%dx%d lattice, Nyquist %d), A nnz = %d"
         % (n, n_side, n_side, nyq, A.nnz))

    coarse = GalerkinCoarseSpace(A, P, verbose=False)
    log2("  coarse space: n_c = %d -> complement dim = %d" % (coarse.n_c, n - coarse.n_c))
    log2("  lambda_min(A_c) = %.6f, A_c symmetric to %.2e"
         % (coarse.A_c_eigmin, coarse.galerkin_asym))

    log2("\n[1/5] drawing errors and splitting them into e_C + e_F")
    splits: Dict[str, Split] = {}
    for role in ("train", "val", "test"):
        splits[role] = build_split(role, scaled_plan(ROLE_MULT[role]), A, coarse,
                                   angles, nyq, n, args.seed, args.ftol)
        s = splits[role]
        log2("  %-5s N=%4d   frac_F: min %.4f  mean %.4f  max %.4f   (dropped %d)"
             % (role, s.E.shape[0], s.frac_F.min(), s.frac_F.mean(),
                s.frac_F.max(), s.discarded))

    checks = coarse.verify_split(splits["test"].E, splits["test"].EC, splits["test"].EF)
    log2("  projector identities on the test split (relative residuals):")
    for k, v in checks.items():
        log2("    %-28s %.3e" % (k, v))
    for k, v in checks.items():
        if v > 1e-9:
            raise SystemExit("projector identity %s = %.3e not satisfied" % (k, v))

    # The structural reason this is an extrapolation test, measured.
    leak = np.linalg.norm(np.asarray(P.T @ splits["train"].RF.T), axis=0) / np.maximum(
        np.linalg.norm(splits["train"].RF, axis=1), 1e-300)
    inrange = np.linalg.norm(np.asarray(P.T @ splits["train"].RC.T), axis=0) / np.maximum(
        np.linalg.norm(splits["train"].RC, axis=1), 1e-300)
    log2("  training residuals r_F = A e_F:  max ||P^T r_F||/||r_F|| = %.3e  (~0)"
         % leak.max())
    log2("  coarse residuals    r_C = A e_C:  min ||P^T r_C||/||r_C|| = %.3e  (O(1))"
         % inrange.min())

    if args.inspect_only:
        run_log.close()
        return 0

    log2("\n[2/5] building the network")
    features_np = trunk_features(angles, args.trunk_features, coords, args.trunk_modes)
    features = torch.from_numpy(np.ascontiguousarray(features_np, dtype=np.float64))

    frozen_T = None
    if args.trunk_freeze == "fourier-orth":
        # Orthonormalise the feature map on the DoF points.  Q is (n, k) with
        # orthonormal columns; with Fourier modes up to the fine Nyquist the
        # features already span R^n, so Q spans R^n and the correction set is
        # unrestricted.
        Q, _ = np.linalg.qr(features_np)
        frozen_T = np.ascontiguousarray(Q[:, :min(n, features_np.shape[1])])
        if frozen_T.shape[1] < args.p:
            log2("  note: frozen trunk spans %d < --p %d; p is set to the span"
                 % (frozen_T.shape[1], args.p))
        args.p = frozen_T.shape[1]

    inner = DeepONetSmoother(
        n_dof=n, trunk_dim=features_np.shape[1], p=args.p, width=args.width,
        depth=args.depth, base_smoother=args.base_smoother,
        diag=A.diagonal() if args.base_smoother == "jacobi" else None)
    trunk_width = args.trunk_width if args.trunk_width > 0 else args.width
    if frozen_T is not None:
        inner.trunk = FrozenTrunk(frozen_T)
        trunk_desc = "%d, frozen/orthonormal" % frozen_T.shape[1]
    else:
        # The trunk is given its own geometry.  A shared narrow width would let
        # rank(T) <= trunk width hold mu_F up on its own -- the trunk, not the
        # smoother, would be under test.
        trunk_desc = "%d, depth %d" % (trunk_width, args.trunk_depth)
        if (trunk_width, args.trunk_depth) != (args.width, args.depth):
            inner.trunk = Trunk(features_np.shape[1], trunk_width, args.p, args.trunk_depth)
    inner = inner.to(dtype)
    model = Stage1Net(inner, features)
    n_par = sum(p.numel() for p in model.parameters())
    log2("  trunk %s -> dim %d ; trunk width %s ; frozen = %s"
         % (args.trunk_features, features_np.shape[1], trunk_desc, frozen_T is not None))
    log2("  branch width %d depth %d ; p = %d ; base = %s"
         % (args.width, args.depth, args.p, args.base_smoother))
    log2("  parameters %d ; correction rank <= p = %d ; complement dim = %d"
         % (n_par, args.p, n - coarse.n_c))
    if args.width < n - coarse.n_c:
        log2("  NOTE: branch width %d < complement dim %d -- the branch cannot "
             "represent the exact correction, and mu_F is capped by that alone "
             "(tools/branch_rank_probe.py)" % (args.width, n - coarse.n_c))

    A_t = to_torch_sparse(A, dtype=torch.float64)
    AT_t = to_torch_sparse(A.T.tocsr(), dtype=torch.float64)

    log2("\n[3/5] training on the complement component only")
    t_train = time.perf_counter()
    res = train(model, splits["train"], splits["val"], A_t, AT_t, epochs=args.epochs,
                batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                patience=args.patience, seed=args.seed, verbose_every=args.log_every,
                log_fn=log2)
    train_wall = time.perf_counter() - t_train
    log2("  best validation mu_F = %.6f at epoch %d (%d epochs run, %.1f s)"
         % (res["best"]["val_mu_F"], res["best"]["epoch"],
            res["best"]["epochs_run"], train_wall))

    log2("\n[4/5] evaluating every smoother on the SAME test errors")
    test = splits["test"]
    cs = ClassicalSmoothers(A)
    omega_info = pick_jacobi_omega(cs, splits["val"], A, float(ld.lam_max))
    log2("  damped Jacobi: omega = %.6f chosen on validation mu_F (stability limit "
         "%.6f)" % (omega_info["omega"], omega_info["stability_limit"]))

    model.eval()
    with torch.no_grad():
        corr_F = model(torch.from_numpy(np.ascontiguousarray(test.RF))).numpy()
        corr_C = model(torch.from_numpy(np.ascontiguousarray(test.RC))).numpy()

    rho: Dict[str, Dict[str, np.ndarray]] = {}
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for kind in CLASSICAL:
        rho[kind] = {
            "F": rho_of(test.EF, apply_classical(kind, test.EF, test.RF, cs,
                                                 omega_info["omega"], args.ssor_omega), A),
            "C": rho_of(test.EC, apply_classical(kind, test.EC, test.RC, cs,
                                                 omega_info["omega"], args.ssor_omega), A),
        }
    rho["deeponet"] = {"F": rho_of(test.EF, test.EF - corr_F, A),
                       "C": rho_of(test.EC, test.EC - corr_C, A)}
    for kind in TABLE_ORDER:
        stats[kind] = {c: reduction_stats(rho[kind][c]) for c in ("F", "C")}

    # Reference and diagnostic, deliberately outside the headline table.
    rho["exact_solve"] = {"F": np.zeros(test.E.shape[0]), "C": np.zeros(test.E.shape[0])}
    stats["exact_solve"] = {c: reduction_stats(rho["exact_solve"][c]) for c in ("F", "C")}

    ridge_info: Dict[str, object] = {"skipped": True}
    if not args.no_ridge:
        ridge_info = fit_ridge(splits["train"], splits["val"], A.tocsc(), args.ridge_grid)
        pred, W = ridge_info["predict"], ridge_info["W"]
        rho["ridge"] = {"F": rho_of(test.EF, test.EF - pred(test.RF, W), A),
                        "C": rho_of(test.EC, test.EC - pred(test.RC, W), A)}
        stats["ridge"] = {c: reduction_stats(rho["ridge"][c]) for c in ("F", "C")}
        log2("  ridge diagnostic: lambda = %g -> mu_F %.6f (train %.6f, val %.6f)"
             % (ridge_info["lambda"], stats["ridge"]["F"]["mean"],
                ridge_info["train_mu_F"], ridge_info["val_mu_F"]))

    with torch.no_grad():
        T = model.inner.trunk(features).numpy()
    trunk_floor_val, rank_T = trunk_floor(T, test.EF, A)
    log2("  trunk span: rank(T) = %d of p = %d -> best-possible mu_F >= %.6f"
         % (rank_T, args.p, trunk_floor_val))

    per_mech: Dict[str, Dict[str, float]] = {}
    for k in ("deeponet", "damped_jacobi", "gauss_seidel"):
        per_mech[k] = {}
        for m in list(MECHANISMS) + ["mixed"]:
            idx = np.array([i for i, mm in enumerate(test.mechanism) if mm == m])
            per_mech[k][m] = (float(rho[k]["F"][idx].mean()) if idx.size
                              else float("nan"))

    log2("\n[5/5] writing results")
    plots = os.path.join(args.out_dir, "plots")
    plot_loss_curves(res["history"], os.path.join(plots, "loss_curves.png"))
    plot_histograms(rho, os.path.join(plots, "reduction_histograms.png"))
    plot_selectivity(stats, os.path.join(plots, "selectivity_map.png"))
    plot_per_mechanism(per_mech, os.path.join(plots, "per_mechanism.png"))

    with open(os.path.join(args.out_dir, "history.csv"), "w", encoding="utf-8") as fh:
        fh.write("epoch,train_loss,val_mse,lr\n")
        for i in range(len(res["history"]["epoch"])):
            fh.write("%d,%.10g,%.10g,%.10g\n"
                     % (res["history"]["epoch"][i], res["history"]["train_loss"][i],
                        res["history"]["val_mse"][i], res["history"]["lr"][i]))

    def row(k: str) -> Dict[str, float]:
        return {
            "mu_F": stats[k]["F"]["mean"], "mu_C": stats[k]["C"]["mean"],
            "median_rho_F": stats[k]["F"]["median"], "std_rho_F": stats[k]["F"]["std"],
            "p95_rho_F": stats[k]["F"]["p95"], "max_rho_F": stats[k]["F"]["max"],
            "P_rhoF_gt_1": stats[k]["F"]["frac_amplified"],
            "median_rho_C": stats[k]["C"]["median"], "max_rho_C": stats[k]["C"]["max"],
            "P_rhoC_gt_1": stats[k]["C"]["frac_amplified"],
        }

    metrics = {
        "stage": "I -- learning the smoothing property",
        "level": args.level,
        "n_dofs": int(n), "n_coarse": int(coarse.n_c),
        "complement_dim": int(n - coarse.n_c),
        "coarse_operator": "Galerkin A_c = P^T A P",
        "loss": "L_smooth = mean_i ||e_F - B(A e_F, X)||_A^2 / ||e_F||_A^2",
        "training_inputs": "complement component ONLY; e_C is never used in training",
        "headline_table": {k: row(k) for k in TABLE_ORDER},
        "reference_exact_solve": row("exact_solve"),
        "ridge_diagnostic": ({k: v for k, v in ridge_info.items()
                              if k not in ("W", "predict")}
                             if not args.no_ridge else {"skipped": True}),
        "ridge_table": row("ridge") if "ridge" in stats else None,
        "split_sizes": {r: int(splits[r].E.shape[0]) for r in splits},
        "split_frac_F": {r: {"min": float(splits[r].frac_F.min()),
                             "mean": float(splits[r].frac_F.mean()),
                             "max": float(splits[r].frac_F.max())} for r in splits},
        "projector_check": checks,
        "training_residual_in_ker_Pt_max": float(leak.max()),
        "coarse_residual_Pt_min": float(inrange.min()),
        "trunk": {"mode": args.trunk_features, "modes": args.trunk_modes, "p": args.p,
                  "dim": int(features_np.shape[1]), "description": trunk_desc,
                  "frozen": frozen_T is not None,
                  "span_is_Rn": bool(frozen_T is not None and rank_T == n),
                  "rank_T": rank_T, "best_possible_mu_F": trunk_floor_val,
                  "note": "floor on mu_F for ANY branch against this trunk; the trunk "
                          "span can be the binding constraint, not the net"},
        "model": {"p": args.p, "branch_width": args.width, "depth": args.depth,
                  "base_smoother": args.base_smoother, "parameters": int(n_par),
                  "homogeneous_in_r": True,
                  "normalisation": "branch sees r/||r||_2; output rescaled by ||r||_2",
                  "branch_width_vs_complement": {
                      "branch_width": args.width, "complement_dim": int(n - coarse.n_c),
                      "can_represent_exact_correction": bool(args.width >= n - coarse.n_c)}},
        "training": {"epochs_requested": args.epochs,
                     "epochs_run": res["best"]["epochs_run"],
                     "best_epoch": res["best"]["epoch"],
                     "best_val_mu_F": res["best"]["val_mu_F"],
                     "batch": args.batch, "lr": args.lr, "seed": args.seed,
                     "model_init_seeded_before_construction": True,
                     "wall_seconds": train_wall},
        "damped_jacobi_omega": omega_info,
        "per_mechanism_mu_F": per_mech,
        "scope": {
            "one_smoothing_step": True,
            "no_coarse_grid_correction_in_loss": True,
            "no_vcycle": "not in this stage",
            "bound_to_one_dof_count_and_ordering": True,
            "norms": "v^T A v on coefficient vectors; M_h is not exported, so no "
                     "continuous L^2(Gamma) norm is available",
        },
        "wall_seconds": time.perf_counter() - t_start,
        "versions": {"numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "torch": torch.__version__},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    # ---- the table the brief asks for -------------------------------------
    log2("")
    log2("Reduction factors on %d test errors (level %d, n = %d)"
         % (test.E.shape[0], args.level, n))
    log2("%-18s %8s %8s %9s %11s %9s %8s %8s %9s"
         % ("smoother", "mu_F", "mu_C", "max rho_F", "P(rho_F>1)",
            "med rho_F", "sd rho_F", "p95 rho_F", "mu_C/mu_F"))
    log2("-" * 100)
    for k in TABLE_ORDER:
        r = metrics["headline_table"][k]
        log2("%-18s %8.4f %8.4f %9.4f %11.4f %9.4f %8.4f %8.4f %9.2f"
             % (LABEL[k], r["mu_F"], r["mu_C"], r["max_rho_F"], r["P_rhoF_gt_1"],
                r["median_rho_F"], r["std_rho_F"], r["p95_rho_F"],
                r["mu_C"] / max(r["mu_F"], 1e-12)))
    log2("-" * 100)
    for k, lab in (("exact_solve", "A^-1 (reference)"), ("ridge", "ridge (diagnostic)")):
        if k not in stats:
            continue
        r = metrics["reference_exact_solve"] if k == "exact_solve" else metrics["ridge_table"]
        log2("%-18s %8.4f %8.4f %9.4f %11.4f %9.4f %8.4f %8.4f %9.2f"
             % (lab, r["mu_F"], r["mu_C"], r["max_rho_F"], r["P_rhoF_gt_1"],
                r["median_rho_F"], r["std_rho_F"], r["p95_rho_F"],
                r["mu_C"] / max(r["mu_F"], 1e-12)))
    log2("")
    dn = metrics["headline_table"]["deeponet"]
    dj = metrics["headline_table"]["damped_jacobi"]
    log2("Does the DeepONet learn preferential reduction of the complement?")
    log2("  DeepONet      mu_F = %.4f   mu_C = %.4f   ratio mu_C/mu_F = %.2f"
         % (dn["mu_F"], dn["mu_C"], dn["mu_C"] / max(dn["mu_F"], 1e-12)))
    log2("  Damped Jacobi mu_F = %.4f   mu_C = %.4f   ratio mu_C/mu_F = %.2f"
         % (dj["mu_F"], dj["mu_C"], dj["mu_C"] / max(dj["mu_F"], 1e-12)))
    log2("  (ratio > 1 == the complement is reduced more than the coarse component, "
         "i.e. selective)")
    log2("  floor imposed by the trunk's span: mu_F >= %.4f" % trunk_floor_val)
    log2("")
    log2("Wrote %s/{metrics.json,history.csv,run.log,plots/}" % args.out_dir)
    log2("total wall clock: %.1f s" % (time.perf_counter() - t_start))
    run_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
