#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""A DeepONet used as the smoothing step of a geometric multigrid cycle.

What this program does
----------------------
On one fixed multigrid level it learns the residual-to-correction map

    B_theta : r_h  ->  delta_e_h ,        r_h = A_h e_h ,

so that the smoothing step ``e_h <- e_h - B_theta(A_h e_h)`` reduces the
*algebraic* error in the A_h energy norm ``||v||_A^2 = v^T A_h v``.

A_h comes from a deal.II discretisation of the surface (Laplace-Beltrami) problem

    -div_Gamma(kappa grad_Gamma u) + u = f

on a torus, exported by ``src/main.cpp`` (``export_level_data``) and converted by
``tools/convert_level_data.py``.  Row i of A_h and coordinate i of
``dof_coordinates.npy`` refer to the same DoF; the program verifies that rather
than assuming it.

Scope, stated once and precisely
--------------------------------
* The branch network consumes a **fixed-length** residual vector, so the trained
  object is tied to **one DoF count and one DoF ordering**.  It is not a
  discretisation-independent operator and this program makes **no** cross-mesh or
  mesh-refinement generalisation claim.  Using it at another level means
  retraining.  The coarser levels of a V-cycle are smoothed classically for
  exactly this reason -- see ``--vcycle``.
* The network does not learn the infinite-dimensional space H^1(Gamma).  It
  learns a map on one finite element space V_h, from residual *coefficient
  vectors* to correction *coefficient vectors*.
* Errors are **not** split into "high frequency" and "low frequency".  The
  generators produce smooth, multi-scale, localised and purely algebraic errors
  and per-sample mixtures of them, so a sample is usually not a pure member of
  one class.  Where a coarse-space object appears it is the *A_h-orthogonal
  complement of range(P)* -- a precisely defined subspace, not a Fourier band.
* The objective is ``L_CGC + lambda * L_E`` (``--loss-mode cgc``, the default):
  the energy left after one smoothing step **and** one exact coarse-grid
  correction, plus a weighted full-energy term.  ``L_CGC`` is what a two-grid
  smoother exists to reduce, and it is what model selection follows.  The
  denominator of both terms is ``||e||_A^2`` -- never ``||Q e||_A^2``, which
  vanishes for errors in range(P).  Which coarse operator the loss contains is an
  explicit choice (``--coarse-operator``): the **assembled** coarse matrix the
  C++ V-cycle actually uses, or ``P^T A P``, for which the corrector is an exact
  A_h-orthogonal projection.  On a curved surface these differ, the difference is
  measured and reported, and the default is the operator that is really executed.
* A small loss is not evidence of a working smoother on its own.  Evaluation
  therefore reports the reduction per error mechanism *including the worst sample
  and the share of samples that grew*, repeated application of the smoother, the
  reduction after one and after several V-cycles, and the classical smoothers on
  the same test set.  Where a number is only a smoothing result, it is labelled as
  one.
* Any ``coeff_l2``/``coeff_l2_ratio`` number here is the Euclidean norm of a
  **coefficient vector** in R^n.  It is **not** the continuous ``L^2(Gamma)``
  norm: that needs the mass matrix M_h, which the exporter does not write.  The
  exporter writes A_l, the support points, kappa at those points, refinement
  edges and P.  So the honest label is used everywhere, and
  ``metrics.json`` records that no M_h is available.
* A full V-cycle claim needs the whole hierarchy.  When ``level_data/L0..LL``
  are present the program runs a real V-cycle from the exported per-level
  matrices and prolongations and reports the *measured* reduction; when they are
  absent it says so and skips, rather than extrapolating from a two-grid step.

Every number in ``metrics.json`` comes from a run of this program.

Quick start
-----------
    # 1. export from deal.II (degree 1 = linear tent functions, 5 cycles)
    docker run --rm --entrypoint /home/dealii/solver/build/solver \\
        -v "$PWD/level_dumps:/work" -w /work lb-exporter 1 5

    # 2. convert level 3 (1024 DoFs) and its coarser levels
    for L in 0 1 2 3 4; do
      python tools/convert_level_data.py --data-dir level_dumps --level $L \\
          --cycle 4 --out-dir level_data/L$L
    done

    # 3. train and evaluate
    python deeponet_smoother.py --data-dir level_data/L3 --out-dir results/L3

``DEEPONET_SMOOTHER.md`` documents the arguments and every output file.

This supersedes the ``train_deeponet_smoother.py`` prototype, which densified A_h
with ``.toarray()`` (unusable beyond a few thousand DoFs), trained on white noise
only, normalised its coarse-space term by a quantity that vanishes on range(P),
and restricted with the prolongation in the wrong direction.  That file is kept as
a thin compatibility entry point that forwards the old command line here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# 0. Torus geometry -- must match TorusGeometry in src/main.cpp
# ---------------------------------------------------------------------------
R_MAJOR = 2.0
R_MINOR = 1.0


def coords_to_angles(X: np.ndarray) -> np.ndarray:
    """Torus support points (n,3) -> parameter angles (n,2) = (xi, eta).

    Inverts  x = (R + r cos eta) cos xi,  y = r sin eta,
             z = (R + r cos eta) sin xi
    exactly, and is the natural chart for building periodic data: xi and eta are
    both 2*pi-periodic, so any trigonometric polynomial in them is a
    well-defined function on the torus with no seam.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    rho = np.sqrt(x * x + z * z)
    xi = np.arctan2(z, x)
    eta = np.arctan2(y, rho - R_MAJOR)
    return np.stack([xi, eta], axis=1)


def trunk_features(angles: np.ndarray, mode: str, coords: np.ndarray,
                   n_modes: int = 0) -> np.ndarray:
    """Trunk input, i.e. a fixed feature map *of the DoF coordinates*.

    ``trig`` encodes the torus periodicity exactly (sin/cos of the two angles) so
    the trunk has no artificial jump at the xi = +-pi seam, but its basis is
    *smooth*: a network built on it can only produce corrections that are smooth
    in space.  ``fourier`` adds sin/cos of all modes |m|,|n| <= n_modes, which
    gives the trunk the spatial frequencies needed to represent an oscillatory
    correction -- see the note in DEEPONET_SMOOTHER.md on why this matters for a
    smoother specifically.  ``xyz`` is the raw support point and ``angles`` the
    raw angles, both as literally specified in the brief.
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
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# 1. Level data, inspected rather than assumed
# ---------------------------------------------------------------------------
@dataclass
class LevelData:
    A: sp.csr_matrix
    coords: np.ndarray
    coeff: Optional[np.ndarray]
    edge: Optional[np.ndarray]
    P: Optional[sp.csr_matrix]
    angles: np.ndarray
    lam_min: Optional[float] = None
    lam_max: Optional[float] = None

    @property
    def n(self) -> int:
        return self.A.shape[0]


def sparse_spd_report(S: sp.csc_matrix) -> Tuple[bool, Optional[float], Optional[float]]:
    """Decide whether a sparse symmetric matrix is positive definite WITHOUT
    densifying it.

    An earlier version of this file called ``np.linalg.cholesky(S.toarray())``,
    which is fine at 1 024 DoFs (8 MB) and fatal at 16 384 (2.1 GB of dense
    matrix plus an O(n^3) factorisation) -- the exact densification this program
    promises not to do, and it silently made the largest level unrunnable.

    Two sparse tests instead:
      * definiteness from a SuperLU factorisation taken in symmetric mode with no
        pivoting -- its U diagonal holds the pivots, which are all positive
        exactly when S is positive definite;
      * the extreme eigenvalues from sparse Lanczos, for reporting only.
    """
    is_spd = False
    try:
        lu = spla.splu(S.tocsc(), diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        is_spd = bool(np.all(lu.U.diagonal() > 0.0))
    except RuntimeError:
        is_spd = False                     # singular or structurally unsuitable

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

    Shapes, symmetry, diagonal sign, definiteness and finiteness are all checked.
    A failed check is a hard error, not a warning: the whole point of the
    exercise is to build on a matrix whose properties are known.
    """
    A_path = os.path.join(data_dir, "level_matrix.npz")
    X_path = os.path.join(data_dir, "dof_coordinates.npy")
    if not os.path.exists(A_path):
        raise SystemExit("missing %s -- run tools/convert_level_data.py first" % A_path)
    if not os.path.exists(X_path):
        raise SystemExit("missing %s" % X_path)

    A = sp.load_npz(A_path).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    coords = np.load(X_path)

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
    scale = max(float(abs(A).max()), 1.0)
    if asym_max > 1e-9 * scale:
        raise SystemExit("max|A - A^T| = %.3e is too large for a symmetric operator" % asym_max)

    diag = A.diagonal()
    if np.any(diag <= 0.0):
        raise SystemExit("A has a non-positive diagonal entry (min %.3e)" % diag.min())

    # A = D^{1/2} (D^{-1/2} A D^{-1/2}) D^{1/2}, so A is SPD iff the
    # Jacobi-scaled matrix is -- and that one is far better conditioned.  It is
    # tested sparsely; see sparse_spd_report.
    d_is = 1.0 / np.sqrt(diag)
    S = (sp.diags(d_is) @ A @ sp.diags(d_is)).tocsc()
    S = ((S + S.T) * 0.5).tocsc()
    spd_ok, lam_min, lam_max = sparse_spd_report(S)
    if not spd_ok:
        raise SystemExit("A is not positive definite (smallest scaled eigenvalue %s)"
                         % ("%.3e" % lam_min if lam_min is not None else "unavailable"))

    coeff = None
    c_path = os.path.join(data_dir, "coeff_values.npy")
    if os.path.exists(c_path):
        coeff = np.load(c_path)
        if coeff.shape[0] != A.shape[0]:
            raise SystemExit("coeff_values.npy has %d entries, need %d"
                             % (coeff.shape[0], A.shape[0]))

    edge = None
    e_path = os.path.join(data_dir, "refinement_edge.npy")
    if os.path.exists(e_path):
        edge = np.load(e_path)

    P = None
    p_path = os.path.join(data_dir, "prolongation.npz")
    if os.path.exists(p_path):
        P = sp.load_npz(p_path).tocsr()
        if P.shape[0] != A.shape[0]:
            raise SystemExit("prolongation has %d rows, A has %d" % (P.shape[0], A.shape[0]))

    return LevelData(A=A, coords=coords, coeff=coeff, edge=edge, P=P,
                     angles=coords_to_angles(coords), lam_min=lam_min, lam_max=lam_max)


# ---------------------------------------------------------------------------
# 2. Sparse linear algebra in torch
# ---------------------------------------------------------------------------
def to_torch_sparse(M: sp.spmatrix, dtype=torch.float64) -> torch.Tensor:
    M = M.tocoo()
    idx = torch.from_numpy(np.vstack([M.row, M.col]).astype(np.int64))
    val = torch.from_numpy(np.asarray(M.data, dtype=np.float64)).to(dtype)
    return torch.sparse_coo_tensor(idx, val, size=M.shape, dtype=dtype).coalesce()


class SparseApply(torch.autograd.Function):
    """y = M v for a constant sparse M, with the transpose passed in explicitly.

    Hand-written rather than relying on torch's sparse autograd coverage: the
    backward pass of M v is M^T g, and supplying M^T as a constant makes that
    exact for any pattern.  Only the dense operand (an error or a correction)
    needs a gradient; A_h never does.
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, M: torch.Tensor, MT: torch.Tensor) -> torch.Tensor:
        ctx.MT = MT
        return torch.sparse.mm(M, v)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return torch.sparse.mm(ctx.MT, g), None, None


def spmm(M: torch.Tensor, MT: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return SparseApply.apply(v, M, MT)


def A_apply(V: torch.Tensor, A: torch.Tensor, AT: torch.Tensor) -> torch.Tensor:
    """A V for V shaped (B, n) -- batch-major, one sample per row.

    torch's sparse mm wants the dense operand as (n, B), so the transpose is done
    here once instead of being spread over every call site.
    """
    return spmm(A, AT, V.t()).t()


def energy_norm(v: torch.Tensor, A: torch.Tensor, AT: torch.Tensor) -> torch.Tensor:
    """v^T A v for each row of v (B,n) -> (B,).  Sparse throughout: A is never
    densified."""
    return (v * A_apply(v, A, AT)).sum(dim=1)


def coeff_l2(v: torch.Tensor) -> torch.Tensor:
    """Euclidean norm of each coefficient vector (a row of v).  NOT the
    L^2(Gamma) norm -- see the module docstring."""
    return torch.sqrt((v * v).sum(dim=1))


def coarse_operator_matrix(ld_fine: "LevelData", ld_coarse: Optional["LevelData"],
                           mode: str):
    """The coarse operator A_H that the LOSS will contain, as
    ``(M, lam_min, lam_max)``.

    ``galerkin``
        ``P^T A_h P``, built from the exported fine operator and prolongation.
        With this A_H the two-grid propagator is the exact A_h-orthogonal
        projection onto the complement of range(P), which is what the theory
        statement is about.

    ``assembled``
        Level ``l-1``'s own matrix, assembled by deal.II on the coarse mesh.  This
        is the operator ``src/main.cpp`` puts into the V-cycle, so this is the
        choice that makes the trained loss agree with the executed step.  The two
        differ on a curved surface (see ``Hierarchy.galerkin_check``), and the
        difference is reported rather than assumed away.

    The result is symmetrised and its definiteness is verified sparsely before it
    is returned; a non-SPD coarse operator is a hard error, not a warning.
    """
    if mode == "galerkin":
        M = ld_fine.P.T @ ld_fine.A @ ld_fine.P
    elif mode == "assembled":
        if ld_coarse is None:
            raise SystemExit("--coarse-operator assembled needs level l-1 on disk")
        M = ld_coarse.A
    else:
        raise ValueError("unknown coarse operator %r" % mode)
    M = sp.csr_matrix((M + M.T) * 0.5)
    ok, lam_min, lam_max = sparse_spd_report(M.tocsc())
    if not ok:
        raise SystemExit("coarse operator (%s) is not positive definite" % mode)
    return M, lam_min, lam_max


def dense_cholesky_spd(M, dtype, max_dense: int, what: str):
    """Cholesky factor of a SMALL SPD matrix, verified sparsely first.

    The fine A_h is never densified -- that is the property this program promises
    throughout.  The coarse operator is 16..4096 here, and a dense Cholesky factor
    is what makes the coarse solve differentiable: torch has no sparse-LU
    autograd, and the loss needs d/d(correction) through A_H^{-1}.  The size guard
    turns "the machine ran out of memory" into a clear message.
    """
    n = M.shape[0]
    if n > max_dense:
        raise SystemExit(
            "coarse operator %s is %d x %d, above --max-dense-coarse %d "
            "(the differentiable coarse solve would need %.2f GB)"
            % (what, n, n, max_dense, n * n * M.dtype.itemsize / 1e9))
    ok, _, _ = sparse_spd_report(sp.csc_matrix(M))
    if not ok:
        raise SystemExit("coarse operator %s is not positive definite" % what)
    dense = np.asarray(M.todense() if sp.issparse(M) else M)
    return torch.linalg.cholesky(torch.from_numpy(dense).to(dtype))


# ---------------------------------------------------------------------------
# 3. Error generators
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


def draw_sample(rng, mechanism, angles, nyq, n_dof):
    """One error sample.  Returns (values at DoFs, generator params, signature)."""
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


def normalise_and_residualise(vals, A):
    """Scale an error to unit A-energy and form r = A e.

    Explicitly guards the degenerate cases: a vanishing or non-finite draw
    returns None so the caller skips it and counts it, instead of emitting
    inf/nan silently.
    """
    energy = float(vals @ (A @ vals))
    if not np.isfinite(energy) or energy <= 0.0:
        return None, None
    e = vals / np.sqrt(energy)
    return e, A @ e


# ---------------------------------------------------------------------------
# 4. Datasets with provably disjoint splits
# ---------------------------------------------------------------------------
def make_dataset(A, angles, nyq, n_dof, plan, master_seed):
    """``plan`` is a list of (mechanism, count) for one role.

    Each mechanism gets its own RNG derived deterministically from
    (master_seed, mechanism, count), so roles are independent streams, and the
    discrete signatures are recorded so the caller can assert that no base
    function is shared between roles.
    """
    E_list, R_list, mech_list, sig_list, par_list = [], [], [], [], []
    discarded = 0
    for mechanism, count in plan:
        rng = np.random.default_rng(stable_seed(master_seed, "role", mechanism, count))
        got, attempts = 0, 0
        while got < count and attempts < count * 20:
            attempts += 1
            vals, p, sig = draw_sample(rng, mechanism, angles, nyq, n_dof)
            e, r = normalise_and_residualise(vals, A)
            if e is None:
                discarded += 1
                continue
            E_list.append(e)
            R_list.append(r)
            mech_list.append(mechanism)
            sig_list.append(repr(sig))
            par_list.append(json.dumps(p, default=float))
            got += 1
        if got < count:
            raise SystemExit("could not draw %d usable %s samples" % (count, mechanism))
    return {"E": np.array(E_list), "R": np.array(R_list), "mechanism": mech_list,
            "signature": sig_list, "params": par_list, "discarded": discarded}


def split_audit(splits: Dict[str, dict]) -> Dict[str, int]:
    """The check the brief asks for: no base function may appear in two splits
    (a shared signature would mean the same underlying function, possibly
    rescaled, is in both)."""
    seen: Dict[str, str] = {}
    overlaps = 0
    for role, ds in splits.items():
        for sig in ds["signature"]:
            if seen.setdefault(sig, role) != role:
                overlaps += 1
    return {"cross_split_signature_overlaps": overlaps,
            "unique_signatures": len(seen),
            "total_samples": sum(len(d["signature"]) for d in splits.values())}


def to_tensors(ds, dtype):
    return (torch.from_numpy(np.asarray(ds["E"], dtype=np.float64)).to(dtype),
            torch.from_numpy(np.asarray(ds["R"], dtype=np.float64)).to(dtype))


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------
class Branch(nn.Module):
    """Encodes the residual.  ``bias=False`` throughout, so a zero residual maps
    to a zero branch output: an exact discrete solution stays a fixed point of
    the smoothing step by construction, not merely by training."""

    def __init__(self, in_dim, width, p, depth=3):
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

    def __init__(self, in_dim, width, p, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONetSmoother(nn.Module):
    """delta_e = sum_k branch_k(r) * trunk_k(x), evaluated at every DoF at once.

    ``n_dof`` is baked into the first branch layer, which is precisely why the
    model is bound to one DoF count and one DoF ordering.

    ``base_smoother="jacobi"`` adds a diagonal skip connection with a learnable
    coefficient:

        delta_e = omega * D^{-1} r  +  branch(r) . trunk(x)

    so the network only has to learn what a damped Jacobi step leaves behind --
    the standard residual-learning arrangement, and the ablation that answers
    "how much does the learned part actually add?".  The branch biases stay False,
    so delta_e still vanishes at r = 0 either way and an exact discrete solution
    remains a fixed point by construction.

    ``use_coeff`` is the one case where that argument does not carry itself.  The
    branch input is then ``[r, kappa]``, and kappa is non-zero even when r is, so
    a bias-free branch can still return a non-zero correction at r = 0 and the
    fixed point is lost.  The fix is structural rather than a penalty: the
    correction is taken RELATIVE to the branch's response at zero residual,

        delta_e = [ branch(r, kappa) - branch(0, kappa) ] . trunk(x) ,

    which is exactly zero at r = 0 for every kappa, and still lets kappa modulate
    the response.  It costs one extra (cheap) branch evaluation, and only when the
    coefficient input is switched on.  For a single fixed operator, where kappa is
    the same vector for every sample and therefore carries no per-sample
    information, the honest default is to leave it off.
    """

    def __init__(self, n_dof, trunk_dim, p=64, width=128, depth=3, use_coeff=False,
                 base_smoother="none", diag=None, omega_init=2.0 / 3.0):
        super().__init__()
        self.n_dof = n_dof
        self.use_coeff = use_coeff
        self.p = p
        self.base_smoother = base_smoother
        in_dim = n_dof * (2 if use_coeff else 1)
        self.branch = Branch(in_dim, width, p, depth)
        self.trunk = Trunk(trunk_dim, width, p, depth)
        if base_smoother == "jacobi":
            if diag is None:
                raise ValueError("base_smoother='jacobi' needs the matrix diagonal")
            self.register_buffer("dinv", torch.from_numpy(1.0 / np.asarray(diag, dtype=np.float64)))
            self.omega = nn.Parameter(torch.tensor(float(omega_init), dtype=torch.float64))

    def _branch_features(self, residual, coeff):
        c = coeff.expand(residual.shape[0], -1)
        return torch.cat([residual, c], dim=1), torch.cat([torch.zeros_like(residual), c], dim=1)

    def forward(self, residual, features, coeff=None):
        t = self.trunk(features)
        if self.use_coeff:
            if coeff is None:
                raise ValueError("model was built with use_coeff=True")
            b_in, b_zero = self._branch_features(residual, coeff)
            out = (self.branch(b_in) - self.branch(b_zero)) @ t.t()
        else:
            out = self.branch(residual) @ t.t()
        if self.base_smoother == "jacobi":
            out = self.omega.to(out.dtype) * (residual * self.dinv.to(out.dtype)) + out
        return out


# ---------------------------------------------------------------------------
# 6. Losses and metrics
# ---------------------------------------------------------------------------
def energy_loss(correction, errors, A, AT, eps=1e-12):
    """L_E -- relative A-energy after ONE learned smoothing step.

        e_tilde = e - delta_e ,    L_E = mean( ||e_tilde||_A^2 / ||e||_A^2 )

    The denominator is the energy of the ORIGINAL error and ``eps`` only floors
    that denominator; it is deliberately NOT added to the numerator, so a perfect
    correction gives a loss of zero rather than a floor of eps.
    """
    e_tilde = errors - correction
    num = energy_norm(e_tilde, A, AT)
    den = energy_norm(errors, A, AT).clamp_min(eps)
    return (num / den).mean()


def coarse_grid_correction_loss(correction, errors, A, AT, P, PT, A_H_chol, eps=1e-12):
    """L_CGC -- relative A-energy left after ONE learned smoothing step FOLLOWED BY
    one exact coarse-grid correction.  This is the quantity a two-grid smoother
    exists to reduce, and the primary training objective.

        e_tilde = e - delta_e
        e_cgc   = (I - P A_H^{-1} P^T A) e_tilde
        L_CGC   = mean( ||e_cgc||_A^2 / ||e||_A^2 )

    ``P`` maps coarse to fine, ``PT = P^T``, so the coarse residual really is
    ``P^T A e_tilde``: the shape is (n_fine, n_coarse) and the coarse space has
    the SHORTER dimension.  Getting that backwards silently trains a different
    operator.

    ``A_H_chol`` is the Cholesky factor of the coarse operator, built ONCE before
    training (not per batch).  Which operator it is matters -- see
    ``coarse_operator_matrix``:

      * ``A_H = P^T A P`` (Galerkin) makes ``I - P A_H^{-1} P^T A`` the exact
        A_h-orthogonal projection onto the complement of range(P).  On a smooth
        plane and a nested space it is the textbook two-grid propagator.
      * ``A_H`` assembled on the coarse mesh is what the deal.II hierarchy and the
        V-cycle in ``src/main.cpp`` actually use.  On a curved surface it is close
        to, but not equal to, the Galerkin product, so the projection property is
        then only approximate.  When the goal is to optimise the step that is
        actually executed, this is the operator the loss must contain.

    The DENOMINATOR IS ||e||_A^2, NOT ||Q e||_A^2.  The coarse complement of the
    fixed point of this loss is range(P), and ||Q e||_A vanishes for e in range(P)
    -- exactly the samples the loss should call "already fine" -- so normalising
    by it divides by a number that goes to zero on a legitimate part of the space.
    Every stored error here has unit A-energy, so ||e||_A^2 = 1 and the loss is the
    remaining energy itself, comparable across samples and mechanisms.
    """
    remaining = errors - correction                     # (B, n_fine)
    z = spmm(PT, P, spmm(A, AT, remaining.t()))         # P^T A (e - de) -> (n_H, B)
    w = torch.cholesky_solve(z, A_H_chol)               # A_H^{-1} P^T A (e - de)
    after = (remaining.t() - spmm(P, PT, w)).t()        # (B, n_fine)
    return (energy_norm(after, A, AT) / energy_norm(errors, A, AT).clamp_min(eps)).mean()


def total_smoother_loss(correction, errors, A, AT, P, PT, A_H_chol, lambda_energy=0.1):
    """L = L_CGC + lambda * L_E, returning all three so the two terms can be
    logged separately.

    L_CGC is the primary objective: it scores exactly what the two-grid step does.
    L_E is the auxiliary term; it constrains the correction in the directions the
    coarse grid would have removed anyway, which the primary term alone does not
    see (with an exact coarse solve, a correction that is pure range(P) noise
    leaves L_CGC unchanged but changes what the smoother does inside a recursive
    V-cycle, where the coarse solve is not exact).

    ``lambda_energy`` is an EXPERIMENTAL setting, not a derived optimum: run 0.0
    (pure two-grid objective) and a small positive value and compare both terms
    and the measured V-cycle, rather than assuming one wins.
    """
    l_energy = energy_loss(correction, errors, A, AT)
    l_cgc = coarse_grid_correction_loss(correction, errors, A, AT, P, PT, A_H_chol)
    return l_cgc + lambda_energy * l_energy, l_cgc, l_energy


def cgc_ratios_numpy(E, correction, A, P, A_H=None, solve=None, eps=1e-300):
    """||(I - P A_H^{-1} P^T A)(E - correction)||_A / ||E||_A, per sample.

    The evaluation-side counterpart of ``coarse_grid_correction_loss``: plain
    NumPy with a sparse LU coarse solve, so it can score ANY coarse operator
    (Galerkin or assembled) on the test set without densifying anything and
    without autograd.  Used to quantify how much the choice of A_H moves the
    number the loss reports, and to compare smoothing-only reductions against
    smoothing-plus-coarse-correction reductions.  Pass ``solve`` to reuse one
    factorisation across methods; otherwise it is built from ``A_H``.
    """
    if solve is None:
        solve = spla.splu(sp.csc_matrix(A_H)).solve
    R = A @ (E - correction).T                      # (n, B)
    W = solve(np.asarray(P.T @ R))
    after = (E - correction).T - np.asarray(P @ W)
    num = np.einsum("ib,ib->b", after, A @ after)
    den = np.einsum("bi,bi->b", E, (A @ E.T).T)
    return np.sqrt(np.maximum(num, 0.0) / np.maximum(den, eps))


@torch.no_grad()
def evaluate_correction(correction, errors, residuals, A, AT, eps=1e-30):
    e_tilde = errors - correction
    e0 = energy_norm(errors, A, AT)
    e1 = energy_norm(e_tilde, A, AT)
    r_new = residuals - A_apply(correction, A, AT)
    return {"energy_ratio": torch.sqrt((e1 + eps) / (e0 + eps)),
            "coeff_ratio": coeff_l2(e_tilde) / (coeff_l2(errors) + eps),
            "residual_ratio": coeff_l2(r_new) / (coeff_l2(residuals) + eps)}


# ---------------------------------------------------------------------------
# 7. Classical smoothers
# ---------------------------------------------------------------------------
class ClassicalSmoothers:
    """One application of each classical smoother, expressed as a correction.

    Jacobi is a diagonal scaling and is batched; Gauss-Seidel and SSOR are
    sequential triangular solves and are applied one sample at a time.  That
    difference is part of the honest cost comparison, not an implementation
    accident, so it is reported rather than hidden.
    """

    def __init__(self, A: sp.csr_matrix):
        self.A = A.tocsr()
        self.n = A.shape[0]
        self.diag = A.diagonal()
        self.D = sp.diags(self.diag)
        self.L = sp.tril(self.A, k=-1, format="csc")
        self.U = sp.triu(self.A, k=1, format="csc")

    def jacobi(self, r, omega):
        return omega * r / self.diag

    def best_jacobi_omega(self, E, R, omegas=(0.4, 0.5, 0.6, 2.0 / 3.0, 0.7, 0.8, 0.9)):
        """Give the classical method its best shot, and report the sweep."""
        best, table = None, {}
        e0 = np.einsum("bi,bi->b", E, (self.A @ E.T).T)
        for w in omegas:
            t = E - self.jacobi(R, w)
            r = np.sqrt(np.maximum(np.einsum("bi,bi->b", t, (self.A @ t.T).T), 0.0) / e0)
            m = float(np.mean(r))
            table[float(w)] = m
            if best is None or m < best[1]:
                best = (float(w), m)
        return best[0], table

    def gauss_seidel(self, r, omega=1.0):
        M = (self.D + omega * self.L).tocsc()
        return spla.spsolve_triangular(M, r, lower=True)

    def symmetric_gs(self, r):
        y = spla.spsolve_triangular((self.D + self.L).tocsc(), r, lower=True)
        return spla.spsolve_triangular((self.D + self.U).tocsc(), self.diag * y, lower=False)

    def ssor(self, r, omega=1.0):
        """With omega = 1 this is exactly symmetric Gauss-Seidel."""
        y = spla.spsolve_triangular((self.D + omega * self.L).tocsc(), r, lower=True)
        z = spla.spsolve_triangular((self.D + omega * self.U).tocsc(), self.diag * y, lower=False)
        return omega * (2.0 - omega) * z

    def apply_all(self, E, R, jacobi_omega, ssor_omega=1.0):
        out = {k: [] for k in ("jacobi", "gs", "sgs", "ssor")}
        for e, r in zip(E, R):
            out["jacobi"].append(e - self.jacobi(r, jacobi_omega))
            out["gs"].append(e - self.gauss_seidel(r))
            out["sgs"].append(e - self.symmetric_gs(r))
            out["ssor"].append(e - self.ssor(r, ssor_omega))
        return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 8. The exported hierarchy: two-grid step and V-cycle
# ---------------------------------------------------------------------------
class Hierarchy:
    """Real coarse-grid corrections from the exported levels.

    Nothing is synthesised: if a level's files are missing the corresponding
    feature is reported unavailable rather than faked.
    """

    def __init__(self, levels: List[LevelData]):
        self.levels = levels
        self.fine_level = len(levels) - 1
        self._coarse_lu = None
        self._coarsest_lu = None

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def galerkin_check(self, lvl: int) -> Dict[str, float]:
        """Compare the assembled coarse operator with the Galerkin product
        P^T A P.  For a nodal Q1 space with standard prolongation these agree to
        round-off; a large discrepancy would mean P is not the transfer the
        hierarchy was assembled with, and would invalidate every coarse-grid
        claim built on it."""
        A = self.levels[lvl].A
        P = self.levels[lvl].P
        A_H = self.levels[lvl - 1].A
        G = (P.T @ A @ P).tocsr()
        d = abs(G - A_H)
        mx = float(d.max()) if d.nnz else 0.0
        return {"max_abs_diff": mx,
                "rel_max_abs_diff": mx / max(float(abs(A_H).max()), 1e-300),
                "coarse_nnz_assembled": int(A_H.nnz),
                "coarse_nnz_galerkin": int(G.nnz)}

    def _lu(self):
        if self._coarse_lu is None:
            A_H = self.levels[self.fine_level - 1].A
            self._coarse_lu = spla.splu(A_H.tocsc())
        return self._coarse_lu

    def coarse_solve(self, rhs):
        return self._lu().solve(rhs)

    def two_grid_stages(self, e: np.ndarray, smoother_correction: np.ndarray, lvl: int):
        """One two-grid step, returning the error after each stage so the
        composition is measured rather than asserted."""
        A = self.levels[lvl].A
        P = self.levels[lvl].P
        r = A @ (e - smoother_correction)
        e_H = self.coarse_solve(P.T @ r)
        e_smooth = e - smoother_correction
        return {"smooth": e_smooth, "coarse": e_H, "two_grid": e_smooth - P @ e_H}

    def vcycle_apply(self, r, lvl, smooth_fine, pre=1, post=1, omega=2.0 / 3.0,
                     smooth_coarse=None):
        """One V-cycle applied to the RESIDUAL: returns an approximate solution
        (correction) of A_l x = r.

        Written in the correction form -- ``x`` is the unknown, pre- and
        post-smoothing update ``x <- x + S(r - A x)``, and the coarse level
        receives the *restricted residual* and returns a *correction*.  The
        error-propagation form is equivalent but easy to get wrong: handing
        ``P^T r`` to the coarse level as if it were an error silently omits the
        coarse solve, and the cycle then diverges instead of converging.

        ``smooth_fine`` is a residual -> correction callable used at
        ``self.fine_level`` only.  Coarser levels use damped Jacobi by default,
        because a smoother trained on one DoF count and ordering cannot act at
        another level: the concrete consequence of the fixed-length branch.
        ``smooth_coarse`` overrides that choice, which is what a fair comparison
        against the solver's own smoother needs -- ``src/main.cpp`` uses
        symmetric SOR with two steps at EVERY level, not damped Jacobi, so
        measuring a cycle against Jacobi below flatters the learned smoother.
        """
        A = self.levels[lvl].A

        def S(rr):
            if lvl == self.fine_level:
                return smooth_fine(rr)
            if smooth_coarse is not None:
                return smooth_coarse(rr)
            return omega * rr / A.diagonal()

        if lvl == 0:
            if self._coarsest_lu is None:
                self._coarsest_lu = spla.splu(A.tocsc())
            return self._coarsest_lu.solve(r)          # exact solve, 16 DoFs

        x = np.zeros_like(r)
        for _ in range(pre):
            x = x + S(r - A @ x)
        r1 = r - A @ x
        x = x + self.levels[lvl].P @ self.vcycle_apply(
            self.levels[lvl].P.T @ r1, lvl - 1, smooth_fine, pre, post, omega,
            smooth_coarse)
        for _ in range(post):
            x = x + S(r - A @ x)
        return x


# ---------------------------------------------------------------------------
# 9. Training
# ---------------------------------------------------------------------------
def train(model, tr, va, args, A, AT, P, PT, A_H_chol, features, Ctr, Cva, out_dir, log):
    """Train the smoother.

    The objective is ``L_CGC + lambda * L_E`` when ``--loss-mode cgc`` (the
    default) and ``L_E`` alone when ``--loss-mode energy``.  Both terms are logged
    separately at every log point and written to ``history.csv``: a single total
    cannot tell "the smoother got better" from "the coarse correction is carrying
    the run", and those are different claims.

    Model selection follows the objective actually being trained.  Under
    ``--loss-mode energy`` that is the validation A-energy, which is what earlier
    releases reported as ``val_energy``; that column is kept either way, so old
    and new runs stay comparable.
    """
    Etr, Rtr = tr
    Eva, Rva = va
    n_tr = Etr.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(50, args.early_stop // 4))
    g = torch.Generator().manual_seed(args.seed)

    use_cgc = args.loss_mode == "cgc"
    if use_cgc and (P is None or A_H_chol is None):
        raise SystemExit("--loss-mode cgc needs the coarse operator; export P or "
                         "use --loss-mode energy")

    history, best, best_epoch, bad = [], float("inf"), -1, 0
    best_energy = float("inf")
    n_val = min(args.val_batch, Eva.shape[0])

    for epoch in range(args.epochs + 1):
        model.train()
        idx = torch.randperm(n_tr, generator=g)[:args.batch_size]
        Eb, Rb = Etr[idx], Rtr[idx]
        Cb = None if Ctr is None else Ctr
        corr = model(Rb, features, Cb)
        if use_cgc:
            loss, tr_cgc, tr_energy = total_smoother_loss(
                corr, Eb, A, AT, P, PT, A_H_chol, args.lambda_energy)
        else:
            tr_energy = energy_loss(corr, Eb, A, AT)
            tr_cgc, loss = None, tr_energy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                vc = model(Rva[:n_val], features, None if Cva is None else Cva)
                v_energy = float(energy_loss(vc, Eva[:n_val], A, AT))
                if use_cgc:
                    v_cgc = float(coarse_grid_correction_loss(
                        vc, Eva[:n_val], A, AT, P, PT, A_H_chol))
                    vloss = v_cgc + args.lambda_energy * v_energy
                else:
                    v_cgc, vloss = None, v_energy
            sched.step(vloss)
            row = {"epoch": epoch, "train_loss": float(loss.item()),
                   "train_cgc": None if tr_cgc is None else float(tr_cgc.item()),
                   "train_energy": float(tr_energy.item()),
                   "val_total": vloss, "val_cgc": v_cgc, "val_energy": v_energy,
                   "lr": float(opt.param_groups[0]["lr"])}
            history.append(row)
            # Always report at --log-every: a multi-thousand-epoch run that prints
            # nothing until it finishes looks hung.
            if use_cgc:
                log("  epoch %6d  total %.6f (cgc %.6f + %.2f*energy %.6f)  val %.6f "
                    "(cgc %.6f)  lr %.2e"
                    % (epoch, loss.item(), tr_cgc.item(), args.lambda_energy,
                       tr_energy.item(), vloss, v_cgc, opt.param_groups[0]["lr"]))
            else:
                log("  epoch %6d  train %.6f  val %.6f  lr %.2e"
                    % (epoch, loss.item(), vloss, opt.param_groups[0]["lr"]))
            if vloss < best - args.min_delta:
                best, best_epoch, bad = vloss, epoch, 0
                best_energy = v_energy
                torch.save({"state_dict": model.state_dict(), "config": vars(args),
                            "epoch": epoch, "val_total": vloss, "val_cgc": v_cgc,
                            "val_energy": v_energy},
                           os.path.join(out_dir, "best_model.pt"))
            else:
                bad += 1
                if bad >= args.early_stop:
                    log("  early stop at epoch %d (best %d, val %.6f)" % (epoch, best_epoch, best))
                    break
    return history, best, best_epoch, best_energy


# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
def plot_history(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(ep, [h["train_loss"] for h in history], label="train (objective)")
    ax.semilogy(ep, [h["val_energy"] for h in history], label=r"val $L_E$ (A-energy)")
    if any(h.get("val_cgc") is not None for h in history):
        # The two terms separately: the primary objective and the auxiliary one,
        # because their ratio is what says which mechanism is doing the work.
        ax.semilogy(ep, [h["val_cgc"] for h in history],
                    label=r"val $L_{CGC}$ (after coarse correction)")
        ax.semilogy(ep, [h["train_cgc"] for h in history], ls=":", label=r"train $L_{CGC}$")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"relative squared A-energy")
    ax.set_title("DeepONet smoother training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_reduction_by_mechanism(per_mech, methods, path):
    mechs = [m for m in per_mech if m not in ("mixed", "ALL")] + \
            ([m for m in ("mixed", "ALL") if m in per_mech])
    x = np.arange(len(mechs))
    w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, m in enumerate(methods):
        ax.bar(x + i * w, [per_mech[k][m]["energy_ratio_mean"] for k in mechs], w, label=m)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(mechs, rotation=20)
    ax.set_ylabel(r"mean $\|e-\delta e\|_A/\|e\|_A$")
    ax.set_title("A-energy reduction after one smoother application (lower is better)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_histograms(ratios_by_mech, path):
    mechs = sorted(ratios_by_mech)
    if not mechs:
        return
    fig, axes = plt.subplots(1, len(mechs), figsize=(3.0 * len(mechs), 3.2), sharey=True)
    if len(mechs) == 1:
        axes = [axes]
    for ax, m in zip(axes, mechs):
        ax.hist(ratios_by_mech[m], bins=24, range=(0.0, 1.05), alpha=0.85)
        ax.axvline(1.0, color="k", lw=1, ls="--")
        ax.set_title(m, fontsize=10)
        ax.set_xlabel(r"$\|e-\delta e\|_A/\|e\|_A$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Distribution of the energy reduction ratio (DeepONet, test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_error_fields(angles, e, e_after, mechanism, path, n_show=3):
    """Representative before/after error fields in the (xi, eta) chart.

    Drawn as a scatter over the DoF support points, so it is independent of
    deal.II's DoF numbering and makes no assumption about it.
    """
    n_show = min(n_show, e.shape[0])
    fig, axes = plt.subplots(n_show, 3, figsize=(10.5, 3.1 * n_show), squeeze=False)
    vmax = max(float(np.abs(e[:n_show]).max()), 1e-12)
    for i in range(n_show):
        corr = e[i] - e_after[i]
        for j, (vals, title) in enumerate([(e[i], r"error $e$"),
                                           (e_after[i], r"after: $e-\delta e$"),
                                           (corr, r"correction $\delta e$")]):
            sc = axes[i][j].scatter(angles[:, 0], angles[:, 1], c=vals, s=16,
                                    cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[i][j].set_title("%s -- %s" % (title, mechanism), fontsize=9)
            axes[i][j].set_aspect("equal")
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            plt.colorbar(sc, ax=axes[i][j], fraction=0.046)
    fig.suptitle(r"Error on the torus in the $(\xi,\eta)$ chart")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Driver
# ---------------------------------------------------------------------------
def sha256_16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Train a DeepONet as a multigrid smoother on one FEM level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", default="level_data/L3")
    ap.add_argument("--hierarchy-root", default=None,
                    help="dir holding L0..LL for the coarse-grid analysis "
                         "(default: the parent of --data-dir)")
    ap.add_argument("--out-dir", default="results/smoother")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=250,
                    help="split over the four mechanisms plus a mixed group")
    ap.add_argument("--train-mechanism", default="mixed",
                    help="mechanism used for the training split (default: a per-sample "
                         "mixture of all four)")
    ap.add_argument("--reuse-data", action="store_true",
                    help="reload dataset_*.npz from --out-dir instead of regenerating")
    ap.add_argument("--width", type=int, default=128, help="hidden width of branch and trunk")
    ap.add_argument("--p", type=int, default=64,
                    help="branch/trunk output width. This BOUNDS THE RANK of the "
                         "correction: delta_e lies in the p-dimensional row space of the "
                         "trunk regardless of r, so p must be comparable to n_dof for the "
                         "model to represent a general smoother. See DEEPONET_SMOOTHER.md")
    ap.add_argument("--depth", type=int, default=3, help="hidden layers in branch and trunk")
    ap.add_argument("--trunk-features", choices=["trig", "fourier", "xyz", "angles"],
                    default="trig",
                    help="trig encodes the torus periodicity exactly; fourier adds all "
                         "modes up to --trunk-modes; xyz is the raw support point")
    ap.add_argument("--trunk-modes", type=int, default=4,
                    help="max |m|,|n| for --trunk-features fourier")
    ap.add_argument("--use-coefficient", action="store_true")
    ap.add_argument("--base-smoother", choices=["none", "jacobi"], default="none",
                    help="jacobi adds a learnable diagonal skip omega*D^-1 r, so the "
                         "network learns only the increment over a classical step")
    ap.add_argument("--omega-init", type=float, default=2.0 / 3.0)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--early-stop", type=int, default=400)
    ap.add_argument("--min-delta", type=float, default=1e-6)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--loss-mode", choices=["cgc", "energy"], default="cgc",
                    help="cgc: L_CGC + lambda*L_E, where L_CGC is the A-energy left "
                         "after one smoothing step AND one coarse-grid correction; "
                         "energy: L_E alone (the earlier objective)")
    ap.add_argument("--lambda-energy", type=float, default=0.1,
                    help="weight of the auxiliary full-energy term under "
                         "--loss-mode cgc. EXPERIMENTAL, not derived: compare 0.0 "
                         "against a small positive value and report both terms")
    ap.add_argument("--coarse-operator", choices=["assembled", "galerkin"],
                    default="assembled",
                    help="coarse operator inside the loss. assembled = level l-1's "
                         "own matrix, the one the C++ V-cycle uses; galerkin = P^T A P, "
                         "for which the two-grid propagator is an exact A-orthogonal "
                         "projection. The two differ on a curved surface")
    ap.add_argument("--max-dense-coarse", type=int, default=4096,
                    help="refuse to densify a coarse operator larger than this")
    ap.add_argument("--coarse-loss-weight", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--smoother-iterations", type=int, default=5,
                    help="repeated applications in the stability check: a smoother is "
                         "used inside an iteration, not once")
    ap.add_argument("--n-coarse-eval", type=int, default=64,
                    help="test samples used for the two-grid / V-cycle analysis")
    ap.add_argument("--vcycle", action="store_true")
    ap.add_argument("--inspect-only", action="store_true",
                    help="validate the level data and stop")
    ap.add_argument("--gen-only", action="store_true",
                    help="generate and save the datasets, then stop")
    args = ap.parse_args(argv)

    if args.coarse_loss_weight is not None:
        raise SystemExit(
            "--coarse-loss-weight is gone.  It weighted the old total\n"
            "    L = L_E + w * L_old-coarse-complement ,\n"
            "which is the OPPOSITE arrangement to the intended objective and used a\n"
            "denominator that vanishes on range(P).  The objective is now\n"
            "    L = L_CGC + lambda * L_E ,\n"
            "so pass --loss-mode cgc --lambda-energy <value> instead "
            "(see --help).")

    torch.set_num_threads(args.threads)
    try:                       # the sparse-invariant notice is noise for a fixed pattern
        torch.sparse.check_sparse_tensor_invariants.disable()
    except Exception:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    os.makedirs(args.out_dir, exist_ok=True)
    pdir = os.path.join(args.out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    log_fh = open(os.path.join(args.out_dir, "run.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    t_start = time.perf_counter()
    log("=" * 76)
    log("DeepONet smoother -- surface (Laplace-Beltrami) multigrid level")
    log("=" * 76)
    log("SCOPE: the branch consumes a fixed-length residual, so this model is bound")
    log("       to ONE DoF count and ONE DoF ordering -- no cross-mesh claim is made.")
    log("       Coefficient-space l^2 is NOT L^2(Gamma): no mass matrix is exported.")

    # ---- load and inspect ------------------------------------------------
    # load_level has already verified SPD sparsely and the spectrum bounds come
    # with it; recomputing them here would densify an n x n matrix, which is what
    # made the 16 384-DoF level unrunnable before.
    ld = load_level(args.data_dir, verbose=False)
    n_dof = ld.n
    asym = abs(ld.A - ld.A.T)
    log("")
    log("Level data: %s" % args.data_dir)
    log("  A           : %d x %d, nnz %d (%.2f/row)" % (ld.A.shape[0], ld.A.shape[1],
                                                       ld.A.nnz, ld.A.nnz / n_dof))
    log("  symmetry    : max|A - A^T| = %.3e -> symmetric" % (float(asym.max()) if asym.nnz else 0.0))
    if ld.lam_min is not None and ld.lam_max is not None:
        log("  definiteness: Jacobi-scaled spectrum [%.4e, %.4e], cond %.2e -> SPD "
            "(sparse factorisation + Lanczos, no densification)"
            % (ld.lam_min, ld.lam_max, ld.lam_max / ld.lam_min))
    else:
        log("  definiteness: SPD (verified by sparse factorisation; spectrum bounds "
            "unavailable)")
    log("  coordinates : %r, |x| in [%.4f, %.4f]"
        % (ld.coords.shape, np.linalg.norm(ld.coords, axis=1).min(),
           np.linalg.norm(ld.coords, axis=1).max()))
    if ld.coeff is not None:
        log("  kappa       : [%.4f, %.4f]" % (ld.coeff.min(), ld.coeff.max()))
    if ld.edge is not None:
        log("  ref. edge   : %d constrained DoFs%s" % (ld.edge.size,
            " (uniform mesh)" if ld.edge.size == 0 else ""))
    log("  prolongation: %s" % ("%r" % (ld.P.shape,) if ld.P is not None else "not exported"))
    log("  MASS MATRIX : not exported -> L^2(Gamma) norms are unavailable; "
        "coefficient l^2 is reported instead")
    if args.inspect_only:
        log_fh.close()
        return 0

    # The DoFs form a square lattice in the parameter plane.  Two numbers matter
    # and must not be confused: n_side is how many DoFs there are per direction,
    # nyq is the largest wavenumber the lattice can represent, which is n_side/2.
    # Capping generated wavenumbers at a fraction of n_side rather than of
    # n_side/2 would alias every mode above the true Nyquist: cos(k xi) sampled
    # on the lattice is indistinguishable from cos((n_side - k) xi), so the
    # recorded generator parameters would describe a different function than the
    # one actually sampled.
    n_side = max(2, int(round(np.sqrt(n_dof))))
    nyq = max(1, n_side // 2)
    angles = ld.angles
    features_np = trunk_features(angles, args.trunk_features, ld.coords, args.trunk_modes)
    log("  DoFs %d -> %dx%d periodic lattice, Nyquist = %d modes/direction"
        % (n_dof, n_side, n_side, nyq))
    log("  trunk features: %s -> dim %d" % (args.trunk_features, features_np.shape[1]))

    A_t = to_torch_sparse(ld.A, dtype)
    AT_t = to_torch_sparse(ld.A.T.tocsr(), dtype)
    features = torch.from_numpy(features_np).to(dtype)
    C_all = torch.from_numpy(np.asarray(ld.coeff, dtype=np.float64)).to(dtype) \
        if (args.use_coefficient and ld.coeff is not None) else None
    if args.use_coefficient and ld.coeff is None:
        raise SystemExit("--use-coefficient needs coeff_values.npy")

    # ---- hierarchy -------------------------------------------------------
    hroot = args.hierarchy_root or os.path.dirname(os.path.abspath(args.data_dir))
    base = os.path.basename(os.path.abspath(args.data_dir))
    level_idx = int(base[1:]) if (base.startswith("L") and base[1:].isdigit()) else None
    levels: List[LevelData] = []
    if level_idx is not None:
        for l in range(level_idx + 1):
            d = os.path.join(hroot, "L%d" % l)
            if not os.path.exists(os.path.join(d, "level_matrix.npz")):
                levels = []
                break
            levels.append(load_level(d, verbose=False))
    hierarchy = Hierarchy(levels) if levels else None
    galerkin = None
    log("")
    if hierarchy is not None and level_idx > 0:
        log("Hierarchy: %d levels, DoFs %s" % (hierarchy.n_levels, [lv.n for lv in levels]))
        log("  Galerkin check per level: P^T A_l P versus the separately assembled A_{l-1}")
        galerkin_all = {}
        for l in range(1, hierarchy.n_levels):
            g = hierarchy.galerkin_check(l)
            galerkin_all["level_%d" % l] = g
            log("    level %d -> %d : max|P^T A P - A_H| = %.3e  (relative %.3e)"
                % (l, l - 1, g["max_abs_diff"], g["rel_max_abs_diff"]))
        galerkin = galerkin_all["level_%d" % level_idx]
        log("  A nonzero difference is expected and is reported rather than hidden: on a")
        log("  curved surface the coarse matrix is assembled with the isoparametric")
        log("  mapping, and a coarse basis function is not reproduced exactly by the fine")
        log("  cells' maps, so the hierarchy is not exactly Galerkin.  The consequence")
        log("  for the loss is spelled out below: which A_H it contains is a choice, and")
        log("  the two choices do not give the same operator.")
    else:
        log("Hierarchy: unavailable (need %s/L0..L%d) -- two-grid and V-cycle "
            "analysis will be SKIPPED, not approximated" % (hroot, level_idx or 0))

    # ---- the coarse operator that goes INTO the loss ---------------------
    # Consistency with the C++ cycle, checked rather than assumed:
    #   * src/main.cpp uses MGTransferPrebuilt, whose restriction is the
    #     transpose of the prolongation, so R = P^T exactly;
    #   * it applies no interface/edge matrices unless the hierarchy has
    #     refinement-edge DoFs -- this exported hierarchy is uniformly refined
    #     and reports 0 of them, so the coarse-grid correction it executes is
    #     I - P A_H^{-1} P^T A_h with A_H the ASSEMBLED coarse matrix.
    # Therefore "assembled" is the default: the loss then scores the step that is
    # actually executed.  "galerkin" (P^T A P) is kept available because it is the
    # operator for which the correction is an exact A-orthogonal projection, and
    # the difference between the two is measured and reported.
    P_t = PT_t = A_H_chol = None
    A_H_used = A_H_used_report = None
    if hierarchy is not None and level_idx and level_idx > 0 and ld.P is not None:
        P_t = to_torch_sparse(ld.P, dtype)
        PT_t = to_torch_sparse(ld.P.T.tocsr(), dtype)
        ld_coarse = levels[level_idx - 1]
        n_edge = 0 if ld.edge is None else int(np.asarray(ld.edge).size)
        log("")
        log("Loss target -- the coarse-grid correction the loss simulates:")
        log("  restriction      : R = P^T (MGTransferPrebuilt in src/main.cpp), "
            "P is %d x %d (fine x coarse)" % ld.P.shape)
        log("  interface/edge   : %d refinement-edge DoFs -> %s"
            % (n_edge, "no edge matrices are applied" if n_edge == 0
               else "edge matrices ARE applied, so the executed correction is NOT "
                    "exactly I - P A_H^{-1} P^T A_h"))
        A_H_used, lam_min, lam_max = coarse_operator_matrix(ld, ld_coarse,
                                                            args.coarse_operator)
        A_H_chol = dense_cholesky_spd(A_H_used, dtype, args.max_dense_coarse,
                                      args.coarse_operator)
        log("  coarse operator  : %s (%d x %d, nnz %d), spectrum [%.3e, %.3e]"
            % (args.coarse_operator, A_H_used.shape[0], A_H_used.shape[1],
               A_H_used.nnz, lam_min, lam_max))
        log("  matches the C++ V-cycle operator: %s"
            % ("YES (assembled A_H)" if args.coarse_operator == "assembled" else
               "NO -- P^T A P is used instead; the two-grid step the V-cycle runs "
               "uses the assembled A_H"))
        # Only the COARSE operator is densified: it is small (16..4096 here) and a
        # Cholesky factor is the cleanest differentiable coarse solve.  The fine
        # A_h stays sparse throughout -- that is the requirement that matters.
        log("  densified for the differentiable coarse solve: %.2f MB; the fine A_h "
            "remains sparse throughout"
            % (A_H_chol.numel() * A_H_chol.element_size() / 1e6))
        A_H_used_report = A_H_used

    if args.loss_mode == "cgc" and A_H_chol is None:
        raise SystemExit(
            "--loss-mode cgc (the default) needs the coarse operator, i.e. "
            "prolongation.npz at %s and levels L0..L%d under %s.  Without them the "
            "two-grid objective cannot be formed; re-run with --loss-mode energy to "
            "train on the full-energy objective alone."
            % (args.data_dir, (level_idx or 0), hroot))

    # ---- data ------------------------------------------------------------
    log("")
    have = all(os.path.exists(os.path.join(args.out_dir, "dataset_%s.npz" % s))
               for s in ("train", "val", "test"))
    if args.reuse_data and have:
        splits = {}
        for s in ("train", "val", "test"):
            z = np.load(os.path.join(args.out_dir, "dataset_%s.npz" % s))
            splits[s] = {k: z[k] for k in z.files}
        log("Datasets reloaded from %s" % args.out_dir)
    else:
        log("Generating errors. train/val/test use independent seeds and have")
        log("DISJOINT discrete signatures, so no base function -- nor a rescaled")
        log("copy of one -- can appear in two splits.")
        q = max(1, args.n_test // 5)
        plans = {
            "train": [(args.train_mechanism, args.n_train)],
            "val": [("mixed", args.n_val)],
            "test": [(m, q) for m in MECHANISMS] + [("mixed", q)],
        }
        splits = {role: make_dataset(ld.A, angles, nyq, n_dof, plan, args.seed)
                  for role, plan in plans.items()}
        for role, ds in splits.items():
            np.savez_compressed(
                os.path.join(args.out_dir, "dataset_%s.npz" % role),
                E=ds["E"].astype(np.float32), R=ds["R"].astype(np.float32),
                mechanism=np.array(ds["mechanism"]),
                signature=np.array(ds["signature"]),
                params=np.array(ds["params"]))

    audit = split_audit(splits)
    log("  train %d | val %d | test %d"
        % tuple(len(splits[s]["mechanism"]) for s in ("train", "val", "test")))
    log("  cross-split signature overlaps: %d (must be 0)" % audit["cross_split_signature_overlaps"])
    if audit["cross_split_signature_overlaps"] != 0:
        raise SystemExit("train/val/test share a base function -- refusing to train")
    for mech in MECHANISMS + ("mixed",):
        c = int(np.sum(np.array(splits["test"]["mechanism"]) == mech))
        if c:
            log("    test[%-11s] %4d" % (mech, c))
    if args.gen_only:
        log_fh.close()
        return 0

    tr = to_tensors(splits["train"], dtype)
    va = to_tensors(splits["val"], dtype)
    te = to_tensors(splits["test"], dtype)
    Ctr = C_all if args.use_coefficient else None
    Cva = C_all if args.use_coefficient else None

    # ---- model -----------------------------------------------------------
    model = DeepONetSmoother(n_dof, features_np.shape[1], args.p, args.width, args.depth,
                             args.use_coefficient, args.base_smoother,
                             ld.A.diagonal(), args.omega_init).to(dtype)
    n_par = sum(p.numel() for p in model.parameters())
    log("")
    log("Model: DeepONet, branch %d -> %d, trunk %d -> %d, depth %d"
        % (n_dof * (2 if args.use_coefficient else 1), args.p,
           features_np.shape[1], args.p, args.depth))
    log("  parameters %d | dtype %s | threads %d | trunk %s"
        % (n_par, args.dtype, args.threads, args.trunk_features))
    if args.use_coefficient:
        log("  coefficient input ON: the branch sees [r, kappa], and the correction is")
        log("  taken relative to its r=0 response, so delta_e(0, kappa) = 0 holds by")
        log("  construction rather than by hope.")
        # What matters is not whether kappa varies over the mesh -- it does -- but
        # whether it varies over SAMPLES.  Here every sample is a vector on the
        # same level with the same operator, so the same kappa vector is fed in
        # every time: it is a constant input, and a constant input carries no
        # information about which sample this is.
        log("  NOTE: every sample in this program shares one A_h and hence one kappa,")
        log("        so kappa is a CONSTANT input: it widens the branch's first layer")
        log("        from %d to %d inputs (%d extra weights) and distinguishes nothing"
            % (n_dof, 2 * n_dof, n_dof * args.width))
        log("        between samples.  Leave it off for the fixed-operator study; a")
        log("        coefficient-family experiment (several kappa, therefore several")
        log("        matrices) is what would justify it.")

    # ---- train -----------------------------------------------------------
    t0 = time.perf_counter()
    history, best_val, best_epoch, best_val_energy = train(
        model, tr, va, args, A_t, AT_t, P_t, PT_t,
        A_H_chol, features, Ctr, Cva, args.out_dir, log)
    train_seconds = time.perf_counter() - t0
    log("  training %.1f s (best epoch %d, val %.6f)" % (train_seconds, best_epoch, best_val))
    with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "train_cgc",
                                           "train_energy", "val_total", "val_cgc",
                                           "val_energy", "lr"])
        w.writeheader()
        w.writerows(history)
    plot_history(history, os.path.join(pdir, "loss_curves.png"))

    ck = torch.load(os.path.join(args.out_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    learned_omega = None
    if args.base_smoother == "jacobi":
        learned_omega = float(model.omega)
        log("  learned diagonal skip: omega = %.5f (initial %.5f)"
            % (learned_omega, args.omega_init))

    # ---- evaluate --------------------------------------------------------
    Ete, Rte = te
    with torch.no_grad():
        corr_model = model(Rte, features, C_all if args.use_coefficient else None)
    E_np, R_np = Ete.numpy(), Rte.numpy()
    e0 = np.einsum("bi,bi->b", E_np, (ld.A @ E_np.T).T)

    smoothers = ClassicalSmoothers(ld.A)
    jac_w, jac_table = smoothers.best_jacobi_omega(E_np, R_np)
    log("")
    log("Classical smoother reference: best damped Jacobi omega = %.3f (mean ratio %.5f)"
        % (jac_w, jac_table[jac_w]))
    log("  omega sweep: %s" % {k: round(v, 5) for k, v in jac_table.items()})

    t0 = time.perf_counter()
    cls = smoothers.apply_all(E_np, R_np, jac_w, 1.0)
    classical_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.no_grad():
        model(Rte, features, C_all if args.use_coefficient else None)
    model_seconds = time.perf_counter() - t0

    e_after_map = {
        "DeepONet": (Ete - corr_model).numpy(),
        "damped Jacobi": cls["jacobi"],
        "Gauss-Seidel": cls["gs"],
        "sym. Gauss-Seidel": cls["sgs"],
        "SSOR": cls["ssor"],
    }
    methods = {}
    for name, e_after in e_after_map.items():
        e1 = np.einsum("bi,bi->b", e_after, (ld.A @ e_after.T).T)
        r_new = R_np - (ld.A @ (e_after - E_np).T).T
        methods[name] = {
            "energy_ratio": np.sqrt(np.maximum(e1, 0.0) / np.maximum(e0, 1e-300)),
            "coeff_ratio": np.linalg.norm(e_after, axis=1) / np.linalg.norm(E_np, axis=1),
            "residual_ratio": np.linalg.norm(r_new, axis=1) / np.linalg.norm(R_np, axis=1),
        }

    mech_arr = np.array(splits["test"]["mechanism"])
    per_mech = {}
    for mech in list(MECHANISMS) + ["mixed", "ALL"]:
        mask = np.ones(len(mech_arr), bool) if mech == "ALL" else (mech_arr == mech)
        if not mask.any():
            continue
        per_mech[mech] = {name: {
            "energy_ratio_mean": float(np.mean(m["energy_ratio"][mask])),
            "energy_ratio_median": float(np.median(m["energy_ratio"][mask])),
            "energy_ratio_p90": float(np.percentile(m["energy_ratio"][mask], 90)),
            # A mean over a set hides the samples a smoother makes WORSE.  Report
            # the worst one and the share that grew, per mechanism, so "works on
            # average" cannot be read as "works".
            "energy_ratio_max": float(np.max(m["energy_ratio"][mask])),
            "frac_amplified": float(np.mean(m["energy_ratio"][mask] > 1.0)),
            "coeff_l2_ratio_mean": float(np.mean(m["coeff_ratio"][mask])),
            "residual_ratio_mean": float(np.mean(m["residual_ratio"][mask])),
            "n": int(mask.sum()),
        } for name, m in methods.items()}

    log("")
    log("Mean A-energy reduction over the test set, ||e - de||_A / ||e||_A:")
    log("  %-20s %-8s %-8s %-11s %-10s %-8s" %
        ("method", "ALL", "smooth", "multiscale", "localized", "algebraic"))
    for name in methods:
        log("  %-20s %-8.4f %-8.4f %-11.4f %-10.4f %-8.4f" % (
            name, per_mech["ALL"][name]["energy_ratio_mean"],
            per_mech["smooth"][name]["energy_ratio_mean"],
            per_mech["multiscale"][name]["energy_ratio_mean"],
            per_mech["localized"][name]["energy_ratio_mean"],
            per_mech["algebraic"][name]["energy_ratio_mean"]))

    # ---- what the loss measures, on the real test set --------------------
    # The trained objective is evaluated here on data it never saw, under BOTH
    # coarse operators, so the trainer/target mismatch is a number rather than a
    # caveat.  Smoothing-only reduction (above) and smoothing+coarse-correction
    # reduction (below) are different claims and are reported separately.
    loss_target = {"loss_mode": args.loss_mode, "lambda_energy": args.lambda_energy,
                   "coarse_operator_for_loss": args.coarse_operator if A_H_chol is not None else None,
                   # L_E on the test set, per method: the mean of the SQUARED ratios,
                   # which is the quantity the loss is.  (The per-mechanism table
                   # reports the mean of the ratios instead, which is the usual way
                   # to quote a reduction and is not the same number.)
                   "energy_test": {name: float(np.mean(m["energy_ratio"] ** 2))
                                   for name, m in methods.items()}}
    if hierarchy is not None and level_idx and level_idx > 0 and ld.P is not None:
        ops = {"assembled": levels[level_idx - 1].A,
               "galerkin": (ld.P.T @ ld.A @ ld.P).tocsr()}
        loss_target["cgc_test"] = {}
        for op_name, A_H in ops.items():
            solve = spla.splu(sp.csc_matrix(A_H)).solve
            entry = {}
            for name, e_after_in in e_after_map.items():
                corr_m = E_np - e_after_in            # the correction that method applied
                r = cgc_ratios_numpy(E_np, corr_m, ld.A, ld.P, solve=solve)
                # the ENERGY (mean of squared ratios) is the quantity the loss is,
                # so that is what is stored; the ratio mean is kept for reference
                entry[name] = {"energy_mean": float(np.mean(r ** 2)),
                               "ratio_mean": float(np.mean(r)),
                               "ratio_max": float(np.max(r)),
                               "frac_amplified": float(np.mean(r > 1.0))}
            loss_target["cgc_test"][op_name] = entry
        loss_target["note"] = (
            "cgc_test[op][method] is one smoother application FOLLOWED BY one exact "
            "coarse-grid correction, over the test set: energy_mean is the mean of "
            "||(I - P A_H^-1 P^T A)(e - de)||_A^2 / ||e||_A^2, which is exactly the "
            "quantity L_CGC is the mean of.  'assembled' is the operator the C++ "
            "V-cycle uses, 'galerkin' the one for which the corrector is an exact "
            "A-orthogonal projection; the gap between them is the trainer/target "
            "mismatch, measured rather than assumed away.")
        log("")
        log("Loss target on the test set -- smoothing THEN coarse correction:")
        log("  %-20s %-12s %-14s %-14s" % ("method", "L_E", "L_CGC(asm)", "L_CGC(gal)"))
        for name in e_after_map:
            le = float(np.mean(methods[name]["energy_ratio"] ** 2))
            log("  %-20s %-12.6f %-14.6f %-14.6f"
                % (name, le, loss_target["cgc_test"]["assembled"][name]["energy_mean"],
                   loss_target["cgc_test"]["galerkin"][name]["energy_mean"]))
        log("  (all three columns are energies, i.e. the mean squared ratio, so they are")
        log("   directly comparable; the trainer minimises the 'assembled' column when")
        log("   --coarse-operator assembled, which is the default)")
        dk = "DeepONet"
        log("  worst test sample after smoothing + coarse correction (assembled): "
            "ratio %.4f, %.1f%% of samples amplified"
            % (loss_target["cgc_test"]["assembled"][dk]["ratio_max"],
               100.0 * loss_target["cgc_test"]["assembled"][dk]["frac_amplified"]))

    # ---- is it stable when iterated? -------------------------------------
    # A mean over one application says nothing about repeated use, which is how a
    # smoother is actually employed.  e <- e - B(A e), repeated, is the honest
    # cheap test; a stationary iteration like this is not guaranteed to converge
    # for an arbitrary learned B, so the result is reported, not assumed.
    iterations = {}
    if args.smoother_iterations > 0:
        log("")
        log("Repeated application: e <- e - B(A e), %d iterations (mean energy ratio "
            "per step)" % args.smoother_iterations)
        with torch.no_grad():
            e_it = Ete.clone()
            ratio_hist = []
            for _ in range(args.smoother_iterations):
                r_it = A_apply(e_it, A_t, AT_t)
                e_it = e_it - model(r_it, features, C_all if args.use_coefficient else None)
                ratio_hist.append(
                    torch.sqrt(energy_norm(e_it, A_t, AT_t) /
                               energy_norm(Ete, A_t, AT_t).clamp_min(1e-300)))
        jac_it = E_np.copy()
        jac_hist = []
        for _ in range(args.smoother_iterations):
            # ld.A is (n,n) and jac_it is (B,n): the sparse product needs the
            # (n,k) orientation, hence the transpose on both sides
            jac_it = jac_it - smoothers.jacobi((ld.A @ jac_it.T).T, jac_w)
            jac_hist.append(np.sqrt(np.einsum("bi,bi->b", jac_it, (ld.A @ jac_it.T).T)
                                    / np.maximum(e0, 1e-300)))
        for k in range(args.smoother_iterations):
            iterations["iter_%d" % (k + 1)] = {
                "deeponet_mean": float(ratio_hist[k].mean()),
                "deeponet_max": float(ratio_hist[k].max()),
                "deeponet_frac_amplified": float((ratio_hist[k] > 1.0).double().mean()),
                "jacobi_mean": float(jac_hist[k].mean()),
            }
        log("  %-6s %-12s %-12s %-14s %-12s" % ("iter", "DeepONet", "worst", "frac > 1",
                                                "Jacobi ref"))
        for k in range(args.smoother_iterations):
            d = iterations["iter_%d" % (k + 1)]
            log("  %-6d %-12.5f %-12.5f %-14.4f %-12.5f"
                % (k + 1, d["deeponet_mean"], d["deeponet_max"],
                   d["deeponet_frac_amplified"], d["jacobi_mean"]))
        mono = all(iterations["iter_%d" % (k + 2)]["deeponet_mean"]
                   < iterations["iter_%d" % (k + 1)]["deeponet_mean"]
                   for k in range(args.smoother_iterations - 1))
        iterations["monotone_mean"] = bool(mono)
        log("  mean ratio decreases at every step: %s" % ("yes" if mono else
            "NO -- the iteration stalls or grows; one application is not evidence "
            "of a usable smoother"))
        iterations["final_energy_ratio_mean"] = float(ratio_hist[-1].mean())

    # ---- coarse-grid correction and V-cycle ------------------------------
    twogrid, vcycle = {}, {}
    mg_errors = {}
    if hierarchy is not None and level_idx and level_idx > 0:
        log("")
        log("Two-grid analysis with the REAL P and coarse operator from the "
            "exported hierarchy:")
        n_ce = min(args.n_coarse_eval, E_np.shape[0])
        sel = np.arange(n_ce)
        e_sel = E_np[sel]
        e0_sel = np.einsum("bi,bi->b", e_sel, (ld.A @ e_sel.T).T)
        for name in ("DeepONet", "damped Jacobi", "sym. Gauss-Seidel"):
            smooth_corr = e_sel - e_after_map[name][sel]
            e_sm = np.array([hierarchy.two_grid_stages(e, c, level_idx)["smooth"]
                             for e, c in zip(e_sel, smooth_corr)])
            e_tg = np.array([hierarchy.two_grid_stages(e, c, level_idx)["two_grid"]
                             for e, c in zip(e_sel, smooth_corr)])
            e_co = np.array([hierarchy.two_grid_stages(e, np.zeros_like(e), level_idx)["two_grid"]
                             for e in e_sel])

            def red(x):
                return float(np.mean(np.sqrt(np.maximum(
                    np.einsum("bi,bi->b", x, (ld.A @ x.T).T), 0.0) / e0_sel)))
            twogrid[name] = {"smoother_only": red(e_sm), "coarse_only": red(e_co),
                             "two_grid": red(e_tg)}
            log("  smoother %-18s only %.4f | coarse only %.4f | two-grid %.4f"
                % (name, twogrid[name]["smoother_only"], twogrid[name]["coarse_only"],
                   twogrid[name]["two_grid"]))

        # Cross-check: the loss is computed with a differentiable Cholesky solve,
        # this with the hierarchy's own LU-solve path.  Both are means of squared
        # ratios -- computing one as the square of a mean ratio would make the two
        # differ by a Jensen gap and turn a real agreement into a spurious one.
        if A_H_used_report is not None and args.coarse_operator == "assembled":
            corr_sel = e_sel - e_after_map["DeepONet"][sel]
            loss_side = float(np.mean(cgc_ratios_numpy(
                e_sel, corr_sel, ld.A, ld.P, A_H=A_H_used_report) ** 2))
            e_tg_sel = np.array([hierarchy.two_grid_stages(e, c, level_idx)["two_grid"]
                                 for e, c in zip(e_sel, corr_sel)])
            measured = float(np.mean(
                np.einsum("bi,bi->b", e_tg_sel, (ld.A @ e_tg_sel.T).T) / e0_sel))
            twogrid["loss_vs_measured"] = {
                "loss_L_CGC_selected_samples": loss_side,
                "measured_two_grid_energy_ratio_sq": measured,
                "abs_diff": abs(loss_side - measured),
            }
            log("  consistency: L_CGC on these samples %.10f vs the measured two-grid "
                "energy %.10f (difference %.2e)" % (loss_side, measured,
                                                    abs(loss_side - measured)))
            log("  (the loss and the cycle are the same operator: the loss factorises")
            log("   the coarse operator once with Cholesky, the hierarchy LU-solves it)")

        if args.vcycle:
            log("")
            log("V-cycle over %d exported levels, iterated %d times. The learned "
                "smoother acts at level %d" % (hierarchy.n_levels,
                                               args.smoother_iterations, level_idx))
            log("only; coarser levels use damped Jacobi, because a fixed-length branch")
            log("cannot be applied at another DoF count or ordering.")
            log("A single V-cycle is one step of a stationary iteration; the loss")
            log("covers one smoothing pass plus one EXACT coarse solve, so the loss")
            log("being small does not by itself imply the recursive cycle contracts.")

            def learned_smooth(rr):
                with torch.no_grad():
                    t = torch.from_numpy(rr).to(dtype)[None, :]
                    return model(t, features, C_all if args.use_coefficient else None)[0].numpy()

            def jac_smooth(rr):
                return jac_w * rr / ld.A.diagonal()

            def symgs_smooth(rr):
                # One smoother per LEVEL: a ClassicalSmoothers object holds the
                # factorisation of one matrix, and a smoother is bound to one DoF
                # count -- the same constraint that stops the learned branch acting
                # below the fine level.  Reusing the fine level's object here
                # raises a broadcasting error, not a clear one, so the dictionary
                # is built from the hierarchy rather than from ld.
                return symgs_by_n[rr.shape[0]].symmetric_gs(rr)

            # The baselines include symmetric Gauss-Seidel at EVERY level, twice
            # per side.  That is what src/main.cpp actually runs --
            # mg::SmootherRelaxation<PreconditionSOR> with set_steps(2) and
            # set_symmetric(true), i.e. symmetric SOR at omega = 1 -- so a cycle
            # measured only against damped Jacobi below would be flattering
            # itself.  Both are reported.
            # symmetric Gauss-Seidel per level, for the "everywhere" baselines
            symgs_by_n = {lv.n: ClassicalSmoothers(lv.A) for lv in levels}

            # (tag, fine smoother, coarse smoother, pre, post, omega-for-Jakobi-below)
            # omega is 2/3 for the learned row -- the module's documented default
            # for undamped-below -- so that row reproduces exactly the cycle this
            # program reported before this list grew.
            variants = [
                ("learned@L%d + Jacobi below" % level_idx, learned_smooth, None,
                 1, 1, 2.0 / 3.0),
                ("damped Jacobi everywhere x1", jac_smooth, None, 1, 1, 2.0 / 3.0),
                ("symGS everywhere x1", symgs_smooth, symgs_smooth, 1, 1, 1.0),
                ("symGS everywhere x2 (the C++ smoother)", symgs_smooth, symgs_smooth,
                 2, 2, 1.0),
            ]
            for tag, fn, coarse_fn, pre, post, om in variants:
                per_iter = []
                for e in e_sel[:32]:
                    # V-cycle as a preconditioner, applied repeatedly:
                    # e <- e - M^{-1}(A e).  Ratios are relative to the same
                    # sample's initial energy, so the sequence is the iteration's
                    # actual contraction history, not a per-step ratio.
                    ev = e.copy()
                    hist = []
                    for _ in range(args.smoother_iterations):
                        ev = ev - hierarchy.vcycle_apply(ld.A @ ev, level_idx, fn,
                                                         pre, post, om,
                                                         smooth_coarse=coarse_fn)
                        hist.append(float(np.sqrt(max(ev @ (ld.A @ ev), 0.0) /
                                                  max(e @ (ld.A @ e), 1e-300))))
                    per_iter.append(hist)
                per_iter = np.array(per_iter)                 # (n_samples, n_iter)
                rr = per_iter[:, 0]
                vcycle[tag] = {"mean_reduction": float(rr.mean()),
                               "min_reduction": float(rr.min()),
                               "max_reduction": float(rr.max()), "n": len(rr),
                               "iterations": [float(per_iter[:, k].mean())
                                              for k in range(per_iter.shape[1])],
                               "contraction_per_iteration": [
                                   float(per_iter[:, k].mean() / per_iter[:, k - 1].mean())
                                   for k in range(1, per_iter.shape[1])],
                               "worst_sample_final": float(per_iter[:, -1].max())}
                log("  %-38s first %.4f | after %d: %.5f | worst sample %.4f"
                    % (tag, vcycle[tag]["mean_reduction"], args.smoother_iterations,
                       vcycle[tag]["iterations"][-1], vcycle[tag]["worst_sample_final"]))
                log("  %-38s per-iteration mean: %s"
                    % ("", " ".join("%.4f" % x for x in vcycle[tag]["iterations"])))
            log("  the last two rows are the solver's own smoother; a learned cycle is")
            log("  only interesting if it beats those, not merely the Jacobi row")

            # ---- an error distribution that is not synthetic at all ----------
            # Every generator in section 3 makes errors; these are the errors a
            # real V-cycle leaves behind as it iterates.  No generator produced
            # them and their statistics were never tuned, so they are the closest
            # thing here to an independent test of the smoother in the setting it
            # would actually be used -- and a smoother trained on algebraic noise
            # is under no obligation to do well on them.
            n_mg = min(32, e_sel.shape[0])
            e_mg = e_sel[:n_mg].copy()
            mg_errors = {}
            with torch.no_grad():
                for k in range(args.smoother_iterations):
                    e0_mg = np.einsum("bi,bi->b", e_mg, (ld.A @ e_mg.T).T)
                    Rm = (ld.A @ e_mg.T).T
                    corr_l = model(torch.from_numpy(Rm).to(dtype), features,
                                   C_all if args.use_coefficient else None).numpy()
                    cand = [("DeepONet", e_mg - corr_l),
                            ("damped Jacobi", e_mg - smoothers.jacobi(Rm, jac_w)),
                            ("sym. Gauss-Seidel",
                             e_mg - np.array([smoothers.symmetric_gs(r) for r in Rm]))]
                    stage = {}
                    for nm, e_after in cand:
                        r = np.sqrt(np.maximum(
                            np.einsum("bi,bi->b", e_after, (ld.A @ e_after.T).T), 0.0)
                            / np.maximum(e0_mg, 1e-300))
                        stage[nm] = {"energy_ratio_mean": float(r.mean()),
                                     "energy_ratio_max": float(r.max()),
                                     "frac_amplified": float(np.mean(r > 1.0))}
                    mg_errors["after_%d_vcycles" % k] = {
                        "initial_energy_ratio_vs_test_error": float(np.sqrt(
                            np.mean(e0_mg / e0_sel[:n_mg]))),
                        "methods": stage}
                    # advance the iteration: one V-cycle with damped Jacobi at every
                    # level, i.e. what the multigrid does without the network
                    e_mg = e_mg - np.array([hierarchy.vcycle_apply(ld.A @ e, level_idx,
                                                                  jac_smooth)
                                            for e in e_mg])
            log("")
            log("Independent test on REAL multigrid-iteration errors (not generated):")
            log("  %-22s %-12s %-12s %-12s" % ("error after k V-cycles", "DeepONet",
                                               "dampedJac", "symGS"))
            for k in range(args.smoother_iterations):
                st = mg_errors["after_%d_vcycles" % k]
                log("  k = %-18d %-12.5f %-12.5f %-12.5f"
                    % (k, st["methods"]["DeepONet"]["energy_ratio_mean"],
                       st["methods"]["damped Jacobi"]["energy_ratio_mean"],
                       st["methods"]["sym. Gauss-Seidel"]["energy_ratio_mean"]))
            log("  (mean ||e - de||_A / ||e||_A on the error left by k V-cycles; each row")
            log("   uses that row's own error as the 100% reference.  These errors are")
            log("   smooth by construction of the cycle, so they live largely in the")
            log("   coarse space -- a smoother is not expected to remove what the coarse")
            log("   grid is there to remove, and this table is where that shows.)")

    # ---- plots -----------------------------------------------------------
    plot_reduction_by_mechanism(per_mech, list(methods),
                               os.path.join(pdir, "reduction_by_mechanism.png"))
    ratios = {m: methods["DeepONet"]["energy_ratio"][mech_arr == m]
              for m in MECHANISMS if (mech_arr == m).any()}
    plot_histograms(ratios, os.path.join(pdir, "energy_reduction_hist.png"))
    e_mod = (Ete - corr_model).numpy()
    for mech in MECHANISMS:
        idx = np.where(mech_arr == mech)[0][:3]
        if idx.size:
            plot_error_fields(angles, E_np[idx], e_mod[idx], mech,
                              os.path.join(pdir, "error_fields_%s.png" % mech))

    # ---- report ----------------------------------------------------------
    metrics = {
        "scope": {
            "fixed_dof_count": n_dof,
            "bound_to_dof_ordering": True,
            "cross_mesh_generalisation_claimed": False,
            "continuous_L2_claimed": False,
            "mass_matrix_available": False,
            "coeff_l2_meaning": "Euclidean norm of the coefficient vector, not L^2(Gamma)",
            "frequency_partitioning_used": False,
            "learns_infinite_dimensional_space": False,
        },
        "provenance": {
            "data_dir": os.path.abspath(args.data_dir),
            "A_npz_sha256_16": sha256_16(os.path.join(args.data_dir, "level_matrix.npz")),
            "coords_sha256_16": sha256_16(os.path.join(args.data_dir, "dof_coordinates.npy")),
            "A_shape": list(ld.A.shape), "A_nnz": int(ld.A.nnz),
            "prolongation_present": ld.P is not None,
            "prolongation_shape": list(ld.P.shape) if ld.P is not None else None,
            "n_levels_available": hierarchy.n_levels if hierarchy else 1,
            "refinement_edge_dofs": int(ld.edge.size) if ld.edge is not None else None,
            "kappa_range": [float(ld.coeff.min()), float(ld.coeff.max())] if ld.coeff is not None else None,
            "master_seed": args.seed,
            "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test,
            "train_mechanism": args.train_mechanism,
            "test_mechanisms": list(MECHANISMS) + ["mixed"],
            "nyquist_modes_per_direction": nyq,
            "dofs_per_side": n_side,
            "multiscale_wavenumber_cap": "0.75 * nyquist",
            "split_audit": audit,
            "topology": "torus R=%.1f r=%.1f (matches TorusGeometry in src/main.cpp)"
                        % (R_MAJOR, R_MINOR),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "model": {"parameters": n_par, "width": args.width, "p": args.p,
                  "depth": args.depth, "trunk_features": args.trunk_features,
                  "trunk_modes": args.trunk_modes if args.trunk_features == "fourier" else None,
                  "trunk_dim": int(features_np.shape[1]),
                  "base_smoother": args.base_smoother,
                  "learned_omega": learned_omega,
                  "use_coefficient": args.use_coefficient, "dtype": args.dtype,
                  "zero_residual_fixed_point": (
                      "branch biases are all False" + (
                          " and the skip vanishes at r=0" if args.base_smoother == "jacobi" else "")
                      + ("; the coefficient input is taken relative to the branch's "
                         "response at r=0, so delta_e(0, kappa) = 0 exactly"
                         if args.use_coefficient else "")),
                  "output_rank_bound": "delta_e lies in the row space of the trunk, "
                                       "dimension <= p (%d) against n=%d" % (args.p, n_dof)},
        "training": {"log_points": len(history),
                     "last_epoch": history[-1]["epoch"] if history else None,
                     "best_epoch": best_epoch,
                     "best_val_total": best_val,
                     # kept under its historical name and meaning (validation L_E at
                     # the selected epoch) so runs across the objective change stay
                     # comparable, and so tools/summarize_runs.py keeps working
                     "best_val_energy": best_val_energy,
                     "train_seconds": train_seconds,
                     "objective": ("L_CGC + %.3g * L_E" % args.lambda_energy
                                   if args.loss_mode == "cgc" else "L_E"),
                     "model_selection_metric": ("val_total (L_CGC + lambda*L_E)"
                                                if args.loss_mode == "cgc"
                                                else "val_energy (L_E)"),
                     "note": "val_energy is reported either way, so runs under "
                             "--loss-mode energy and --loss-mode cgc stay comparable "
                             "on the column earlier releases used."},
        "loss_target": loss_target,
        "smoother_iterations": iterations,
        "mg_iteration_errors": mg_errors,
        "smoothers": per_mech,
        "jacobi_omega": {"best": jac_w, "sweep": jac_table},
        "two_grid": twogrid,
        "vcycle": vcycle,
        "timing": {
            "training_seconds": train_seconds,
            "deeponet_inference_seconds_full_test_set": model_seconds,
            "deeponet_seconds_per_sample": model_seconds / max(len(Ete), 1),
            "four_classical_smoothers_seconds_full_test_set": classical_seconds,
            "classical_seconds_per_sample_per_method": classical_seconds / max(len(Ete) * 4, 1),
            "note": "Training time is reported separately from inference. Jacobi is a "
                    "diagonal scaling and is batched; Gauss-Seidel and SSOR are "
                    "sequential triangular solves, so their per-sample cost is "
                    "inherently serial.",
        },
        "hierarchy_galerkin": galerkin,
        "data_notes": {
            "generators": {
                "smooth": "1-2 periodic trigonometric modes, |m|,|n| <= min(2, nyq/4)",
                "multiscale": "3-6 modes, amplitudes ~ |k|^-decay, wavenumbers capped "
                              "below Nyquist so sampling does not alias",
                "localized": "1-2 periodic Gaussian bumps, width 1-3 mesh cells",
                "algebraic": "iid standard-normal coefficients in the FE basis",
                "mixed": "per-sample random convex mixture of the four; each component "
                         "is normalised before mixing so it stays a genuine mixture",
            },
            "normalisation": "every stored error has unit A_h energy; degenerate draws "
                             "are skipped and counted in discarded_draws",
            "discarded_draws": int(splits["train"].get("discarded", 0)
                                   + splits["val"].get("discarded", 0)
                                   + splits["test"].get("discarded", 0))
            if isinstance(splits["train"], dict) and "discarded" in splits["train"] else None,
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    with open(os.path.join(args.out_dir, "splits.json"), "w", encoding="utf-8") as fh:
        json.dump({r: {"mechanism": splits[r]["mechanism"],
                       "signature": splits[r]["signature"],
                       "params": splits[r]["params"]} for r in ("train", "val", "test")},
                  fh, indent=1, default=str)

    log("")
    log("Wrote %s/{metrics.json,history.csv,splits.json,best_model.pt,run.log,plots/}"
        % args.out_dir)
    log("total wall clock: %.1f s" % (time.perf_counter() - t_start))
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
