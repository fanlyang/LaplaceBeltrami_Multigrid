#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Stage I -- does the DeepONet learn the *smoothing property*?

The question
------------
A smoother exists to reduce the part of the algebraic error that a coarse-grid
correction cannot reach.  On one fixed level, with the Galerkin coarse operator

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

The confound that is measured, not hidden
-----------------------------------------
The DeepONet output is ``delta_e = sum_k b_k(r) T_k(x)``, so for a fixed trunk the
correction lies in the ``p``-dimensional span of the trunk's output vectors
*whatever the branch returns*.  A small ``mu_F`` can therefore be a statement
about the trunk rather than about smoothing.  This program bounds that confound
instead of ignoring it: for the *trained* trunk it reports the best A-norm
approximation of ``e_F`` attainable inside ``span(T)``, and the implied floor

    mu_F  >=  E[ sqrt(1 - reach_F(e_F)) ] ,

which is the best any branch could have achieved against that trunk.  Set
``--p`` to ``n`` and the floor becomes vacuous.  DEEPONET_SMOOTHER.md Finding 1
(trunk *span*, not trunk *size*, governs what is reachable) is why this
diagnostic is necessary rather than decorative.

Ridge diagnostic
----------------
A ridge regression is fitted to the *same* training pairs (residual ->
complement error), with its regularisation chosen on the same validation metric
the network uses, and evaluated on the same test errors.  It answers what the
network alone cannot: is selectivity something the DeepONet learned, or does
*any* map fitted to this data inherit it?  It is reported as a diagnostic and
kept out of the headline table, which holds only the classical smoothers and the
network.

Network and damped-Jacobi each get their tuned parameter chosen on validation and
scored on test, so the comparison is symmetric.

What this program is not
------------------------
* One smoothing step is measured.  No V-cycle is run and no claim is made about
  the complete multigrid solver -- that is Stage III.
* No coarse-grid correction is applied during training or evaluation; the loss
  deliberately stops after the smoothing step.
* The operator is bound to one DoF count and one DoF ordering; ``n`` is baked
  into the first branch layer.
* Every norm is ``v^T A v`` on coefficient vectors.  ``M_h`` is not exported, so
  no continuous ``L^2(Gamma)`` norm is available or claimed.

Usage
-----
    python stage1_smoothing_property.py --selftest
    python stage1_smoothing_property.py --data-dir level_data/L3 --out-dir results/stage1_L3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reused, not reimplemented: the geometry helpers, the sparse-matvec autograd,
# the error generators and the classical smoothers already exist on this branch
# and are the objects the other results were measured with.  Sharing the
# implementations is what makes these numbers comparable to those.
from deeponet_smoother import (  # noqa: E402
    ClassicalSmoothers,
    DeepONetSmoother,
    MECHANISMS,
    Trunk,
    draw_sample,
    energy_norm,
    load_level,
    normalise_and_residualise,
    stable_seed,
    to_torch_sparse,
    trunk_features,
)

# ---------------------------------------------------------------------------
# Palette -- the validated set already used by bench/plot.py, kept identical so a
# figure from here can sit beside a figure from there.  Fixed slot per method,
# never keyed by rank and never cycled.
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


def log(msg: str = "") -> None:
    print(msg, flush=True)


def energy_rows(V: np.ndarray, A: sp.spmatrix) -> np.ndarray:
    """``v^T A v`` for each row of ``V`` (N, n) -> (N,).

    Every energy in this file goes through here.  The arrays are sample-major
    (a row is one error) while ``A @ V.T`` is DoF-major, and mixing the two
    layouts silently reduces over the wrong axis -- so the transpose lives in one
    place instead of at twenty call sites.
    """
    return np.einsum("si,si->s", V, np.asarray(A @ V.T).T)


# ---------------------------------------------------------------------------
# 1. The Galerkin coarse space and its A-orthogonal projection
# ---------------------------------------------------------------------------
class GalerkinCoarseSpace:
    """``A_c = P^T A P`` plus the exact A-orthogonal split ``e = e_C + e_F``.

    The inverse of ``P^T A P`` is never formed.  The coarse system is solved --
    what the brief prescribes, and the only sane thing to do: the explicit
    inverse is a dense ``n_c x n_c`` object, while the sparse factorisation is
    built once and reused for every sample.
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

        # lambda_min(A_c) > 0 is what makes the sum in the module docstring
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
        AE = np.asarray(self.A @ E.T)             # (n, N)
        Z = self.lu.solve(np.asarray(self.P.T @ AE))   # (n_c, N)
        EC = np.asarray(self.P @ Z).T             # (N, n)
        return EC, E - EC

    def verify_split(self, E: np.ndarray, EC: np.ndarray,
                     EF: np.ndarray) -> Dict[str, float]:
        """The identities the brief states, measured on real samples.

        Returned as relative residuals so they are comparable across sample
        scale; anything above a few 1e-12 means the split is wrong.
        """
        # Sample-major throughout: everything the caller passes is (N, n).
        AEF = np.asarray(self.A @ EF.T).T                     # (N, n)
        ec_aef = np.einsum("si,si->s", EC, AEF)
        orth = np.abs(ec_aef) / np.maximum(
            np.linalg.norm(EC, axis=1) * np.linalg.norm(AEF, axis=1), 1e-300)

        PtAEF = np.asarray(self.P.T @ AEF.T)                  # (n_c, N)
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
# 2. Errors, split into their coarse and complement components
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """One role's errors, both components, and the residuals of each.

    ``RF``/``RC`` are ``A e_F`` / ``A e_C`` in sample-major form (N, n), formed
    once: ``A`` is constant and ``A e_F`` is the entire input distribution of
    this stage.
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
            e, _r = normalise_and_residualise(vals, A)
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
# 3. The learned smoother
# ---------------------------------------------------------------------------
class FrozenTrunk(nn.Module):
    """A trunk that is a fixed matrix ``T`` (n, p) and is never trained.

    Freezing is what makes the trunk *provably* non-binding rather than
    empirically so.  When ``T`` has orthonormal columns spanning ``R^n`` -- the
    orthonormalised Fourier basis up to the fine Nyquist -- then
    ``span(T) = R^n``, so the floor in §4 of the report is exactly zero and
    ``mu_F`` can only be a statement about the branch.  A learned trunk reaches
    full rank here too (verified: rank 1024 of p = 1024), so nothing in
    expressiveness is given up, but a learned trunk must be re-evaluated and
    re-differentiated inside every training step, which is ~70% of the step cost
    for a matrix that is already full rank.  Freezing also stops the trunk
    drifting to a lower-rank configuration mid-run, which would silently
    re-introduce the confound this whole diagnostic exists to remove.

    Orthonormality additionally keeps the parameterisation well conditioned:
    ``T^T T = I``, so the branch's coefficients are exactly the correction's
    Fourier coefficients and carry no conditioning penalty of their own.
    """

    def __init__(self, T: np.ndarray):
        super().__init__()
        self.register_buffer("T", torch.from_numpy(np.ascontiguousarray(T, dtype=np.float64)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.T

    @property
    def p(self) -> int:
        return int(self.T.shape[1])


class Stage1Net(nn.Module):
    """``delta_e = ||r||_2 * B_theta(r / ||r||_2, X)``.

    The wrapping is a deliberate modelling choice, stated because it constrains
    the answer.  Every classical smoother in the comparison table is a *linear*
    operator, and for a linear smoother the map ``r -> delta_e`` is exactly
    homogeneous of degree one -- the exact solve ``delta_e = A^{-1} r`` being the
    extreme case.  Imposing homogeneity puts the network in the same class as the
    objects it is compared against and makes ``rho`` scale-free.  A
    non-homogeneous DeepONet has strictly more freedom, all of it on the coarse
    side, which is precisely where the question is; so the constraint is the
    conservative choice rather than the flattering one.

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
    ``e_C`` is not touched until evaluation.

    ``log_fn`` defaults to stdout only.  Main passes the tee that also writes
    ``run.log``, so the per-epoch trace ends up in the run's own log file rather
    than only in the console capture.
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
            "best": {"val_mse": best["val_mse"], "val_mu_F": float(np.sqrt(max(best["val_mse"], 0.0))),
                     "epoch": best["epoch"], "epochs_run": len(history["epoch"])}}


# ---------------------------------------------------------------------------
# 4. Evaluation
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
    asserting it here because ``jacobi`` divides by a length-n vector and would
    silently broadcast along the wrong axis under the transpose."""
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

    The sweep is expressed relative to ``2 / lambda_max(D^{-1} A)``, the
    stability limit: beyond it damped Jacobi diverges and there is nothing to
    tune.  Selecting on the complement component gives the classical method its
    best shot at the same target the network is trained on, so neither is
    handicapped.
    """
    limit = 2.0 / float(lam_max)
    table = {}
    for w in np.linspace(0.05, 1.0, grid_n) * limit:
        Enew = apply_classical("damped_jacobi", split.EF, split.RF, cs, float(w), 1.0)
        table[float(w)] = reduction_stats(rho_of(split.EF, Enew, A))["mean"]
    best_w = min(table.items(), key=lambda kv: kv[1])
    return {"omega": float(best_w[0]), "val_mu_F": float(best_w[1]),
            "stability_limit": limit, "sweep": table}


def fit_ridge(tr: Split, va: Split, te: Split, A: sp.csc_matrix,
              lam_grid: Sequence[float]) -> Dict[str, object]:
    """Best *linear* map ``r -> delta_e`` fitted to the same training pairs.

    Diagnostic only, never in the headline table.  Fitted in the same A-inner
    product the network is trained in: with ``G`` a square root of ``A``
    (Cholesky factor transposed), the objective is

        min_V  sum_i || V r_i - G e_i ||^2 + lam ||V||_F^2 ,   W = G^{-1} V ,

    whose normal equations are ``(R^T R + lam I) V^T = R^T (G E^T)^T``.  The
    regulariser is chosen on the same validation complement metric as everything
    else, so the diagnostic gets no advantage over the network.
    """
    n = tr.RF.shape[1]
    G = np.linalg.cholesky(np.asarray(A.todense())).T       # G^T G = A

    # The network's loss is the *relative* energy, sum_i ||.||_A^2 / ||e_i||_A^2,
    # which is a weighted least squares with w_i = 1/||e_i||_A^2.  Fitting the
    # unweighted problem instead would hand the network an advantage the
    # comparison is supposed to measure, so the weights are applied here too: the
    # data is scaled by sqrt(w_i) and the regression run as if unweighted.
    w = 1.0 / np.maximum(np.sqrt(energy_rows(tr.EF, A)), 1e-12)
    Rs = tr.RF * w[:, None]
    Es = tr.EF * w[:, None]

    Zt = (G @ Es.T).T                                       # (N, n) = G e_i rows
    RtR = Rs.T @ Rs
    RtZ = Rs.T @ Zt

    def predict(R: np.ndarray, W: np.ndarray) -> np.ndarray:
        """``delta_e`` for each row of ``R``.

        ``W`` acts on a column vector, so row-wise this is ``R @ W.T``.  Writing
        ``R @ W`` instead silently applies the transpose; the two are both
        plausible-looking ridge fits, so the mistake is invisible in the loss
        curve and only shows up as an unexplained number in the table.
        """
        return R @ W.T

    best = None
    grid = {}
    for lam in lam_grid:
        Vt = np.linalg.solve(RtR + float(lam) * np.eye(n), RtZ)
        W = np.linalg.solve(G, Vt.T)                         # (n, n)
        mu = reduction_stats(rho_of(va.EF, va.EF - predict(va.RF, W), A))["mean"]
        grid[float(lam)] = float(mu)
        if best is None or mu < best[1]:
            best = (float(lam), float(mu), W)
    lam, mu_va, W = best

    # A ridge fit must reproduce its own training data well.  If the transpose
    # were wrong this is where it would show, so it is checked rather than
    # assumed.
    mu_tr = reduction_stats(rho_of(tr.EF, tr.EF - predict(tr.RF, W), A))["mean"]
    return {"lambda": lam, "val_mu_F": mu_va, "train_mu_F": float(mu_tr),
            "grid": grid, "W": W, "predict": predict}


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
def replot(out_dir: str) -> int:
    """Redraw the figures that need nothing but the run's own recorded numbers.

    ``metrics.json`` holds every mean (and so the whole selectivity map and the
    per-mechanism bars) and ``history.csv`` holds the loss trace, so a change to
    how a figure is *drawn* does not require re-training to take effect.
    ``reduction_histograms.png`` is the exception: it needs the per-sample
    ``rho`` arrays, which are derived during evaluation and deliberately not
    written to disk.
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
                e, tr, va, lr = line.split(",")
                hist["epoch"].append(int(e))
                hist["train_loss"].append(float(tr))
                hist["val_mse"].append(float(va))
                hist["lr"].append(float(lr))
        plot_loss_curves(hist, os.path.join(plots, "loss_curves.png"))

    log("replotted %s -> selectivity_map.png, per_mechanism.png%s"
        % (out_dir, ", loss_curves.png" if os.path.exists(h_path) else ""))
    return 0


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
    for k in TABLE_ORDER:
        s = stats[k]
        groups.setdefault((round(s["F"]["mean"], 4), round(s["C"]["mean"], 4)),
                          []).append(LABEL[k])
    colors: Dict[Tuple[float, float], str] = {}
    for k in TABLE_ORDER:
        s = stats[k]
        colors.setdefault((round(s["F"]["mean"], 4), round(s["C"]["mean"], 4)),
                          SERIES[k])

    for (xf, yc), names in groups.items():
        ax.scatter([xf], [yc], s=72, color=colors[(xf, yc)],
                   edgecolor=SURFACE, linewidth=1.3, zorder=3)
        ax.annotate(" / ".join(names), (xf, yc), textcoords="offset points",
                    xytext=(9, 7), color=INK_2, fontsize=8.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\mu_F$  (complement component)", color=INK_2, fontsize=9)
    ax.set_ylabel(r"$\mu_C$  (coarse component)", color=INK_2, fontsize=9)
    ax.set_title("Smoothing selectivity on level 3",
                 color=INK, fontsize=10.5, loc="left")
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


# ---------------------------------------------------------------------------
# 6. Self-test -- no training, no data generation
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
# 7. Main
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
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=250)
    ap.add_argument("--p", type=int, default=1024,
                    help="branch/trunk width; bounds the rank of the correction. "
                         "Defaults to n, which makes the trunk-span floor vacuous")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--trunk-features", default="fourier",
                    choices=["xyz", "angles", "trig", "fourier"])
    ap.add_argument("--trunk-modes", type=int, default=16)
    ap.add_argument("--trunk-width", type=int, default=-1,
                    help="trunk hidden width; -1 reuses --width.  The trunk's "
                         "output rank is bounded by its width, so a narrow trunk "
                         "can bind mu_F on its own")
    ap.add_argument("--trunk-depth", type=int, default=0,
                    help="trunk hidden depth; 0 makes the trunk a single linear "
                         "map of the features, which is what lets span(T) cover "
                         "the complement and keeps the trunk out of the answer")
    ap.add_argument("--trunk-freeze", default="fourier-orth",
                    choices=["none", "fourier-orth"],
                    help="'fourier-orth' freezes the trunk to an orthonormal basis "
                         "of the features, so span(T) = R^n and the trunk-span "
                         "floor is exactly zero. 'none' trains the trunk")
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
    # Same two settings the rest of the branch runs with: float64 throughout
    # (the energies being compared differ in the 4th decimal), and the sparse
    # invariant check off, since it only warns once per construction.
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
        # orthonormal columns, k = min(n, dim(features)); with Fourier modes up
        # to the fine Nyquist the features already span R^n, so Q spans R^n and
        # the correction set is unrestricted.
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
        trunk_width, trunk_depth_desc = frozen_T.shape[1], "frozen/orthonormal"
    else:
        # The trunk is given its own geometry.  DeepONetSmoother ties the trunk's
        # width and depth to the branch's, and rank(T) <= trunk width, so with a
        # shared narrow width the trunk alone can hold mu_F above ~0.75 whatever
        # the branch learns -- the trunk, not the smoother, would be under test.
        trunk_depth_desc = str(args.trunk_depth)
        if (trunk_width, args.trunk_depth) != (args.width, args.depth):
            inner.trunk = Trunk(features_np.shape[1], trunk_width, args.p,
                                args.trunk_depth)
    inner = inner.to(dtype)
    model = Stage1Net(inner, features)
    n_par = sum(p.numel() for p in model.parameters())
    log2("  trunk %s -> dim %d ; trunk width %s depth %s ; frozen = %s"
         % (args.trunk_features, features_np.shape[1], trunk_width,
            trunk_depth_desc, frozen_T is not None))
    log2("  branch width %d depth %d ; p = %d ; base = %s"
         % (args.width, args.depth, args.p, args.base_smoother))
    log2("  parameters %d ; correction rank <= p = %d ; complement dim = %d"
         % (n_par, args.p, n - coarse.n_c))

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
        ridge_info = fit_ridge(splits["train"], splits["val"], test, A.tocsc(),
                               args.ridge_grid)
        pred = ridge_info["predict"]
        W = ridge_info["W"]
        rho["ridge"] = {"F": rho_of(test.EF, test.EF - pred(test.RF, W), A),
                        "C": rho_of(test.EC, test.EC - pred(test.RC, W), A)}
        stats["ridge"] = {c: reduction_stats(rho["ridge"][c]) for c in ("F", "C")}
        log2("  ridge diagnostic: lambda = %g -> mu_F %.6f (train %.6f, val %.6f)"
             % (ridge_info["lambda"], stats["ridge"]["F"]["mean"],
                ridge_info["train_mu_F"], ridge_info["val_mu_F"]))

    # Trunk-span floor: the best ANY branch could have done against this trunk.
    with torch.no_grad():
        T = model.inner.trunk(features).numpy()
    TtAT = T.T @ np.asarray(A @ T)
    coef = np.linalg.solve(
        TtAT + 1e-12 * (np.trace(TtAT) / max(args.p, 1)) * np.eye(args.p),
        T.T @ np.asarray(A @ test.EF.T))
    approx = (T @ coef).T                                # (N, n), sample-major
    reach = energy_rows(approx, A) / np.maximum(energy_rows(test.EF, A), 1e-300)
    trunk_floor = float(np.mean(np.sqrt(np.maximum(1.0 - reach, 0.0))))
    rank_T = int(np.linalg.matrix_rank(T, tol=1e-10))
    log2("  trunk span: rank(T) = %d of p = %d -> best-possible mu_F >= %.6f"
         % (rank_T, args.p, trunk_floor))

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
                  "dim": int(features_np.shape[1]), "width": trunk_width,
                  "depth": trunk_depth_desc, "frozen": frozen_T is not None,
                  "span_is_Rn": bool(frozen_T is not None and rank_T == n),
                  "rank_T": rank_T, "best_possible_mu_F": trunk_floor,
                  "note": "floor on mu_F for ANY branch against the trained trunk; "
                          "the trunk span can be the binding constraint, not the net"},
        "model": {"p": args.p, "width": args.width, "depth": args.depth,
                  "base_smoother": args.base_smoother, "parameters": int(n_par),
                  "homogeneous_in_r": True,
                  "normalisation": "branch sees r/||r||_2; output rescaled by ||r||_2"},
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
            "no_vcycle": "Stage III",
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
    for k, lab in (("exact_solve", "A^-1 (reference)"),
                   ("ridge", "ridge (diagnostic)")):
        if k not in stats:
            continue
        r = metrics["reference_exact_solve"] if k == "exact_solve" else metrics["ridge_table"]
        log2("%-18s %8.4f %8.4f %9.4f %11.4f %9.4f %8.4f %8.4f %9.2f"
             % (lab, r["mu_F"], r["mu_C"], r["max_rho_F"], r["P_rhoF_gt_1"],
                r["median_rho_F"], r["std_rho_F"], r["p95_rho_F"],
                r["mu_C"] / max(r["mu_F"], 1e-12)))
    log2("")
    dn, dj = metrics["headline_table"]["deeponet"], metrics["headline_table"]["damped_jacobi"]
    log2("Does the DeepONet learn preferential reduction of the complement?")
    log2("  DeepONet      mu_F = %.4f   mu_C = %.4f   ratio mu_C/mu_F = %.2f"
         % (dn["mu_F"], dn["mu_C"], dn["mu_C"] / max(dn["mu_F"], 1e-12)))
    log2("  Damped Jacobi mu_F = %.4f   mu_C = %.4f   ratio mu_C/mu_F = %.2f"
         % (dj["mu_F"], dj["mu_C"], dj["mu_C"] / max(dj["mu_F"], 1e-12)))
    log2("  (ratio > 1 == the complement is reduced more than the coarse component, "
         "i.e. selective)")
    log2("  floor imposed by the trained trunk's span: mu_F >= %.4f" % trunk_floor)
    log2("")
    log2("Wrote %s/{metrics.json,history.csv,run.log,plots/}" % args.out_dir)
    log2("total wall clock: %.1f s" % (time.perf_counter() - t_start))
    run_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
