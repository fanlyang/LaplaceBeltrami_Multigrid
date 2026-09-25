"""The penalised loss, the scale-equivariant wrapper, and Q_epsilon in torch.

The objective is the specification's

    L = E ||Q_epsilon e^+||_{A_epsilon}^2  +  lambda E ||e^+||_{A_epsilon}^2 ,
    e^+ = e - B_theta(r),   r = A_epsilon e,

with lambda > 0. The first term is what a smoother is for -- it ignores whatever
the coarse grid would fix -- and the second exists because that indifference is
not free: in a real cycle the coarse correction is applied only to the component
it can see, and error the smoother pushes into range(P) is not removed for
nothing. lambda is selected on validation data; see ``select_lambda``.

**Q_epsilon in the training graph.** Training needs Q to be differentiable with
respect to the correction, so it cannot be a scipy coarse solve. The coarse
operator A_H = P^T A P is 256 x 256 here, so its CHOLESKY FACTOR is formed once
and reused: Q v = v - P (A_H^{-1} (P^T A v)) is then a few small dense products
and two triangular solves. No inverse is formed and no n x n projector is ever
materialised -- the factor is a factor. ``selftest`` checks the torch path
against the independent scipy sparse-solve path on random vectors, so the two
agree to machine precision before any number depends on it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .model import A_apply, energy_norm, to_torch_sparse


class ScaleEquivariant(nn.Module):
    """Wrap a residual-to-correction map so that B(lambda r) = lambda B(r).

    The specification asks for explicit handling of residual scaling. Without
    it a tanh network is not scale-equivariant, so a model trained on unit
    residuals need not behave sensibly when the solver hands it a residual ten
    times larger -- and a V-cycle does exactly that, since residuals shrink by
    orders of magnitude as the iteration proceeds.

    The wrapper normalises the residual, applies the network, and rescales:

        B(r) = ||r||_2 * N(r / ||r||_2),      B(0) = 0

    which is positively homogeneous of degree one by construction, for every
    parameter value, not merely at convergence. The norm is the Euclidean norm
    of the residual rather than the A^{-1} norm of it: the A^{-1} norm is the
    more natural scale (it equals ||e||_A for r = A e) but it costs a solve per
    evaluation, and homogeneity -- the property actually needed -- is obtained
    either way. ``scale_test`` measures the residual scaling error at amplitudes
    the model never trained on.

    Zero maps to zero exactly, including in the backward pass: an exact discrete
    solution stays a fixed point by construction, not by training.
    """

    def __init__(self, net: nn.Module, tiny: float = 1e-300):
        super().__init__()
        self.net = net
        self.tiny = tiny

    def forward(self, residual: torch.Tensor, features: torch.Tensor,
                coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm = torch.linalg.norm(residual, dim=1, keepdim=True)
        safe = norm.clamp_min(self.tiny)
        unit = residual / safe
        out = self.net(unit, features, coeff) * norm
        # Where the residual is exactly zero the normalised vector is undefined;
        # force the correction to zero there rather than returning nan.
        return torch.where(norm > 0.0, out, torch.zeros_like(out))


class QProjector:
    """Q_epsilon applied to a batch, differentiably, by a coarse solve."""

    def __init__(self, problem, dtype: torch.dtype = torch.float64):
        self.A = to_torch_sparse(problem.A, dtype)
        self.AT = to_torch_sparse(problem.A.T.tocsr(), dtype)
        self.P = torch.from_numpy(
            np.asarray(problem.P.todense(), dtype=np.float64)).to(dtype)
        A_H = np.asarray((problem.P.T @ (problem.A @ problem.P)).todense(),
                         dtype=np.float64)
        self.A_H_chol = torch.linalg.cholesky(
            torch.from_numpy(A_H).to(dtype))
        self.n = problem.n
        self.m = problem.P.shape[1]

    def __call__(self, V: torch.Tensor) -> torch.Tensor:
        """V of shape (b, n) -> Q_epsilon V, same shape."""
        b = self.A_apply(V)                 # A V           (b, n)
        rhs = b @ self.P                    # P^T A V       (b, m)
        z = torch.cholesky_solve(rhs.t(), self.A_H_chol).t()   # (b, m)
        return V - z @ self.P.t()

    def A_apply(self, V: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.A, V.t()).t()

    def energy(self, V: torch.Tensor) -> torch.Tensor:
        """v^T A v per row of V, sparsely -- A is never densified."""
        return (V * self.A_apply(V)).sum(dim=1)


def penalised_loss(
    correction: torch.Tensor,
    errors: torch.Tensor,
    Q: QProjector,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L = mean ||Q e^+||_A^2 + lambda * mean ||e^+||_A^2.

    Returns (total, complement_term, full_term), all scalars. The two terms are
    returned separately so the training log can show which one is moving: a
    single total cannot distinguish "the smoother improved" from "the penalty
    took over", and those are different claims.

    Both terms are already relative: every training error is normalised to
    ||e||_A = 1 before it reaches here, so no denominator is needed and the two
    terms are directly comparable.
    """
    e_plus = errors - correction
    complement = Q.energy(Q(e_plus)).mean()
    full = Q.energy(e_plus).mean()
    return complement + lam * full, complement, full


def scaling_error(net: nn.Module, R: torch.Tensor, features: torch.Tensor,
                  amplitudes) -> float:
    """Worst relative deviation from positive homogeneity over ``amplitudes``.

    ``B(lambda r) = lambda B(r)`` should hold exactly for the wrapped network.
    This measures it rather than assuming it, and the result is reported for
    amplitudes deliberately chosen off the training scale.
    """
    with torch.no_grad():
        base = net(R, features, None)
        worst = 0.0
        for lam in amplitudes:
            scaled = net(R * lam, features, None)
            denom = float(scaled.abs().max().clamp_min(1e-300))
            worst = max(worst, float((scaled - base * lam).abs().max()) / denom)
    return worst


def select_lambda(
    problem,
    candidates,
    build_and_train,
    validate,
    log=print,
) -> Tuple[float, list]:
    """Choose the full-error penalty weight on VALIDATION data.

    ``build_and_train(lam)`` returns a trained model and ``validate(model)``
    returns the validation figure of merit -- the reduction the model achieves
    in the two-grid step on held-out validation errors, which is the end-use
    quantity the penalty is protecting. The sweep table is returned with the
    winner so the choice is auditable; the test set is not consulted.

    The search is run at a reduced training budget and the winner is retrained
    at the full budget. That is stated here because it is a real limitation: a
    lambda that wins after 800 epochs is not guaranteed to win after 3000, and
    the sweep table is what lets a reader judge how flat the choice was.
    """
    table = []
    best_lam, best_score = None, float("inf")
    for lam in candidates:
        model = build_and_train(lam)
        score = float(validate(model))
        table.append({"lambda": float(lam), "validation_score": score})
        log("      lambda = %-8g validation two-grid reduction %.6f"
            % (lam, score))
        if score < best_score:
            best_lam, best_score = float(lam), score
    return best_lam, table
