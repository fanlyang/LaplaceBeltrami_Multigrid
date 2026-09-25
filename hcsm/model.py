"""The existing MLP DeepONet branch and trunk, and the Stage-I loss.

Nothing about the architecture is redesigned here. ``Branch``, ``Trunk`` and
``DeepONetSmoother`` are imported unchanged from ``deeponet_smoother.py``; so is
the sparse apply machinery and the loss kernel. What this module adds is the
narrower *objective* the specification asks for.

The one thing the specification changes relative to the existing framework is
the loss. The existing default is ``L_CGC + lambda * L_E``, which contains a
coarse-grid correction. Stage I must use the smoothing loss alone:

    L_smooth = mean[ ||e_F - B_theta(A e_F, X)||_A^2 / ||e_F||_A^2 ]

so ``L_CGC`` is not imported, not computed and not available from this module.
That is a deliberate omission, not an oversight: the question Stage I asks is
what one smoothing step does on its own, and a loss that already contains the
coarse correction cannot answer it.

Why the input is A e_F and not e_F
----------------------------------
The smoother under study is defined as

    S_theta(e) = e - B_theta(A_epsilon e, X),

i.e. the network consumes the *residual* of the error. For an error that solves
A e = r this is the same information, but it is the form the specification uses
and the form the existing framework uses, so it is the form kept here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

import deeponet_smoother as ds
from deeponet_smoother import (  # noqa: F401  (re-exported for the other modules)
    Branch,
    DeepONetSmoother,
    Trunk,
    A_apply,
    energy_loss,
    energy_norm,
    spmm,
    to_torch_sparse,
    trunk_features,
)

from .problem import Problem

#: Trunk input. 'trig' = (sin xi, cos xi, sin eta, cos eta). It encodes the torus
#: periodicity exactly, so a correction has no artificial seam at xi = +-pi, and
#: it is the existing framework's default for this geometry.
DEFAULT_TRUNK_FEATURES = "trig"


def build_features(problem: Problem, mode: str = DEFAULT_TRUNK_FEATURES,
                   n_modes: int = 0) -> np.ndarray:
    """The trunk input: a fixed feature map of the DoF support points."""
    return np.ascontiguousarray(
        ds.trunk_features(problem.angles, mode, problem.coords, n_modes),
        dtype=np.float64,
    )


def build_model(
    problem: Problem,
    trunk_dim: int,
    p: int = 64,
    width: int = 128,
    depth: int = 3,
    base_smoother: str = "none",
    omega_init: float = 2.0 / 3.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> DeepONetSmoother:
    """Instantiate the existing architecture for one epsilon.

    ``base_smoother='jacobi'`` switches on the framework's learned damped-Jacobi
    skip connection, ``delta_e = omega D^{-1} r + branch(r) . trunk(x)``, with
    ``omega`` learnable. Both settings are trained and reported: the skip is part
    of the existing architecture, but a reader is entitled to see what the MLP
    contributes on its own, so the plain variant is carried through every table
    as an ablation rather than being quietly dropped.
    """
    torch.manual_seed(seed)
    diag = problem.A.diagonal() if base_smoother == "jacobi" else None
    model = DeepONetSmoother(
        n_dof=problem.n,
        trunk_dim=trunk_dim,
        p=p,
        width=width,
        depth=depth,
        use_coeff=False,
        base_smoother=base_smoother,
        diag=diag,
        omega_init=omega_init,
    )
    return model.to(dtype)


def stage1_loss(
    correction: torch.Tensor,
    e_F: torch.Tensor,
    A: torch.Tensor,
    AT: torch.Tensor,
    floor: float = 1e-12,
) -> torch.Tensor:
    """L_smooth, exactly as specified.

        mean[ (e_F - delta_e)^T A (e_F - delta_e) / (e_F^T A e_F) ]

    ``energy_loss`` is the existing framework's implementation of precisely this
    ratio. Its ``floor`` guards the denominator only; it is never added to the
    numerator, so a perfect correction scores zero rather than the floor.

    The floor is unreachable in this experiment: ``samplers`` rejects any sample
    whose e_F has relative A-energy below 1e-8 and then rescales the survivors to
    ||e_F||_A = 1, so every denominator here is 1 up to rounding. That is
    asserted by the caller rather than assumed.
    """
    return energy_loss(correction, e_F, A, AT, eps=floor)


@torch.no_grad()
def corrections(
    model: DeepONetSmoother,
    residuals: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """B_theta(r, X) for a batch of residuals."""
    model.eval()
    return model(residuals, features, None)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
