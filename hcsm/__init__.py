"""Stage-I high-contrast smoothing experiment.

This package answers one question, and is organised so that the answer is a
consequence of measurements rather than an assumption:

    As the scalar diffusion coefficient becomes increasingly heterogeneous,
    which coarse-space-complement error components become difficult for
    classical point smoothers, and can the existing DeepONet smoother reduce
    those same error components more effectively?

The modules follow the specification directly:

    problem.py    the coefficient family kappa_epsilon, the level operator
                  A_epsilon, the Galerkin coarse operator P^T A P, and the
                  A_epsilon-orthogonal decomposition e = e_C + e_F
    samplers.py   the sampled error distribution and the disjoint train / val /
                  test splits, all in coarse-space-complement form
    model.py      the existing MLP DeepONet branch and trunk, and the Stage-I
                  smoothing loss L_smooth
    train.py      the training loop (unchanged optimiser/schedule framework)
    evaluate.py   rho_F, rho_C, the controlled-mode diagnostic rho_k, the
                  Jacobi-hard error selection, and the ideal two-grid rho_TG
    plots.py      the required figures
    artifacts.py  CSV/NPZ output and the cross-epsilon summary

Nothing in this package densifies a matrix, forms a matrix inverse, or uses the
independently assembled coarse operator.
"""

from .problem import EPSILONS, Problem, eps_tag, kappa_epsilon, load_problem

__all__ = [
    "EPSILONS",
    "Problem",
    "eps_tag",
    "kappa_epsilon",
    "load_problem",
]
