"""The anisotropic experiment for one epsilon.

    python -m hcsm.run_tensor --eps 1e-2 --out results/tensor/eps-1e-02

Order of operations, and why:

1. load and verify A_eps, P, A_H = P^T A P, and the hierarchy for the V-cycle
2. draw the four error families -- gaussian, fourier, weak, survivor -- in equal
   proportions, each projected by Q_eps and normalised in A, with the projection
   ratio recorded for every candidate
3. generate the HELD-OUT stress set: analytic near-worst-case modes, used for
   nothing except the final evaluation
4. measure the classical smoothers -- damped Jacobi, forward GS, symmetric GS --
   on every family: full and complement norm ratios, upper quantiles, worst
   case, and wall-clock time
5. compute eta(e) and check the specification's bound against it
6. select lambda on VALIDATION errors, then train the DeepONet on the penalised
   loss at the selected weight
7. measure the learned smoother on exactly the same errors
8. run the REAL V-cycle on the exported hierarchy -- classical and learned --
   from random right-hand sides, with a flexible outer solver for the learned
   preconditioner

Steps 4 and 8 answer different questions and are reported separately: step 4 is
what a smoother does to a manufactured error, step 8 is what the multigrid
method does to a real solve.
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch

from . import aniso_errors as AE
from .aniso_evaluate import (evaluate_families, jacobi_bound_check, jacobi_eta,
                             ratio_stats, estimate_condition_relevant)
from .evaluate import MethodSet, Smoothers, _corrections
from .losses import QProjector, ScaleEquivariant, penalised_loss, scaling_error
from .model import build_features, build_model, parameter_count
from .problem import TENSOR_EPSILONS, eps_tag, load_problem
from .train import TrainConfig, train_penalised, write_history
from .vcycle import (ClassicalSmoothing, Hierarchy, LearnedSmoothing,
                     measure_vcycle, solve_with_preconditioner)


def log(msg: str) -> None:
    print(msg, flush=True)


class WrappedSmoother:
    """The trained (scale-equivariant) network behind the smoother interface.

    ``_corrections`` only ever asks a DeepONet for ``from_residual``, so that is
    all this has to provide. Kept as a tiny class rather than a subclass of the
    framework's ``DeepONetSmoother`` because the wrapper changes what the
    forward pass *is*, and inheriting would leave the parent's ``from_error``
    silently calling the wrong (unwrapped) object.
    """

    def __init__(self, model, features_np):
        self.model = model
        self.X = torch.from_numpy(np.asarray(features_np, dtype=np.float64))

    @torch.no_grad()
    def from_residual(self, R: np.ndarray) -> np.ndarray:
        one = np.asarray(R).ndim == 1
        Rt = torch.from_numpy(np.atleast_2d(np.asarray(R, dtype=np.float64)))
        out = self.model(Rt, self.X, None).numpy()
        return out[0] if one else out


class Mixture:
    """Several families pooled, keeping the per-sample family label."""

    def __init__(self, families: Dict[str, AE.Family]):
        self.families = families
        self.E = np.concatenate([f.E for f in families.values()], axis=0)
        self.R = np.concatenate([f.R for f in families.values()], axis=0)
        self.label: List[str] = []
        for name, f in families.items():
            self.label.extend([name] * len(f))
        # Pooled in family blocks; the trainer shuffles per epoch, and this
        # order is only used to slice the labels back out.
        self.index = {name: np.arange(sum(len(g) for g in
                                          list(families.values())[:i]),
                                      sum(len(g) for g in
                                          list(families.values())[:i + 1]))
                      for i, name in enumerate(families)}

    def __len__(self) -> int:
        return self.E.shape[0]


def hierarchy_dirs(root: str, eps: float, level: int) -> List[str]:
    return [os.path.join(root, "tensor-eps-%s-L%d" % (eps_tag(eps), l))
            for l in range(level + 1)]


def run_one(eps: float, root: str, out_dir: str, seed: int = 0, level: int = 3,
            n_families: int = 96, epochs: int = 3000, p_dim: int = 64,
            width: int = 128, depth: int = 3,
            lambda_candidates: Sequence[float] = (0.0, 0.01, 0.1, 1.0),
            lambda_epochs: int = 600,
            vcycle_rhs: int = 4, verbose: bool = True) -> Dict[str, object]:
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    tag = eps_tag(eps)

    log("=" * 78)
    log("epsilon = %g   anisotropy %.0f:1 in the eta over the xi direction"
        % (eps, 1.0 / eps))
    log("=" * 78)

    # ---- 1. the problem -------------------------------------------------
    problem = load_problem(root, eps, level=level, verbose=verbose)
    cond = estimate_condition_relevant(problem)
    log("  A_eps: %d DoFs, nnz %d, diag ratio %.3f, cond(A_H) %.1f"
        % (problem.n, problem.A.nnz, cond["diag_ratio"], cond["A_H_condition"]))

    # ---- 2. the error families -----------------------------------------
    plan = AE.equal_plan(n_families)
    jac_omega = 2.0 / 3.0
    log("  drawing families (equal proportions: %s)" % plan)
    train_fam = AE.draw_mixture(problem, plan, seed, "train", jac_omega, verbose)
    val_fam = AE.draw_mixture(problem, plan, seed, "val", jac_omega, verbose)
    test_fam = AE.draw_mixture(problem, plan, seed, "test", jac_omega, verbose)
    audit = AE.audit({"train": _pool(train_fam), "val": _pool(val_fam),
                      "test": _pool(test_fam)})
    log("  cross-role exact duplicates: %d"
        % audit["cross_role_exact_duplicates"])
    if audit["cross_role_exact_duplicates"]:
        raise SystemExit("a sample appears in two roles; refusing to report a "
                         "test number that is partly a training number")
    for name in AE.FAMILIES:
        t, v, s = train_fam[name], val_fam[name], test_fam[name]
        log("    %-9s train %3d  val %3d  test %3d   median ||Qz||/||z|| %.3f"
            % (name, len(t), len(v), len(s),
               float(np.median(s.Qz_ratio)) if len(s) else float("nan")))

    stress = AE.stress_set(problem)
    log("  stress set: %d held-out analytic modes (%d skipped as already in "
        "range(P))" % (stress["E"].shape[0], len(stress["skipped"])))

    # ---- 3. classical smoothers ----------------------------------------
    smoothers = Smoothers(problem.A)
    methods = MethodSet(smoothers=smoothers, deeponets={},
                        jacobi_omega=jac_omega, ssor_omega=1.0)

    test_mix = Mixture(test_fam)
    classical = evaluate_families(problem, methods, test_fam)
    stress_cls = evaluate_families(
        problem, methods, {"stress": _as_family(stress)})

    bound = jacobi_bound_check(problem, methods, test_mix.E)
    eta = jacobi_eta(problem, test_mix.E, smoothers)

    log("  --- classical smoothers on the held-out test families ---")
    log("      %-11s %-24s %8s %8s %8s %8s" %
        ("family", "method", "mean", "p95", "worst", "s"))
    for fam_name in AE.FAMILIES:
        for meth in ("jacobi", "gs", "sgs_ssor"):
            r = classical[fam_name][meth]["full"]
            c = classical[fam_name][meth]["complement"]
            log("      %-11s %-24s %8.4f %8.4f %8.4f" %
                (fam_name, meth + " full", r.mean, r.p95, r.worst))
            log("      %-11s %-24s %8.4f %8.4f %8.4f" %
                ("", meth + " complement", c.mean, c.p95, c.worst))
    log("  eta(e): mean %.4f  p95 %.4f  max %.4f" %
        (bound["eta_mean"], bound["eta_p95"], bound["eta_max"]))
    log("  bound ||Q S_J e||/||e|| >= 1 - omega*eta holds for %.1f%% of samples; "
        "worst violation %.4f"
        % (100 * bound["fraction_where_bound_holds"], bound["worst_violation"]))

    # ---- 4. the real V-cycle, classical ---------------------------------
    hdirs = hierarchy_dirs(root, eps, level)
    hierarchy = Hierarchy.from_dirs(hdirs)
    log("  hierarchy: %s" % " -> ".join(str(A.shape[0]) for A in hierarchy.A))

    vcycles: Dict[str, object] = {}
    outer: Dict[str, object] = {}
    rng = np.random.default_rng(seed + 7)
    b = rng.normal(size=hierarchy.A[-1].shape[0])
    for kind in ("jacobi", "gs", "sgs"):
        sm = ClassicalSmoothing(hierarchy, kind=kind, omega=jac_omega)
        vcycles[kind] = measure_vcycle(hierarchy, sm, level, n_rhs=vcycle_rhs,
                                       seed=seed)
        outer[kind] = solve_with_preconditioner(hierarchy, sm, level, b,
                                                flexible=False)
        log("      V-cycle [%-7s] stationary rate %.4f (worst %.4f), "
            "%.1f iterations, %.2f s/rhs | outer %d iters"
            % (kind, vcycles[kind]["mean_rate"], vcycles[kind]["worst_rate"],
               vcycles[kind]["mean_iterations"],
               vcycles[kind]["seconds_per_rhs"], outer[kind]["iterations"]))

    # ---- 5. lambda on validation, then train ----------------------------
    features_np = build_features(problem)
    Q = QProjector(problem)
    val_mix = Mixture(val_fam)
    train_mix = Mixture(train_fam)
    cfg = TrainConfig(epochs=epochs, seed=seed)
    arch = {"p": p_dim, "width": width, "depth": depth}

    lam_table = []
    lam = 0.0
    if lambda_candidates:
        log("  selecting lambda on VALIDATION errors (short budget: %d epochs)"
            % lambda_epochs)
        best = float("inf")
        for cand in lambda_candidates:
            m = _fit(problem, train_mix, val_mix, features_np, Q, arch,
                     float(cand), lambda_epochs, seed, out_dir)
            score = _validation_reduction(problem, val_mix, m, features_np)
            lam_table.append({"lambda": float(cand),
                              "validation_full_reduction": float(score)})
            log("      lambda = %-8g validation full reduction %.6f"
                % (cand, score))
            if score < best:
                best, lam = float(score), float(cand)
        log("    selected lambda = %g" % lam)

    log("  training the DeepONet on the penalised loss (lambda = %g, %d updates)"
        % (lam, epochs))
    model = build_model(problem, features_np.shape[1], p=p_dim, width=width,
                        depth=depth, base_smoother="jacobi", seed=seed)
    model = ScaleEquivariant(model)
    history, info = train_penalised(
        model, problem, train_mix, val_mix, cfg, Q, lam,
        features_np=features_np, out_dir=out_dir,
        log=log if verbose else (lambda s: None),
        checkpoint_name="best_model.pt")
    write_history(history, os.path.join(out_dir, "history.csv"))
    log("    best epoch %d, validation total %.6f, %d parameters, %.0f s"
        % (info["best_epoch"], info["best_val_loss"], parameter_count(model),
           info["seconds"]))

    # scaling behaviour at amplitudes the model never trained on
    with torch.no_grad():
        R = torch.from_numpy(train_mix.R[:64])
        X = torch.from_numpy(features_np)
        scale_err = scaling_error(model, R, X, (1e-6, 1e-3, 1e3, 1e6))
    log("    scale-equivariance error over amplitudes 1e-6..1e6: %.3e" % scale_err)

    # ---- 6. the learned smoother on the same errors ---------------------
    #
    # The correction map is the WRAPPED network, because that is the object
    # that was trained and the object that is scale-equivariant. Calling the
    # inner network here would evaluate a different function from the one the
    # loss optimised.
    learned = WrappedSmoother(model, features_np)

    methods_learned = MethodSet(smoothers=smoothers,
                                deeponets={"scale": learned},
                                jacobi_omega=jac_omega, ssor_omega=1.0)
    dn_families = evaluate_families(problem, methods_learned, test_fam)
    dn_stress = evaluate_families(problem, methods_learned,
                                  {"stress": _as_family(stress)})

    log("  --- DeepONet ---")
    for fam_name in AE.FAMILIES:
        r = dn_families[fam_name]["deeponet_scale"]["full"]
        c = dn_families[fam_name]["deeponet_scale"]["complement"]
        log("      %-11s full mean %.4f p95 %.4f worst %.4f | complement mean "
            "%.4f" % (fam_name, r.mean, r.p95, r.worst, c.mean))

    # ---- 7. the learned V-cycle -----------------------------------------
    sm_learned = LearnedSmoothing(hierarchy, model, features_np, learned_level=level)
    vc_learned = measure_vcycle(hierarchy, sm_learned, level, n_rhs=vcycle_rhs,
                                seed=seed)
    out_learned = solve_with_preconditioner(hierarchy, sm_learned, level, b,
                                            flexible=True)
    log("      V-cycle [learned] stationary rate %.4f (worst %.4f), %.1f "
        "iterations, %.2f s/rhs | FGMRES %d iters (converged %s)"
        % (vc_learned["mean_rate"], vc_learned["worst_rate"],
           vc_learned["mean_iterations"], vc_learned["seconds_per_rhs"],
           out_learned["iterations"], out_learned["converged"]))
    log("      learned smoother used on %d level-applications, classical "
        "fallback on %d" % (sm_learned.levels_learned, sm_learned.levels_classical))

    # ---- 8. write --------------------------------------------------------
    def pack(block):
        return {fam: {meth: {"full": d["full"].as_dict(),
                             "complement": d["complement"].as_dict(),
                             "n": d["n"], "seconds": d["seconds"]}
                      for meth, d in meths.items()}
                for fam, meths in block.items()}

    metrics = {
        "epsilon": eps, "anisotropy": 1.0 / eps, "tag": tag,
        "seed": seed, "level": level, "n_dof": problem.n,
        "problem": problem.diagnostics,
        "condition": cond,
        "family_plan": plan,
        "sample_sha256": {
            role: hashlib.sha256(np.ascontiguousarray(mix.E).tobytes()).hexdigest()
            for role, mix in (("train", train_mix), ("val", val_mix), ("test", test_mix))},
        "family_audit": audit,
        "stress_labels": stress["label"],
        "bound_check": bound,
        "eta": {"mean": float(np.mean(eta)), "p95": float(np.percentile(eta, 95)),
                "max": float(np.max(eta))},
        "classical_families": pack(classical),
        "classical_stress": pack(stress_cls),
        "deeponet_families": pack(dn_families),
        "deeponet_stress": pack(dn_stress),
        "training": {**info, "lambda": lam, "lambda_table": lam_table,
                     "architecture": arch, "trunk_features": "trig",
                     "config": vars(cfg),
                     "parameters": parameter_count(model),
                     "scale_equivariance_error": scale_err},
        "vcycle": {"classical": vcycles, "learned": vc_learned,
                   "hierarchy_dofs": [int(A.shape[0]) for A in hierarchy.A],
                   "leaned_levels": sm_learned.levels_learned,
                   "classical_levels": sm_learned.levels_classical},
        "outer_solver": {"classical": outer, "learned": out_learned},
        "scope": [
            "One fixed fine level (level %d, %d DoFs). The branch consumes a "
            "fixed-length residual, so the learned smoother is bound to this "
            "DoF count and ordering; every coarser level of the V-cycle is "
            "smoothed classically, and that is reported."
            % (level, problem.n),
            "Q_epsilon is the GALERKIN complement projector built from "
            "P^T A_eps P. The V-cycle runs on the independently REDISCRETIZED "
            "level operators, which differ on a curved surface; the difference "
            "is recorded and the two are never conflated.",
            "Ratios named 'full' and 'complement' are NORM ratios "
            "||.||_A/||e||_A. Squared-energy ratios are reported separately "
            "under 'mean_squared_ratio'.",
            "The stress set is analytic and held out: it is used for evaluation "
            "only, never for training, model selection or thresholding.",
        ],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    np.savez_compressed(
        os.path.join(out_dir, "stress.npz"),
        **{"stress_" + k: v for k, v in stress.items()
           if isinstance(v, np.ndarray)})
    log("  metrics -> %s   (%.0f s total)" % (os.path.join(out_dir, "metrics.json"),
                                              time.time() - t_start))
    return metrics


def _pool(families: Dict[str, AE.Family]) -> AE.Family:
    """Pool families into one Family for the cross-role duplicate audit."""
    return AE.Family(
        name="pooled",
        E=np.concatenate([f.E for f in families.values()], axis=0),
        R=np.concatenate([f.R for f in families.values()], axis=0),
        Qz_ratio=np.concatenate([f.Qz_ratio for f in families.values()]),
        subkind=[k for f in families.values() for k in f.subkind],
        digest=[d for f in families.values() for d in f.digest])


def _as_family(stress: Dict[str, object]) -> AE.Family:
    return AE.Family(name="stress", E=stress["E"], R=stress["R"],
                     Qz_ratio=stress["Qz_ratio"], subkind=list(stress["label"]),
                     digest=["stress"] * stress["E"].shape[0])


def _fit(problem, train_mix, val_mix, features_np, Q, arch, lam, epochs,
         seed, out_dir):
    """Short-budget fit used only to rank the lambda candidates."""
    m = ScaleEquivariant(build_model(
        problem, features_np.shape[1], p=arch["p"], width=arch["width"],
        depth=arch["depth"], base_smoother="jacobi", seed=seed))
    short = TrainConfig(epochs=epochs, seed=seed, log_every=max(50, epochs // 4))
    train_penalised(m, problem, train_mix, val_mix, short, Q, lam,
                    features_np=features_np, out_dir=out_dir, log=lambda s: None,
                    checkpoint_name="lambda_probe_%g.pt" % lam)
    return m


def _validation_reduction(problem, val_mix, model, features_np) -> float:
    """Mean full-error reduction on VALIDATION errors -- the selection criterion.

    The penalty exists to stop the smoother from pushing error into range(P),
    which the first loss term cannot see; the quantity it protects is therefore
    the FULL error, and that is what lambda is selected on. Validation only.
    """
    with torch.no_grad():
        R = torch.from_numpy(val_mix.R)
        delta = model(R, torch.from_numpy(features_np), None).numpy()
    e_plus = val_mix.E - delta
    num = np.sqrt(np.maximum(np.einsum("bi,bi->b", e_plus,
                                       (problem.A @ e_plus.T).T), 0.0))
    den = np.sqrt(np.maximum(np.einsum("bi,bi->b", val_mix.E,
                                       (problem.A @ val_mix.E.T).T), 1e-300))
    return float(np.mean(num / den))


def _validation_score(problem, val_mix, level) -> float:
    return float("nan")   # placeholder kept for symmetry with the tables


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--data-root", default="level_data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--samples-per-family", type=int, default=24,
                    help="samples in EACH of the four families, for each role")
    ap.add_argument("--families", type=int, default=None,
                    help="deprecated: TOTAL samples per role, divisible by four")
    ap.add_argument("--max-steps", "--epochs", dest="epochs", type=int, default=3000,
                    help="exact maximum number of optimizer updates")
    ap.add_argument("--lambda-steps", "--lambda-epochs", dest="lambda_epochs",
                    type=int, default=600, help="optimizer updates per lambda probe")
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    total = args.families if args.families is not None else 4 * args.samples_per_family
    if total < 4 or total % 4:
        ap.error("total sample count must be positive and divisible by four")
    if args.epochs < 1 or args.lambda_epochs < 1:
        ap.error("training step counts must be positive")

    torch.set_num_threads(args.threads)
    out = args.out or os.path.join("results", "tensor", "eps-%s" % eps_tag(args.eps))
    run_one(eps=args.eps, root=args.data_root, out_dir=out, seed=args.seed,
            level=args.level, n_families=total, epochs=args.epochs,
            lambda_epochs=args.lambda_epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
