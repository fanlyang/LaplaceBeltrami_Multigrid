"""Run the whole Stage-I experiment for one epsilon.

    python -m hcsm.run_epsilon --eps 1e-1 --out results/stage1/eps-1e-01

The order of the steps is the order of the specification, and the order matters:

    1. load and verify A_epsilon, P and A_H = P^T A P        (sections D, E)
    2. draw disjoint train / validation / test samples, each
       in coarse-space-complement form                       (sections F, G)
    3. tune the classical damping on VALIDATION errors       (section I)
    4. train one model per architecture on L_smooth alone    (section H)
    5. measure rho_F and rho_C on the unseen test set        (sections I, J)
    6. measure the controlled modes sin(k xi), sin(k eta)    (section K)
    7. select the empirically Jacobi-hard errors             (section L)
    8. run the ideal Galerkin two-grid method                (section M)
    9. write every number, sample and figure                 (sections N, O)

Nothing is selected on the test set: the damping parameters come from the
validation split, model selection follows the validation loss, and the test set
is touched only to produce the numbers that are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from . import plots
from .evaluate import (DeepONetSmoother, MethodSet, Smoothers,
                       controlled_mode_scan, cross_evaluate_hard,
                       evaluate_methods, hard_errors, oracle_classical,
                       per_sample_rho, repeated_steps, rho_stats,
                       tune_classical, two_grid, two_grid_on_ef)
from .model import build_features, build_model, parameter_count
from .problem import Problem, load_problem
from .samplers import DEFAULT_PLAN, Split, draw_split, split_audit
from .train import TrainConfig, train_smoother, write_history

#: The DeepONet configurations trained per epsilon. The branch and trunk are the
#: existing framework's; what varies is which of its existing switches are on.
#:
#: ``plain`` is the specification's literal architecture: the MLP branch over the
#: residual, the MLP trunk over the evaluation point, nothing else.
#:
#: ``skip`` adds the framework's learned damped-Jacobi skip,
#: ``delta = omega D^{-1} r + branch(r).trunk(x)``, which is the configuration the
#: existing framework's own measurements identified as the competitive one.
#:
#: ``skip_fourier`` adds spatial reach. With the ``trig`` trunk the input is only
#: (sin xi, cos xi, sin eta, cos eta) -- four numbers -- so every correction the
#: network can produce is a smooth function of the torus angles. That is a poor
#: match for this experiment, whose training targets are *coarse-space-complement*
#: errors, i.e. exactly the oscillatory part. Without a frequency-aware trunk,
#: "the learned smoother does not help" would be a statement about the trunk
#: rather than about the architecture, so the ``fourier`` trunk (the framework's
#: documented alternative, carrying all modes |m|,|n| <= 4) is carried alongside.
#: Both outcomes are informative: if reach closes the gap, the trig result was a
#: trunk artefact; if it does not, the conclusion survives the trunk choice.
ARCHITECTURES: Dict[str, Dict[str, object]] = {
    "plain": {"trunk": "trig", "n_modes": 0, "base": "none"},
    "skip": {"trunk": "trig", "n_modes": 0, "base": "jacobi"},
    "skip_fourier": {"trunk": "fourier", "n_modes": 4, "base": "jacobi"},
}
DEFAULT_ARCHITECTURES = ("plain", "skip", "skip_fourier")


def log(msg: str) -> None:
    print(msg, flush=True)


def _splits_for(problem: Problem, plan: Sequence, seed: int,
                verbose: bool) -> Dict[str, Split]:
    """Train / validation / test with an identical mechanism mix and disjoint draws."""
    def scaled(divisor: int):
        return tuple((m, max(4, c // divisor)) for m, c in plan)

    return {
        role: draw_split(problem, scaled(div), seed=seed, role=role, verbose=verbose)
        for role, div in (("train", 1), ("val", 4), ("test", 2))
    }


def run_one(
    eps: float,
    root: str,
    out_dir: str,
    seed: int = 0,
    level: int = 3,
    epochs: int = 3000,
    p_dim: int = 64,
    width: int = 128,
    depth: int = 3,
    n_hard: int = 20,
    steps_secondary: int = 3,
    architectures: Sequence[str] = DEFAULT_ARCHITECTURES,
    verbose: bool = True,
) -> Dict[str, object]:
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    tag = "%.0e" % eps

    log("=" * 78)
    log("epsilon = %g   (kappa in [%g, %g], contrast %.0f)"
        % (eps, eps, 1 + eps, (1 + eps) / eps))
    log("=" * 78)

    # ---- 1. the problem --------------------------------------------------
    problem = load_problem(root, eps, level=level, verbose=verbose)
    log("  A_epsilon : %d DoFs, %d nonzeros, level %d, P is %d x %d"
        % (problem.n, problem.A.nnz, level, problem.P.shape[0], problem.P.shape[1]))
    log("  A_H=P^T A P: %d x %d, SPD, condition %.1f"
        % (problem.A_H.shape[0], problem.A_H.shape[1],
           problem.diagnostics["A_H_condition"]))
    log("  coefficient: [%.6g, %.6g], measured contrast %.1f  (spec: %.0f)"
        % (problem.coeff.min(), problem.coeff.max(),
           problem.coeff.max() / problem.coeff.min(), (1 + eps) / eps))

    # ---- 2. samples ------------------------------------------------------
    splits = _splits_for(problem, DEFAULT_PLAN, seed, verbose)
    audit = split_audit(splits)
    log("  samples: train %d, val %d, test %d" % (
        len(splits["train"]), len(splits["val"]), len(splits["test"])))
    log("  disjointness: %d exact sample duplicates across roles, "
        "%d RNG-stream-seed collisions  (%d signature overlaps, diagnostic only)"
        % (audit["cross_split_exact_duplicates"],
           audit["role_stream_seed_collisions"],
           audit["cross_split_signature_overlaps"]))
    if audit["cross_split_exact_duplicates"] or audit["role_stream_seed_collisions"]:
        raise SystemExit("a sample appears in two roles -- refusing to report a "
                         "test number that is partly a training number")
    log("  ||e_F||_A / ||e||_A over the accepted test samples: "
        "mean %.3f, min %.3f, max %.3f"
        % (splits["test"].ef_ratio.mean(), splits["test"].ef_ratio.min(),
           splits["test"].ef_ratio.max()))

    # ---- 3. classical tuning on VALIDATION -------------------------------
    smoothers = Smoothers(problem.A)
    tuning = tune_classical(problem, smoothers, splits["val"].E_F)
    jacobi_omega = float(tuning["jacobi_omega"])
    ssor_omega = float(tuning["ssor_omega"])
    log("  classical tuning on validation e_F (NOT on test):")
    log("    damped Jacobi  omega = %.2f   mean rho_F %.4f"
        % (jacobi_omega, tuning["jacobi_table"][jacobi_omega]))
    log("    SSOR           omega = %.2f   mean rho_F %.4f"
        % (ssor_omega, tuning["ssor_table"][ssor_omega]))

    oracle = oracle_classical(problem, smoothers, splits["test"].E_F)
    log("    [bound only, not a method result] best test-set omega: "
        "Jacobi %.2f -> %.4f, SSOR %.2f -> %.4f"
        % (oracle["jacobi_omega"], oracle["jacobi_table"][oracle["jacobi_omega"]],
           oracle["ssor_omega"], oracle["ssor_table"][oracle["ssor_omega"]]))

    # ---- 4. train --------------------------------------------------------
    cfg = TrainConfig(epochs=epochs, seed=seed)
    histories: Dict[str, List[Dict[str, object]]] = {}
    training: Dict[str, Dict[str, object]] = {}
    deeponets: Dict[str, DeepONetSmoother] = {}
    features_np: Dict[str, np.ndarray] = {}

    for arch in architectures:
        spec = ARCHITECTURES[arch]
        trunk_mode, base = str(spec["trunk"]), str(spec["base"])
        feats = build_features(problem, mode=trunk_mode, n_modes=int(spec["n_modes"]))
        features_np[arch] = feats
        ckpt = "best_model_%s.pt" % arch

        log("  training DeepONet [%s] (trunk=%s dim %d, base_smoother=%s) ..."
            % (arch, trunk_mode, feats.shape[1], base))
        model = build_model(problem, feats.shape[1], p=p_dim, width=width,
                            depth=depth, base_smoother=base, seed=seed)
        history, info = train_smoother(
            model, problem, splits["train"], splits["val"], cfg,
            features_np=feats, out_dir=out_dir,
            log=log if verbose else (lambda s: None),
            checkpoint_name=ckpt)
        write_history(history, os.path.join(out_dir, "history_%s.csv" % arch))
        histories[arch] = history

        # Evaluate the SAVED checkpoint, not the in-memory model, so every
        # published number belongs to the published weights.
        saved = torch.load(os.path.join(out_dir, ckpt), map_location="cpu",
                           weights_only=False)
        reloaded = build_model(problem, feats.shape[1], p=p_dim, width=width,
                               depth=depth, base_smoother=base, seed=seed)
        reloaded.load_state_dict(saved["state_dict"])
        deeponets[arch] = DeepONetSmoother(reloaded, problem, feats)

        info["parameters"] = parameter_count(reloaded)
        info["base_smoother"] = base
        info["trunk_features"] = trunk_mode
        info["trunk_dim"] = int(feats.shape[1])
        if base == "jacobi":
            info["omega_trained"] = float(saved["state_dict"]["omega"].item())
        training[arch] = info
        log("    best epoch %d, val L_smooth %.6f, %d params, %.0fs%s"
            % (info["best_epoch"], info["best_val_loss"], info["parameters"],
               info["seconds"],
               ("  omega = %.4f" % info["omega_trained"]) if base == "jacobi" else ""))

    methods = MethodSet(smoothers=smoothers, deeponets=deeponets,
                        jacobi_omega=jacobi_omega, ssor_omega=ssor_omega)

    # ---- 5. rho_F and rho_C ---------------------------------------------
    E_F, E_C = splits["test"].E_F, splits["test"].E_C
    rho_F = evaluate_methods(problem, methods, E_F)
    rho_C = evaluate_methods(problem, methods, E_C)
    rho_F_all = per_sample_rho(problem, methods, E_F)
    rho_C_all = per_sample_rho(problem, methods, E_C)
    rho_F_repeated = repeated_steps(problem, methods, E_F, steps=steps_secondary)

    log("  --- Section I: rho_F on %d unseen test errors (one smoothing step) ---" % len(E_F))
    log("      %-24s %7s %7s %7s %7s %7s %8s" %
        ("method", "mean", "median", "std", "p95", "max", "frac>1"))
    for name in methods.names():
        s = rho_F[name]
        log("      %-24s %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f"
            % (plots.style.METHOD_LABEL.get(name, name), s["mean"], s["median"],
               s["std"], s["p95"], s["max"], s["frac_gt_1"]))

    log("  --- Section J: rho_C on the corresponding coarse components ---")
    for name in methods.names():
        s = rho_C[name]
        log("      %-24s %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f"
            % (plots.style.METHOD_LABEL.get(name, name), s["mean"], s["median"],
               s["std"], s["p95"], s["max"], s["frac_gt_1"]))

    # ---- 6. controlled modes --------------------------------------------
    scans = controlled_mode_scan(problem, methods, families=("xi", "eta"))
    for fam in ("xi", "eta"):
        rec = scans[fam]
        if not rec.get("k", np.zeros(0)).size:
            log("  controlled modes (%s): none survived the projection" % fam)
            continue
        log("  --- Section K: controlled %s modes, k = %s ---"
            % (fam, ", ".join("%d" % k for k in rec["k"])))
        for name in methods.names():
            y = np.asarray(rec[name])
            log("      %-24s mean %.4f, min %.4f, max %.4f"
                % (plots.style.METHOD_LABEL.get(name, name),
                   y.mean(), y.min(), y.max()))

    # ---- 7. hard errors --------------------------------------------------
    hard = hard_errors(problem, methods, splits["test"], n_worst=n_hard,
                       reference="jacobi")
    log("  --- Section L: the %d hardest test errors for damped Jacobi ---" % n_hard)
    log("      rho_J on the hard set: mean %.4f (all test: %.4f); "
        "%.0f%% of them are already amplified (all test: %.0f%%)"
        % (hard["hard_mean_rho_jacobi"], hard["all_mean_rho_jacobi"],
           100 * hard["hard_frac_gt_1"], 100 * hard["all_frac_gt_1"]))
    log("      energy inside the lowest-kappa quartile: hard %.1f%%, "
        "whole test set %.1f%%  -> enrichment %.2fx"
        % (100 * hard["error_energy_in_low_kappa_hard"],
           100 * hard["error_energy_in_low_kappa_all"],
           hard["low_kappa_enrichment"]))
    hard_eval = cross_evaluate_hard(problem, methods, splits["test"], hard["index"])
    for name in methods.names():
        s, j = hard_eval[name], rho_F[name]
        log("      %-24s hard-set mean %7.4f  vs  full-test mean %7.4f"
            % (plots.style.METHOD_LABEL.get(name, name), s["mean"], j["mean"]))

    # ---- 8. ideal Galerkin two-grid -------------------------------------
    tg = two_grid(problem, methods, splits["test"].E)
    tg_ef = two_grid_on_ef(problem, methods, splits["test"])
    log("  --- Section M: rho_TG, one smoothing step + exact coarse solve ---")
    log("      %-24s %7s %7s %7s %7s %7s %8s" %
        ("method", "mean", "median", "std", "p95", "max", "frac>1"))
    for name in list(methods.names()) + ["coarse_only"]:
        s = tg[name]
        log("      %-24s %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f"
            % (plots.style.METHOD_LABEL.get(name, name), s["mean"], s["median"],
               s["std"], s["p95"], s["max"], s["frac_gt_1"]))

    # ---- 9. write everything --------------------------------------------
    # Only what cannot be reconstructed is stored. The error itself is
    # E = E_C + E_F exactly (checked to machine precision in hcsm.selftest), and
    # the residual is R_F = A_epsilon E_F with A_epsilon sitting in level_data/,
    # so storing either again would double the file for nothing. The train and
    # validation arrays are regenerated exactly by re-running with the same seed,
    # and their composition is recorded in metrics.json.
    np.savez_compressed(
        os.path.join(out_dir, "samples.npz"),
        test_E_C=splits["test"].E_C,
        test_E_F=splits["test"].E_F,
        test_ef_ratio=splits["test"].ef_ratio,
        test_mechanism=np.asarray(splits["test"].mechanism),
        test_digest=np.asarray(splits["test"].digest),
        reconstruction="test_E = test_E_C + test_E_F ; "
                       "test_R_F = A_epsilon @ test_E_F with A_epsilon from "
                       "level_data/eps-<E>-L3/level_matrix.npz",
        **{"rho_F_" + k: v for k, v in rho_F_all.items()},
        **{"rho_C_" + k: v for k, v in rho_C_all.items()},
        hard_index=hard["index"],
        hard_E_F=hard["E_F"], hard_R_F=hard["R_F"],
        hard_S_jacobi=hard["S_jacobi"], hard_S_gs=hard["S_gs"],
        **{"hard_" + k: v for k, v in hard.items() if k.startswith("S_deeponet")},
    )
    # The method columns are taken from what was actually run, not from a
    # hard-coded list: a name typed here that no model produced would otherwise
    # write a silently short row.
    ordered = [k for k in ("jacobi", "gs", "sgs_ssor") if k in rho_F_all]
    ordered += [k for k in rho_F_all if k.startswith("deeponet_")]

    with open(os.path.join(out_dir, "test_samples.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "mechanism", "signature", "params", "ef_ratio"]
                   + ["rho_F_" + k for k in ordered])
        for i in range(len(splits["test"])):
            w.writerow([
                i, splits["test"].mechanism[i], splits["test"].signature[i],
                splits["test"].params[i], "%.10g" % splits["test"].ef_ratio[i],
                *["%.10g" % rho_F_all[k][i] for k in ordered],
            ])
    with open(os.path.join(out_dir, "hard_errors.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "test_index", "mechanism"] + ["rho_" + k for k in ordered])
        for rank, i in enumerate(hard["index"]):
            w.writerow([rank, int(i), splits["test"].mechanism[int(i)]]
                       + ["%.10g" % hard["rho_" + k][rank] for k in ordered])

    metrics: Dict[str, object] = {
        "epsilon": eps,
        "contrast": (1 + eps) / eps,
        "tag": tag,
        "seed": seed,
        "level": level,
        "problem": problem.diagnostics,
        "split_audit": audit,
        "splits": {role: s.summary() for role, s in splits.items()},
        "tuning": {
            "jacobi_omega": jacobi_omega,
            "ssor_omega": ssor_omega,
            "jacobi_sweep_validation": tuning["jacobi_table"],
            "ssor_sweep_validation": tuning["ssor_table"],
        },
        "oracle_bound_on_test": {
            "jacobi_omega": oracle["jacobi_omega"],
            "jacobi_mean_rho_F": oracle["jacobi_table"][oracle["jacobi_omega"]],
            "ssor_omega": oracle["ssor_omega"],
            "ssor_mean_rho_F": oracle["ssor_table"][oracle["ssor_omega"]],
            "note": "best achievable on the test set itself; a bound, never a "
                    "reported method result",
        },
        "training": training,
        "hyperparameters": {
            "p": p_dim, "width": width, "depth": depth,
            "epochs": epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "weight_decay": cfg.weight_decay, "grad_clip": cfg.grad_clip,
            "early_stop": cfg.early_stop, "log_every": cfg.log_every,
            "loss": "L_smooth only",
        },
        "rho_F": rho_F,
        "rho_C": rho_C,
        "rho_F_repeated_%d_steps" % steps_secondary: rho_F_repeated,
        "rho_TG": tg,
        "rho_TG_on_eF": tg_ef,
        # The scalar and per-sample entries go into metrics.json; the full
        # (n_worst, n_dof) fields would add megabytes of JSON that nobody reads,
        # so they live in samples.npz instead and this says so.
        "hard_errors": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in hard.items()
                        if not (isinstance(v, np.ndarray) and v.ndim > 1)},
        "hard_errors_arrays": "the e_F, r_F and S(.) fields of the selected hard "
                              "errors are in samples.npz under hard_*",
        "hard_errors_cross_evaluated": hard_eval,
        "controlled_modes": {
            fam: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in scans[fam].items()}
            for fam in scans
        },
        "scope": [
            "One fixed multigrid level (level 3, 1024 DoFs on a 32x32 periodic "
            "DoF lattice). The branch network consumes a fixed-length residual, "
            "so the model is bound to this DoF count AND ordering; no cross-mesh "
            "or mesh-refinement generalisation is claimed or measured.",
            "The coarse space is range(P) with P the exported level-2 -> level-3 "
            "prolongation, and the coarse operator is the Galerkin P^T A P. The "
            "independently assembled coarse operator is NOT used.",
            "e_F is the A_epsilon-orthogonal coarse-space-complement component. "
            "It is NOT a frequency band and is never called 'high frequency'.",
            "The energy norm is the algebraic ||v||_A^2 = v^T A_epsilon v. No "
            "mass matrix is exported, so no continuous L^2(Gamma) norm is "
            "available and none is reported.",
            "rho_F and rho_TG are ratios for ONE smoothing step at the stated "
            "damping; the multi-step numbers are reported separately.",
            "All errors are canonicalised to unit A-energy. The classical "
            "reductions are invariant under that; the DeepONet's is not exactly, "
            "because its branch is a tanh network.",
            "The conclusion is about this torus, this discretisation, this "
            "coefficient family, this training distribution and this DeepONet "
            "configuration, and is not generalised beyond them.",
        ],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    # ---- figures ---------------------------------------------------------
    plots.plot_training({(eps, a): histories[a] for a in histories},
                        os.path.join(plot_dir, "training_curves.png"))
    plots.plot_mode_scan({tag: scans}, "xi",
                         os.path.join(plot_dir, "modes_xi.png"))
    plots.plot_mode_scan({tag: scans}, "eta",
                         os.path.join(plot_dir, "modes_eta.png"))
    plots.plot_hard_errors(problem, hard,
                           os.path.join(plot_dir, "hard_errors_on_torus.png"))
    plots.plot_before_after(problem, hard, 0,
                            os.path.join(plot_dir, "hard_error_before_after.png"))
    plots.plot_rho_distribution(rho_F_all,
                                os.path.join(plot_dir, "rho_F_distribution.png"),
                                "Per-sample reduction on the unseen test set, "
                                r"$\varepsilon = 10^{%d}$"
                                % int(round(np.log10(eps))))

    log("  metrics -> %s" % os.path.join(out_dir, "metrics.json"))
    log("  total %.1f s" % (time.time() - t_start))

    return {"metrics": metrics, "plots_dir": plot_dir, "out_dir": out_dir}


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--data-root", default="level_data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-hard", type=int, default=20)
    ap.add_argument("--architectures", default=",".join(DEFAULT_ARCHITECTURES),
                    help="comma-separated subset of %s"
                         % ", ".join(ARCHITECTURES))
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = args.out or os.path.join("results", "stage1", "eps-%.0e" % args.eps)
    run_one(eps=args.eps, root=args.data_root, out_dir=out,
            seed=args.seed, level=args.level, epochs=args.epochs,
            p_dim=args.p, width=args.width, depth=args.depth,
            n_hard=args.n_hard,
            architectures=tuple(a for a in args.architectures.split(",") if a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
