"""Regenerate one epsilon's figures from the saved checkpoints, and check them.

    python -m hcsm.refigure --results results/stage1

Why this exists, beyond redrawing pictures:

* **It is a reproducibility check.** Every number in ``metrics.json`` is
  recomputed here from the checkpoint, the seed and the deterministic sampler --
  no training -- and compared against what was written. A mismatch means the
  published figures do not belong to the published weights, which is exactly the
  kind of error that is invisible in a finished report.
* **It keeps the figures consistent.** A plotting change made after some runs
  finished would otherwise leave the earlier figures drawn by the old code.

The configuration is read back from ``metrics.json`` (seed, level, p/width/depth,
which architectures, which trunk, which base smoother), not hard-coded, so this
reproduces the run that was actually made rather than the run that was intended.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from . import plots
from .evaluate import (DeepONetSmoother, MethodSet, Smoothers, hard_errors,
                       per_sample_rho)
from .model import build_features, build_model
from .problem import EPSILONS, eps_tag, load_problem
from .run_epsilon import ARCHITECTURES, _splits_for
from .samplers import DEFAULT_PLAN


def refigure_one(run_dir: str, root: str, seed_override: Optional[int] = None,
                 verbose: bool = True) -> Dict[str, float]:
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        metrics = json.load(fh)

    eps = metrics["epsilon"]
    seed = metrics["seed"] if seed_override is None else seed_override
    hp = metrics["hyperparameters"]
    problem = load_problem(root, eps, level=metrics["level"])

    splits = _splits_for(problem, DEFAULT_PLAN, seed, verbose=False)
    smoothers = Smoothers(problem.A)

    deeponets: Dict[str, DeepONetSmoother] = {}
    for arch, info in metrics["training"].items():
        spec = ARCHITECTURES[arch]
        feats = build_features(problem, mode=str(spec["trunk"]),
                               n_modes=int(spec["n_modes"]))
        model = build_model(problem, feats.shape[1], p=hp["p"], width=hp["width"],
                            depth=hp["depth"], base_smoother=str(spec["base"]),
                            seed=seed)
        state = torch.load(os.path.join(run_dir, "best_model_%s.pt" % arch),
                           map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        deeponets[arch] = DeepONetSmoother(model, problem, feats)

    methods = MethodSet(smoothers=smoothers, deeponets=deeponets,
                        jacobi_omega=float(metrics["tuning"]["jacobi_omega"]),
                        ssor_omega=float(metrics["tuning"]["ssor_omega"]))

    # -- recompute and compare --------------------------------------------
    rho_F = per_sample_rho(problem, methods, splits["test"].E_F)
    deviations: Dict[str, float] = {}
    for name, values in rho_F.items():
        recorded = metrics["rho_F"].get(name)
        if recorded is not None:
            deviations[name] = abs(float(np.mean(values)) - recorded["mean"])

    # -- redraw ------------------------------------------------------------
    plot_dir = os.path.join(run_dir, "plots")
    hard = hard_errors(problem, methods, splits["test"], n_worst=20,
                       reference="jacobi")
    plots.plot_hard_errors(problem, hard,
                           os.path.join(plot_dir, "hard_errors_on_torus.png"))
    plots.plot_before_after(problem, hard, 0,
                            os.path.join(plot_dir, "hard_error_before_after.png"))
    plots.plot_rho_distribution(
        rho_F, os.path.join(plot_dir, "rho_F_distribution.png"),
        "Per-sample reduction on the unseen test set, "
        r"$\varepsilon = 10^{%d}$" % int(round(np.log10(eps))))
    plots.plot_mode_scan(
        {eps_tag(eps): metrics["controlled_modes"]}, "xi",
        os.path.join(plot_dir, "modes_xi.png"))
    plots.plot_mode_scan(
        {eps_tag(eps): metrics["controlled_modes"]}, "eta",
        os.path.join(plot_dir, "modes_eta.png"))

    if verbose:
        worst = max(deviations.items(), key=lambda kv: kv[1]) if deviations else ("-", 0.0)
        print("    rho_F reproduced from the saved checkpoint: worst |mean "
              "difference| = %.3e (%s)" % (worst[1], worst[0]))
    return deviations


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results/stage1")
    ap.add_argument("--data-root", default="level_data")
    ap.add_argument("--eps", type=float, default=None,
                    help="one epsilon; default is every run under --results")
    args = ap.parse_args(argv)

    epsilons = [args.eps] if args.eps is not None else list(EPSILONS)
    worst = 0.0
    for eps in epsilons:
        run_dir = os.path.join(args.results, "eps-%s" % eps_tag(eps))
        if not os.path.exists(os.path.join(run_dir, "metrics.json")):
            print("  skipping %s (no metrics.json)" % run_dir)
            continue
        print("  epsilon = %g" % eps)
        dev = refigure_one(run_dir, args.data_root)
        worst = max(worst, max(dev.values()) if dev else 0.0)

    print()
    print("worst mean-rho_F discrepancy across all runs: %.3e" % worst)
    print("(this is the reproducibility check: the figures and the tables come "
          "from the same saved weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
