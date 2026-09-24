#!/usr/bin/env python3
"""V-cycle contraction against the classical baselines the C++ actually uses.

The V-cycle comparison inside ``deeponet_smoother.py`` contrasts the learned
smoother at the fine level (Jacobi below) with **damped Jacobi everywhere**.  That
is a weak baseline, and it is not the one the solver uses:
``src/main.cpp`` builds the multigrid smoother as

    mg::SmootherRelaxation<PreconditionSOR<SparseMatrix<double>>, ...>
    mg_smoother->set_steps(2);
    mg_smoother->set_symmetric(true);

i.e. **symmetrised SOR with two steps** on every level.  A V-cycle claim measured
only against damped Jacobi therefore does not answer the question a reader will
ask -- "does this beat the solver's own smoother?" -- so this tool measures the
reduced V-cycle for the learned smoother and for each classical smoother, at one
and at two steps, using the exported hierarchy and a trained model.

It trains nothing: it reads a run's ``best_model.pt`` and the level data, so the
numbers correspond exactly to the model that run reports.

    python tools/vcycle_baselines.py --data-dir level_data/L3 \\
        --run results/L3_cgc_l0.1 --n-samples 32 --iterations 5

Output is a table of mean ||e - M^{-1} A e||_A / ||e||_A per iteration, so a
method that stalls or diverges is visible rather than averaged away.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deeponet_smoother as ds  # noqa: E402


def load_model(run_dir, ld, features_np, dtype):
    """Rebuild the model a run reports, from its own best_model.pt."""
    ck = torch.load(os.path.join(run_dir, "best_model.pt"), weights_only=False)
    cfg = ck["config"]
    model = ds.DeepONetSmoother(ld.n, features_np.shape[1], cfg["p"], cfg["width"],
                                cfg["depth"], cfg.get("use_coefficient", False),
                                cfg.get("base_smoother", "none"),
                                ld.A.diagonal(), cfg.get("omega_init", 2.0 / 3.0)).to(dtype)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck.get("epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run", required=True, help="results dir holding best_model.pt")
    ap.add_argument("--hierarchy-root", default=None)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    args = ap.parse_args()

    torch.sparse.check_sparse_tensor_invariants.disable()
    hroot = args.hierarchy_root or os.path.dirname(os.path.abspath(args.data_dir))
    lvl = int(os.path.basename(os.path.abspath(args.data_dir))[1:])
    levels = [ds.load_level(os.path.join(hroot, "L%d" % k)) for k in range(lvl + 1)]
    hierarchy = ds.Hierarchy(levels)
    ld = levels[lvl]
    A = ld.A
    dtype = torch.float64

    features_np = ds.trunk_features(ld.angles, "trig", ld.coords)
    model, cfg, epoch = load_model(args.run, ld, features_np, dtype)
    features = torch.from_numpy(features_np).to(dtype)
    C = None
    if cfg.get("use_coefficient") and ld.coeff is not None:
        C = torch.from_numpy(np.asarray(ld.coeff, dtype=np.float64)).to(dtype)

    # The errors are the RUN'S OWN test errors when it saved them, so these rows
    # can be read against the V-cycle rows the run itself reports.  Generating
    # fresh errors here instead would put two different error distributions in one
    # table and make the comparison meaningless.
    ds_path = os.path.join(args.run, "dataset_test.npz")
    if os.path.exists(ds_path):
        E = np.load(ds_path)["E"][:args.n_samples].astype(np.float64)
        # The stored errors are already at unit A-energy; the normalisation here
        # only removes the float32 rounding the npz carries, and reproduces what
        # the run itself fed its model (it casts the same array to float64).
        E = np.array([e / np.sqrt(e @ (A @ e)) for e in E])
        src = "%s (the run's own test errors)" % ds_path
    else:
        rng = np.random.default_rng(args.seed)
        nyq = max(1, max(2, int(round(np.sqrt(ld.n)))) // 2)
        E = []
        while len(E) < args.n_samples:
            v, _, _ = ds.draw_sample(rng, "mixed", ld.angles, nyq, ld.n)
            e, _ = ds.normalise_and_residualise(v, A)
            if e is not None:
                E.append(e)
        E = np.array(E)
        src = "generated mixed errors (no dataset_test.npz in the run dir)"

    # A smoother is bound to ONE DoF count -- the same constraint that stops the
    # learned branch acting below the fine level -- so every classical smoother
    # here is built per level and dispatches on the DoF count.  Using the fine
    # level's diagonal at a coarse level does not raise a clear error, it raises a
    # broadcasting error, which is why this is spelled out.
    smoothers_by_n = {lv.n: ds.ClassicalSmoothers(lv.A) for lv in levels}
    diag_by_n = {lv.n: lv.A.diagonal() for lv in levels}
    jac_w = 0.7
    learned_omega = float(model.omega) if cfg.get("base_smoother") == "jacobi" else None

    def learned(rr):
        with torch.no_grad():
            t = torch.from_numpy(rr).to(dtype)[None, :]
            return model(t, features, C)[0].numpy()

    def jacobi(rr):
        return jac_w * rr / diag_by_n[rr.shape[0]]

    def symgs(rr):
        # symmetric Gauss-Seidel.  At omega = 1 this is exactly SSOR, and it is
        # what PreconditionSOR + set_symmetric(true) does in src/main.cpp.
        return smoothers_by_n[rr.shape[0]].symmetric_gs(rr)

    # (label, fine-level smoother, coarse-level smoother, pre, post, omega)
    # `omega` is only used when the coarse smoother is left at its default
    # (damped Jacobi below); the learned rows use it so that they reproduce the
    # module's own V-cycle exactly -- omega = 2/3 below, not 1.0, which would be
    # undamped Jacobi and a different (worse) baseline.
    #
    # The C++ configuration is symmetric SOR at omega = 1, two steps, on every
    # level -- "symGS everywhere x2" below.  That is the row that decides whether
    # the learned smoother earns its place.  The learned rows are given one AND
    # two steps, because comparing one learned step against two classical ones
    # would settle the question by arithmetic rather than by quality: with two
    # steps the learned smoother is evaluated twice per side, so its inference
    # cost doubles too -- which metrics.json reports separately.
    cases = [
        ("learned @L%d, Jacobi below x1" % lvl, learned, None, 1, 1, 2.0 / 3.0),
        ("learned @L%d, symGS below x1" % lvl, learned, symgs, 1, 1, 2.0 / 3.0),
        ("learned @L%d, symGS below x2" % lvl, learned, symgs, 2, 2, 2.0 / 3.0),
        ("symGS everywhere x1", symgs, symgs, 1, 1, 1.0),
        ("symGS everywhere x2  (the C++ smoother)", symgs, symgs, 2, 2, 1.0),
        ("damped Jacobi everywhere x1", jacobi, None, 1, 1, jac_w),
    ]

    print("V-cycle as a preconditioner: %d samples, %d iterations" %
          (E.shape[0], args.iterations))
    print("level %d (%d DoFs), %d levels, model from %s (epoch %s)"
          % (lvl, ld.n, hierarchy.n_levels, args.run, epoch))
    if learned_omega is not None:
        print("learned diagonal skip omega = %.5f" % learned_omega)
    print()
    results = {}
    for label, fine, coarse, pre, post, om in cases:
        hist = []
        for e in E:
            ev = e.copy()
            row = []
            for _ in range(args.iterations):
                ev = ev - hierarchy.vcycle_apply(A @ ev, lvl, fine, pre, post, om,
                                                 smooth_coarse=coarse)
                row.append(float(np.sqrt(max(ev @ (A @ ev), 0.0) /
                                         max(e @ (A @ e), 1e-300))))
            hist.append(row)
        hist = np.array(hist)
        results[label] = {
            "mean_per_iteration": [float(hist[:, k].mean()) for k in range(hist.shape[1])],
            "worst_sample_final": float(hist[:, -1].max()),
            "n_samples": int(hist.shape[0]), "pre": pre, "post": post,
        }

    print("%-38s %s   %s" % ("variant", " ".join("it%d" % (k + 1)
                                                 for k in range(args.iterations)),
                             "worst"))
    print("-" * 72)
    for label, r in results.items():
        print("%-38s %s   %.4f" % (label,
                                   " ".join("%.4f" % x for x in r["mean_per_iteration"]),
                                   r["worst_sample_final"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"data_dir": args.data_dir, "run": args.run, "level": lvl,
                       "n_dofs": ld.n, "jacobi_omega": jac_w,
                       "learned_omega": learned_omega, "variants": results},
                      fh, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
