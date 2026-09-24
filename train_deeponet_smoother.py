#!/usr/bin/env python3
"""Compatibility entry point for the DeepONet smoother trainer.

This file used to BE the trainer.  It is now a thin adapter: it accepts the old
command line, refuses the flags whose meaning has changed, and forwards everything
to ``deeponet_smoother.py``, which is the implementation and the only place the
losses are defined.  Runs started from here therefore produce the corrected
objective, not the old one.

Why it was rewritten
--------------------
The prototype that lived here trained ``B_theta : r -> delta_e`` with

    loss = alpha * high_frequency_loss + beta * energy_loss

and that arrangement had five defects, each of which changes the number reported:

1. ``A_sp.toarray()`` densified the FEM operator.  Fine at 1 024 DoFs, fatal at
   16 384 (a 2.1 GB dense matrix plus an O(n^3) factorisation).  Everything here
   is sparse now, and the fine ``A_h`` is never densified anywhere.
2. ``high_frequency_loss`` restricted with the prolongation in the wrong
   direction.  ``P`` is (n_fine, n_coarse) -- the coarse space is the SHORTER one
   -- so the coarse residual is ``P^T A e``, not ``A P e`` or ``P A e``.
3. Its normaliser was the coarse complement of the ORIGINAL error,
   ``||Q e||_A^2``.  That quantity vanishes for errors already in range(P), which
   is a legitimate part of the space, so the ratio divided by ~0 exactly where it
   should have reported "nothing left to remove".
4. The coarse term was a *penalty added to* the full-energy term.  The intended
   objective is the other way round: the coarse-grid-corrected energy is the
   primary term and the full energy is the auxiliary one,

       L = L_CGC + lambda * L_E .

   So the old ``--alpha``/``--beta`` weights cannot express the intended
   objective, and adding ``--coarse-loss-weight 1`` to the module did not either.
5. It trained and validated on white noise only, and treated a low training loss
   as evidence that the smoother works.

Two further points that are properties of the experiment rather than of the code,
and are handled in the module: a fixed grid bounds the correction's rank by ``p``,
so ``p`` must be comparable to the number of DoFs (raising ``p`` does not fix a
narrow branch, and no loss change can); and the loss only covers one smoothing
pass plus one EXACT coarse solve, so it must be checked against a real, recursive
V-cycle.

The module also checks, rather than assumes, that the coarse-grid correction
inside the loss is the one the C++ multigrid executes -- ``R = P^T``
(MGTransferPrebuilt), no interface/edge matrices on this uniformly refined
hierarchy, and the assembled coarse operator as the default (``P^T A P`` is
available and its difference is measured, not hidden).

Usage
-----
    # the old command line still works, with the corrected objective:
    python train_deeponet_smoother.py --data-dir level_data/L3 --epochs 4000 \\
        --p 128 --width 256 --seed 0 --out-dir results/myrun

    # see the objective actually used, and the module command it becomes:
    python train_deeponet_smoother.py --data-dir level_data/L3 --dry-run

Only the flags you pass are forwarded; everything else comes from the module's
defaults, which are the studied configuration.  Unrecognised flags are passed
through verbatim (``--log-every``, ``--grad-clip``, ...), so the module's full
command line works here too; ``--help`` lists the new knobs
(``--lambda-energy``, ``--coarse-operator``, ``--loss-mode``, ``--vcycle``, ...).
"""

import argparse
import os
import sys

import deeponet_smoother as ds


def build_parser():
    ap = argparse.ArgumentParser(
        description="DeepONet smoother trainer (adapter over deeponet_smoother.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -- flags carried over from the prototype, all defaulting to None so that
    #    only what the caller actually asks for is forwarded -------------------
    ap.add_argument("--data-dir", default=None,
                    help="one LEVEL directory (level_data/L3), not the root that "
                         "holds L0..Ln")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--p", type=int, default=None,
                    help="branch/trunk output width. Bounds the RANK of the "
                         "correction by p regardless of the residual, so it must be "
                         "comparable to the DoF count -- see DEEPONET_SMOOTHER.md")
    ap.add_argument("--width", type=int, default=None, help="hidden width")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--use-coefficient", action="store_true",
                    help="feed kappa into the branch.  For a single fixed operator "
                         "kappa is the same vector for every sample and adds nothing "
                         "but parameters; the module reports it explicitly")
    ap.add_argument("--trunk-features", default=None,
                    choices=["trig", "fourier", "xyz", "angles"])
    ap.add_argument("--trunk-modes", type=int, default=None)
    ap.add_argument("--base-smoother", default=None, choices=["none", "jacobi"],
                    help="jacobi adds a learnable diagonal skip omega*D^-1 r")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=["float32", "float64"])

    # -- the objective --------------------------------------------------------
    ap.add_argument("--loss-mode", default=None, choices=["cgc", "energy"])
    ap.add_argument("--lambda-energy", type=float, default=None,
                    help="weight of the auxiliary full-energy term in "
                         "L = L_CGC + lambda * L_E.  Experimental: compare 0.0 with a "
                         "small positive value")
    ap.add_argument("--coarse-operator", default=None,
                    choices=["assembled", "galerkin"])
    ap.add_argument("--vcycle", action="store_true")
    ap.add_argument("--smoother-iterations", type=int, default=None)
    ap.add_argument("--reuse-data", action="store_true")

    # -- refused legacy flags -------------------------------------------------
    ap.add_argument("--use-hf", action="store_true",
                    help="REMOVED -- see the error message")
    ap.add_argument("--alpha", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--beta", type=float, default=None, help=argparse.SUPPRESS)

    ap.add_argument("--dry-run", action="store_true",
                    help="print the effective objective and the module command, "
                         "then stop")
    return ap


def refuse_legacy(args):
    """Stop, with the reason, rather than silently training something else."""
    if args.use_hf or args.alpha is not None or args.beta is not None:
        raise SystemExit(
            "--use-hf / --alpha / --beta are gone.\n"
            "  They expressed   loss = alpha * (coarse complement) + beta * (energy),\n"
            "  i.e. the coarse-grid-corrected term as a PENALTY on top of the full\n"
            "  energy, normalised by ||Q e||_A^2 -- which is zero for errors in\n"
            "  range(P), so the ratio is unstable exactly where it matters.\n"
            "  The intended objective is the other way round:\n"
            "      L = L_CGC + lambda * L_E ,   L_CGC the primary term,\n"
            "  and it is now the default (--loss-mode cgc).  Use --lambda-energy to\n"
            "  set the weight of the auxiliary energy term; start by comparing\n"
            "  --lambda-energy 0.0 against 0.1 and report L_CGC and L_E separately.")


def check_data_dir(data_dir):
    """The prototype defaulted to the ROOT (level_data) while the module needs one
    level directory; say so instead of failing on a missing npz."""
    if data_dir is None:
        return
    if os.path.exists(os.path.join(data_dir, "level_matrix.npz")):
        return
    levels = [d for d in ("L%d" % i for i in range(12))
              if os.path.exists(os.path.join(data_dir, d, "level_matrix.npz"))]
    if levels:
        raise SystemExit(
            "%s is the hierarchy ROOT, not a level directory: it holds %s.\n"
            "Training happens on one level, so pass the level itself, e.g.\n"
            "    --data-dir %s"
            % (data_dir, ", ".join(levels), os.path.join(data_dir, levels[-1])))
    raise SystemExit("%s has no level_matrix.npz -- run tools/convert_level_data.py "
                     "first (see DEEPONET_SMOOTHER.md section 3)" % data_dir)


def main(argv=None):
    # parse_known_args, so every flag this file does not itself interpret is
    # forwarded verbatim to the module -- the adapter must not become a second,
    # silently-out-of-date copy of the module's command line.
    args, unknown = build_parser().parse_known_args(argv)
    refuse_legacy(args)
    check_data_dir(args.data_dir)

    fwd = []
    for key in ("data-dir", "out-dir", "seed", "epochs", "batch-size", "lr", "p",
                "width", "depth", "trunk-features", "trunk-modes", "base-smoother",
                "n-train", "n-val", "n-test", "threads", "dtype", "loss-mode",
                "lambda-energy", "coarse-operator", "smoother-iterations"):
        val = getattr(args, key.replace("-", "_"))
        if val is not None:
            fwd += ["--" + key, str(val)]
    if args.use_coefficient:
        fwd.append("--use-coefficient")
    if args.vcycle:
        fwd.append("--vcycle")
    if args.reuse_data:
        fwd.append("--reuse-data")
    fwd += list(unknown)          # anything else goes to the module unchanged

    effective = {
        "objective": ("L_CGC + lambda * L_E" if (args.loss_mode or "cgc") == "cgc"
                      else "L_E (full energy only)"),
        "lambda_energy": args.lambda_energy if args.lambda_energy is not None else 0.1,
        "coarse_operator": args.coarse_operator or "assembled",
        "coarse_operator_meaning": (
            "the coarse matrix the C++ V-cycle uses (default)"
            if (args.coarse_operator or "assembled") == "assembled" else
            "P^T A P: exact A-orthogonal projection, but not the executed operator"),
        "model_selection": ("validation L_CGC + lambda*L_E"
                            if (args.loss_mode or "cgc") == "cgc"
                            else "validation L_E"),
        "note": "p bounds the correction's rank; the loss cannot change that",
    }
    print("=" * 74)
    print("DeepONet smoother trainer -> deeponet_smoother.py")
    print("=" * 74)
    for k, v in effective.items():
        print("  %-24s %s" % (k, v))
    print("  module command            python deeponet_smoother.py %s" % " ".join(fwd))
    print("=" * 74, flush=True)

    if args.dry_run:
        return 0
    return ds.main(fwd)


if __name__ == "__main__":
    sys.exit(main())
