"""Compare the plain MLP at the default budget against the 4x-budget re-run.

    python -m hcsm.compare_budget --main results/stage1 --budget results/stage1_budget

Writes ``results/stage1_budget/budget_check.csv`` and prints the comparison.

The question this answers is narrow and worth stating plainly: the main run
trains the plain MLP DeepONet for the existing framework's default 3000 epochs,
and at that budget its validation loss is still falling. So "the plain MLP is
worse than damped Jacobi" is only a usable statement if it survives a much
larger budget. If the 4x run closes the gap, the main comparison is an
under-training artefact and must be reported as one. If it does not, the gap is
a property of what this architecture can express at this trunk width, and the
curve that shows it is the evidence.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, Optional, Sequence

from .problem import EPSILONS, eps_tag


def load(path: str) -> Dict[str, Dict[str, object]]:
    out = {}
    for p in sorted(glob.glob(os.path.join(path, "eps-*", "metrics.json"))):
        with open(p) as fh:
            m = json.load(fh)
        out[m["tag"]] = m
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main", default="results/stage1")
    ap.add_argument("--budget", default="results/stage1_budget")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    main = load(args.main)
    budget = load(args.budget)
    if not budget:
        raise SystemExit("no runs found under %s/eps-*/" % args.budget)

    out_csv = args.out or os.path.join(args.budget, "budget_check.csv")
    rows = []

    print("=" * 100)
    print("Plain MLP DeepONet: default budget vs 4x budget")
    print("=" * 100)
    print("%-8s %10s %10s %10s | %8s %8s | %8s %8s %8s"
          % ("eps", "ep(3000)", "ep(12000)", "val(12000)", "rho 3k", "rho 12k",
             "Jaco", "SGS", "TG 12k"))
    print("-" * 100)

    for eps in EPSILONS:
        tag = eps_tag(eps)
        if tag not in budget:
            continue
        b = budget[tag]
        m = main.get(tag, {})
        t_b = b.get("training", {}).get("plain", {})
        t_m = m.get("training", {}).get("plain", {})
        rho_b = b.get("rho_F", {}).get("deeponet_plain", {})
        rho_m = m.get("rho_F", {}).get("deeponet_plain", {})
        jac = m.get("rho_F", {}).get("jacobi", {})
        sgs = m.get("rho_F", {}).get("sgs_ssor", {})
        tg = b.get("rho_TG", {}).get("deeponet_plain", {})

        print("%-8s %10s %10s %10.4f | %8.4f %8.4f | %8.4f %8.4f %8.4f"
              % (tag, t_m.get("best_epoch"), t_b.get("best_epoch"),
                 t_b.get("best_val_loss", float("nan")),
                 rho_m.get("mean", float("nan")), rho_b.get("mean", float("nan")),
                 jac.get("mean", float("nan")), sgs.get("mean", float("nan")),
                 tg.get("mean", float("nan"))))

        rows.append({
            "epsilon": eps,
            "epochs_default": t_m.get("best_epoch"),
            "val_default": t_m.get("best_val_loss"),
            "rho_F_default": rho_m.get("mean"),
            "epochs_4x": t_b.get("best_epoch"),
            "val_4x": t_b.get("best_val_loss"),
            "rho_F_4x": rho_b.get("mean"),
            "rho_F_jacobi": jac.get("mean"),
            "rho_F_sgs_ssor": sgs.get("mean"),
            "rho_TG_4x": tg.get("mean"),
        })

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("-" * 100)
    print("wrote %s" % out_csv)
    print()
    print("Reading: if rho(12k) is still well above the Jacobi and SGS columns, the")
    print("plain MLP's deficit is not a budget artefact and the main comparison stands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
