"""Collect the four per-epsilon runs into the cross-epsilon results.

    python -m hcsm.collect --results results/stage1 --out results/stage1

This produces the figures that need more than one epsilon (the coefficient
panels, the epsilon-sweeps, the controlled-mode scans), the machine-readable
summary tables, and the markdown tables that ``README.md`` quotes. Every number
in the summary is read back from the per-run ``metrics.json`` files, so nothing
is transcribed by hand and nothing can drift from the run that produced it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import plots
from .problem import EPSILONS, eps_tag, load_problem

#: The statistics written to the summary CSV, in order.
STATS = ("n", "mean", "median", "std", "p95", "max", "frac_gt_1", "mean_energy")


def fem_csv_names(eps: float) -> List[str]:
    """Candidate filenames for one epsilon's verification table.

    ``experiments/run_fem_exports.sh`` names them from its shell array, so they
    are ``fem_eps-1e-1.csv`` and friends -- NOT ``fem_eps-0.1.csv``, which is what
    ``"%s" % 1e-1`` would produce. Both spellings are tried, and both are listed
    here rather than one being assumed, because getting this wrong does not
    raise: it silently drops the verification figure, which is the one figure the
    specification says must be checked before anything else is interpreted.
    """
    n = int(round(-np.log10(eps)))
    names = ["fem_eps-1e-%d.csv" % n, "fem_eps-%g.csv" % eps, "fem_eps-%s.csv" % eps]
    return list(dict.fromkeys(names))          # dedupe, keep order


def read_fem_table(path: str) -> List[Dict[str, float]]:
    """One solver verification CSV -> a list of per-cycle rows."""
    rows: List[Dict[str, float]] = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            rows.append({k: (v if k == "method" or k == "converged" else float(v))
                         for k, v in row.items()})
    return rows


def load_metrics(results_dir: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "eps-*", "metrics.json"))):
        with open(p) as fh:
            m = json.load(fh)
        out[m["tag"]] = m
    if not out:
        raise SystemExit("no metrics.json found under %s/eps-*/" % results_dir)
    return out


def summary_rows(metrics: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """One flat row per (epsilon, section, method) -- the tidy form of everything."""
    rows: List[Dict[str, object]] = []
    for tag, m in metrics.items():
        eps = m["epsilon"]
        for section in ("rho_F", "rho_C", "rho_TG", "rho_TG_on_eF",
                        "rho_F_repeated_3_steps"):
            block = m.get(section)
            if not block:
                continue
            for method, stats in block.items():
                for s in STATS:
                    if s in stats:
                        rows.append({
                            "epsilon": eps, "tag": tag, "section": section,
                            "method": method, "statistic": s,
                            "value": stats[s],
                        })
        for arch, info in m.get("training", {}).items():
            rows.append({"epsilon": eps, "tag": tag, "section": "training",
                         "method": "deeponet_" + arch, "statistic": "best_val_loss",
                         "value": info["best_val_loss"]})
            rows.append({"epsilon": eps, "tag": tag, "section": "training",
                         "method": "deeponet_" + arch, "statistic": "best_epoch",
                         "value": info["best_epoch"]})
            if "omega_trained" in info:
                rows.append({"epsilon": eps, "tag": tag, "section": "training",
                             "method": "deeponet_" + arch,
                             "statistic": "omega_trained",
                             "value": info["omega_trained"]})
    return rows


def write_summary_csv(rows: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epsilon", "tag", "section", "method",
                                           "statistic", "value"])
        w.writeheader()
        w.writerows(rows)


def markdown_table(metrics: Dict[str, Dict[str, object]], section: str,
                   statistic: str, methods: Sequence[str]) -> str:
    """A README-ready table of one statistic across epsilons."""
    tags = sorted(metrics.keys(), key=lambda t: -float(t))
    head = "| method | " + " | ".join(
        r"$\varepsilon=10^{%d}$" % int(round(np.log10(float(t)))) for t in tags) + " |"
    sep = "|" + "---|" * (len(tags) + 1)
    lines = [head, sep]
    for meth in methods:
        cells = []
        for t in tags:
            block = metrics[t].get(section, {})
            cell = block.get(meth)
            cells.append("--" if cell is None else "%.4f" % cell[statistic])
        lines.append("| %s | %s |" % (plots.style.METHOD_LABEL.get(meth, meth),
                                      " | ".join(cells)))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results/stage1")
    ap.add_argument("--data-root", default="level_data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fem-csv", default="results")
    args = ap.parse_args(argv)

    out = args.out or args.results
    plot_dir = os.path.join(out, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    metrics = load_metrics(args.results)
    print("loaded %d epsilon runs: %s" % (len(metrics), ", ".join(sorted(metrics))))

    # -- tidy summary ------------------------------------------------------
    rows = summary_rows(metrics)
    write_summary_csv(rows, os.path.join(out, "summary.csv"))
    print("  wrote summary.csv (%d rows)" % len(rows))

    # -- 1. kappa on the torus --------------------------------------------
    problems = [load_problem(args.data_root, e) for e in EPSILONS]
    plots.plot_kappa_torus(problems, os.path.join(plot_dir, "kappa_on_torus.png"))

    # -- 2-4. epsilon sweeps ----------------------------------------------
    summary = {
        tag: {sec: m.get(sec, {}) for sec in ("rho_F", "rho_C", "rho_TG")}
        for tag, m in metrics.items()
    }
    plots.plot_eps_curves(
        summary, "rho_F", "mean", r"mean $\rho_F$",
        os.path.join(plot_dir, "eps_vs_mean_rhoF.png"),
        "Mean reduction of the coarse-space-complement error, one smoothing step",
        reference_line=1.0, reference_label="no reduction")
    plots.plot_eps_curves(
        summary, "rho_F", "p95", r"95th percentile $\rho_F$",
        os.path.join(plot_dir, "eps_vs_p95_rhoF.png"),
        "95th-percentile reduction: the tail a mean hides",
        reference_line=1.0, reference_label="no reduction")
    plots.plot_eps_curves(
        summary, "rho_F", "max", r"max $\rho_F$",
        os.path.join(plot_dir, "eps_vs_max_rhoF.png"),
        "Worst-case reduction over the test set",
        reference_line=1.0, reference_label="no reduction")
    plots.plot_eps_curves(
        summary, "rho_TG", "mean", r"mean $\rho_{TG}$",
        os.path.join(plot_dir, "eps_vs_mean_rhoTG.png"),
        "Ideal Galerkin two-grid: one smoothing step plus an exact coarse solve",
        reference_line=1.0, reference_label="no reduction")
    plots.plot_eps_curves(
        summary, "rho_C", "mean", r"mean $\rho_C$",
        os.path.join(plot_dir, "eps_vs_mean_rhoC.png"),
        "The smoother applied to the COARSE component instead (diagnostic only; "
        "the coarse correction is what handles these in a real cycle)",
        reference_line=1.0, reference_label="no reduction")

    # -- 5-6. controlled modes across epsilons ----------------------------
    scans = {tag: m["controlled_modes"] for tag, m in metrics.items()}
    plots.plot_mode_scan(scans, "xi", os.path.join(plot_dir, "modes_xi_all_eps.png"))
    plots.plot_mode_scan(scans, "eta", os.path.join(plot_dir, "modes_eta_all_eps.png"))

    # -- training curves, all epsilons and both variants on one chart ------
    histories: Dict[Sequence, List[Dict[str, object]]] = {}
    for tag, m in metrics.items():
        for arch in ("plain", "skip"):
            p = os.path.join(args.results, "eps-%s" % tag, "history_%s.csv" % arch)
            if os.path.exists(p):
                rows = []
                with open(p) as fh:
                    for r in csv.DictReader(fh):
                        rows.append({"epoch": int(float(r["epoch"])),
                                     "val_loss": float(r["val_loss"]),
                                     "train_loss": float(r["train_loss"])})
                histories[(m["epsilon"], arch)] = rows
    if histories:
        plots.plot_training(histories, os.path.join(plot_dir, "training_all.png"))

    # -- the finite-element verification ----------------------------------
    fem = {}
    for eps in EPSILONS:
        for name in fem_csv_names(eps):
            p = os.path.join(args.fem_csv, name)
            if os.path.exists(p):
                fem[eps_tag(eps)] = read_fem_table(p)
                break
        else:
            print("  WARNING: no verification table for epsilon=%g; looked for %s"
                  % (eps, ", ".join(fem_csv_names(eps))))
    if fem:
        plots.plot_fem_convergence(fem, os.path.join(plot_dir, "fem_verification.png"))
        with open(os.path.join(out, "fem_verification.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epsilon", "cycle", "dofs", "h1_error", "l2_error",
                        "linf_error", "h1_rate", "l2_rate"])
            for tag, rows in sorted(fem.items()):
                for i, r in enumerate(rows):
                    def rate(key):
                        if i == 0:
                            return ""
                        return "%.4f" % (np.log2(rows[i - 1][key] / r[key]))
                    w.writerow([tag, r["cycle"], r["dofs"], r["h1_error"],
                                r["l2_error"], r["linf_error"], rate("h1_error"),
                                rate("l2_error")])
        print("  wrote fem_verification.csv")

    # -- the DeepONet-variant ablation -------------------------------------
    plots.plot_variant_bars(
        summary, plots.VARIANT_METHODS, "rho_F", "mean",
        os.path.join(plot_dir, "variant_ablation_rhoF.png"),
        r"mean $\rho_F$ (one smoothing step)",
        "Which DeepONet configuration, if any, matches the classical smoothers?\n"
        "bars are one hue by design: the identity is the row label, not the colour")
    plots.plot_variant_bars(
        summary, plots.VARIANT_METHODS, "rho_TG", "mean",
        os.path.join(plot_dir, "variant_ablation_rhoTG.png"),
        r"mean $\rho_{TG}$ (ideal Galerkin two-grid)",
        "The same variants inside the ideal two-grid method\n"
        "the coarse-correction floor is the dashed 'coarse correction only' row")

    # -- README tables -----------------------------------------------------
    table_methods = ["jacobi", "gs", "sgs_ssor", "deeponet_plain", "deeponet_skip",
                     "deeponet_skip_fourier"]
    md = []
    for title, section, stat in (
        ("Mean rho_F (one smoothing step, unseen test set)", "rho_F", "mean"),
        ("95th-percentile rho_F", "rho_F", "p95"),
        ("Worst-case rho_F over the test set", "rho_F", "max"),
        ("Fraction of test errors AMPLIFIED (rho_F > 1)", "rho_F", "frac_gt_1"),
        ("Mean rho_C (the smoother on the coarse component)", "rho_C", "mean"),
        ("Mean rho_TG (ideal Galerkin two-grid)", "rho_TG", "mean"),
        ("95th-percentile rho_TG", "rho_TG", "p95"),
        ("Mean rho_TG run on the e_F samples", "rho_TG_on_eF", "mean"),
        ("Mean rho_F after 3 smoothing steps", "rho_F_repeated_3_steps", "mean"),
    ):
        md.append("#### %s\n\n%s\n" % (title, markdown_table(metrics, section, stat,
                                                             table_methods)))
    with open(os.path.join(out, "readme_tables.md"), "w") as fh:
        fh.write("\n".join(md))
    print("  wrote readme_tables.md")

    # -- a compact console summary ----------------------------------------
    print()
    print("=" * 92)
    print("%-28s %s" % ("mean rho_F", "  ".join(
        "eps=%-8s" % t for t in sorted(metrics, key=lambda x: -float(x)))))
    print("-" * 92)
    for meth in table_methods:
        cells = []
        for t in sorted(metrics, key=lambda x: -float(x)):
            cell = metrics[t].get("rho_F", {}).get(meth)
            cells.append("%-12.4f" % cell["mean"] if cell else "--")
        print("%-28s %s" % (plots.style.METHOD_LABEL.get(meth, meth),
                            "  ".join(cells)))
    print("-" * 92)
    print("%-28s %s" % ("mean rho_TG", "  ".join(
        "eps=%-8s" % t for t in sorted(metrics, key=lambda x: -float(x)))))
    print("-" * 92)
    for meth in table_methods + ["coarse_only"]:
        cells = []
        for t in sorted(metrics, key=lambda x: -float(x)):
            cell = metrics[t].get("rho_TG", {}).get(meth)
            cells.append("%-12.4f" % cell["mean"] if cell else "--")
        print("%-28s %s" % (plots.style.METHOD_LABEL.get(meth, meth),
                            "  ".join(cells)))
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
