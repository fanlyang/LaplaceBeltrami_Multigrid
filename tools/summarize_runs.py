#!/usr/bin/env python3
"""Collate several deeponet_smoother.py runs into one comparison table.

Every figure is read back out of each run's ``metrics.json``, so the summary
cannot drift from what was actually measured -- nothing here is retyped by hand.

    python tools/summarize_runs.py results/trunk_trig_p1024 results/trunk_fourier8 ...
    python tools/summarize_runs.py --glob 'results/*/metrics.json' --csv summary.csv
"""
import argparse
import glob
import json
import os
import sys

MECHS = ("ALL", "smooth", "multiscale", "localized", "algebraic")
CLASSICAL = ("damped Jacobi", "Gauss-Seidel", "sym. Gauss-Seidel", "SSOR")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def describe(m, path):
    mod = m.get("model", {})
    tr = m.get("training", {})
    name = mod.get("trunk_features", "?")
    if name == "fourier":
        name = "fourier%d" % (mod.get("trunk_modes") or 0)
    if mod.get("base_smoother", "none") != "none":
        name += "+" + mod["base_smoother"]
    # two_grid gained a "loss_vs_measured" entry, which is a consistency statement
    # rather than a smoother row: keep only the per-method rows here so the table
    # below cannot trip over a dict without "smoother_only".
    tg = {k: v for k, v in m.get("two_grid", {}).items()
          if isinstance(v, dict) and "smoother_only" in v}
    return {
        "run": os.path.basename(os.path.dirname(path)),
        "trunk": name,
        "p": mod.get("p"),
        "params": mod.get("parameters"),
        # log_points replaced an earlier, misleadingly named "epochs_run"
        "log_points": tr.get("log_points", tr.get("epochs_run")),
        "last_epoch": tr.get("last_epoch"),
        "best_epoch": tr.get("best_epoch"),
        "val_energy": tr.get("best_val_energy"),
        "objective": tr.get("objective"),
        "n_dof": m.get("scope", {}).get("fixed_dof_count"),
        "train_s": tr.get("train_seconds"),
        "omega": mod.get("learned_omega"),
        "smooth": m.get("smoothers", {}),
        "loss_target": m.get("loss_target", {}),
        "iterations": m.get("smoother_iterations", {}),
        "mg_errors": m.get("mg_iteration_errors", {}),
        "two_grid": tg,
        "two_grid_consistency": m.get("two_grid", {}).get("loss_vs_measured"),
        "vcycle": m.get("vcycle", {}),
        "galerkin": m.get("hierarchy_galerkin", {}),
        "overlaps": m.get("provenance", {}).get("split_audit", {}).get(
            "cross_split_signature_overlaps"),
    }


def cgc_energy(r, op, method="DeepONet"):
    """Mean L_CGC on the test set under coarse operator ``op``, or None."""
    return (r["loss_target"].get("cgc_test", {}).get(op, {})
            .get(method, {}).get("energy_mean"))


def le_energy(r, method="DeepONet"):
    """Mean L_E on the test set (the mean of the squared ratios), or None."""
    return r["loss_target"].get("energy_test", {}).get(method)


def cell(v, fmt="%.5f"):
    return "-" if v is None else fmt % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="result directories")
    ap.add_argument("--glob", default=None, help="glob for metrics.json files")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    paths = []
    for d in args.dirs:
        p = d if d.endswith(".json") else os.path.join(d, "metrics.json")
        if not os.path.exists(p):
            sys.exit("no metrics.json at %s" % p)
        paths.append(p)
    if args.glob:
        paths += sorted(glob.glob(args.glob))
    if not paths:
        sys.exit("nothing to summarize")

    runs = [describe(load(p), p) for p in paths]

    print("# Run comparison\n")
    # "log points" counts reported evaluations; best epoch is the one that mattered.
    # A last_epoch column was tried and dropped: runs predating that key all show "-".
    print("| run | trunk | p | params | objective | log pts | best epoch | val energy | train s |")
    print("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for r in runs:
        print("| %s | %s | %s | %s | %s | %s | %s | %.5f | %.0f |" % (
            r["run"], r["trunk"], r["p"], r["params"], r["objective"] or "-",
            r["log_points"], r["best_epoch"], r["val_energy"], r["train_s"]))

    print("\n## Mean test energy ratio  ||e - de||_A / ||e||_A  (lower is better)\n")
    hdr = "| run | " + " | ".join(MECHS) + " |"
    print(hdr)
    print("| --- |" + " ---: |" * len(MECHS))
    for r in runs:
        cells = []
        for m in MECHS:
            c = r["smooth"].get(m, {}).get("DeepONet")
            cells.append("%.4f" % c["energy_ratio_mean"] if c else "-")
        print("| %s | %s |" % (r["run"], " | ".join(cells)))

    # The objective, on the test set.  L_E is smoothing only; L_CGC is smoothing
    # followed by one exact coarse-grid correction, under each coarse operator.
    # These are energies (mean squared ratios), so unlike the table above they are
    # the numbers the trainer actually minimises.
    print("\n## Test-set objective: L_E and L_CGC (energies; lower is better)\n")
    print("| run | L_E | L_CGC assembled | L_CGC galerkin | coarse op in loss |")
    print("| --- | ---: | ---: | ---: | --- |")
    for r in runs:
        print("| %s | %s | %s | %s | %s |" % (
            r["run"], cell(le_energy(r)), cell(cgc_energy(r, "assembled")),
            cell(cgc_energy(r, "galerkin")),
            r["loss_target"].get("coarse_operator_for_loss") or "-"))

    print("\n## Per mechanism, best classical reference for the same runs\n")
    print("| run | mechanism | DeepONet | " + " | ".join(CLASSICAL) + " |")
    print("| --- | --- |" + " ---: |" * (len(CLASSICAL) + 1))
    for r in runs:
        for m in MECHS:
            row = r["smooth"].get(m, {})
            if "DeepONet" not in row:
                continue
            cells = ["%.4f" % row["DeepONet"]["energy_ratio_mean"]]
            for c in CLASSICAL:
                cells.append("%.4f" % row[c]["energy_ratio_mean"] if c in row else "-")
            print("| %s | %s | %s |" % (r["run"], m, " | ".join(cells)))

    print("\n## Worst case per mechanism (a mean hides the samples a smoother worsens)\n")
    print("| run | mechanism | worst ratio | fraction amplified |")
    print("| --- | --- | ---: | ---: |")
    for r in runs:
        for m in MECHS:
            c = r["smooth"].get(m, {}).get("DeepONet")
            if not c or "energy_ratio_max" not in c:
                continue
            print("| %s | %s | %.4f | %.4f |" % (
                r["run"], m, c["energy_ratio_max"], c["frac_amplified"]))

    # Repeated application: a smoother is used inside an iteration, not once.
    if any(r["iterations"] for r in runs):
        print("\n## Repeated application  e <- e - B(A e):  mean energy ratio\n")
        print("| run | it. 1 | it. 2 | it. 3 | it. 4 | it. 5 | monotone |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in runs:
            it = r["iterations"]
            if not it:
                continue
            ks = sorted((k for k in it if k.startswith("iter_")),
                        key=lambda k: int(k.split("_")[1]))
            vals = [cell(it[k]["deeponet_mean"], "%.4f") for k in ks[:5]]
            vals += ["-"] * (5 - len(vals))
            print("| %s | %s | %s |" % (r["run"], " | ".join(vals),
                                        it.get("monotone_mean")))

    if any(r["two_grid"] for r in runs):
        print("\n## Two-grid / V-cycle (real P and coarse operator)\n")
        print("| run | smoother | smoother only | coarse only | two-grid |")
        print("| --- | --- | ---: | ---: | ---: |")
        for r in runs:
            for k, v in r["two_grid"].items():
                print("| %s | %s | %.4f | %.4f | %.4f |" % (
                    r["run"], k, v["smoother_only"], v["coarse_only"], v["two_grid"]))
        for r in runs:
            for k, v in r["vcycle"].items():
                # newer runs iterate the cycle and carry the whole contraction
                # history; older ones only the first application
                extra = ""
                if "iterations" in v:
                    extra = " (after %d: %.5f, worst sample %.4f)" % (
                        len(v["iterations"]), v["iterations"][-1],
                        v.get("worst_sample_final", float("nan")))
                print("| %s | V-cycle %s | - | - | %.4f%s |" % (
                    r["run"], k, v["mean_reduction"], extra))

    # Real multigrid-iteration errors: no generator produced these.
    if any(r["mg_errors"] for r in runs):
        print("\n## On the errors a real V-cycle leaves behind (not generated)\n")
        print("| run | error after | DeepONet | damped Jacobi | sym. Gauss-Seidel |")
        print("| --- | --- | ---: | ---: | ---: |")
        for r in runs:
            for k in sorted(r["mg_errors"]):
                st = r["mg_errors"][k]["methods"]
                print("| %s | %s | %.4f | %.4f | %.4f |" % (
                    r["run"], k.replace("_", " "),
                    st["DeepONet"]["energy_ratio_mean"],
                    st["damped Jacobi"]["energy_ratio_mean"],
                    st["sym. Gauss-Seidel"]["energy_ratio_mean"]))

    # Does the loss contain the correction the cycle executes?
    if any(r["two_grid_consistency"] for r in runs):
        print("\n## Loss versus measured cycle (same samples; must agree)\n")
        for r in runs:
            c = r["two_grid_consistency"]
            if c:
                print("- %s: L_CGC %.10f vs measured two-grid energy %.10f "
                      "(difference %.2e)"
                      % (r["run"], c["loss_L_CGC_selected_samples"],
                         c["measured_two_grid_energy_ratio_sq"], c["abs_diff"]))

    if any(r["galerkin"] for r in runs):
        print("\n## Hierarchy Galerkin check (P^T A P vs the assembled coarse operator)\n")
        for r in runs:
            if r["galerkin"]:
                print("- %s: relative max|P^T A P - A_H| = %.3e"
                      % (r["run"], r["galerkin"].get("rel_max_abs_diff", float("nan"))))

    overlaps = {r["run"]: r["overlaps"] for r in runs}
    print("\n## Split audit (cross-split signature overlaps; 0 required)\n")
    for k, v in overlaps.items():
        print("- %s: %s" % (k, v))

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            import csv as _csv
            w = _csv.writer(fh)
            w.writerow(["run", "trunk", "p", "params", "log_points", "objective",
                        "val_energy", "train_s",
                        "L_E", "L_CGC_assembled", "L_CGC_galerkin",
                        "vcycle_first", "vcycle_after_k"]
                       + ["DeepONet_" + m for m in MECHS]
                       + ["jacobi_" + m for m in MECHS]
                       + ["sgs_" + m for m in MECHS])
            for r in runs:
                vc_learned = next((v for k, v in r["vcycle"].items()
                                   if k.startswith("learned")), None)
                row = [r["run"], r["trunk"], r["p"], r["params"], r["log_points"],
                       r["objective"] or "", r["val_energy"], r["train_s"],
                       le_energy(r), cgc_energy(r, "assembled"),
                       cgc_energy(r, "galerkin"),
                       None if vc_learned is None else vc_learned.get("mean_reduction"),
                       None if vc_learned is None else
                       (vc_learned.get("iterations") or [None])[-1]]
                for meth, key in (("DeepONet", "DeepONet"), ("jacobi", "damped Jacobi"),
                                  ("sgs", "sym. Gauss-Seidel")):
                    for m in MECHS:
                        c = r["smooth"].get(m, {}).get(key)
                        row.append(c["energy_ratio_mean"] if c else "")
                w.writerow(row)
        print("\nwrote %s" % args.csv)


if __name__ == "__main__":
    main()
