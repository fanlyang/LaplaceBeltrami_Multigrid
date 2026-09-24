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
    return {
        "run": os.path.basename(os.path.dirname(path)),
        "trunk": name,
        "p": mod.get("p"),
        "params": mod.get("parameters"),
        "epochs": tr.get("epochs_run"),
        "best_epoch": tr.get("best_epoch"),
        "val_energy": tr.get("best_val_energy"),
        "n_dof": m.get("scope", {}).get("fixed_dof_count"),
        "train_s": tr.get("train_seconds"),
        "omega": mod.get("learned_omega"),
        "smooth": m.get("smoothers", {}),
        "two_grid": m.get("two_grid", {}),
        "vcycle": m.get("vcycle", {}),
        "galerkin": m.get("hierarchy_galerkin", {}),
        "overlaps": m.get("provenance", {}).get("split_audit", {}).get(
            "cross_split_signature_overlaps"),
    }


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
    print("| run | trunk | p | params | epochs | best epoch | val energy | train s |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in runs:
        print("| %s | %s | %s | %s | %s | %s | %.5f | %.0f |" % (
            r["run"], r["trunk"], r["p"], r["params"], r["epochs"], r["best_epoch"],
            r["val_energy"], r["train_s"]))

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
                print("| %s | V-cycle %s | - | - | %.4f |" % (r["run"], k, v["mean_reduction"]))

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
            w.writerow(["run", "trunk", "p", "params", "epochs", "val_energy", "train_s"]
                       + ["DeepONet_" + m for m in MECHS]
                       + ["jacobi_" + m for m in MECHS]
                       + ["sgs_" + m for m in MECHS])
            for r in runs:
                row = [r["run"], r["trunk"], r["p"], r["params"], r["epochs"],
                       r["val_energy"], r["train_s"]]
                for meth, key in (("DeepONet", "DeepONet"), ("jacobi", "damped Jacobi"),
                                  ("sgs", "sym. Gauss-Seidel")):
                    for m in MECHS:
                        c = r["smooth"].get(m, {}).get(key)
                        row.append(c["energy_ratio_mean"] if c else "")
                w.writerow(row)
        print("\nwrote %s" % args.csv)


if __name__ == "__main__":
    main()
