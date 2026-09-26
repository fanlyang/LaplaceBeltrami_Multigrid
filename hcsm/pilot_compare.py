"""Re-evaluate both checkpoints with identical corrected tests and FGMRES.

python -m hcsm.pilot_compare --baseline results/corrected-pilot/baseline \
    --corrected results/corrected-pilot/corrected --out results/corrected-pilot
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch
import pyamg

from .aniso_errors import draw_mixture, equal_plan, stress_set
from .aniso_evaluate import evaluate_families
from .evaluate import MethodSet, Smoothers
from .losses import ScaleEquivariant
from .model import build_features, build_model
from .problem import load_problem
from .run_tensor import Mixture, WrappedSmoother, _as_family, hierarchy_dirs
from .train import restore_best
from .vcycle import Hierarchy, ClassicalSmoothing, LearnedSmoothing, measure_vcycle, solve_with_preconditioner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    p = load_problem("level_data", 1e-4, level=3, verbose=False)
    features = build_features(p)
    nets = {}
    checkpoints = {}
    for name in ("baseline", "corrected"):
        path = Path(getattr(args, name)) / "best_model.pt"
        model = ScaleEquivariant(build_model(p, features.shape[1], base_smoother="jacobi", seed=0))
        saved = restore_best(model, path)
        nets[name] = WrappedSmoother(model, features)
        checkpoints[name] = {"saved_step_label": saved["epoch"],
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    sample_hash = {}
    for role in ("train", "val", "test"):
        families = draw_mixture(p, equal_plan(96), 0, role)
        sample_hash[role] = hashlib.sha256(Mixture(families).E.tobytes()).hexdigest()
    expected = json.loads((Path(args.corrected)/"metrics.json").read_text())["sample_sha256"]
    if sample_hash != expected:
        raise AssertionError("common evaluation dataset differs from corrected training run")
    families["stress"] = _as_family(stress_set(p))
    methods = MethodSet(Smoothers(p.A), nets, 2/3, 1.)
    measured = evaluate_families(p, methods, families)
    rows = []
    for family, block in measured.items():
        for method, d in block.items():
            for metric in ("full", "complement"):
                rows.append({"family": family, "method": method, "metric": metric,
                             **d[metric].as_dict(), "seconds_per_sample": d["seconds"]/d["n"]})
    hierarchy = Hierarchy.from_dirs(hierarchy_dirs("level_data", 1e-4, 3))
    smoothers = {k: ClassicalSmoothing(hierarchy, k) for k in ("jacobi", "gs", "sgs")}
    smoothers.update({k: LearnedSmoothing(hierarchy, net.model, features, 3)
                     for k, net in nets.items()})
    cycle = {}
    b = np.random.default_rng(7).normal(size=p.n)
    for name, smoother in smoothers.items():
        cycle[name] = {"stationary": measure_vcycle(hierarchy, smoother, 3, seed=0),
                       "outer": solve_with_preconditioner(hierarchy, smoother, 3, b, flexible=True)}
        print(name, cycle[name], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metrics = {"epsilon": 1e-4, "level": 3, "seed": 0, "threads": 4,
               "sample_sha256": sample_hash, "checkpoints": checkpoints,
               "smoothing": rows, "cycles": cycle,
               "environment": {"python": platform.python_version(), "platform": platform.system(),
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "torch": torch.__version__, "pyamg": pyamg.__version__},
               "scope": "One seed and one mesh; all common outer solves use FGMRES. Stored FEM artifacts reused."}
    (out/"comparison.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
