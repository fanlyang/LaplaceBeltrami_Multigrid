#!/usr/bin/env python3
"""Break down the cost of one training step, to explain why larger levels are slow.

This does not train anything.  It builds the same model configuration the
experiments use, times each component of a single optimisation step, and reports
the analytic operation counts and the memory each part holds.  The point is to
replace "level 5 felt slow" with a measured cost model.

    python tools/profile_step.py --levels level_data/L3 level_data/L4 level_data/L5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deeponet_smoother as ds  # noqa: E402

try:                       # the sparse-invariant notice is noise for a fixed pattern
    torch.sparse.check_sparse_tensor_invariants.disable()
except Exception:
    pass


def macs(n_dof, width, p, feat_dim, batch):
    """Multiply-accumulates for one forward pass."""
    trunk = n_dof * (feat_dim * width + width * width + width * p)
    branch = batch * (n_dof * width + width * width + width * p)
    out = batch * p * n_dof
    return {"trunk": trunk, "branch": branch, "output matmul": out,
            "total": trunk + branch + out}


def timed(fn, n_warmup=2, n_iter=5):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter


def human(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return "%.1f %s" % (b, unit) if unit != "B" else "%d B" % b
        b /= 1024.0


def profile(level_dir, p, trunk_features, batch, n_train):
    ld = ds.load_level(level_dir)
    n = ld.n
    feats_np = ds.trunk_features(ld.angles, trunk_features, ld.coords, 4)
    feat_dim = feats_np.shape[1]
    model = ds.DeepONetSmoother(n, feat_dim, p, 128, 3, False,
                                "jacobi", ld.A.diagonal(), 2.0 / 3.0).double()
    A_t = ds.to_torch_sparse(ld.A)
    AT_t = ds.to_torch_sparse(ld.A.T.tocsr())
    F = torch.from_numpy(feats_np).double()
    g = torch.Generator().manual_seed(0)
    R = torch.randn(batch, n, generator=g, dtype=torch.float64)
    E = torch.randn(batch, n, generator=g, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_par = sum(q.numel() for q in model.parameters())
    with torch.no_grad():
        t_trunk = timed(lambda: model.trunk(F))
        t_branch = timed(lambda: model.branch(R))
        b = model.branch(R)
        tvec = model.trunk(F)
        t_out = timed(lambda: b @ tvec.t())
        corr = b @ tvec.t()
        t_loss = timed(lambda: ds.energy_loss(corr, E, A_t, AT_t))

    def step():
        opt.zero_grad(set_to_none=True)
        loss = ds.energy_loss(model(R, F, None), E, A_t, AT_t)
        loss.backward()
        opt.step()

    t_step = timed(step, n_warmup=1, n_iter=3)

    m = macs(n, 128, p, feat_dim, batch)
    # memory actually held, in float64
    param_b = n_par * 8
    adam_b = 2 * param_b
    grad_b = param_b
    data_b = 2 * n_train * n * 8 + 2 * n * 8          # E and R for the training split
    trunk_act_b = n * p * 8                            # the (n, p) trunk output
    return {
        "level": os.path.basename(os.path.abspath(level_dir)),
        "n_dof": n, "p": p, "feat_dim": feat_dim, "batch": batch,
        "params": n_par,
        "macs": m,
        "trunk_share": m["trunk"] / m["total"],
        "times": {"trunk fwd": t_trunk, "branch fwd": t_branch, "output matmul": t_out,
                  "energy loss": t_loss, "full step (fwd+bwd+opt)": t_step},
        "mem": {"parameters": param_b, "adam state (2x)": adam_b, "gradients": grad_b,
                "trunk output (n,p)": trunk_act_b,
                "training data E+R": data_b,
                "total": param_b + adam_b + grad_b + trunk_act_b + data_b},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+",
                    default=["level_data/L3", "level_data/L4", "level_data/L5"])
    ap.add_argument("--p", nargs="+", type=int, default=[128, 256, 256])
    ap.add_argument("--batch", nargs="+", type=int, default=[32, 16, 8])
    ap.add_argument("--n-train", nargs="+", type=int, default=[1024, 512, 256])
    ap.add_argument("--trunk-features", default="trig")
    args = ap.parse_args()

    results = []
    print("Cost of ONE training step, same configuration the experiments use.\n")
    for lvl, p, b, nt in zip(args.levels, args.p, args.batch, args.n_train):
        if not os.path.isdir(lvl):
            print("skip %s (missing)" % lvl)
            continue
        r = profile(lvl, p, args.trunk_features, b, nt)
        results.append(r)
        print("=" * 78)
        print("%s   n_dof=%d  p=%d  batch=%d  params=%d"
              % (r["level"], r["n_dof"], r["p"], r["batch"], r["params"]))
        print("  analytic multiply-accumulates per forward pass:")
        for k in ("trunk", "branch", "output matmul", "total"):
            print("    %-16s %15.4g" % (k, r["macs"][k]))
        print("    trunk share of the FLOPs: %.1f%%" % (100 * r["trunk_share"]))
        print("  measured time per component [ms]:")
        for k, v in r["times"].items():
            print("    %-28s %8.2f" % (k, 1000 * v))
        print("  memory held [float64]:")
        for k, v in r["mem"].items():
            print("    %-22s %-10s" % (k, human(v)))

    if len(results) >= 2:
        print("\n" + "=" * 78)
        print("Scaling from %s to %s:" % (results[0]["level"], results[-1]["level"]))
        a, b = results[0], results[-1]
        print("  n_dof      x %.1f" % (b["n_dof"] / a["n_dof"]))
        print("  params     x %.1f" % (b["params"] / a["params"]))
        print("  FLOPs/step x %.1f" % (b["macs"]["total"] / a["macs"]["total"]))
        print("  time/step  x %.1f" % (b["times"]["full step (fwd+bwd+opt)"] /
                                        a["times"]["full step (fwd+bwd+opt)"]))
        print("  memory     x %.1f" % (b["mem"]["total"] / a["mem"]["total"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
