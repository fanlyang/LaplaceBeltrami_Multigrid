#!/usr/bin/env python3
"""Convert the deal.II level-data dumps into the NumPy/SciPy files that the
learned-smoother training script expects.

The solver's ``export_level_data`` writes, for each multigrid level ``l`` and
refinement cycle ``c``:

    A-level-<l>-cycle-<c>.coo           sparse level operator A_l (COO)
    dof-coords-level-<l>-cycle-<c>.txt  physical support points, one "x y z" a line
    coeff-level-<l>-cycle-<c>.txt       kappa sampled at those support points
    refinement-edge-level-<l>-cycle-<c>.txt   constrained DoF indices
    P-level-<l>-cycle-<c>.coo           prolongation from level l-1 to l

This script reads one fixed level (and the highest cycle, unless told otherwise)
and produces, in ``--out-dir`` (default ``level_data``):

    level_matrix.npz    scipy CSR A_l
    dof_coordinates.npy (n_dofs, 3) float64
    coeff_values.npy    (n_dofs,)   float64
    refinement_edge.npy (n_edge,)   int64   (empty on a uniform mesh)
    prolongation.npz    scipy CSR P (level l-1 -> l), only if a P-*.coo exists

Run from the directory that holds the solver dumps, e.g.

    python tools/convert_level_data.py --level 3
"""

import argparse
import glob
import os
import re

import numpy as np
import scipy.sparse as sp


def read_coo(path):
    # First line: n_rows n_cols; then "row column value" (zero-based, scientific).
    with open(path) as f:
        n_rows, n_cols = (int(x) for x in f.readline().split())
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    rows = data[:, 0].astype(int)
    cols = data[:, 1].astype(int)
    vals = data[:, 2]
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(n_rows, n_cols)).tocsr()


def read_vector(path):
    # First line: count; then one value per line.
    with open(path) as f:
        count = int(f.readline())
    data = np.loadtxt(path, skiprows=1)
    data = np.atleast_1d(data)
    assert data.shape[0] == count, (path, data.shape, count)
    return data


def read_coords(path):
    # First line: n_dofs; then one "x y z" per line.
    with open(path) as f:
        count = int(f.readline())
    data = np.loadtxt(path, skiprows=1)
    assert data.shape == (count, 3), (path, data.shape, count)
    return data


def find_cycle(data_dir, prefix, level):
    pat = os.path.join(data_dir, "%s-level-%d-cycle-*.coo" % (prefix, level))
    files = glob.glob(pat) or glob.glob(pat.replace(".coo", ".txt"))
    if not files:
        return None
    return max(int(re.search(r"cycle-(\d+)", f).group(1)) for f in files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--level", type=int, required=True,
                    help="multigrid level to export (the fixed training level)")
    ap.add_argument("--cycle", type=int, default=None,
                    help="refinement cycle (default: highest found)")
    ap.add_argument("--out-dir", default="level_data")
    args = ap.parse_args()

    level = args.level
    cycle = args.cycle
    if cycle is None:
        cycle = find_cycle(args.data_dir, "A", level)
        if cycle is None:
            raise SystemExit("no A-level-%d-cycle-*.coo in %s"
                             % (level, args.data_dir))

    tag = "level-%d-cycle-%d" % (level, cycle)
    d = args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)

    A_path = os.path.join(d, "A-%s.coo" % tag)
    coords_path = os.path.join(d, "dof-coords-%s.txt" % tag)
    coeff_path = os.path.join(d, "coeff-%s.txt" % tag)
    edge_path = os.path.join(d, "refinement-edge-%s.txt" % tag)
    P_path = os.path.join(d, "P-%s.coo" % tag)

    for p in (A_path, coords_path, coeff_path, edge_path):
        if not os.path.exists(p):
            raise SystemExit("missing %s" % p)

    A = read_coo(A_path)
    coords = read_coords(coords_path)
    coeff = read_vector(coeff_path)
    edge = read_vector(edge_path).astype(int)

    n = coords.shape[0]
    assert A.shape == (n, n), (A.shape, n)
    assert coeff.shape[0] == n, (coeff.shape, n)

    sp.save_npz(os.path.join(args.out_dir, "level_matrix.npz"), A)
    np.save(os.path.join(args.out_dir, "dof_coordinates.npy"), coords)
    np.save(os.path.join(args.out_dir, "coeff_values.npy"), coeff)
    np.save(os.path.join(args.out_dir, "refinement_edge.npy"), edge)

    print("level %d (cycle %d): %d dofs, %d nonzeros"
          % (level, cycle, n, A.nnz))
    print("  wrote level_matrix.npz, dof_coordinates.npy, "
          "coeff_values.npy, refinement_edge.npy -> %s" % args.out_dir)

    if os.path.exists(P_path):
        P = read_coo(P_path)
        assert P.shape[0] == n, (P.shape, n)
        sp.save_npz(os.path.join(args.out_dir, "prolongation.npz"), P)
        print("  wrote prolongation.npz (%d x %d) -> %s"
              % (P.shape[0], P.shape[1], args.out_dir))
    else:
        print("  no P-%s.coo; train with the energy loss alone" % tag)


if __name__ == "__main__":
    main()
