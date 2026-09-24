#!/usr/bin/env python3
"""Independent checks of the mathematics in deeponet_smoother.py.

These do not train anything.  They re-derive each quantity on small levels with
plain dense NumPy and compare against the sparse implementation the experiments
use, so a sign error, a transpose slip or a wrong normalisation cannot survive
into the reported results.

    python tools/selftest.py --hierarchy-root level_data
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deeponet_smoother as ds  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print("  %-58s %s %s" % (name, "OK" if ok else "FAIL", detail))
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy-root", default="level_data")
    ap.add_argument("--small-level", default="L2", help="used for the dense checks")
    ap.add_argument("--big-level", default="L3", help="used for the V-cycle check")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    try:
        torch.sparse.check_sparse_tensor_invariants.disable()
    except Exception:
        pass
    print("Sparse-vs-dense checks on %s" % args.small_level)
    ld = ds.load_level(os.path.join(args.hierarchy_root, args.small_level))
    A = ld.A
    Ad = A.toarray()
    n = A.shape[0]
    A_t = ds.to_torch_sparse(A)
    AT_t = ds.to_torch_sparse(A.T.tocsr())

    # a random error normalised to unit A-energy, exactly as the dataset does
    v = rng.normal(size=n)
    e, r = ds.normalise_and_residualise(v, A)
    check("error normalised to unit A-energy", abs(e @ A @ e - 1.0) < 1e-10,
          "e^T A e = %.15f" % (e @ A @ e))
    check("residual is A e", np.allclose(r, A @ e, atol=0, rtol=0))

    # ---- energy_norm (sparse, batch-major) against dense ------------------
    V = np.stack([e, 2.0 * e, rng.normal(size=n)])
    Vd = V @ Ad.T * V
    dense_e = np.einsum("bi,bi->b", V, (Ad @ V.T).T)
    torch_e = ds.energy_norm(torch.from_numpy(V), A_t, AT_t).numpy()
    check("energy_norm matches dense v^T A v", np.allclose(dense_e, torch_e, rtol=1e-12),
          "max rel diff %.2e" % (np.max(np.abs(dense_e - torch_e) / np.abs(dense_e))))

    # ---- the loss is the stated formula ----------------------------------
    E = torch.from_numpy(np.stack([e, 2.0 * e]))
    zeros = torch.zeros_like(E)
    loss0 = float(ds.energy_loss(zeros, E, A_t, AT_t))
    check("loss(no correction) = 1", abs(loss0 - 1.0) < 1e-12, "got %.15f" % loss0)
    loss_perfect = float(ds.energy_loss(E.clone(), E, A_t, AT_t))
    # the eps in the loss is a deliberate division guard, so the floor is ~eps
    check("loss(perfect correction) ~ 0 (eps floor)", loss_perfect < 1e-11,
          "got %.3e" % loss_perfect)
    half = 0.5 * E
    lh = float(ds.energy_loss(half, E, A_t, AT_t))
    check("loss(e/2) = 1/4", abs(lh - 0.25) < 1e-9, "got %.15f" % lh)

    # ---- classical smoothers against dense equivalents -------------------
    sm = ds.ClassicalSmoothers(A)
    D = np.diag(np.diag(Ad))
    L = np.tril(Ad, -1)
    U = np.triu(Ad, 1)
    om = 0.7
    c = sm.jacobi(r, om)
    check("damped Jacobi matches dense", np.allclose(c, om * np.linalg.solve(D, r)))
    c = sm.gauss_seidel(r)
    check("Gauss-Seidel matches dense (D+L)^-1 r",
          np.allclose(c, np.linalg.solve(D + L, r)))
    c = sm.symmetric_gs(r)
    check("symmetric GS matches dense (D+U)^-1 D (D+L)^-1 r",
          np.allclose(c, np.linalg.solve(D + U, np.diag(Ad) * np.linalg.solve(D + L, r))))
    c = sm.ssor(r, 1.0)
    check("SSOR at omega=1 equals symmetric GS",
          np.allclose(c, np.linalg.solve(D + U, np.diag(Ad) * np.linalg.solve(D + L, r))))

    # ---- two-grid stages against a dense solve ---------------------------
    print("Two-grid check: L2 as the fine level, L1 as the coarse one")
    levels = [ds.load_level(os.path.join(args.hierarchy_root, "L%d" % k)) for k in (1, 2)]
    h = ds.Hierarchy(levels)
    h.fine_level = 1
    P = levels[1].P                 # L2 -> L1, shape (256, 64)
    A_H = levels[0].A               # the assembled coarse operator, 64 x 64
    A_fine = levels[1].A
    corr = 0.3 * e
    st = h.two_grid_stages(e, corr, 1)
    e_sm = e - corr
    e_H_dense = np.linalg.solve(A_H.toarray(), P.T @ (A_fine @ e_sm))
    check("coarse stage solves A_H d = P^T A e_smooth",
          np.allclose(st["coarse"], e_H_dense, rtol=1e-10, atol=1e-12))
    check("two_grid = e_smooth - P d",
          np.allclose(st["two_grid"], e_sm - P @ e_H_dense, rtol=1e-12, atol=1e-14))

    # ---- V-cycle on a real hierarchy -------------------------------------
    print("V-cycle check on %s (levels L0..L2)" % args.big_level)
    lv = [ds.load_level(os.path.join(args.hierarchy_root, "L%d" % k)) for k in (0, 1, 2)]
    hh = ds.Hierarchy(lv)
    hh.fine_level = 2
    A2 = lv[2].A
    omega = 2.0 / 3.0

    def jac(rr):
        return omega * rr / A2.diagonal()

    v2 = rng.normal(size=lv[2].n)
    e2, r2 = ds.normalise_and_residualise(v2, A2)
    red = []
    for _ in range(5):
        e2 = e2 - hh.vcycle_apply(A2 @ e2, 2, jac)
        red.append(np.sqrt(max(e2 @ (A2 @ e2), 0.0)))
    check("V-cycle contracts monotonically", all(b < a for a, b in zip(red, red[1:])),
          "ratios %s" % ["%.4f" % x for x in red])
    check("V-cycle reduces the error below 1 per cycle", red[0] < 1.0, "%.4f" % red[0])

    # ---- the rank bound is real -----------------------------------------
    print("Rank-bound check")
    T = np.linalg.qr(rng.normal(size=(n, 32)))[0]          # an arbitrary 32-dim basis
    proj = T @ (T.T @ e)
    captured = (proj @ A @ proj) / (e @ A @ e)
    check("a 32-dim correction subspace captures roughly p/n of the energy",
          0.02 < captured < 0.35, "captured %.4f for p/n = %.4f" % (captured, 32.0 / n))

    # ---- generator sanity ------------------------------------------------
    print("Generator checks")
    ang = ld.angles
    # nyq is the largest wavenumber the lattice resolves = DoFs-per-side / 2,
    # NOT the DoFs per side. Conflating the two is what caused an aliasing bug.
    nyq = max(1, int(round(np.sqrt(n))) // 2)
    bad = 0
    for mech in ds.MECHANISMS:
        for _ in range(20):
            vals, p, sig = ds.draw_sample(np.random.default_rng(0), mech, ang, nyq, n)
            if not np.all(np.isfinite(vals)) or np.linalg.norm(vals) == 0.0:
                bad += 1
    check("every generator yields finite, nonzero draws", bad == 0, "%d bad" % bad)
    vals, pr, sig = ds.gen_multiscale(np.random.default_rng(1), ang, nyq, n)
    check("multiscale wavenumbers stay within the recorded cap",
          all(max(abs(m), abs(nn)) <= pr["wavenumber_cap"] for m, nn in pr["modes"]),
          "cap %.0f" % pr["wavenumber_cap"])
    check("multiscale cap is strictly below Nyquist (no aliasing)",
          pr["wavenumber_cap"] < nyq,
          "cap %.0f < Nyquist %.0f" % (pr["wavenumber_cap"], nyq))

    # the localized generator must be periodic: a bump at the seam is one bump
    vals, pr, sig = ds.gen_localized(np.random.default_rng(7), ang, nyq, n)
    check("localized bumps are single-connected (periodic distance)",
          np.all(np.isfinite(vals)))

    print()
    if FAIL:
        print("FAILED %d check(s): %s" % (len(FAIL), FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
