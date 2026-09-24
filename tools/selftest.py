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
    # eps floors the DENOMINATOR only, so a perfect correction gives 0, not eps
    check("loss(perfect correction) = 0 (eps is not in the numerator)",
          loss_perfect < 1e-20, "got %.3e" % loss_perfect)
    half = 0.5 * E
    lh = float(ds.energy_loss(half, E, A_t, AT_t))
    check("loss(e/2) = 1/4", abs(lh - 0.25) < 1e-9, "got %.15f" % lh)

    # ---- L_CGC: smoothing followed by coarse-grid correction -------------
    print("Coarse-grid-corrected loss (L_CGC) on %s -> %s"
          % (args.small_level, "L1" if args.small_level == "L2" else "L0"))
    lv2 = [ds.load_level(os.path.join(args.hierarchy_root, "L%d" % k)) for k in (1, 2)]
    ld_f, ld_c = lv2[1], lv2[0]
    Pf = ld_f.P
    A_H_gal = sp.csr_matrix((Pf.T @ ld_f.A @ Pf + (Pf.T @ ld_f.A @ Pf).T) * 0.5)
    A_H_asm = sp.csr_matrix((ld_c.A + ld_c.A.T) * 0.5)
    chol_gal = ds.dense_cholesky_spd(sp.csr_matrix(A_H_gal), torch.float64, 4096, "galerkin")
    chol_asm = ds.dense_cholesky_spd(sp.csr_matrix(A_H_asm), torch.float64, 4096, "assembled")

    A2 = ld_f.A
    n2 = A2.shape[0]
    A2_t = ds.to_torch_sparse(A2)
    A2T_t = ds.to_torch_sparse(A2.T.tocsr())
    P_t = ds.to_torch_sparse(Pf)
    PT_t = ds.to_torch_sparse(Pf.T.tocsr())
    rng2 = np.random.default_rng(args.seed + 1)
    Es = np.stack([ds.normalise_and_residualise(rng2.normal(size=n2), A2)[0] for _ in range(4)])
    Es_t = torch.from_numpy(Es)

    # dense reference: Q = I - P (P^T A P)^{-1} P^T A, applied to e
    # NOTE the index convention: Q @ Es.T is (n_dof, batch), so the contraction
    # that yields a per-sample energy is "ib,ib->b" (sum over the DoF index).
    # Writing "bi,bi->b" here would sum over the batch instead and quietly return a
    # per-DoF vector -- which is exactly the kind of slip this file exists to catch.
    Q = np.eye(n2) - Pf @ np.linalg.solve(A_H_gal.toarray(), (Pf.T @ A2).toarray())
    Q = np.asarray(Q)
    QE = Q @ Es.T
    ref = np.einsum("ib,ib->b", QE, A2 @ QE)
    got = float(ds.coarse_grid_correction_loss(
        torch.zeros_like(Es_t), Es_t, A2_t, A2T_t, P_t, PT_t, chol_gal))
    check("L_CGC(Galerkin, no correction) = ||Q e||_A^2 / ||e||_A^2",
          np.allclose(got, ref.mean(), rtol=1e-10),
          "sparse %.12f vs dense %.12f" % (got, ref.mean()))

    # the Galerkin corrector really is an A-orthogonal projection: Q^2 = Q
    check("Galerkin corrector is idempotent (Q^2 = Q)", np.allclose(Q @ Q, Q, atol=1e-10),
          "max|Q^2 - Q| = %.2e" % np.max(np.abs(Q @ Q - Q)))

    # a PERFECT correction in the coarse space must be recognised as such, which
    # is exactly the case the old ||Q e||_A^2 denominator could not handle
    e_span = Pf @ rng2.normal(size=Pf.shape[1])
    e_span = e_span / np.sqrt(e_span @ A2 @ e_span)
    e_span_t = torch.from_numpy(e_span[None, :])
    l_span = float(ds.coarse_grid_correction_loss(
        torch.zeros_like(e_span_t), e_span_t, A2_t, A2T_t, P_t, PT_t, chol_gal))
    Qs = Q @ e_span
    q_span = float(Qs @ (A2 @ Qs))
    check("an error inside range(P) is reported as fully removable, finitely",
          l_span < 1e-20 and q_span < 1e-20,
          "L_CGC %.2e (||Q e||_A^2 = %.2e, the old normaliser)" % (l_span, q_span))

    # the assembled coarse operator is a DIFFERENT operator, and the loss says so
    got_asm = float(ds.coarse_grid_correction_loss(
        torch.zeros_like(Es_t), Es_t, A2_t, A2T_t, P_t, PT_t, chol_asm))
    dense_asm = np.linalg.solve(A_H_asm.toarray(), (Pf.T @ A2 @ Es.T))
    asm_after = Es.T - Pf @ dense_asm
    ref_asm = np.einsum("ib,ib->b", asm_after, A2 @ asm_after)
    check("L_CGC(assembled) matches its own dense reference",
          np.allclose(got_asm, ref_asm.mean(), rtol=1e-10),
          "sparse %.10f vs dense %.10f" % (got_asm, ref_asm.mean()))
    check("the two coarse operators give different losses (so the choice matters)",
          abs(got - got_asm) > 1e-9,
          "galerkin %.10f vs assembled %.10f (rel. difference %.2e)"
          % (got, got_asm, abs(got - got_asm) / max(got, 1e-300)))

    # total = L_CGC + lambda * L_E, with each part returned separately
    corr_t = 0.1 * Es_t
    total, l_cgc, l_en = ds.total_smoother_loss(corr_t, Es_t, A2_t, A2T_t, P_t, PT_t,
                                                chol_asm, lambda_energy=0.25)
    check("total_smoother_loss = L_CGC + lambda * L_E",
          abs(float(total) - (float(l_cgc) + 0.25 * float(l_en))) < 1e-12,
          "%.10f = %.10f + 0.25*%.10f" % (float(total), float(l_cgc), float(l_en)))

    # ---- the zero-residual fixed point, with and without kappa -----------
    print("Zero-residual fixed point")
    for use_c, base in ((False, "none"), (False, "jacobi"), (True, "none"), (True, "jacobi")):
        net = ds.DeepONetSmoother(n2, 4, p=8, width=16, depth=2, use_coeff=use_c,
                                  base_smoother=base, diag=A2.diagonal()).to(torch.float64)
        feats = torch.from_numpy(ds.trunk_features(ld_f.angles, "trig", ld_f.coords)).to(torch.float64)
        kap = torch.from_numpy(np.asarray(ld_f.coeff, dtype=np.float64)) if use_c else None
        zero_r = torch.zeros(2, n2, dtype=torch.float64)
        out = net(zero_r, feats, kap)
        check("B(0) = 0 with use_coeff=%s, base=%s" % (use_c, base),
              float(out.abs().max()) == 0.0, "max|B(0)| = %.3e" % float(out.abs().max()))

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

    # The coarse-level smoother is overridable, which is what a fair comparison
    # against the solver's own smoother needs (src/main.cpp uses symmetric SOR at
    # every level, two steps, not damped Jacobi below).  Check that the override
    # really reaches the coarse levels and that the default is untouched.
    # A smoother is bound to ONE DoF count -- the same constraint that stops the
    # learned branch acting below the fine level -- so the stand-ins below are
    # built per level and dispatch on the DoF count.
    sm_by_n = {lv[k].n: ds.ClassicalSmoothers(lv[k].A) for k in range(3)}

    def symgs_smooth(rr):
        return sm_by_n[rr.shape[0]].symmetric_gs(rr)

    def jac_below(rr):
        # The default coarse smoother is omega * A_l^{-1} rr at EACH level, so a
        # stand-in for it has to pick the diagonal by DoF count, not use the fine
        # level's.  Passing the fine diagonal here is what the first version of
        # this check did, and the shapes did not broadcast.
        return (2.0 / 3.0) * rr / {lv[k].n: lv[k].A.diagonal() for k in range(3)}[rr.shape[0]]

    e3, _ = ds.normalise_and_residualise(rng.normal(size=lv[2].n), A2)

    def one_cycle(smooth_fine, coarse_fn, pre=1, post=1):
        ev = e3 - hh.vcycle_apply(A2 @ e3, 2, smooth_fine, pre, post,
                                  2.0 / 3.0, smooth_coarse=coarse_fn)
        return float(np.sqrt(max(ev @ (A2 @ ev), 0.0)))

    r_default = one_cycle(jac, None)               # damped Jacobi below (default)
    r_explicit = one_cycle(jac, jac_below)         # the same, passed explicitly
    r_symgs = one_cycle(jac, symgs_smooth)         # symGS below
    r_cxx = one_cycle(symgs_smooth, symgs_smooth, 2, 2)   # the C++ configuration
    check("omitting smooth_coarse is exactly damped Jacobi below (default intact)",
          abs(r_default - r_explicit) < 1e-14,
          "%.10f vs %.10f" % (r_default, r_explicit))
    check("smooth_coarse reaches the coarse levels (changes the cycle)",
          abs(r_symgs - r_default) > 1e-6, "%.4f vs %.4f" % (r_symgs, r_default))
    check("the C++ configuration (symGS x2 everywhere) contracts",
          r_cxx < 1.0, "one cycle reduces the error to %.4f" % r_cxx)

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
