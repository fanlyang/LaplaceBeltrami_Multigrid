#!/usr/bin/env python3
"""Train a DeepONet as a learned multigrid smoother on one fixed level.

The network learns the residual-to-correction map

        B_theta : (r, Gamma) -> du ,

so that the smoothing step  S_theta(e) = e - B_theta(A e)  reduces the
algebraic error in the energy norm  ||v||_A^2 = v^T A v .

Training data is generated algebraically: no PDE solution is needed.  For each
sample an error vector e is drawn, the residual r = A e is formed, the network
predicts du = B_theta(r), and the remaining error e_tilde = e - du is penalised
in the energy norm.  An optional coarse-space (high-frequency) term is added
when a prolongation operator is available.

Layout (see tools/convert_level_data.py for how the files are produced):

    level_data/level_matrix.npz     sparse level operator A (scipy CSR)
    level_data/dof_coordinates.npy  (n_dofs, 3) support points
    level_data/coeff_values.npy     (n_dofs,)   kappa at the support points
    level_data/prolongation.npz     (optional)  P: level-1 -> level

Run:

    python train_deeponet_smoother.py --data-dir ./level_data --epochs 3000
"""

import argparse
import csv
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Model: DeepONet with a fully-connected branch over the residual and a
# trunk over the 3D surface coordinates.
# ----------------------------------------------------------------------
class Branch(nn.Module):
    """Encoder of the input function (the residual).  bias=False everywhere so
    that a zero residual maps to zero branch output -- the structural form of
    the zero-residual fixed point."""

    def __init__(self, in_dim, width, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width, bias=False), nn.Tanh(),
            nn.Linear(width, width, bias=False), nn.Tanh(),
            nn.Linear(width, out_dim, bias=False))

    def forward(self, x):
        return self.net(x)


class Trunk(nn.Module):
    """Encoder of the evaluation point (the level support point)."""

    def __init__(self, in_dim, width, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, out_dim))

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, n_dofs, coord_dim, p, width, use_coefficient=False):
        super().__init__()
        self.n_dofs = n_dofs
        self.use_coefficient = use_coefficient
        branch_in = n_dofs * (2 if use_coefficient else 1)
        self.branch = Branch(branch_in, width, p)
        self.trunk = Trunk(coord_dim, width, p)

    def forward(self, residual, coords, coeff=None):
        # residual: (B, n_dofs); coords: (n_dofs, coord_dim); coeff: (n_dofs,)
        if self.use_coefficient:
            assert coeff is not None
            b_in = torch.cat([residual, coeff.expand(residual.shape[0], -1)],
                             dim=1)
        else:
            b_in = residual
        b = self.branch(b_in)               # (B, p)
        t = self.trunk(coords)              # (n_dofs, p)
        return torch.einsum('bp,ip->bi', b, t)  # (B, n_dofs)


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
def energy_loss(correction, errors, A, eps=1e-8):
    e_tilde = errors - correction
    num = torch.einsum('bi,ij,bj->b', e_tilde, A, e_tilde)
    den = torch.einsum('bi,ij,bj->b', errors, A, errors)
    return ((num + eps) / (den + eps)).mean()


def high_frequency_loss(correction, errors, A, P, A_coarse, eps=1e-8):
    # Q = I - P A_c^{-1} P^T A  is the energy-projection complement.
    e_tilde = errors - correction

    def Q(v):
        rhs_c = torch.einsum('ci,ij,bj->bc', P, A, v)     # P^T A v
        x = torch.linalg.solve(A_coarse[None], rhs_c[..., None])[..., 0]
        return v - torch.einsum('ic,bc->bi', P, x)

    Qe = Q(errors)
    Qet = Q(e_tilde)
    num = torch.einsum('bi,ij,bj->b', Qet, A, Qet)
    den = torch.einsum('bi,ij,bj->b', Qe, A, Qe)
    return (num / (den + eps)).mean()


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def sample_errors(n_dofs, batch):
    # White noise carries all frequencies; this is the plain "artificial
    # algebraic error" distribution.  A smoother-relevant distribution (for
    # example one that favours oscillatory components) can be injected here.
    return torch.randn(batch, n_dofs)


def weighted_jacobi_step(errors, A, omega=2.0 / 3.0):
    # Classical reference: correction = omega D^{-1} r, with r = A e.
    r = torch.einsum('ij,bj->bi', A, errors)
    return omega * (1.0 / torch.diag(A))[None] * r


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="level_data")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p", type=int, default=32, help="branch/trunk width")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--use-coefficient", action="store_true",
                    help="feed the coefficient into the branch alongside r")
    ap.add_argument("--use-hf", action="store_true",
                    help="add the coarse-space high-frequency loss")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight of the high-frequency loss")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="weight of the energy loss")
    ap.add_argument("--out-dir", default="smoother_results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- load the exported level data ----------------------------------
    A_sp = sp.load_npz(os.path.join(args.data_dir, "level_matrix.npz"))
    coords = np.load(os.path.join(args.data_dir, "dof_coordinates.npy"))
    n_dofs = A_sp.shape[0]
    assert coords.shape == (n_dofs, 3), coords.shape

    coeff_path = os.path.join(args.data_dir, "coeff_values.npy")
    coeff = np.load(coeff_path) if os.path.exists(coeff_path) else None

    A = torch.tensor(A_sp.toarray(), dtype=torch.float32)
    X = torch.tensor(coords, dtype=torch.float32)
    C = torch.tensor(coeff, dtype=torch.float32) if coeff is not None else None

    # -- optional prolongation / coarse operator -----------------------
    P = None
    A_coarse = None
    p_path = os.path.join(args.data_dir, "prolongation.npz")
    if os.path.exists(p_path):
        P_sp = sp.load_npz(p_path)
        P = torch.tensor(P_sp.toarray(), dtype=torch.float32)
        A_coarse = P.t() @ A @ P
        print("Prolongation loaded (%d x %d); high-frequency loss available."
              % P.shape)
    if args.use_hf and P is None:
        raise SystemExit("--use-hf needs prolongation.npz in %s" % args.data_dir)

    # -- model & optimiser ---------------------------------------------
    net = DeepONet(n_dofs, 3, args.p, args.width, args.use_coefficient)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters())
    print("DeepONet: %d dofs, %d parameters" % (n_dofs, n_params))

    # -- fixed validation errors ---------------------------------------
    n_val = 128
    E_val = sample_errors(n_dofs, n_val)
    with torch.no_grad():
        R_val = torch.einsum('ij,bj->bi', A, E_val)   # r = A e

    os.makedirs(args.out_dir, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        net.train()
        E = sample_errors(n_dofs, args.batch_size)
        with torch.no_grad():
            R = torch.einsum('ij,bj->bi', A, E)
        correction = net(R, X, C)
        loss = energy_loss(correction, E, A)
        if args.use_hf:
            loss = (args.alpha * high_frequency_loss(correction, E, A, P, A_coarse)
                    + args.beta * loss)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                val_energy = energy_loss(net(R_val, X, C), E_val, A).item()
                jac_energy = energy_loss(weighted_jacobi_step(E_val, A),
                                         E_val, A).item()
            history.append((epoch, loss.item(), val_energy, jac_energy))
            print("epoch %5d  loss %.5f  val-energy %.5f  jacobi-energy %.5f"
                  % (epoch, loss.item(), val_energy, jac_energy))
            if val_energy < best_val:
                best_val = val_energy
                torch.save(net.state_dict(),
                           os.path.join(args.out_dir, "best_smoother.pt"))

    with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_energy", "jacobi_energy"])
        w.writerows(history)
    print("Saved %s/best_smoother.pt and %s/history.csv"
          % (args.out_dir, args.out_dir))


if __name__ == "__main__":
    main()
