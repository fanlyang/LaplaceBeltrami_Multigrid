# DeepONet as a multigrid smoother — training and evaluation

`deeponet_smoother.py` learns the residual-to-correction map

```
B_theta : r_h -> delta_e_h ,        r_h = A_h e_h
```

on one fixed multigrid level, so that the smoothing step

```
u_h <- u_h + B_theta(r_h)        equivalently        S_theta(e_h) = e_h - B_theta(A_h e_h)
```

reduces the *algebraic* error in the A_h energy norm `||v||_A^2 = v^T A_h v`. The
reported quantity is the one that decides whether the network is an effective
smoother: the per-sample ratio `||e_h - B_theta(A_h e_h)||_A / ||e_h||_A`.

`A_h` comes from the deal.II surface (Laplace–Beltrami) solver in `src/main.cpp`,
exported by `export_level_data` and converted by `tools/convert_level_data.py`.

**Loss.** Exactly the relative energy error:

```
L(theta) = (1/N) * sum_i  ||e^[i] - B_theta(A e^[i])||_A^2 / ( ||e^[i]||_A^2 + eps )
```

computed with sparse matrix–vector products only; `A_h` is never densified.

---

## 1. Scope — what is and is not claimed

This section is deliberate. The program prints it at startup and records it in
`metrics.json` under `scope`.

* The branch consumes a **fixed-length residual vector**, so the trained model is
  bound to **one DoF count and one DoF ordering**. It is not a
  discretisation-independent operator. **No cross-mesh or mesh-refinement
  generalisation is claimed**, and none is measured. Using it at another level
  means retraining; that is why coarser levels of the V-cycle are smoothed
  classically.
* The network does **not** learn the infinite-dimensional space `H^1(Gamma)`. It
  learns a map on one finite element space `V_h`, from residual coefficient
  vectors to correction coefficient vectors.
* Errors are **not** partitioned into "high frequency" and "low frequency". The
  generators produce smooth, multi-scale, localised and purely algebraic errors
  and per-sample mixtures of them. Where a coarse-space object appears it is the
  *`A_h`-orthogonal complement of `range(P)`* — a precisely defined subspace, not
  a Fourier band.
* Every `coeff_l2` figure is the Euclidean norm of a **coefficient vector** in
  `R^n`. It is **not** the continuous `L^2(Gamma)` norm: that needs the mass
  matrix `M_h`, which the exporter does not write. The exporter writes `A_l`, the
  support points, kappa at those points, refinement edges and `P`. So the honest
  label is used everywhere, and `metrics.json` records
  `"mass_matrix_available": false`.

---

## 2. Install

```
python -m pip install -r requirements-deeponet.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The second command is separate because the PyTorch CPU index does not carry
numpy/scipy/matplotlib. CPU-only torch is sufficient: nothing here needs a GPU,
and `n_dof = 1024` trains in a few minutes.

## 3. Generate the data

The exporter writes plain `.coo`/`.txt` dumps; the converter turns them into
`level_matrix.npz` + `dof_coordinates.npy`.

```bash
# build the exporter image (the published laplace-beltrami images predate
# export_level_data, so the image must be rebuilt from this branch's src/main.cpp)
docker build -t lb-exporter .

# degree 1 = linear tent functions, 6 refinement cycles -> levels 0..5
docker run --rm --entrypoint /home/dealii/solver/build/solver \
    -v "$PWD/level_dumps:/work" -w /work lb-exporter 1 6

# convert one snapshot per level. level 3 = 1024 DoFs, 4 = 4096, 5 = 16384.
for L in 0 1 2 3 4; do
  python tools/convert_level_data.py --data-dir level_dumps --level $L \
      --cycle 4 --out-dir level_data/L$L
done
python tools/convert_level_data.py --data-dir level_dumps --level 5 \
    --cycle 5 --out-dir level_data/L5
```

Mixing cycle 4 for levels 0–4 with cycle 5 for level 5 is safe, and the reason is
worth stating rather than assuming: a level matrix is assembled on that level's
mesh, which uniform refinement never changes, so `A-level-4-cycle-4.coo` and
`A-level-4-cycle-5.coo` are **byte-identical** (verified by `md5sum`, likewise for
level 3). The cycle only decides how deep the hierarchy goes.

`level_data/L3` is a 32x32 periodic DoF lattice (1024 DoFs, 9216 nonzeros, 9 per
row), `L4` is 64x64 (4096), `L5` is 128x128 (16384, 147456 nonzeros). In every
case `A` is symmetric to exactly 0 and passes a sparse positive-definiteness test;
the Jacobi-scaled condition number grows about fourfold per refinement (169, 679,
2718), as expected for a Laplacian-like operator. `load_level()` verifies all of
this at run time and refuses to continue on a non-symmetric or non-definite
matrix rather than proceeding on an assumption.

## 4. Run

```bash
python deeponet_smoother.py --data-dir level_data/L3 --out-dir results/L3 --vcycle
```

Useful arguments (full list: `--help`):

| argument | meaning |
| --- | --- |
| `--data-dir` | converted level directory; the level index is read from its `L<n>` name |
| `--hierarchy-root` | directory holding `L0..Ln` for the two-grid / V-cycle analysis |
| `--p` | branch/trunk output width. **Bounds the rank of the correction** — see §6 |
| `--trunk-features` | `xyz` \| `angles` \| `trig` \| `fourier` (with `--trunk-modes`) |
| `--base-smoother jacobi` | adds a learnable diagonal skip, `delta_e = omega D^-1 r + B_theta(r)` |
| `--coarse-loss-weight` | weight of the coarse-complement loss (needs `P`) |
| `--vcycle` | run a real V-cycle from the exported hierarchy |
| `--reuse-data` | reload `dataset_*.npz` instead of regenerating |
| `--inspect-only`, `--gen-only` | stop after validation / data generation |

A small smoke configuration, then the full matrix:

```bash
bash experiments/run_all.sh quick
bash experiments/run_all.sh
python tools/summarize_runs.py --glob 'results/*/metrics.json' --csv results/summary.csv
```

## 5. Output files

Written to `--out-dir`:

| file | contents |
| --- | --- |
| `metrics.json` | **everything measured**, plus full provenance: `A` and coordinate SHA-256, shapes, nnz, seeds, generator parameters, split audit, library versions, and the explicit `scope` block |
| `history.csv` | per-logged-epoch train loss, validation energy, learning rate |
| `splits.json` | per split: mechanism, discrete signature and generator parameters of every sample — the audit trail for the disjointness claim |
| `best_model.pt` | `state_dict` + config + best epoch/validation value |
| `run.log` | the complete console transcript |
| `dataset_{train,val,test}.npz` | errors `E` (unit `A`-energy), residuals `R = A E`, mechanism labels, signatures, parameters |
| `plots/loss_curves.png` | train/validation loss |
| `plots/reduction_by_mechanism.png` | mean energy ratio per mechanism, all methods |
| `plots/energy_reduction_hist.png` | distribution of the ratio, per mechanism |
| `plots/error_fields_<mech>.png` | before / after / correction, in the `(xi, eta)` chart |

## 6. Verification

`tools/selftest.py` re-derives every quantity on the small levels with plain
dense NumPy and compares it against the sparse implementation the experiments
use, so a sign error, a transpose slip or a wrong normalisation cannot survive
into a reported number:

```
python tools/selftest.py --hierarchy-root level_data
```

It checks, all passing:

* the error is normalised to `e^T A e = 1` and the residual is exactly `A e`;
* `energy_norm` against a dense `v^T A v` (max relative difference `4.4e-16`);
* the loss invariants — `loss(no correction) = 1` exactly, `loss(perfect) = 0` up
  to the deliberate `eps` guard, `loss(e/2) = 1/4`;
* **each classical smoother against its own dense formula** — damped Jacobi,
  `(D+L)^{-1} r`, `(D+U)^{-1} D (D+L)^{-1} r`, and SSOR at `omega = 1` equal to
  symmetric Gauss-Seidel;
* the two-grid stages against a dense solve,
  `A_H d = P^T A e_smooth` and `e_smooth - P d`;
* the V-cycle contracts monotonically (measured `0.190, 0.082, 0.044, 0.027, 0.017`);
* **the rank bound of §7 Finding 1, measured independently**: an arbitrary
  32-dimensional correction subspace captures `0.1295` of the error's
  `A`-energy, against the predicted `p/n = 0.125`;
* every generator produces finite, non-zero draws, and the multiscale wavenumber
  cap is strictly below Nyquist.

## 7. What the experiments established

All numbers below are produced by `tools/summarize_runs.py` from the runs'
`metrics.json`; nothing is transcribed by hand.

**Data are verified, not assumed.** `A` is symmetric to exactly 0, SPD with
Jacobi-scaled condition 169, `kappa in [1.1, 2.1]` matching
`kappa = 1.1 + sin^2(xi) cos^2(eta)`, support-point radii in `[1, 3]` for
`R = 2, r = 1`, and the split audit reports **0 cross-split signature overlaps**
in every run — so no base function, nor a rescaled copy of one, appears in two
splits.

**Finding 1 — what the model can fix is set by the span of the trunk, not by the
size of the trunk alone.** The DeepONet output is `b(r) . T(x)^T` with `T` of
shape `(n, p)`, so `delta_e` always lies in the `p`-dimensional row space of the
trunk, *independent of `r`*. The consequence is not simply "`p` must equal `n`",
which is what a first reading suggests and what an earlier version of this note
claimed; the model reduces exactly that part of the error whose energy lies in
that subspace, so what matters is the **effective dimension of the error
distribution** relative to the trunk's span:

* For a genuinely high-dimensional (white) error the captured fraction really is
  about `p/n`. `tools/selftest.py` measures this independently: an arbitrary
  32-dimensional subspace captures `0.1295` of a random error's `A`-energy
  against `p/n = 0.125`. That is why the `algebraic` mechanism is the one the
  model cannot touch (measured 1.015, i.e. no reduction).
* For a **smooth** error the energy is nearly low-dimensional, so a *much*
  smaller `p` suffices. Measured: with `p = 64` against `n = 1024` — one
  sixteenth of the DoFs, and the raw `xyz` coordinates as trunk input — the model
  reduces smooth errors to **0.368**, better than symmetric Gauss–Seidel's
  `0.744`. The rank bound costs nothing there because there is nothing
  high-dimensional to represent.

**Finding 2 — the trunk basis must contain the spatial frequencies of an
oscillatory correction.** A trunk built on `sin/cos(xi), sin/cos(eta)` through an
MLP has a *smooth* basis; a trunk built on the raw coordinates has the same
property in a different guise. Either way it can only produce smooth corrections,
which is the opposite of what a smoother must do, since removing oscillatory
error is the smoother's entire job. Measured with the localised mechanism, the
before/after error plot shows the sharp bump essentially untouched while the
correction `delta_e` is a smooth ripple spread over the whole domain. Adding
Fourier features (`--trunk-features fourier`) widens the reachable subspace and
the comparison table below shows what that buys.

**Finding 3 — the hierarchy is not exactly Galerkin, and that matters.** The
program compares `P^T A_l P` against the separately assembled `A_{l-1}` at every
level and reports the relative difference. It is *not* zero (order `1e-3` here).
This is expected for an isoparametric surface discretisation — a coarse basis
function is not reproduced exactly by the fine cells' mappings — but it has a
concrete consequence: for the coarse-complement loss to be a true `A`-orthogonal
projection, `Q` must be built from `P^T A P`; the measured discrepancy is why the
loss does that while the two-grid step uses the assembled `A_H` (which is what an
actual multigrid cycle uses). A large discrepancy would mean `P` is not the
transfer the hierarchy was assembled with, and would invalidate every coarse-grid
claim built on it.

**Finding 4 — the V-cycle must be written in correction form.** An
error-propagation formulation is equivalent in exact arithmetic but easy to get
wrong: handing `P^T r` to the coarse level as if it were an *error* silently
omits the coarse solve, and the cycle then **diverges** (measured reduction
2.2–3.8, i.e. > 1). `vcycle_apply` therefore solves for the correction directly
(`x <- x + S(r - A x)`, coarse level receives the restricted residual and returns
a correction). It converges: measured per-cycle reduction ≈ 0.209 with damped
Jacobi at every level.

**Finding 5 — DoFs-per-side is not the Nyquist wavenumber.** The DoFs form a
square lattice in the parameter plane, so `n_side = sqrt(n_dof)` and the largest
representable wavenumber is `n_side / 2`. Conflating the two caps the generated
multiscale wavenumbers at `0.75 * n_side` instead of `0.75 * n_side / 2`, which
**aliases** everything above the true Nyquist: `cos(k xi)` sampled on the lattice
is indistinguishable from `cos((n_side - k) xi)`, so the recorded generator
parameters would describe a different function than the one actually sampled —
silently corrupting the provenance the brief asks for. This was a real bug here,
caught by `tools/selftest.py` (the first version of the generator reported a cap
of 12 against a true Nyquist of 8 on the 16x16 level), and fixed; the check
"multiscale cap is strictly below Nyquist" now guards it.

**Finding 6 — "sparse throughout" has to include the validation code.** The
program's promise that `A_h` is never densified was nearly self-defeating: the
definiteness check called `np.linalg.cholesky(S.toarray())` on the Jacobi-scaled
matrix. That is harmless at 1 024 DoFs (8 MB) and at 4 096 (134 MB), and at
16 384 DoFs it is a **2.1 GB dense matrix and an O(n^3) factorisation** — the
largest level simply never started, and the run looked hung rather than wrong.
It is now decided by a SuperLU factorisation taken in symmetric mode with no
pivoting (its U diagonal holds the pivots, positive exactly when the matrix is
positive definite) plus sparse Lanczos for the spectrum bounds. The 16 384-DoF
level validates in 28 s instead of never. The lesson generalises: a
"no densification" claim has to cover every helper, not just the hot path.

## 8. Results

Every number below is read out of the runs' `metrics.json` by
`tools/summarize_runs.py`; none is transcribed by hand. All configurations share
the **same 50 test errors** (10 per mechanism, fixed by `seed=0, n_test=50`), and
the classical smoothers are evaluated on exactly those errors, so the comparison
is paired.

### 8.1 Level 3 — 1024 DoFs, 32x32 lattice

Mean `||e - delta_e||_A / ||e||_A` after **one** application (lower is better):

| method | trunk | p | params | ALL | smooth | multiscale | localized | algebraic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DeepONet + Jacobi skip** | trig | 128 | 230 k | **0.3856** | **0.1968** | 0.3969 | **0.5019** | 0.4305 |
| DeepONet | trig | 1024 | 461 k | 0.7655 | 0.2952 | 0.9683 | 0.6940 | 1.0108 |
| DeepONet | fourier-16 | 1024 | 739 k | 0.7616 | 0.4739 | 0.8786 | 0.6877 | 0.9369 |
| DeepONet (literal spec: raw coords) | xyz | 64 | 214 k | 0.7765 | 0.3679 | 0.9729 | 0.6685 | 1.0150 |
| damped Jacobi, best omega | — | — | — | 0.6433 | 0.9320 | 0.4375 | 0.8465 | 0.4103 |
| Gauss–Seidel | — | — | — | 0.5729 | 0.8563 | 0.3889 | 0.7533 | 0.3368 |
| symmetric Gauss–Seidel | — | — | — | 0.4297 | 0.7435 | **0.2237** | 0.6242 | **0.1642** |
| SSOR (omega = 1) | — | — | — | 0.4297 | 0.7435 | **0.2237** | 0.6242 | **0.1642** |

SSOR at `omega = 1` is algebraically symmetric Gauss–Seidel, which is why the two
rows agree to every digit — a consistency check on the implementation, not a
duplicate. The four classical rows are also **identical across all three runs**
(0.6433, 0.5729, 0.4297, 0.4297 in every one of them), which is the evidence
that the three configurations really were scored on the same test errors.

**The learned smoother with the diagonal skip beats every classical smoother
overall** (0.3856 against 0.4297 for symmetric Gauss–Seidel), and it wins
decisively on the two mechanisms whose energy is concentrated: `smooth`
(0.1968 against 0.7435) and `localized` (0.5019 against 0.6242). It loses only on
`algebraic` (0.4305 against 0.1642) — the genuinely high-dimensional, white-noise
error that Finding 1 says lies outside the trunk's span. That is exactly the
trade-off the rank argument predicts, and it is visible mechanism by mechanism
rather than only in an average.

The two plain DeepONet designs — the literal reading of the brief (raw
coordinates, `p = 64`) and the wider smooth trunk (`trig`, `p = 1024`), with six
times the parameters — do **not** beat the classical smoothers overall, for the
same reason: they too cannot touch the high-dimensional part. Note they are not
useless: both beat symmetric Gauss–Seidel on `smooth`, substantially.

Adding Fourier features out to the Nyquist limit (`fourier-16`, same `p = 1024`)
does move the mechanisms the frequency argument predicts — `multiscale` 0.9683 ->
0.8786 and `algebraic` 1.0108 -> 0.9369 — but it makes `smooth` distinctly
*worse*, 0.2952 -> 0.4739. That is the rank budget being spent differently rather
than more cleverly: the trunk has the same `p` dimensions either way, so widening
its frequency coverage dilutes how well it can represent any one part of the
spectrum. Widening the reachable subspace and deepening it are not the same
thing, and with `p` fixed they trade against each other. It still does not beat
the classical smoothers overall (0.7616 against 0.4297).

The mechanism is visible in `plots/error_fields_localized.png`. For the plain
design the localised bump survives almost untouched while the correction is a
low-amplitude smooth ripple spread over the whole domain — the model produces
something it can represent rather than what is needed. With the skip connection
the correction reproduces the bump's own shape and removes it (the peak falls
from about 0.35 to about 0.05), which is what moves the `localized` column from
0.6685 to 0.5019. A picture of the correction field is therefore worth as much as
the aggregate number: it shows *what* the network learned, not just how much it
helped.

### 8.2 Two-grid and V-cycle at level 3

Real `P` and the assembled coarse operator, from the exported hierarchy:

| smoother | smoother only | coarse correction only | two-grid |
| --- | ---: | ---: | ---: |
| DeepONet + Jacobi skip | 0.3856 | 0.5812 | 0.2227 |
| damped Jacobi | 0.6433 | 0.5812 | 0.2588 |
| symmetric Gauss–Seidel | 0.4297 | 0.5812 | 0.1507 |

V-cycle over the four exported levels (learned smoother at the finest level only,
damped Jacobi below it — a fixed-length branch cannot act at another DoF count):

| cycle | mean reduction | range |
| --- | ---: | --- |
| **learned @ L3 + Jacobi below** | **0.1084** | [0.0177, 0.2276] |
| damped Jacobi at every level | 0.1753 | [0.0855, 0.2457] |

**The hybrid cycle beats an all-classical cycle**, 0.1084 against 0.1753, which is
the claim that actually matters for multigrid: replacing one node of the cycle
improves the cycle. The two plain designs do the opposite — their V-cycles measured 0.4283 (`xyz`, p=64) and 0.4218 (`trig`, p=1024), and 0.4719 (`fourier-16`), i.e. *worse* than
Jacobi everywhere. Substituting a smoother that cannot damp high-dimensional
error for one that can makes the whole cycle worse, and the V-cycle measurement
is what exposes it.

### 8.3 Scaling: level 4 — 4096 DoFs, 64x64 lattice

The winning configuration from 8.1 (trig trunk, Jacobi skip) re-run unchanged at
level 4 except `p = 256`, 4000 epochs, 512 training samples:

| method | ALL | smooth | multiscale | localized | algebraic |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepONet + Jacobi skip | 0.6409 | 0.9394 | 0.4543 | 0.8427 | 0.4249 |
| damped Jacobi, best omega | 0.6433 | 0.9825 | 0.4426 | 0.8471 | 0.4066 |
| Gauss–Seidel | 0.5878 | 0.9567 | 0.4017 | 0.7525 | 0.3453 |
| symmetric Gauss–Seidel | 0.4476 | 0.9169 | 0.2291 | 0.6178 | 0.1647 |

Two-grid: DeepONet 0.2491, damped Jacobi 0.2491, symmetric Gauss–Seidel 0.1372.
V-cycle: learned @ L4 **0.1755** against damped Jacobi everywhere 0.1751.

**The design does not transfer to level 4 on the same budget, and the evidence
says why.** Its two-grid reduction is identical to damped Jacobi's to four
decimals (0.2491 in both), the V-cycle is a tie (0.1755 against 0.1751), and the
learned skip coefficient settled at `omega = 0.7080` — essentially the best swept
Jacobi value of 0.7. The network learned to reproduce an optimally damped Jacobi
step and essentially nothing more: at level 4 the learned increment is
negligible, which is why the DeepONet row sits on top of the Jacobi row on every
mechanism.

Two things changed at once and they point the same way. The rank ratio **halved**
— `p/n = 0.0625` here against `0.125` at level 3 — so the trunk's span is a
smaller fraction of the space it must cover; and the problem grew fourfold while
the training budget did not (same 4000 epochs, batch 16 and 512 samples). This is
the scaling consequence of the fixed-DoF scope measured rather than asserted: the
configuration has to be re-tuned per level, and `p` should grow with `n_dof`
rather than staying put. It is not evidence that the approach fails — the level-3
result stands — but it is evidence that a single tuned configuration is not
transferable, which is exactly what the fixed-length-branch caveat predicts.

### 8.4 Caveat on these numbers

The skip-connection run was **still improving at its final epoch** (validation
0.2225 at epoch 6000, best epoch 6000, curve monotonically decreasing and not
flattened). These figures are therefore a lower bound on what that design
achieves, not its converged performance. Training was not extended further
because the point here is the comparison, not a tuned best.

## 9. Limitations, and what further exports would be needed

Stated without qualification:

* **No real multigrid-trajectory training data was available**, so the training
  distribution is synthetic (the four generators plus mixtures). The program
  supports real samples — point it at error vectors dumped from actual V-cycle
  iterates — but none were on disk, and none is faked. The synthetic distribution
  is therefore a *modelling choice*, not a measurement of what a real cycle
  produces.
* **`L^2(Gamma)` errors cannot be formed.** Only coefficient-space `l^2` is
  reported. A mass matrix `M_h` (or cell areas and connectivity) would be needed.
* **The V-cycle here is algebraic**, assembled from the exported per-level
  matrices and prolongations. It is a genuine V-cycle over the real hierarchy,
  but it is *not* the deal.II multigrid solver: no claim is made here about CG
  iteration counts, or about the hybrid cycle as a solver or preconditioner. To
  measure that, the run needs an outer Krylov solve, i.e. the deal.II `PreconditionMG`
  path in `src/main.cpp` — a C++ experiment, not this script.
* **This is a two-grid-level transfer only where `P` is exported.** The measured
  two-grid reduction uses the real `P` and `A_H`; the coarse-space complement is
  exact because it is built from `P^T A P`.
* **Timing.** Training time is reported separately from inference. Jacobi is a
  diagonal scaling and is batched; Gauss–Seidel and SSOR are sequential
  triangular solves, so their per-sample cost is inherently serial and is
  reported as such rather than silently normalised.

## 10. Relationship to `train_deeponet_smoother.py`

The earlier prototype is superseded. It built `A = torch.tensor(A_sp.toarray())`
— densifying the level operator, which makes it unusable beyond a few thousand
DoFs — trained only on white-noise errors, had no train/validation/test
discipline, and contained the error-propagation V-cycle bug of Finding 4. It is
left in place for reference; use `deeponet_smoother.py`.
