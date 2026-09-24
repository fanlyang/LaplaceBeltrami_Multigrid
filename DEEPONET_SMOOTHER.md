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
* The objective is `L_CGC + lambda * L_E`: the `A`-energy left after one learned
  smoothing step **and** one exact coarse-grid correction, plus a weighted
  full-energy term. Which coarse operator the loss contains is an explicit choice
  (`--coarse-operator`): the **assembled** coarse matrix the C++ V-cycle uses (the
  default), or `P^T A P`, for which the corrector is an exact `A`-orthogonal
  projection. Both are scored on the test set and reported. `lambda` is an
  experimental weight, and Finding 7 reports what setting it to zero costs.
* A low loss is not evidence that the smoother works. Evaluation reports the
  reduction per error mechanism *including the worst sample and the share of
  samples the smoothing makes worse*, repeated application, the V-cycle iterated
  against **the solver's own smoother** as well as Jacobi, and a test on errors a
  real V-cycle left behind. §8.2 is where the distinction between "beats Jacobi"
  and "beats what the solver uses" turns out to matter.
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
| `--loss-mode cgc` | **default**: `L = L_CGC + lambda * L_E`, the coarse-grid-corrected energy first |
| `--loss-mode energy` | `L_E` alone — the objective of releases before this change |
| `--lambda-energy` | weight of the auxiliary `L_E` under `--loss-mode cgc` (default 0.1; experimental — compare 0.0 and a small positive value) |
| `--coarse-operator assembled` | **default**: the coarse matrix the C++ V-cycle uses |
| `--coarse-operator galerkin` | `P^T A P` — the operator for which the corrector is an exact `A`-orthogonal projection |
| `--smoother-iterations` | how many times the smoother, and the V-cycle, are applied in the stability check |
| `--vcycle` | run a real V-cycle from the exported hierarchy |
| `--reuse-data` | reload `dataset_*.npz` instead of regenerating |
| `--inspect-only`, `--gen-only` | stop after validation / data generation |

`--coarse-loss-weight` was removed rather than reweighted: it expressed
`L_E + w * (coarse-complement loss)`, the opposite arrangement to the intended
objective. It now fails with an explanation instead of silently training
something else.

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
| `history.csv` | per-logged-epoch `train_loss`, `train_cgc`, `train_energy`, `val_total`, `val_cgc`, `val_energy`, learning rate — the two objective terms are logged separately, so "the smoother improved" and "the coarse correction is carrying the run" can be told apart |
| `splits.json` | per split: mechanism, discrete signature and generator parameters of every sample — the audit trail for the disjointness claim |
| `best_model.pt` | `state_dict` + config + best epoch and its validation values |
| `run.log` | the complete console transcript |
| `dataset_{train,val,test}.npz` | errors `E` (unit `A`-energy), residuals `R = A E`, mechanism labels, signatures, parameters |
| `plots/loss_curves.png` | objective, `L_CGC` and `L_E`, train and validation |
| `plots/reduction_by_mechanism.png` | mean energy ratio per mechanism, all methods |
| `plots/energy_reduction_hist.png` | distribution of the ratio, per mechanism |
| `plots/error_fields_<mech>.png` | before / after / correction, in the `(xi, eta)` chart |
| `vcycle_baselines.json` | written by `tools/vcycle_baselines.py`, not by the run: the V-cycle against the solver's own smoother (symmetric SOR, two steps) at matched step counts |

`metrics.json` carries five blocks that answer "is this smoother actually usable",
each of which is easy to omit and each of which has changed a conclusion:

| block | question |
| --- | --- |
| `loss_target` | what the trained objective is, and what one smoothing + one coarse correction leaves on the test set under **both** coarse operators |
| `smoothers[mech][method]` | per mechanism, the mean ratio **plus** `energy_ratio_max` and `frac_amplified` — a mean hides the samples a smoother makes worse |
| `smoother_iterations` | the smoother applied repeatedly — a mean over one application says nothing about an iteration |
| `mg_iteration_errors` | the same measured on errors a **real V-cycle** left behind, which no generator produced |
| `two_grid.loss_vs_measured` | the loss and the executed cycle on the same samples, as a number rather than an assurance |

The `vcycle` block holds the learned cycle and the damped-Jacobi baseline. The
rows for the solver's own smoother (symmetric SOR, two steps) were added to the
program after the runs committed here were made, so for those runs the fair
comparison comes from `tools/vcycle_baselines.py`, which reconstructs it from the
saved `best_model.pt`; a fresh run records all four rows itself.

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
* the loss invariants — `loss(no correction) = 1` exactly, `loss(perfect) = 0`
  exactly (the `eps` floors the denominator only), `loss(e/2) = 1/4`;
* **`L_CGC` against a dense `Q = I - P(P^T A P)^{-1} P^T A`**, on the Galerkin
  and on the assembled coarse operator, each against its own dense reference
  (`0.818666352123` and `0.8189453218`);
* the Galerkin corrector is idempotent, `max|Q^2 - Q| = 8.9e-16` — so it really is
  the `A`-orthogonal projection the theory statement is about;
* **the case the old normaliser could not handle**: for an error drawn from
  `range(P)`, `L_CGC = 1.0e-31` while `||Q e||_A^2 = 1.9e-31` — the corrected loss
  reports "fully removable" finitely, where dividing by `||Q e||_A^2` would have
  been a division by ~0;
* `total_smoother_loss = L_CGC + lambda * L_E` with each part returned separately;
* the two coarse operators give **different** losses (relative difference
  `3.4e-04` at L2→L1), so the choice of `--coarse-operator` is a real one;
* **`B(0) = 0` exactly** for all four combinations of `use_coeff` and
  `base_smoother`, i.e. the coefficient input does not cost the zero-residual
  fixed point;
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

The trainer and the V-cycle are then cross-checked inside an actual run:
`two_grid.loss_vs_measured` in `metrics.json` compares the loss's `L_CGC` on the
two-grid samples against the energy the hierarchy's own `two_grid_stages` leaves,
and the two agree to `0.0e+00` — the loss contains the same correction the cycle
executes, up to Cholesky versus LU on the coarse operator.

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

The scaling study in §8.3 sharpens this into a quantitative threshold. Holding
the design fixed and letting `p/n` fall, the advantage is present at
`p/n = 0.125` and entirely gone by `0.0625`, where the learned smoother collapses
onto optimally damped Jacobi to three decimals. The rank ratio, not the level
number, is what governs whether the method earns its keep.

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

**Finding 3 — the hierarchy is not exactly Galerkin, so which coarse operator the
loss contains is a real choice.** The program compares `P^T A_l P` against the
separately assembled `A_{l-1}` at every level and reports the relative
difference. It is *not* zero — `1.9e-3` at L4→3, `8.3e-3` at L3→2, `4.7e-4` at
L5→4. This is expected for an isoparametric surface discretisation: a coarse basis
function is not reproduced exactly by the fine cells' mappings. Two coarse
operators are therefore available, and they are not the same operator:

* `--coarse-operator assembled` (the **default**) puts in the loss the matrix the
  C++ V-cycle actually uses, so the objective scores the step that is executed.
  On this uniformly refined hierarchy the executed coarse-grid correction is
  exactly `I - P A_H^{-1} P^T A_h`: `src/main.cpp` uses `MGTransferPrebuilt`,
  whose restriction is the transpose of the prolongation, and the hierarchy has
  **0 refinement-edge DoFs**, so no interface/edge matrices are applied.
* `--coarse-operator galerkin` uses `P^T A P`, for which the corrector is an
  exact `A`-orthogonal projection — the operator the theory statement is about.
  With this one, only the projection property is exact and the executed step is
  not what the loss measured.

`metrics.json` reports the test-set `L_CGC` under **both**, so the gap is a
number rather than a caveat, and `two_grid.loss_vs_measured` checks the loss
against the cycle on the same samples. A large `P^T A P` vs `A_H` discrepancy
would instead mean `P` is not the transfer the hierarchy was assembled with, and
would invalidate every coarse-grid claim built on it.

**Finding 4 — the V-cycle must be written in correction form.** An
error-propagation formulation is equivalent in exact arithmetic but easy to get
wrong: handing `P^T r` to the coarse level as if it were an *error* silently
omits the coarse solve, and the cycle then **diverges** (measured reduction
2.2–3.8, i.e. > 1). `vcycle_apply` therefore solves for the correction directly
(`x <- x + S(r - A x)`, coarse level receives the restricted residual and returns
a correction). It converges: measured mean first-cycle reduction **0.1753** at
level 3 with damped Jacobi at every level, reaching **0.0149** after five cycles,
with no sample above 1 at any stage.

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

**Finding 7 — the auxiliary energy term is not decoration; without it the model
stops being a smoother.** The objective is `L_CGC + lambda * L_E`, and `lambda`
is an experimental weight, so it was measured rather than assumed: the same
configuration (level 3, `p = 128`, trig trunk, learnable Jacobi skip, 6000
epochs) was trained at `lambda = 0` and at `lambda = 0.1`, on the identical test
set. The pure two-grid objective **wins on its own loss and loses everywhere the
smoother is used**:

| quantity (level 3, test set) | `lambda = 0` | `lambda = 0.1` |
| --- | ---: | ---: |
| `L_CGC` (the primary objective, lower better) | **0.0494** | 0.0582 |
| `L_E` (smoothing only) | 1.069 | **0.166** |
| worst test sample, smoothing only | 1.978 | **0.671** |
| share of test samples the smoothing makes worse | 34.0 % | **0 %** |
| repeated smoothing, 1st / 5th step | 0.902 / 0.479 | **0.385 / 0.110** |
| V-cycle, first / fifth cycle | 0.2316 / 0.0116 | **0.1076 / 0.0090** |
| on real V-cycle-iteration errors, k = 1 .. 4 | 2.13, 3.15, 3.74, **4.12** | 0.65, 0.78, 0.84, **0.86** |

The mechanism is exactly the one the loss is blind to. `L_CGC` contains an *exact*
coarse solve, so any component of the correction lying in `range(P)` is removed
before it is scored: **the loss does not care what the smoother does there** — and
in a recursive V-cycle, where the coarse solve is inexact, that is precisely the
part that matters. The `lambda = 0` model duly learns a correction that is
worthless as a smoother: it increases the energy of a third of the test errors,
and on the errors a real V-cycle actually leaves behind it *diverges* (4.12× after
four cycles). `lambda = 0.1` costs 18 % on the primary loss and is better on every
other line of the table.

This also retires a claim the earlier version of this note made. A small `L_E`
does not imply a good smoother, but a small *`L_CGC`* does not either: the two
terms answer different questions, which is why both are logged separately and the
ablation is reported rather than a single number.

## 8. Results

Every number below is read out of the runs' `metrics.json` by
`tools/summarize_runs.py`; none is transcribed by hand. All configurations share
the **same 50 test errors** (10 per mechanism, fixed by `seed=0, n_test=50`), and
the classical smoothers are evaluated on exactly those errors, so the comparison
is paired. The test set was checked to be **bitwise identical** before and after
the objective change, so the runs below and the earlier ones are scored on the
same errors and remain comparable on any column that does not depend on the loss.

The learned runs in §8.1–§8.3 use the current objective, `L_CGC + 0.1 * L_E`,
with the assembled coarse operator. Two older runs are kept as the
`L_E`-only baseline (`skip_jacobi_trig128`, `L4_skip_jacobi_trig256`,
`L5_skip_jacobi_trig256`); their `objective` column in `results/summary.csv` is
empty, which is how they are told apart.

### 8.1 Level 3 — 1024 DoFs, 32x32 lattice

Mean `||e - delta_e||_A / ||e||_A` after **one** application (lower is better):

| method | trunk | p | params | ALL | smooth | multiscale | localized | algebraic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DeepONet + Jacobi skip** | trig | 128 | 230 k | **0.3849** | **0.1940** | 0.4097 | **0.4316** | 0.4717 |
| DeepONet | trig | 1024 | 461 k | 0.7655 | 0.2952 | 0.9683 | 0.6940 | 1.0108 |
| DeepONet | fourier-16 | 1024 | 739 k | 0.7616 | 0.4739 | 0.8786 | 0.6877 | 0.9369 |
| DeepONet (literal spec: raw coords) | xyz | 64 | 214 k | 0.7765 | 0.3679 | 0.9729 | 0.6685 | 1.0150 |
| damped Jacobi, best omega | — | — | — | 0.6433 | 0.9320 | 0.4375 | 0.8465 | 0.4103 |
| Gauss–Seidel | — | — | — | 0.5729 | 0.8563 | 0.3889 | 0.7533 | 0.3368 |
| symmetric Gauss–Seidel | — | — | — | 0.4297 | 0.7435 | **0.2237** | 0.6242 | **0.1642** |
| SSOR (omega = 1) | — | — | — | 0.4297 | 0.7435 | **0.2237** | 0.6242 | **0.1642** |

SSOR at `omega = 1` is algebraically symmetric Gauss–Seidel, which is why the two
rows agree to every digit — a consistency check on the implementation, not a
duplicate. The classical rows are **identical across every run at a given level**,
in all ten runs here, which is the evidence that they really were scored on the
same test errors.

**On one application, the learned smoother with the diagonal skip beats every
classical smoother overall** (0.3849 against 0.4297 for symmetric Gauss–Seidel),
and it wins on the two mechanisms whose energy is concentrated: `smooth` (0.1940
against 0.7435) and `localized` (0.4316 against 0.6242). It loses on `algebraic`
(0.4717 against 0.1642) — the genuinely high-dimensional, white-noise error that
Finding 1 says lies outside the trunk's span. That is the trade-off the rank
argument predicts, visible mechanism by mechanism rather than only in an average.

**But this column is not the one that decides whether the method is useful**, and
§8.2 is where that shows: a smoother is judged inside a cycle, and there the
ranking changes.

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
from about 0.35 to about 0.05). A picture of the correction field is therefore
worth as much as the aggregate number: it shows *what* the network learned, not
just how much it helped.

### 8.2 Two-grid and V-cycle at level 3

Real `P` and the assembled coarse operator, from the exported hierarchy:

| smoother | smoother only | coarse correction only | two-grid |
| --- | ---: | ---: | ---: |
| DeepONet + Jacobi skip | 0.3849 | 0.5933 | 0.2088 |
| damped Jacobi | 0.6433 | 0.5933 | 0.2588 |
| symmetric Gauss–Seidel | 0.4297 | 0.5933 | **0.1507** |

**Symmetric Gauss–Seidel wins this table.** One smoothing pass followed by one
coarse-grid correction leaves 0.1507 of the error for symGS against 0.2088 for
the learned smoother: the learned smoother's advantage in the smoothing-only
column does not survive the coarse correction being included, which is the
quantity a two-grid method exists to reduce. The loss and the cycle agree exactly
on these samples (`two_grid.loss_vs_measured.abs_diff = 0`), so this is not a
mismatch between what was trained and what is measured.

V-cycle over the four exported levels, iterated five times, on the run's own test
errors. The learned smoother acts at the finest level only — a fixed-length branch
cannot act at another DoF count — while the classical rows act at every level:

| variant | it 1 | it 2 | it 3 | it 4 | it 5 | worst sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned @ L3, Jacobi below (x1) | 0.1076 | 0.0427 | 0.0230 | 0.0139 | 0.0090 | 0.0325 |
| learned @ L3, symGS below (x1) | 0.0890 | 0.0343 | 0.0184 | 0.0111 | 0.0070 | 0.0323 |
| learned @ L3, symGS below (x2) | 0.0390 | 0.0118 | 0.0049 | 0.0023 | 0.0012 | 0.0064 |
| **symGS everywhere (x1)** | **0.0599** | 0.0088 | 0.0018 | 0.0004 | 0.0001 | 0.0005 |
| **symGS everywhere (x2) — the C++ smoother** | **0.0207** | 0.0012 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| damped Jacobi everywhere (x1) | 0.1714 | 0.0698 | 0.0377 | 0.0227 | 0.0145 | 0.0388 |

Produced by `tools/vcycle_baselines.py`, which reads the trained `best_model.pt`
and scores it on the run's own saved test errors; its "learned @ L3, Jacobi below"
row reproduces the run's own reported 0.1076 exactly, which is how the tool is
checked against the program.

Three statements, in order of how much they matter:

1. **The learned smoother does not beat the solver's own smoother.** `src/main.cpp`
   builds the multigrid smoother as `mg::SmootherRelaxation<PreconditionSOR>` with
   `set_steps(2)` and `set_symmetric(true)` — symmetric SOR at `omega = 1`, two
   steps, at *every* level. That is the last-but-one row: **0.0207** per cycle
   against **0.0890** for the learned smoother at matched steps (x1, with symGS
   below). It is not close. Measured against *damped Jacobi* — the baseline an
   earlier version of this note quoted — the learned smoother looks like a
   substantial win (0.0890 against 0.1714); measured against the smoother the
   solver actually uses, it is a loss of more than 4x.
2. **At doubled steps the ordering is subtler than "classical wins".** Learned x2
   (0.0390) is better than symGS x1 (0.0599) per cycle, so the learned smoother is
   worth roughly two symGS steps — at the cost of two network evaluations, which
   are batched and parallel, against two sequential triangular solves. Per-cycle
   contraction is therefore not the whole comparison; `metrics.json`'s `timing`
   block reports the per-sample cost of each, and a fair verdict needs both.
3. **The claim that survives unchanged is the narrow one.** Substituting the
   learned smoother for *damped Jacobi* improves the cycle (0.0890 against 0.1714
   with symGS below; 0.1076 against 0.1714 with Jacobi below). The two plain
   designs do the opposite — their cycles measured 0.4283 (`xyz`, p=64), 0.4218
   (`trig`, p=1024) and 0.4719 (`fourier-16`), i.e. *worse* than Jacobi
   everywhere. But "better than Jacobi" is not the bar a multigrid smoother has to
   clear, and on this evidence the learned smoother does not clear it.

### 8.3 Scaling: levels 4 and 5 — the benefit tracks `p/n`

The level-3 configuration (trig trunk, Jacobi skip) re-run unchanged at 4096 and
16 384 DoFs under the current objective. `p` was allowed to grow only from 128 to
256 while `n_dof` grew 16-fold, so the rank ratio `p/n` falls 4x across the three
— that is the variable this experiment isolates.

| level | n_dof | p | **p/n** | ALL | smooth | multiscale | localized | algebraic | learned omega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L3 | 1 024 | 128 | **0.125** | **0.3849** | **0.1940** | 0.4097 | **0.4316** | 0.4717 | 0.6666 |
| L4 | 4 096 | 256 | 0.0625 | 0.6484 | 0.9835 | 0.4560 | 0.8542 | 0.4051 | 0.6657 |
| L5 | 16 384 | 256 | 0.03125 | 0.6631 | 0.9948 | 0.4491 | 0.8546 | 0.4004 | 0.6636 |

Classical reference on the same test errors at each level:

| level | damped Jacobi (best omega) | Gauss–Seidel | symmetric Gauss–Seidel |
| --- | ---: | ---: | ---: |
| L3 | 0.6433 | 0.5729 | 0.4297 |
| L4 | 0.6433 | 0.5878 | 0.4476 |
| L5 | 0.6615 | 0.6061 | 0.4699 |

Two-grid for the learned smoother against the classical smoothers, and the
V-cycle against the solver's own smoother (`tools/vcycle_baselines.py`, iteration
1 of the table in §8.2):

| level | two-grid (learned) | two-grid (Jacobi) | two-grid (symGS) | V-cycle (learned, symGS below x1) | V-cycle (symGS everywhere x2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| L3 | 0.2088 | 0.2588 | **0.1507** | 0.0890 | **0.0207** |
| L4 | 0.2476 | 0.2491 | **0.1372** | 0.1240 | **0.0220** |
| L5 | 0.2430 | 0.2442 | **0.1267** | 0.1204 | **0.0206** |

**The benefit does not survive the rank ratio falling, and the collapse is total
rather than gradual.** At L4 and L5 the learned smoother becomes numerically
indistinguishable from damped Jacobi: the two-grid reduction agrees to four
decimals (0.2476/0.2491 and 0.2430/0.2442), the overall ratio matches to three
decimals (0.6484 against 0.6433, 0.6631 against 0.6615), and the learned skip
coefficient settles at `omega = 0.6657` / `0.6636` — essentially the module's own
default of 2/3. The network learned to reproduce a damped Jacobi step and nothing
more. Even the `smooth` column, where level 3 gained enormously (0.1940 against
Jacobi's 0.9320), is now dead level: 0.9835 against Jacobi's 0.9825 at L4, and
0.9948 against 0.9956 at L5.

**And at these levels the training does not converge at all — which the earlier
version of this note misread.** The validation curve is flat from the first
logged epoch: at L4 the objective is 0.126141 at epoch 0 and 0.126576 at epoch
4000, and the checkpoint selected as "best" is **epoch 0**. The `L_E`-only
baseline run that this note previously quoted as improving to `best epoch 2000`
was flat in the same way — its validation energy went 0.30222 -> 0.30194 over
2000 epochs, a 0.03 % fluctuation inside a 0.302–0.307 band. Both "best epochs"
were noise, selected because `--min-delta 1e-6` is far below the fluctuation of a
flat curve. So at L4/L5 the honest statement is not "the learned smoother is
overtaken"; it is that **neither objective generalises past the training split**,
while the training loss itself falls (`L_E` 0.330 -> 0.262 at L4) — the model
fits its 512 samples without learning a smoother. Two changes point the same way:
`p/n` fell 4x, and the problem grew 16-fold while the training budget did not
(6000 epochs at L3 against 4000 at L4 and 3000 at L5).

This sharpens Finding 1 into something usable: **the gain appears when the trunk's
span covers a large enough fraction of the space — it is present at `p/n = 0.125`
and gone by `0.0625`.** So the design should be re-tuned per level with `p` grown
alongside `n_dof`, or the trunk replaced by something whose span is not a fixed
fraction of `n_dof`. It is not evidence that the approach fails — the level-3
result stands — but a single tuned configuration is not transferable, which is
exactly what the fixed-length-branch caveat predicts.

### 8.4 Caveat on these numbers

The level-3 run was **still improving at its final epoch** (best epoch 6000,
validation curve still decreasing). Its figures are therefore a lower bound on
what that design reaches, not its converged performance. Training was not
extended further because the point is the comparison, not a tuned best.

At level 3 the learned smoother wins the smoothing-only column and loses the
two-grid and V-cycle ones against symmetric Gauss–Seidel. What remains open, and
what a next experiment should measure rather than assume, is whether the learned
map can be made to beat symGS once `p` is grown with `n_dof` — the rank argument
says that is the axis to move, and §8.3 says nothing else was the obstacle.

### 8.5 Why larger levels take so much longer

`tools/profile_step.py` times each component of one optimisation step and reports
the analytic operation counts, so the cost model is measured rather than
guessed. Same configuration the experiments use:

| level | n_dof | p | batch | params | MACs/step | trunk share | ms/step | memory held |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L3 | 1 024 | 128 | 32 | 230 401 | 4.35e7 | 78.3 % | 27.1 | 24.0 MB |
| L4 | 4 096 | 256 | 16 | 656 513 | 2.29e8 | 88.7 % | 92.7 | 60.1 MB |
| L5 | 16 384 | 256 | 8 | 2 229 377 | 8.64e8 | **94.1 %** | 405.0 | 164.3 MB |

L3 → L5: `n_dof` x16, params x9.7, FLOPs/step x19.9, **wall-clock/step x15.0**,
memory x6.8. Wall-clock tracks the operation count closely, which is the direct
evidence that the runtime is compute-bound rather than memory- or
bandwidth-bound. The raw output is committed as `results/profile_step.txt`.

**The reason is the trunk, and it is a design consequence, not an accident.** The
trunk is evaluated at *every DoF* to produce `T(x)`, so its cost is
`O(n_dof * width * (features + width + p))` and — this is the part that matters —
**it does not depend on the batch size at all**. At L5 the trunk is 94.1 % of the
forward FLOPs; the branch, the only term that scales with the residual vector, is
under 2 %. The practical consequence is that shrinking the batch cannot buy speed
here: going from batch 32 at L3 to batch 8 at L5 cut the branch term by 4x, which
is 2 % of the work. Every step still evaluates the trunk at all 16 384 DoFs
whatever the batch.

Three things follow, and they explain the observed runtime:

* **Compute, not memory, is the binding constraint.** Memory grew 6.8x to 164 MB
  — parameters 17 MB, Adam state 34 MB, gradients 17 MB, the trunk's `(n, p)`
  output 32 MB, and 64 MB of `E`/`R` training data. 164 MB is unremarkable on any
  current machine; the run never approached a memory limit. Reporting that plainly
  matters, because "it needs more memory" is the intuitive explanation and it is
  the wrong one.
* **The step count does not fall with level.** The same ~3 000 epochs are needed,
  so the per-step cost multiplies straight through: 3 000 x 0.4 s is about 20
  minutes of pure step time before the evaluation phase. Level 5's wall clock was
  roughly 40 minutes end to end for exactly this reason — not memory, and not a
  stall.
* **The trunk's input width is a first-order cost.** `--trunk-features trig` has 4
  features; `fourier-16` has 2 176, so its trunk's first layer is 544x larger.
  That is why the Fourier run was the slowest of the level-3 batch even though its
  matrices were identical to the others — the same reason a richer trunk encoding
  is not free.

If this needed to be faster, the lever is the trunk, not the batch: evaluate it on
a subset of DoFs, or replace the global coordinate trunk with a local one whose
cost is `O(n_dof)` with a small constant instead of `O(n_dof * width * p)`.

## 9. Limitations, and what further exports would be needed

Stated without qualification:

* **No real multigrid-trajectory training data was available**, so the training
  distribution is synthetic (the four generators plus mixtures). The program
  supports real samples — point it at error vectors dumped from actual V-cycle
  iterates — but none were on disk, and none is faked. The synthetic distribution
  is therefore a *modelling choice*, not a measurement of what a real cycle
  produces. What the program *does* now provide is a **real multigrid-iteration
  test set** with no synthetic component: `metrics.json`'s `mg_iteration_errors`
  scores every method on the errors a genuine V-cycle leaves behind after k
  cycles, at several stages of the iteration. Those errors were produced by the
  exported hierarchy rather than by a generator, and they are where the
  `lambda = 0` model was caught diverging (Finding 7). They are still not the
  C++ solver's own iterates, which would need the export described below.
* **`L^2(Gamma)` errors cannot be formed.** Only coefficient-space `l^2` is
  reported. A mass matrix `M_h` (or cell areas and connectivity) would be needed.
* **The V-cycle here is algebraic**, assembled from the exported per-level
  matrices and prolongations. It is a genuine V-cycle over the real hierarchy, and
  its classical rows now include the solver's own smoother (symmetric SOR, two
  steps, via `tools/vcycle_baselines.py`), but it is *not* the deal.II multigrid
  solver: no claim is made here about CG iteration counts, or about the hybrid
  cycle as a solver or preconditioner. To measure that, the run needs an outer
  Krylov solve, i.e. the deal.II `PreconditionMG` path in `src/main.cpp` — a C++
  experiment, not this script. Two further points belong to that experiment: the
  learned smoother is a *nonlinear* map, so whether it may be used inside standard
  CG depends on properties this program does not establish (a flexible Krylov
  method is the safe generalisation), and a fair cost comparison must count
  network inference, data conversion and total solve time, not iterations.
* **This is a two-grid-level transfer only where `P` is exported.** The measured
  two-grid reduction uses the real `P` and `A_H`. The coarse operator inside the
  loss is the assembled one by default; `--coarse-operator galerkin` selects
  `P^T A P` instead, and both are reported on the test set so the gap between them
  is a measured number (Finding 3).
* **Timing.** Training time is reported separately from inference. Jacobi is a
  diagonal scaling and is batched; Gauss–Seidel and SSOR are sequential
  triangular solves, so their per-sample cost is inherently serial and is
  reported as such rather than silently normalised. This matters for §8.2: the
  learned smoother loses to symGS on contraction per cycle at matched steps, so
  any remaining argument for it would have to be won on wall-clock cost per cycle,
  which the `timing` block reports but which the comparison above does not settle.
* **The training budget was deliberately held roughly constant across levels**
  (6000 epochs at level 3, 4000 at level 4, 3000 at level 5; `p` grew only from
  128 to 256 while `n_dof` grew 16-fold). That is precisely what the level-4/5
  result in §8.3 turns on, so read those numbers as "this configuration,
  unchanged, at 4096 and 16 384 DoFs" and not as the best this design can do
  there. Given that the validation curve at those levels is flat from the first
  epoch, more epochs would not change that reading either.

## 10. Relationship to `train_deeponet_smoother.py`

The earlier prototype is superseded, and is now a thin adapter that forwards the
old command line to `deeponet_smoother.py` (and refuses the flags whose meaning
changed). Its defects, each of which moved a reported number:

* it built `A = torch.tensor(A_sp.toarray())` — densifying the level operator,
  which makes it unusable beyond a few thousand DoFs;
* it trained only on white-noise errors, with no train/validation/test
  discipline;
* it restricted with the prolongation in the **wrong direction** — `P` is
  `(n_fine, n_coarse)`, so the coarse residual is `P^T A e`;
* its coarse-space normaliser was `||Q e||_A^2`, which vanishes for errors in
  `range(P)` — a legitimate part of the space — so the ratio divided by ~0
  exactly where it should have said "nothing left to remove";
* it added the coarse term as a *penalty on top of* the full energy, whereas the
  intended objective puts the coarse-grid-corrected term first;
* it contained the error-propagation V-cycle bug of Finding 4.

Use `deeponet_smoother.py` directly, or this file as a compatibility entry point.
