# Can a DeepONet learn to smooth?

A learned smoother for the geometric multigrid solver in `src/main.cpp`. This
branch contains **Stage I** of that programme: train a DeepONet on the part of the
algebraic error that a coarse-grid correction *cannot* reach, and measure whether
it reduces that part while leaving the coarse part alone.

That property is called **smoothing selectivity**, and it is the whole job of a
smoother. Stage I is the experiment that establishes whether it is there at all,
before any V-cycle is run.

---

## The answer

**Yes — and with a branch wide enough to represent the correction, the learned
step is the best of the six methods compared.**

One smoothing step, 1024 test errors, level $\ell = 3$ ($n = 1024$ DoFs, Galerkin
coarse operator with $n_c = 256$, so the complement is 768-dimensional). $\mu_F$
is the mean reduction factor on the complement component, $\mu_C$ on the coarse
component; **smaller $\mu_F$ is better, and $\mu_C/\mu_F > 1$ means selective.**

| Smoother | $\mu_F$ | $\mu_C$ | $\max\rho_F$ | $\mathbb{P}(\rho_F>1)$ | $\mu_C/\mu_F$ |
|---|---|---|---|---|---|
| Jacobi | 0.6996 | 0.6734 | 1.4310 | 0.1387 | 0.96 |
| Damped Jacobi | 0.4248 | 0.7380 | 0.7230 | 0.0000 | 1.74 |
| Gauss–Seidel | 0.3664 | 0.6180 | 0.5658 | 0.0000 | 1.69 |
| Symmetric GS / SSOR | 0.1889 | 0.4695 | 0.3848 | 0.0000 | 2.49 |
| **DeepONet (width 768)** | **0.1456** | **1.0928** | 0.3198 | 0.0000 | **7.50** |
| ridge (diagnostic, not a competitor) | 0.0000 | 1.0249 | 0.0000 | 0.0000 | ∞ |

The network removes 85% of the complement's energy in one step — more than
symmetric Gauss–Seidel — and amplifies no sample.

**But that is only half the story, and the other half is the point.** The same
architecture at branch width 256 scores $\mu_F = 0.5803$ — *fifth of six*, behind
every convergent classical smoother. Both runs are committed. See
[§4](#4-why-two-runs-of-the-same-thing) for why, and why it matters.

Full write-up, including everything the tables above do not say, is in
**[`STAGE1_SMOOTHING_PROPERTY.md`](STAGE1_SMOOTHING_PROPERTY.md)**.

---

## 1. The method

### 1.1 The coarse space and the split

On one fixed level, with the assembled fine operator $A$ and the prolongation $P$
from the next coarser level, the coarse operator is the Galerkin one,

$$A_c = P^{T}AP,$$

and the $A$-orthogonal projection onto the prolongated coarse space is

$$\Pi_C = P\left(P^{T}AP\right)^{-1}P^{T}A, \qquad \Pi_F = I - \Pi_C .$$

Every error splits as $e = e_C + e_F$ with $e_C = \Pi_C e \in \operatorname{range}(P)$ and
$e_F = \Pi_F e$ in its $A$-orthogonal complement. These are exact projections, so

$$e_C^{T}Ae_F = 0, \qquad P^{T}Ae_F = 0, \qquad
\lVert e\rVert_A^2 = \lVert e_C\rVert_A^2 + \lVert e_F\rVert_A^2 .$$

The program **asserts** all three on the data it uses rather than assuming them;
`--selftest` prints the residuals, which sit at $10^{-15}$.

### 1.2 The loss

Stage I trains on the complement component only:

$$\mathcal{L}_{\mathrm{smooth}}(\theta) = \frac{1}{N}\sum_{i=1}^{N}
\frac{\bigl\lVert e_{F,i} - B_{\theta}(Ae_{F,i}, X_3)\bigr\rVert_A^2}
     {\lVert e_{F,i}\rVert_A^2},
\qquad \lVert v\rVert_A^2 = v^{T}Av .$$

No coarse-grid correction appears anywhere in this stage. `e_C` is not mentioned
in the loss and is not seen during training.

### 1.3 Why it is a real test rather than a fit

$P^{T}$ is surjective, so $\dim\ker(P^{T}) = n - n_c = 768$, while
$\operatorname{range}(AP)$ has dimension $n_c = 256$; and $P^{T}APz = 0$ forces $z = 0$
because $A_c$ is SPD. Therefore

$$\mathbb{R}^{n} = \operatorname{range}(AP) \;\oplus\; \ker(P^{T}).$$

Since $r_F = Ae_F$ satisfies $P^{T}r_F = P^{T}Ae_F = 0$, **every training input lies
in $\ker(P^{T})$**; every coarse residual $r_C = Ae_C$ lies in
$\operatorname{range}(AP)$. Measured on the actual splits: the training residuals give
$\max\lVert P^{T}r_F\rVert/\lVert r_F\rVert = 2.5\times10^{-14}$, the coarse ones
$\min\lVert P^{T}r_C\rVert/\lVert r_C\rVert = 1.02$. The two input sets meet only at the
origin, so $\mu_C$ is a pure extrapolation measurement.

### 1.4 One consequence worth stating early

Restricted to the complement, the exact map is $r_F \mapsto A^{-1}r_F$ — **linear**.
A linear fit to the same training pairs therefore reaches $\mu_F = 0$ out of
sample, and does: the ridge diagnostic above scores $0.0000$ while also leaving the
coarse component alone ($1.0249$).

So selectivity is not something the network has to *discover* — any map fitted to
the complement data has it. What is actually being measured is how well the
network fits a linear map, subject to the two rank ceilings below.

---

## 2. Two rank ceilings, and why the diagnostics exist

A DeepONet writes its correction as

$$\delta e = \sum_k b_k(r)\,T_k(x),$$

which gives the model two separate rank ceilings. **Either can silently become the
thing under measurement, and neither is visible in the loss curve.**

| ceiling | what it is | how it is measured |
|---|---|---|
| **trunk** | the correction lies in $\operatorname{span}(T)$ whatever the branch returns | `tools/trunk_reach.py` — the floor on $\mu_F$ no branch can beat |
| **branch** | the Jacobian factors through `width` units, and the complement is 768-dimensional | `tools/branch_rank_probe.py` — the best rank-$k$ linear map on the same data |

On the trunk: `range(P)` is what the *coarse* lattice resolves, so its complement
is the band between the coarse and the fine Nyquist (modes $8 \lesssim |k| \le 16$ here).
A Fourier trunk therefore needs modes up to the **fine** Nyquist before it can span
it at all:

| trunk | dim | rank | floor on $\mu_F$ |
|---|---|---|---|
| Fourier $K=4$ | 160 | 80 | 0.989724 |
| Fourier $K=8$ | 576 | 288 | 0.911890 |
| Fourier $K=12$ | 1248 | 624 | 0.563591 |
| **Fourier $K=16$** | 2176 | 1023 | **0.000675** |

The runs here freeze the trunk (`--trunk-freeze fourier-orth`) to an orthonormal
basis of those features, so $\operatorname{span}(T) = \mathbb{R}^n$, the floor is exactly
zero, and the trunk is *provably* not what is being measured.

On the branch, the reference curve is:

| rank $k$ | 64 | 128 | 192 | **256** | 384 | 512 | 640 | **768** |
|---|---|---|---|---|---|---|---|---|
| best $\mu_F$ | 0.8057 | 0.7160 | 0.6467 | **0.5860** | 0.4768 | 0.3666 | 0.2375 | **0.0000** |

---

## 3. Results

### 3.1 The two runs

Identical in data, loss, schedule, trunk and seed. **The only difference is the
branch width**, and it moves $\mu_F$ by a factor of four:

| | width 256 | width 768 |
|---|---|---|
| $\mu_F$ (complement) | 0.5803 — fifth of six | **0.1456 — first of six** |
| $\mu_C$ (coarse) | 1.0400 | 1.0928 |
| $\mu_C/\mu_F$ | 1.79 | **7.50** |
| expressible rank | 256 | 768 (= complement dim) |
| best rank-$k$ linear map | 0.5860 | 0.0000 |
| output | `results/stage1_L3` | `results/stage1_L3_w768` |

At width 256 the network lands within **1%** of the best rank-256 *linear* map:
that number measured the width, not the smoother.

### 3.2 Where the reduction comes from

$\mu_F$ by error mechanism. The width-256 column is what suggested "combine the
learned and classical steps"; the width-768 column shows that reading was wrong:

| mechanism | DeepONet w256 | **DeepONet w768** | Damped Jacobi | Gauss–Seidel |
|---|---|---|---|---|
| smooth | 0.0907 | **0.0129** | 0.4554 | 0.4029 |
| multiscale | 0.6144 | **0.0978** | 0.4345 | 0.3711 |
| localized | 0.4270 | **0.0627** | 0.4461 | 0.4155 |
| algebraic | 0.7675 | **0.2503** | 0.3961 | 0.3325 |
| mixed | 0.6856 | **0.1853** | 0.4166 | 0.3523 |

At width 768 the network beats Gauss–Seidel on **every** mechanism. There was no
complementarity to exploit, only a branch that could not represent the answer.

### 3.3 Two caveats, stated plainly

* **It still misses the linear limit.** The best rank-768 linear map reaches
  $\mu_F = 0.0000$; the network reaches 0.1456. No rank ceiling binds at width 768
  and the trunk floor is $10^{-6}$, so the residual is the optimiser failing to
  close the last 0.15 of a problem it can represent.
* **Selectivity costs amplification.** $\mathbb{P}(\rho_C>1) = 1.0$ in both runs:
  every coarse sample is *grown*, by up to 19.7% at width 768 (worse than 8.4% at
  width 256). The loss never mentioned $e_C$ and the extrapolation to
  $\operatorname{range}(AP)$ is expansive rather than zero.

---

## 4. Why two runs of the same thing?

Because one run gave a confident, plausible, **wrong** answer, and the comparison
is the most useful thing here.

The width-256 run produced a specific and actionable conclusion — "the network is
strong on smooth errors and weak on algebraic ones, so combine it with a classical
smoother" — that is an artifact of the branch's capacity. It is the clearest
argument for publishing a capacity bound next to a result, which is why both
diagnostics in §2 exist and why both runs are committed rather than the flattering
one alone.

---

## 5. Code layout

| file | what it is |
|---|---|
| **`stage1_smoothing_property.py`** | **the whole experiment.** Self-contained: level-data loading and verification, the four error generators, the classical smoothers, the DeepONet, the Galerkin split, training, evaluation, diagnostics and figures |
| `STAGE1_SMOOTHING_PROPERTY.md` | the full write-up — every number in it is printed by the script |
| `tools/trunk_reach.py` | trunk-span floor on $\mu_F$ for a family of Fourier trunks |
| `tools/branch_rank_probe.py` | the best rank-$k$ linear map on the same data (reduced-rank regression) |
| `tools/convert_level_data.py` | turns the C++ solver's `.coo`/`.txt` dumps into the `.npz`/`.npy` the script reads |
| `src/main.cpp` | the deal.II surface solver; `export_level_data` writes the per-level operators |
| `level_data/L3/` | the level this stage runs on: `level_matrix.npz` ($A$), `prolongation.npz` ($P$), `dof_coordinates.npy` ($X_3$) |
| `results/stage1_L3{,_w768}/` | the two runs: `metrics.json` (every number, plus a scope block), `history.csv`, `run.log`, `plots/` |

---

## 6. Running it

```bash
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# checks first: projector identities, ker(P^T) dimension, and every classical
# smoother against the dense matrix that defines it
python stage1_smoothing_property.py --selftest

# the two diagnostics (no training, seconds)
python tools/trunk_reach.py       --data-dir level_data/L3
python tools/branch_rank_probe.py --data-dir level_data/L3

# the width-768 run -- the one that answers the question (~35 min on a laptop CPU)
python stage1_smoothing_property.py \
    --data-dir level_data/L3 --out-dir results/stage1_L3_w768 \
    --width 768 --epochs 1200 --batch 256 --lr 3e-3 --patience 300 --trunk-modes 16

# the width-256 run -- the under-capacity comparison (~17 min)
python stage1_smoothing_property.py \
    --data-dir level_data/L3 --out-dir results/stage1_L3 \
    --width 256 --epochs 1200 --batch 256 --lr 3e-3 --patience 300 --trunk-modes 16

# redraw the metrics-only figures without retraining
python stage1_smoothing_property.py --replot --out-dir results/stage1_L3

# what the flags do
python stage1_smoothing_property.py --help
```

**Runs are deterministic.** `torch.manual_seed` is applied *before* the model is
constructed, not merely inside the training loop — torch seeds its global RNG
nondeterministically at process start, so seeding only inside `train()` leaves the
initial weights varying between runs of identical arguments. Verified: two
independent invocations produce byte-identical results. The error draws were
already deterministic, each mechanism getting its own RNG stream derived from
`(seed, role, mechanism, count)`.

Watch the printed `val mean sq rho_F` column: it is $\mathbb{E}[\rho_F^2]$, and the
run keeps the epoch that minimises it. The first thing to check in any re-run is
the `trunk span: rank(T) = ...` line — if the floor there is not near zero, the
number below it is about the trunk.

---

## 7. Where the data comes from

`A_l`, `P_l` and the DoF coordinates are exported by the deal.II surface
(Laplace–Beltrami) solver in `src/main.cpp`, which solves

$$-\operatorname{div}_\Gamma(\kappa\,\nabla_\Gamma u) + u = f$$

on a torus. To regenerate `level_data/` from scratch:

```bash
# 1. export from deal.II (degree 1 = linear tent functions, 5 cycles)
docker run --rm --entrypoint /home/dealii/solver/build/solver \
    -v "$PWD/level_dumps:/work" -w /work lb-exporter 1 5

# 2. convert level 3 (1024 DoFs)
python tools/convert_level_data.py --data-dir level_dumps --level 3 --cycle 4 \
    --out-dir level_data/L3
```

Only level 3 is committed; the exporter and the converter handle the rest of the
hierarchy if a later stage needs it.

---

## 8. Scope

* **One smoothing step.** No V-cycle is run, and no claim is made about the
  complete multigrid solver.
* **No coarse-grid correction** enters the loss or the evaluation.
* **One DoF count, one DoF ordering.** $n$ is baked into the first branch layer;
  the trained object is tied to this level, and no cross-mesh generalisation is
  claimed or measured.
* **Coefficient-vector norms only.** Every norm is $v^{T}Av$. The mass matrix
  $M_h$ is not exported, so no continuous $L^2(\Gamma)$ norm is available.
* **Both $\mu_F$ values are width-specific** (0.5803 at width 256, 0.1456 at
  width 768). Neither is a statement about DeepONets in general, and the widths
  were chosen to answer the question — 256 because that is what the first run
  used, 768 because it is the complement dimension — not swept.
* **The complement is not a frequency band.** It is the $A$-orthogonal complement
  of $\operatorname{range}(P)$, a precisely defined subspace. §2 shows it is *nearly* the
  band between the coarse and fine Nyquist, but that is an observation about this
  hierarchy, not the definition.

---

## 9. License

LGPL-2.1-or-later. See [`LICENSE`](LICENSE).
