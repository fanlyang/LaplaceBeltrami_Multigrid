# High-contrast scalar diffusion on a torus

**Which coarse-space-complement error components become difficult for classical
point smoothers as the diffusion coefficient becomes heterogeneous, and can the
existing DeepONet smoother reduce those same components more effectively?**

This branch is a controlled Stage-I experiment on the surface
(Laplace–Beltrami) problem

$$-\nabla_g\cdot(\kappa_\varepsilon\,\nabla_g u) + u = f
\qquad\text{on the torus } R=2,\ r=1,$$

with the manufactured solution $u_{\text{exact}}(\xi,\eta)=\sin\xi\sin\eta$ held
fixed and the scalar coefficient made progressively more heterogeneous,

$$\kappa_\varepsilon(\xi,\eta) = \varepsilon + \sin^2\xi\,\cos^2\eta,
\qquad \varepsilon \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\},$$

so that $\kappa_{\min}=\varepsilon$, $\kappa_{\max}=1+\varepsilon$ and the
contrast is $(1+\varepsilon)/\varepsilon$, growing from $11\times$ to
$10001\times$. This is **high-contrast heterogeneous scalar diffusion**, not
directional anisotropy.

<!-- RESULTS-TABLES -->

---

## 1. What is held fixed, and what changes

Only the coefficient changes. Everything that could otherwise explain a
difference is identical across the four runs:

| held fixed | value |
|---|---|
| geometry | torus, $R=2$, $r=1$ |
| finite element | $Q_1$ (linear tents), `MappingQ`, `QGauss(2·degree+1)` |
| mesh hierarchy | uniform refinement, cycles 0–5 → 16 … 16384 DoFs |
| **fine level** | **multigrid level 3: 1024 DoFs on a 32×32 periodic DoF lattice** |
| prolongation $P$ | the exported level-2 → level-3 interpolation (1024×256) |
| DoF numbering and ordering | unchanged between epsilons |
| smoothness of $\kappa_\varepsilon$ | $\kappa_\varepsilon$ is $\kappa_{10^{-1}}$ shifted down by a constant — the *pattern* is identical, only the range moves |
| DeepONet branch / trunk | `deeponet_smoother.Branch` / `Trunk`, unchanged |
| optimiser / schedule | `deeponet_smoother`'s framework, unchanged |
| error norms | the solver's own `integrate_difference` norms |

The one thing that varies is $\varepsilon$, therefore $A_\varepsilon$, therefore
the smoothing behaviour. That is the whole design of the experiment.

**The mesh is coarse on purpose.** 1024 DoFs is the level the existing learned
smoother was developed and measured on, and it is small enough that a
32×32 DoF lattice still resolves the coefficient's pattern (16 DoFs per period
in each angle). Using the same mesh for all four epsilons is what makes the
comparison controlled.

---

## 2. What is measured, precisely

### 2.1 The $A_\varepsilon$-orthogonal decomposition

For any error $e$,

$$b_H = P^{\mathsf T}A_\varepsilon e,\qquad
  A_H^{G} z = b_H,\qquad
  e_C = Pz,\qquad
  e_F = e - e_C .$$

The coarse operator is the **Galerkin** operator $A_H^{G} = P^{\mathsf T}A_\varepsilon P$.
With it, the two defining identities hold exactly in exact arithmetic:

$$P^{\mathsf T}A_\varepsilon e_F = 0,\qquad e_C^{\mathsf T}A_\varepsilon e_F = 0 .$$

That is the sense in which the split is $A_\varepsilon$-orthogonal. The
theoretical projector $\Pi_C = P(P^{\mathsf T}A_\varepsilon P)^{-1}P^{\mathsf T}A_\varepsilon$
is never formed, and no matrix inverse is ever computed: $z$ is obtained by a
single sparse LU factorisation of $A_H^{G}$, reused across right-hand sides.
The identities are verified numerically on every epsilon (the self-test measures
worst relative residuals of $6\times10^{-16}$ and $7\times10^{-17}$ respectively).

The complement is called the **coarse-space-complement component**, never
"high-frequency error". It is a precisely defined subspace, not a frequency
band, and the distinction is kept everywhere in the code and the results.

### 2.2 The smoothers

All are scored by the same quantity, on the same errors, with one smoothing step:

$$\rho(e) = \frac{\lVert S(e)\rVert_A}{\lVert e\rVert_A},
\qquad \lVert v\rVert_A^2 = v^{\mathsf T}A_\varepsilon v .$$

| method | $S(e)$ | notes |
|---|---|---|
| damped Jacobi | $e-\omega D^{-1}A e$ | $\omega$ chosen on **validation** errors |
| Gauss–Seidel | $e-(D+L)^{-1}A e$ | forward sweep |
| symmetric GS / SSOR | $e - \omega(2-\omega)(D+\omega U)^{-1}D(D+\omega L)^{-1}Ae$ | at $\omega=1$ exactly SGS |
| DeepONet (plain MLP) | $e - B_\theta(A e, X)$ | the specification's architecture |
| DeepONet (+ Jacobi skip) | $e - \big[\omega D^{-1}Ae + B_\theta(Ae,X)\big]$ | the existing framework's best-measured configuration |

Two conventions are fixed once:

* **The damping parameters are tuned on validation errors, never on the test
  set.** The oracle values (the best achievable $\omega$ *on the test set*) are
  computed and written to `metrics.json`, but only as a bound; they are never
  reported as a method's result. Without this the classical side would get a
  tuned advantage the network never got.
* **Every error is canonicalised to unit $A$-energy.** The classical ratios are
  invariant under that; the DeepONet's is not exactly, because its branch is a
  $\tanh$ network, so canonicalising is what makes the numbers comparable.

### 2.3 The four diagnostics

| symbol | meaning | section of the specification |
|---|---|---|
| $\rho_F$ | the smoother on the coarse-space-complement component $e_F$ | I |
| $\rho_C$ | the smoother on the coarse component $e_C$ (diagnostic; in a real cycle the coarse grid handles these) | J |
| $\rho_k$ | the smoother on $\sin(k\xi)$ / $\sin(k\eta)$ after projection onto the complement | K |
| $\rho_{TG}$ | one smoothing step **plus an exact Galerkin coarse solve**, $\lVert C_GS(e)\rVert_A/\lVert e\rVert_A$ | M |

Every one is reported with its **mean, median, standard deviation, 95th
percentile, maximum and the fraction of samples amplified** ($\rho>1$). A mean
cannot distinguish "reduces everything a little" from "reduces most things a lot
and amplifies a few", and for a smoother inside a V-cycle that difference is the
one that matters. The plots draw every individual sample for the same reason.

---

## 3. The method, in the order it runs

### 3.1 The finite-element verification comes first

For every $\varepsilon$ the right-hand side is **recomputed in closed form** from
the same $\kappa_\varepsilon$ — never reused from another coefficient. In the
angle coordinates

$$f_\varepsilon = -\frac{1}{h_\xi r}\left[
   \frac{\partial}{\partial\xi}\!\left(\frac{r}{h_\xi}\kappa_\varepsilon u_\xi\right)
 + \frac{\partial}{\partial\eta}\!\left(\frac{h_\xi}{r}\kappa_\varepsilon u_\eta\right)\right] + u,$$

and only $\kappa_\varepsilon$ carries $\varepsilon$; its two angle derivatives do
not, because $\partial_\xi\varepsilon=\partial_\eta\varepsilon=0$. The solver is
run over six uniform refinements and $L^2$, $H^1$-seminorm and $L^\infty$ errors
are recorded per cycle (→ `results/fem_verification.csv`, figure
`plots/fem_verification.png`). **This is checked before any smoother number is
interpreted.**

Each $\varepsilon$ then produces its own $A_\varepsilon$, verified sparse
symmetric and positive definite, with the exported coefficient verified against
$\kappa_\varepsilon$ evaluated independently in Python (max deviation
$4\times10^{-16}$).

### 3.2 Training data: sampled errors, decomposed

The model is **not** trained on a manufactured sine mode. Training samples are
drawn from the existing framework's error families — smooth, multiscale,
localised, purely algebraic, and random mixtures of those four — and each is then

1. canonicalised to unit $A$-energy,
2. split by the $A_\varepsilon$-orthogonal decomposition,
3. **rejected** if its $e_F$ carries less than $10^{-8}$ of the energy (nothing to
   learn from), and
4. canonicalised again, so every target has $\lVert e_F\rVert_A = 1$ and the loss
   denominator is exactly one.

The rejection rate is itself a measurement — it is how much of each error family
the coarse space already spans — and it is reported per mechanism.

The raw error is the *only* thing drawn; $e_F$ is never sampled directly, so the
training distribution is a genuine consequence of the decomposition.

Train / validation / test are separate draws with an identical mechanism mix.
Disjointness is enforced two ways: a **content hash** of every sample vector
(a shared RNG stream would produce bit-identical draws) and a check that no two
roles share an RNG stream seed. Both are recorded in `metrics.json`.

> The first version of that audit gated on the generator *signature* instead, and
> aborted the run. It was wrong: the smooth family records its mode indices in the
> signature but randomises amplitudes and phases, so two genuinely different
> functions routinely share a signature. The content hash measures what was
> actually meant, and the signature count is kept as a diagnostic only. This is
> noted because a check that aborts on a false positive is as damaging as one that
> passes a false negative.

### 3.3 Stage-I loss

Only the smoothing loss:

$$\mathcal L_{\text{smooth}} = \operatorname*{mean}_i
  \frac{(e_F^{[i]} - \delta e_\theta^{[i]})^{\mathsf T}
        A_\varepsilon (e_F^{[i]} - \delta e_\theta^{[i]})}
       {e_F^{[i]\,\mathsf T} A_\varepsilon\, e_F^{[i]}} .$$

$L_{CGC}$ is **not** used, and neither is $L_{CGC} + \lambda L_E$. That is
deliberate: a loss that already contains the coarse-grid correction cannot answer
what one smoothing step does on its own. The module does not import either.

### 3.4 Model selection and budget

The optimiser, schedule, gradient clipping, early stopping and checkpointing are
the existing framework's, unchanged. Model selection follows the **validation**
$\mathcal L_{\text{smooth}}$; nothing is selected on the test set, and the
evaluation reloads the saved checkpoint rather than the in-memory model, so every
published number belongs to the published weights.

Two architectures are trained per epsilon and both are reported everywhere:
the plain MLP DeepONet, and the same network with the framework's learned
Jacobi skip switched on. The plain variant is the specification's architecture;
the skip variant is what the existing framework's own measurements identified as
competitive, and dropping it would guarantee a weak baseline. Carrying both, with
the same budget and the same data, is what makes the comparison readable either
way.

---

## 4. Scope — what is not claimed

* One fixed multigrid level. The branch consumes a **fixed-length** residual, so
  the trained object is bound to one DoF count **and one DoF ordering**. No
  cross-mesh, cross-level or mesh-refinement generalisation is claimed or
  measured.
* A separate model per $\varepsilon$. **No universal operator across
  $\varepsilon$ is trained or claimed** — that is a separate experiment.
* The energy norm is the algebraic $\lVert v\rVert_A^2 = v^{\mathsf T}A_\varepsilon v$.
  No mass matrix is exported, so no continuous $L^2(\Gamma)$ norm is available
  and none is reported.
* $\rho_F$ and $\rho_{TG}$ are ratios for **one** smoothing step at the stated
  damping. Multi-step numbers are reported separately, not conflated.
* The conclusion is about this torus, this discretisation, this coefficient
  family, this training distribution and this DeepONet configuration. It is not
  generalised beyond them.
* The experiment does not begin from "multigrid fails" or "DeepONet is better".
  Both would be conclusions, and the chain that is actually tested is
  $\varepsilon \downarrow \;\to\;$ contrast $\uparrow \;\to\;$ $A_\varepsilon$
  changes $\;\to\;$ smoothing changes $\;\to\;$ *identify* which $e_F$ components
  are poorly reduced $\;\to\;$ compare on exactly those $\;\to\;$ test whether the
  difference survives in the two-grid method.

---

## 5. Repository layout

```
src/main.cpp                  deal.II solver; kappa_0 is a run-time parameter
                              (--epsilon=), so f, A_epsilon, the exported
                              coefficient and every error norm follow it
tools/convert_level_data.py   solver dumps -> level_data/eps-<E>-L3/
deeponet_smoother.py          the EXISTING framework. Branch, Trunk,
                              DeepONetSmoother, energy_loss, ClassicalSmoothers
                              and load_level are imported from here unchanged.
DEEPONET_SMOOTHER.md          its documentation (not this experiment's)
docs/SOLVER.md                the finite-element solver's own documentation

hcsm/                         THIS experiment
  problem.py                  kappa_epsilon, A_epsilon, A_H = P^T A P, and the
                              A-orthogonal decomposition (no inverse anywhere)
  samplers.py                 sampled errors in coarse-space-complement form,
                              disjoint splits with a content-hash audit
  model.py                    the imported architecture + the Stage-I loss
  train.py                    the Stage-I training loop
  evaluate.py                 rho_F, rho_C, controlled modes, hard errors, rho_TG
  plots.py                    the figures
  collect.py                  cross-epsilon aggregation and README tables
  selftest.py                 84 checks of every identity above
  run_epsilon.py              the whole experiment for one epsilon

experiments/
  run_fem_exports.sh          builds the solver image, runs it per epsilon
  run_conversion.sh           dumps -> NumPy
  run_stage1.sh               all four epsilons + collection

level_data/eps-<E>-L3/        the converted per-epsilon level problem
results/fem_eps-<E>.csv       the manufactured-solution verification tables
results/stage1/eps-<E>/       one run: metrics.json, checkpoints, history,
                              per-sample NPZ, CSVs, plots/
results/stage1/summary.csv    every statistic, tidy long form
results/stage1/readme_tables.md   the tables in section 3 of this file
```

### How this relates to the existing DeepONet smoother

`deeponet_smoother.py` is kept on this branch, and is **imported, not copied**:
`Branch`, `Trunk`, `DeepONetSmoother`, `energy_loss`, `to_torch_sparse`,
`A_apply`, `energy_norm`, `ClassicalSmoothers`, `load_level` and
`sparse_spd_report` all come from it, so there is exactly one implementation of
each and the two cannot drift apart.

What this experiment does *not* use is its objective. The existing default is
`L_CGC + lambda * L_E`, which contains a coarse-grid correction; Stage I requires
the smoothing loss alone, and `hcsm.model` neither imports nor computes the CGC
term. Its V-cycle, hierarchy and mechanism-sweep machinery are likewise not used:
Stage I is deliberately narrower, and the two-grid test in section M is a single
exact coarse solve, not an iterated cycle.

---

## 6. Reproducing

Everything below runs from the branch root. On Windows use Git Bash; the scripts
find the Python that has `numpy`/`scipy`/`torch` themselves (they look for
`.venv-deeponet` beside the *main* checkout, so a git worktree works too), and
they set `MSYS_NO_PATHCONV=1` where a container path would otherwise be rewritten.

```bash
# 0. checks — 84 assertions, no results are trusted before these pass
python -m hcsm.selftest

# 1. finite-element side: build the solver, run all four epsilons
bash experiments/run_fem_exports.sh

# 2. dumps -> level_data/eps-<E>-L3/
bash experiments/run_conversion.sh

# 3. the experiment: train and evaluate both architectures per epsilon, then
#    collect the cross-epsilon figures and tables
bash experiments/run_stage1.sh
```

Environment: `pip install -r requirements-deeponet.txt` then
`pip install torch --index-url https://download.pytorch.org/whl/cpu` (the CPU
index carries no numpy/scipy, so the two commands must be separate). CPU-only
torch is sufficient; nothing here needs a GPU. The solver runs in Docker from
`fanyoung/dealii-codim:9.7.1`, which the `Dockerfile` builds on.

Everything is seeded. `--seed 0` is used throughout, the seed is written into
every `metrics.json`, and the per-sample CSVs carry the generator signature of
each test error, so a reported number can be traced back to the sample that
produced it.

### The individual steps

```bash
python -m hcsm.run_epsilon --eps 1e-3 --out results/stage1/eps-1e-03 --epochs 3000
python -m hcsm.collect --results results/stage1 --data-root level_data --fem-csv results
```

To run a different budget (used for the budget check in section 4):

```bash
python -m hcsm.run_epsilon --eps 1e-1 --architectures plain \
    --epochs 12000 --out results/stage1_budget/eps-1e-01
```
