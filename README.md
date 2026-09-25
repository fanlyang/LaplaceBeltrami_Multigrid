# Anisotropic tensor diffusion on a torus

**Which coarse-space-complement error components defeat classical point smoothers
as the operator becomes anisotropic, and what a learned smoother does about
them?**

This branch replaces the earlier *scalar* coefficient with a genuine
**direction-dependent tensor** and reruns the Stage-I experiment:

$$-\nabla_\Gamma\cdot(D_\varepsilon\nabla_\Gamma u)+u=f_\varepsilon,
\qquad
D_\varepsilon=\varepsilon\,e_\xi\otimes e_\xi+e_\eta\otimes e_\eta,$$

on the torus $R=2$, $r=1$, with $e_\xi=(-\sin\xi,0,\cos\xi)$ and
$e_\eta=(-\sin\eta\cos\xi,\cos\eta,-\sin\eta\sin\xi)$ the orthonormal tangent
frame. The strong direction is $\eta$ (weight 1), the weak direction is $\xi$
(weight $\varepsilon$), so the anisotropy is $1/\varepsilon$, swept over
$\varepsilon\in\{1,10^{-1},10^{-2},10^{-3},10^{-4}\}$. $\varepsilon=1$ is the
identity tensor and is the **isotropic control**.

This is the correction to the earlier experiment, and it matters. There the
coefficient was a *scalar*, $\kappa_\varepsilon=\varepsilon+\sin^2\xi\cos^2\eta$;
lowering $\varepsilon$ deepened a thin band around a measure-zero set, the
operator barely moved, and — measured, not assumed — **no error component became
harder for any smoother**. A scalar coefficient cannot make an operator
anisotropic. A tensor can, and does: the multigrid now fails outright.

---

## Headline

### 1. The multigrid fails, and it is the solver that fails

The solver's own CG-preconditioned multigrid, iterations to reach $10^{-10}$:

| $\varepsilon$ | 16 | 64 | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|---|
| $1$ (isotropic) | 1 | 7 | 9 | 10 | **11** | **11** |
| $10^{-1}$ | 1 | 10 | 14 | 20 | 25 | 28 |
| $10^{-2}$ | 1 | 11 | 19 | 29 | 40 | 57 |
| $10^{-3}$ | 1 | 11 | 20 | 35 | 57 | 88 |
| $10^{-4}$ | 1 | 11 | 20 | 36 | 61 | **104** |

At $\varepsilon=1$ the count is bounded and the method is optimal. At
$\varepsilon=10^{-4}$ it **roughly doubles with every refinement**: the multigrid
has stopped being an $O(1)$ method. The per-step reduction tells the same story
— $0.118$ at $\varepsilon=1$, $0.800$ at $\varepsilon=10^{-4}$ at the finest
level.

**This is not a discretisation artefact.** The manufactured solution converges at
$O(h)$ in $H^1$ and $O(h^2)$ in $L^2$ at *every* $\varepsilon$ (measured rates
1.001 and 1.999 throughout), the operator is verified symmetric positive
definite, and the exported prolongation is verified to be exactly the one the
V-cycle uses. The discretisation is fine; the smoother is what fails.

### 2. The failing components are the ones the theory names

Mean norm ratio $\lVert e^+\rVert_A/\lVert e\rVert_A$ after **one** smoothing
step, per error family ($1.0$ = no reduction):

| family | method | $1$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
|---|---|---|---|---|---|---|
| **weak** (oscillatory $\xi$, smooth $\eta$) | damped Jacobi | 0.574 | 0.900 | 0.956 | 0.961 | **0.961** |
| | Gauss–Seidel | 0.388 | 0.786 | 0.904 | 0.915 | 0.916 |
| | symmetric GS | 0.223 | 0.643 | 0.827 | 0.848 | 0.850 |
| **survivor** (what smoothers leave) | damped Jacobi | 0.651 | 0.835 | 0.858 | 0.859 | 0.860 |
| | symmetric GS | 0.310 | 0.522 | 0.550 | 0.551 | 0.551 |
| **gaussian** (nodal noise) | damped Jacobi | 0.393 | 0.517 | 0.536 | 0.538 | 0.538 |
| | symmetric GS | 0.163 | 0.241 | 0.250 | 0.251 | 0.251 |
| **fourier** (mixed modes) | damped Jacobi | 0.379 | 0.467 | 0.473 | 0.473 | 0.473 |
| | symmetric GS | 0.149 | 0.232 | 0.241 | 0.241 | 0.241 |

The degradation is **not uniform, and that is the point**. Nodal noise
(`gaussian`) and Fourier mixtures (`fourier`) saturate: beyond
$\varepsilon\approx10^{-2}$ they stop getting worse. The **weak-direction family**
— oscillatory along $\xi$, the weakly-coupled direction, constant or slow along
the strongly-coupled $\eta$ — goes from 0.574 to 0.961 for damped Jacobi and
never recovers: a point smoother reduces it by 4% at best. This is exactly the
component the classical anisotropic analysis predicts, and the family was
generated and measured rather than assumed. Note also that the weak family is
*not* uniformly difficult at $\varepsilon=1$ (0.574, in line with the others);
it is the **combination** of anisotropy with oscillation in the weak direction
that defeats the smoother.

**The mechanism, measured.** For $D=\operatorname{diag}(A)$ the specification's
quantity

$$\eta(e)=\frac{\lVert D^{-1}Ae\rVert_A}{\lVert e\rVert_A}$$

has mean $1.23$–$1.25$ and 95th percentile $2.32$–$2.53$ across the sweep. With
$\omega=2/3$ this gives $\omega\eta\approx0.83$ typically and $\omega\eta>1$ in
the upper tail, so the bound
$\lVert S_Je\rVert_A\ge(1-\omega\eta(e))\lVert e\rVert_A$ **guarantees nothing
at all** — its right-hand side is negative for a large fraction of samples. The
check itself holds empirically for 100% of samples, with the worst violation
$-6.6\times10^{-3}$ (i.e. no violation); that is reported as a measurement, since
carrying the inequality through $Q_\varepsilon$ is not guaranteed by the
argument — $Q_\varepsilon$ is an $A$-orthogonal projection, so
$\lVert Qv\rVert_A\le\lVert v\rVert_A$ and the step can fail in principle.

### 3. The learned smoother beats Jacobi where Jacobi is bad, and never beats symmetric GS

Mean complement norm ratio $\lVert Q_\varepsilon e^+\rVert_A/\lVert e\rVert_A$,
one step, DeepONet vs the classical smoothers:

| family | $\varepsilon$ | Jacobi | Gauss–Seidel | **symmetric GS** | DeepONet |
|---|---|---|---|---|---|
| weak | $10^{-4}$ | 0.961 | 0.916 | **0.850** | 0.951 |
| gaussian | $10^{-4}$ | 0.488 | 0.362 | **0.231** | 0.393 |
| fourier | $10^{-4}$ | 0.425 | 0.338 | **0.222** | 0.428 |
| survivor | $10^{-4}$ | 0.841 | 0.661 | **0.541** | 0.680 |
| stress (held out) | $10^{-4}$ | 0.821 | 0.724 | **0.609** | 0.681 |

Read carefully, because the answer is not uniform:

* Against **damped Jacobi** the learned smoother wins on the weak, survivor and
  stress families at *every* $\varepsilon$, and on the gaussian and fourier
  families once $\varepsilon\le10^{-1}$ — at $\varepsilon=1$ it is slightly
  *worse* than Jacobi there (0.413 vs 0.393 on gaussian, 0.401 vs 0.379 on
  fourier). So "the learned operator handles what the point smoother reduces
  poorly" holds where the point smoother is genuinely poor, and not where it is
  not.
* Against **symmetric Gauss–Seidel** it loses on every family at every
  $\varepsilon$, without exception. On the weak family it is only marginally
  better than Jacobi (0.951 against 0.961) while symmetric GS reaches 0.850.

The classical smoother that is actually good on this problem therefore remains
better than the learned one everywhere. The specification's first hope is
**partially supported**; the stronger hope is **not**.

### 4. In the real V-cycle the learned smoother is the worst of the four

Stationary V-cycle contraction rate on the exported hierarchy, random
right-hand sides, no Krylov acceleration:

| $\varepsilon$ | Jacobi | Gauss–Seidel | symmetric GS | **learned** |
|---|---|---|---|---|
| $1$ | 0.7510 | 0.4258 | **0.3159** | 0.6974 |
| $10^{-1}$ | 0.9295 | 0.8331 | **0.7566** | 0.9403 |
| $10^{-2}$ | 0.9644 | 0.9278 | **0.8866** | 0.9708 |
| $10^{-3}$ | 0.9722 | 0.9454 | **0.9177** | 0.9776 |
| $10^{-4}$ | 0.9733 | 0.9480 | **0.9222** | 0.9784 |

Every method's cycle degenerates; symmetric GS degenerates least. The learned
smoother is better than Jacobi **only at $\varepsilon=1$** and is the worst of
the four for every anisotropic case. In the outer solve the consequence is
blunt: flexible CG with the learned V-cycle **fails to converge in 500
iterations** at $\varepsilon\in\{10^{-1},10^{-2},10^{-3}\}$.

**Two structural reasons this comparison understates the learned smoother, and
both are reported rather than hidden.** First, the branch network consumes a
fixed-length residual, so the learned smoother can act on **one level only** —
the 1024-DoF finest level; the three coarser levels fall back to classical
symmetric GS. At most a quarter of the cycle's smoothing applications are
learned. Second, the model is trained for the framework's default 3000 epochs
and its validation loss is still falling; the earlier experiment on this
repository found exactly this regime, where the plain network is
budget-limited. Neither caveat changes the direction of the result, but neither
is a reason to read it as "a learned smoother cannot work here".

### 5. The full-error penalty is not optional — without it the model is harmful

The loss is
$\mathcal{L}=\mathbb{E}\lVert Q_\varepsilon e^+\rVert_A^2+\lambda\,\mathbb{E}\lVert e^+\rVert_A^2$,
with $\lambda$ chosen on **validation** errors. The selection table is
unambiguous, at every $\varepsilon$:

| $\varepsilon$ | $\lambda=0$ | $0.01$ | $0.1$ | $1$ |
|---|---|---|---|---|
| $1$ | 2.145 | 0.746 | 0.505 | **0.486** |
| $10^{-2}$ | 10.161 | 0.912 | 0.626 | **0.620** |
| $10^{-4}$ | 10.871 | 0.890 | 0.626 | **0.620** |

With $\lambda=0$ the validation full-error reduction is **greater than one at
every $\varepsilon$** — up to $10.9$, meaning the smoother makes held-out errors
*eleven times worse*. The reason is the one the specification anticipates: a
smoother that only minimises $\lVert Q_\varepsilon e^+\rVert_A$ is free to push
error into $\mathrm{range}(P)$, which the first term cannot see and the coarse
grid does not fully remove. $\lambda=1$ was selected at every $\varepsilon$, and
the sweep is flat between $\lambda=0.1$ and $1$, so the choice is not delicate.

---

## What was verified before any number above was believed

`python -m hcsm.selftest_tensor` — **33 checks, 0 failed**:

| check | result |
|---|---|
| manufactured solution converges at $O(h)/O(h^2)$ at every $\varepsilon$ | H¹ rate 1.001, L² rate 1.999 throughout |
| **$A_\varepsilon$ is affine in $\varepsilon$**: $A_\eta+\varepsilon A_\xi$ | max deviation $1.8\times10^{-15}$ (relative $4.4\times10^{-16}$) |
| $\varepsilon$ changes the assembly | 33% max-norm change, $\varepsilon=1\to10^{-4}$ |
| $A_\varepsilon$ symmetric positive definite at every $\varepsilon$ | symmetric to $4\times10^{-16}$ |
| **$Q_\varepsilon$: the torch loss path == the scipy solve path** | $9\times10^{-15}$ |
| $P^{\mathsf T}A_\varepsilon(Q_\varepsilon v)=0$ through the torch path | relative $9\times10^{-16}$ |
| **exported $P$ == `MGTransferPrebuilt`**, and $P^{\mathsf T}$ == its restriction | **exactly 0**, every level, every $\varepsilon$ |

The **affine check is the one that matters most**: the tensor enters the
xi-stiffness with weight $\varepsilon$ and the eta-stiffness with weight 1, so
$A(\varepsilon)$ must be exactly affine. Any place where the tensor was applied
in the active assembly but not the level assembly, or vice versa, breaks it. It
holds to machine precision.

The **transfer check** was asked for explicitly. The exported prolongation is
compared against the object the V-cycle actually uses, by applying
`MGTransferPrebuilt::prolongate` and `restrict_and_add` to random vectors and
comparing with the matrix re-read from the exported `.coo` file. The difference
is exactly zero — not small, zero — at every level. (Restriction uses the
`dst += Pᵀsrc` convention, which is what `MGTransferPrebuilt` implements.)

The **V-cycle implementation** is checked against the solver independently:
configured to match deal.II's own settings (two symmetric SOR sweeps before and
after, exact coarse solve), the Python cycle gives **0.1017** at $\varepsilon=1$
against deal.II's CG-accelerated **0.0910** at the same level. The right
relationship — the bare stationary cycle is slightly worse than the
CG-accelerated one — and both then degrade together across the sweep.

---

## The error families

Four families, equal proportions, each pushed through
$\widetilde e=Q_\varepsilon z$, $e=\widetilde e/\lVert\widetilde e\rVert_{A_\varepsilon}$,
$r=A_\varepsilon e$, with numerically negligible projections discarded and
counted.

| family | how it is built | why |
|---|---|---|
| `gaussian` | iid nodal values | the classical algebraic error |
| `fourier` | random mixtures in both angles, random phases, decaying amplitudes | broad frequency content |
| `weak` | oscillatory in $\xi$ (upper half of the resolved band), constant or slowly varying in $\eta$ | oscillatory along the **weak-coupling** direction |
| `survivor` | 2, 5 or 10 steps of Jacobi / forward GS / symmetric GS on a projected random error | by construction, the directions classical methods do *not* reduce |

**Aliasing is checked, not assumed.** `assert_no_aliasing` refuses any mode above
the lattice Nyquist: on an $n_{\text{side}}$ lattice $\cos(k\xi)$ and
$\cos((n_{\text{side}}-k)\xi)$ take identical values at every DoF, so a generator
claiming $k>\text{nyquist}$ would have produced a different function than its
recorded parameters describe.

**The `weak` family is not labelled difficult in advance.** The reduction table
above is what decides it, and the family is compared against the others — where
it turns out to be no harder than the rest at $\varepsilon=1$.

**Disjointness.** Train, validation and test roles have independent RNG streams
and a content-hash duplicate audit across roles; the run aborts on any duplicate.

**Held-out stress set.** Analytic near-worst-case modes
$\cos(m\xi)\cos(n\eta)$ for $m\in\{1,2,4,8,12,16\}$,
$n\in\{0,1,2,4\}$, projected and normalised. These are used for **evaluation
only** — never for training, model selection or thresholding.

---

## Model, loss and scaling

The architecture is the existing framework's residual-to-correction DeepONet
(`Branch`, `Trunk`, `DeepONetSmoother`, imported unchanged), with its learned
Jacobi skip, **wrapped in a scale-equivariant normalisation**:

$$B_\theta(r)=\lVert r\rVert_2\,N_\theta\!\left(\frac{r}{\lVert r\rVert_2}\right),
\qquad B_\theta(0)=0 .$$

This is positively homogeneous of degree one **by construction, for every
parameter value**, not merely at convergence, and zero maps to zero exactly
(including in the backward pass), so an exact discrete solution stays a fixed
point. Measured equivariance error over amplitudes $10^{-6}$ to $10^{6}$:
$5.6\times10^{-16}$.

**Operator conditioning.** The framework's architecture supports a per-DoF
coefficient input, but the tensor's coefficient is *position-independent* — it
is $\varepsilon$ in the xi direction and 1 in the eta direction everywhere — so
there is no per-DoF field to supply and the honest input is $\varepsilon$ itself.
For this first controlled comparison **a separate model is trained per
$\varepsilon$**, as the specification directs. **No universal model across
$\varepsilon$ is trained or claimed**; that is a separate experiment and remains
open.

---

## Scope — what is not established

* **One mesh.** The fine level is 1024 DoFs on a 32×32 lattice. The branch
  network consumes a fixed-length residual, so the learned object is bound to
  one DoF count **and one DoF ordering**; it cannot be used on another level, and
  the coarser levels of the V-cycle are smoothed classically. Mesh dependence for
  the *classical* methods is visible in the solver's own iteration counts across
  refinements (the table in §1).
* **The Galerkin diagnostic and the V-cycle are different operators.**
  $Q_\varepsilon$ uses the Galerkin coarse operator $P^{\mathsf T}A_\varepsilon P$;
  the solver's V-cycle uses the independently **rediscretized** level operators.
  On a curved surface these differ. They are never conflated, and the difference
  is recorded in `metrics.json`.
* **Ratios are norm ratios.** Everything labelled `full` or `complement` is
  $\lVert\cdot\rVert_A/\lVert e\rVert_A$. Squared-energy ratios are reported
  separately under `mean_squared_ratio`, because
  $\lVert e^+\rVert_A^2/\lVert e\rVert_A^2$ is what composes over cycles while
  $\lVert e^+\rVert_A/\lVert e\rVert_A$ is what a smoothing-factor table quotes.
* **No claim that anisotropy guarantees failure in general**, and none that a
  DeepONet must win. What is shown is that *this* tensor defeats *these*
  point smoothers on *this* torus, that the failing components are the
  weak-direction ones, and that the learned smoother helps against Jacobi and not
  against symmetric GS.
* The learned smoother's two structural handicaps — one level of four, and a
  training budget at which it has not converged — are stated in §4. A larger
  budget, or a smoother applied on every level, is a different experiment.

---

## Repository layout

```
src/main.cpp                  deal.II solver. --coefficient=tensor applies
                              D_eps = eps e_xi(x)e_xi + e_eta(x)e_eta through
                              the ONE shared cell worker that both active and
                              level assembly use; --coefficient=scalar keeps
                              the earlier kappa_eps experiment reproducible.
                              Also verifies the exported P against
                              MGTransferPrebuilt and writes transfer-check.txt.
tools/convert_level_data.py   solver dumps -> level_data/tensor-eps-<E>-L<l>/
deeponet_smoother.py          the existing framework (Branch, Trunk,
                              DeepONetSmoother, ClassicalSmoothers, load_level)
                              imported unchanged, never copied

hcsm/                         the experiment
  problem.py                  A_eps, P, A_H = P^T A P, Q_eps by coarse solve
  aniso_errors.py             the four families, the aliasing check, the
                              held-out stress set
  losses.py                   the penalised loss, the scale-equivariant
                              wrapper, and Q_eps inside the training graph
  train.py                    the training loops (Stage-I and penalised)
  aniso_evaluate.py           the two ratios, eta(e), the bound check
  vcycle.py                   the real V-cycle, FCG for the nonlinear case
  evaluate.py, model.py, plots.py, style.py, samplers.py
                              shared framework (also used by the scalar
                              experiment on branch high-contrast-smoother)
  selftest_tensor.py          33 checks of the operator and the projector
  check_vcycle.py             the V-cycle vs deal.II's own rate
  run_tensor.py               the whole experiment for one epsilon
  collect_tensor.py           cross-epsilon tables + analysis.tex
  run_epsilon.py, collect.py, refigure.py, compare_budget.py
                              the SCALAR experiment's drivers, kept so that
                              experiment stays reproducible; its data lives on
                              branch high-contrast-smoother

experiments/
  run_tensor_exports.sh       build + run the solver, all five epsilons
  run_tensor_conversion.sh    dumps -> level_data/
  run_tensor_stage1.sh        the experiment, all five epsilons + collection

results/tensor_verif_<E>.csv  manufactured-solution verification per cycle
results/transfer-check-<E>.txt  exported P vs MGTransferPrebuilt
results/tensor/eps-<E>/       metrics.json, checkpoints, history, stress.npz
results/tensor/summary.csv    every statistic, tidy long form
results/tensor/analysis.tex   the LaTeX analysis (numbers read from metrics.json)
```

### The earlier scalar experiment

It is **preserved**, on branch **`high-contrast-smoother`** (pushed), complete
with its code, results and README. Its Stage-I drivers are still on this branch
so it remains reproducible here as well; the solver reproduces its finite-element
side directly with `--coefficient=scalar --epsilon=<E>`. Only its bulky
regenerable data (`results/stage1/`, `level_data/eps-*-L3/`) was removed from
this branch to keep it small.

Its finding, which motivated this experiment: with a **scalar** coefficient, a
$910\times$ increase in pointwise contrast moved every smoother by under 1% and
made no component harder, because $\kappa_\varepsilon$ was smooth with a uniform
lower bound and the operator barely changed (condition number $44\to119$). A
scalar coefficient cannot produce anisotropy. The tensor on this branch does.

---

## Reproducing

```bash
# 0. checks -- run before believing anything
python -m hcsm.selftest_tensor      # 33 checks of operator, projector, transfer
python -m hcsm.check_vcycle         # the V-cycle against deal.II's own rate

# 1. the finite-element side: build the solver, run all five epsilons
bash experiments/run_tensor_exports.sh

# 2. dumps -> level_data/tensor-eps-<E>-L0..L3
bash experiments/run_tensor_conversion.sh

# 3. the experiment: families, classical baselines, lambda selection, training,
#    V-cycle, then the cross-epsilon tables and analysis.tex
bash experiments/run_tensor_stage1.sh

# one epsilon only
bash experiments/run_tensor_stage1.sh python 1e-2
```

Each epsilon takes about 5 minutes (four lambda probes at 600 epochs, then 3000
epochs of training, plus the family generation and the V-cycles); the whole sweep
about 25 minutes on 4 threads. The `analysis.tex` compiles with `pdflatex` plus
`booktabs`.

Everything is seeded (`--seed 0`), the seed is written into every `metrics.json`,
and the per-sample content hashes are recorded so a reported number can be traced
to the sample that produced it.
