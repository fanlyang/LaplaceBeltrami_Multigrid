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

## Headline

Mean $\rho_F$ on the unseen test set, one smoothing step, for the four
coefficients. Lower is better; $1$ means the step achieved nothing.

| method | $\varepsilon{=}10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
|---|---|---|---|---|
| **symmetric GS / SSOR** | **0.1757** | **0.1727** | **0.1733** | **0.1734** |
| DeepONet, fourier trunk + Jacobi skip | 0.3100 | 0.2812 | 0.2789 | 0.2780 |
| Gauss–Seidel | 0.3525 | 0.3461 | 0.3464 | 0.3465 |
| DeepONet, trig trunk + Jacobi skip | 0.3715 | 0.3615 | 0.3559 | 0.3582 |
| damped Jacobi | 0.4187 | 0.4154 | 0.4153 | 0.4153 |
| DeepONet, plain MLP | 0.8994 | 0.8735 | 0.8775 | 0.8973 |
| contrast $(1+\varepsilon)/\varepsilon$ | 11× | 101× | 1001× | 10001× |

Four things follow, and they are the answer to the question this branch set out
to ask.

**1. Over this coefficient family, the contrast changes almost nothing.** A
$910\times$ increase in pointwise contrast moves every classical smoother by less
than 1%: Jacobi $0.4187 \to 0.4153$, Gauss–Seidel $0.3525 \to 0.3465$, SGS/SSOR
$0.1757 \to 0.1734$. The validation-tuned damping does not move either — Jacobi
settles at $\omega = 0.60$ and SSOR at $\omega = 1.00$ at *every* contrast. So no
coarse-space-complement component "becomes difficult" here, and the reason is
measurable rather than mysterious: the coefficient is *smooth* with a uniform
positive lower bound, and dropping $\varepsilon$ only deepens a thin band around
a measure-zero set. The area-weighted mean of $\kappa_\varepsilon$ moves from
0.260 at $\varepsilon=10^{-2}$ to 0.250 at $\varepsilon=10^{-4}$ — 4%, from a
$100\times$ change in the pointwise minimum — and the coarse operator's
condition number rises only from 44.2 to 118.6. **Coefficient contrast is not the
quantity that governs smoothing behaviour; the spectral structure of
$A_\varepsilon$ is, and here it barely moves.**

**2. Symmetric Gauss–Seidel / SSOR is the best smoother at every contrast**, by a
wide margin — $2.4\times$ better than damped Jacobi and $1.6$–$1.8\times$ better
than the best learned model. Nothing in this experiment overturns that.

**3. The learned smoother beats the two weakest classical smoothers and never
catches the strongest.** The best DeepONet configuration (fourier trunk) improves
on damped Jacobi and on forward Gauss–Seidel at every contrast, but it stays well
behind SGS/SSOR throughout. It does get *better* as contrast grows (validation
loss $0.104 \to 0.082$, $\rho_F$ $0.310 \to 0.278$), while the classical smoothers
stay flat — a real trend, but far too small to close the gap.

**4. The plain MLP DeepONet is not a usable smoother at this budget.** It cuts the
error norm by only about 10% ($\rho_F \approx 0.90$) and *amplifies* between 1.2%
and 3.5% of samples — the only method here that ever makes a test error worse. Its
two-grid reduction (0.549–0.574) is barely better than the coarse correction alone
(0.596–0.610), i.e. its smoothing contributes almost nothing.

The trunk experiment is what makes (3) and (4) interpretable rather than an
artefact of one design choice: giving the trunk spatial reach buys a real,
reproducible improvement ($0.372 \to 0.310$ at $\varepsilon=10^{-1}$), which is
evidence for the explanation that the model's reach is set by its trunk — but the
remaining gap to SGS/SSOR is structural, not a matter of tuning.

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
| DeepONet branch and trunk | `deeponet_smoother.Branch` / `Trunk`, unchanged (three configurations of the *existing switches* are trained — see §2.4) |
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

### 2.3 The three DeepONet configurations

The branch and trunk classes are the existing framework's, unchanged. What varies
between the three configurations is which of the framework's *existing* switches
are on — this is not architecture search, and no new layer type is introduced.

| name | trunk input | skip | why it is here |
|---|---|---|---|
| `plain` | `trig` (4 dims) | none | the specification's literal architecture |
| `skip` | `trig` (4 dims) | learned damped Jacobi | the framework's own best-measured configuration |
| `skip_fourier` | `fourier`, \|m\|,\|n\|≤4 (160 dims) | learned damped Jacobi | spatial reach |

`skip_fourier` is not decoration. With the `trig` trunk the trunk input is only
$(\sin\xi,\cos\xi,\sin\eta,\cos\eta)$ — four numbers — so every correction the
network can produce is a smooth function of the angles, while the training
targets here are *coarse-space-complement* errors, i.e. precisely the oscillatory
part. Without a frequency-aware arm, "the learned smoother does not help" would
be a statement about the trunk rather than about the architecture. Both outcomes
are informative: if reach closes the gap, the `trig` result was a trunk artefact;
if it does not, the conclusion survives the trunk choice.

The main comparison figures draw `skip`, chosen **in advance** as the
framework's default configuration. It is not the variant that happened to score
best on the test set; that would be selecting on the test set. All three appear
in every table and in `plots/variant_ablation_rhoF.png`.

### 2.4 The four diagnostics

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

All three configurations of §2.3 are trained per epsilon, on the same data, with
the same budget and the same seed, and all three are reported in every table.
Carrying the strongest one matters: reporting only the specification's literal
`plain` variant would be reporting a baseline nobody would deploy, and reporting
only the framework's default would leave open whether the result is really about
the trunk rather than the architecture. Model selection for each is its own
validation loss; none of them sees the test set during training.

---

## 4. Results

### 4.1 The discretisation is sound at every contrast

The manufactured-solution check that must pass before any smoother number is
interpreted. Left, the errors; right, the observed rates.

| $\varepsilon$ | contrast | $H^1$ rate | $L^2$ rate | $L^\infty$ (finest) |
|---|---|---|---|---|
| $10^{-1}$ | 11× | 1.003 | 1.999 | 7.93e-04 |
| $10^{-2}$ | 101× | 1.006 | 1.998 | 8.07e-04 |
| $10^{-3}$ | 1001× | 1.008 | 1.998 | 8.10e-04 |
| $10^{-4}$ | 10001× | 1.008 | 1.998 | 8.11e-04 |

Rates are measured between the two finest levels (4096 → 16384 DoFs) and are
$O(h)$ for the $H^1$ seminorm and $O(h^2)$ for $L^2$ — optimal for $Q_1$, and
**essentially independent of the contrast**. That is the expected behaviour for a
coefficient that is smooth but has a large dynamic range, and it is the reason
every smoother difference reported below can be attributed to the smoothing and
not to a discretisation that has degraded. The full table per cycle is in
`results/stage1/fem_verification.csv`; the figure is
`results/stage1/plots/fem_verification.png`.

### 4.2 The contrast sweep, and why it is flat

The full statistics — mean, median, standard deviation, 95th percentile, maximum
and the fraction amplified — are in `results/stage1/summary.csv` (one tidy row
per epsilon, section, method and statistic) and rendered in
`results/stage1/readme_tables.md`. The figures are
`plots/eps_vs_mean_rhoF.png`, `plots/eps_vs_p95_rhoF.png`,
`plots/eps_vs_max_rhoF.png` and `plots/variant_ablation_rhoF.png`.

Mean $\rho_F$ across the sweep:

| method | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.4187 | 0.4154 | 0.4153 | 0.4153 |
| Gauss–Seidel | 0.3525 | 0.3461 | 0.3464 | 0.3465 |
| symmetric GS / SSOR | 0.1757 | 0.1727 | 0.1733 | 0.1734 |
| DeepONet (plain MLP) | 0.8994 | 0.8735 | 0.8775 | 0.8973 |
| DeepONet (trig trunk, + skip) | 0.3715 | 0.3615 | 0.3559 | 0.3582 |
| DeepONet (fourier trunk, + skip) | 0.3100 | 0.2812 | 0.2789 | 0.2780 |

The same flatness holds in the tail, which is the part a mean hides: the
95th percentile of $\rho_F$ for SGS/SSOR is 0.2398, 0.2371, 0.2379, 0.2380, and
its maximum is 0.3305, 0.3295, 0.3296, 0.3296. After three smoothing steps the
ordering is unchanged and still flat (SGS/SSOR 0.0427–0.0437, Gauss–Seidel
0.0808–0.0826, Jacobi 0.1660–0.1699).

**Why.** Three measurements say the same thing. The validation-tuned damping does
not move ($\omega_J = 0.60$ and $\omega_{\text{SSOR}} = 1.00$ at all four
contrasts). The coarse operator's condition number moves from 44.2 to 118.6 —
a factor of 2.7 against a factor of 910 in the coefficient contrast. And the
area-weighted distribution of $\kappa_\varepsilon$ barely changes, because
$\kappa_\varepsilon$ is $\kappa_{10^{-1}}$ minus a constant: its *median* is
0.2295, 0.1395, 0.1305, 0.1296 and its area-weighted mean is 0.3499, 0.2599,
0.2509, 0.2500. Lowering $\varepsilon$ deepens a thin band around the
measure-zero set $\{\sin\xi\cos\eta = 0\}$; it does not restructure the operator.
A smoothing factor is governed by the spectral structure of $A_\varepsilon$, and
on this family that structure is nearly contrast-independent.

This is a statement about **this** coefficient family, and it is worth being
explicit about what would be different. A coefficient whose low-$\kappa$ region
has *positive measure* — a channel, a barrier, a layered medium — varies the
energy weighting across a finite area and would be expected to change the
smoothing factors substantially. The family in the specification does not, and
that is a property of the family, measured rather than assumed.

### 4.3 The other three diagnostics agree

**The coarse component ($\rho_C$).** The smoother run on $e_C$ instead of $e_F$,
as a diagnostic — in a real cycle the coarse grid handles these. Every classical
smoother reduces them only partly and the ordering is the same (SGS/SSOR
0.3400–0.3761, Gauss–Seidel 0.5034–0.5476, Jacobi 0.6398–0.6883), again flat in
contrast. The plain MLP *amplifies* them: $\rho_C = 1.0089$–$1.0099$, worse than
doing nothing, at every contrast.

**The controlled modes ($\rho_k$).** `plots/modes_xi_all_eps.png` and
`plots/modes_eta_all_eps.png` plot $\rho_k$ for $\sin(k\xi)$ and $\sin(k\eta)$,
each first projected onto the coarse-space complement. For the $\xi$ modes the
worst case is around $k = 8$ of 16: Jacobi peaks at 0.699, Gauss–Seidel 0.463,
SGS/SSOR 0.289, and the fourier-trunk DeepONet at 0.631 (peaking at $k=9$). For
the $\eta$ modes the curves are flatter but the ordering identical — Jacobi spans
0.298–0.435 and SGS/SSOR 0.149–0.207. The curves at the four contrasts are
visually the same curve.

**The Jacobi-hard errors, and where they live.** Selecting the 20 test errors
with the largest empirical $\rho_J$ and then scoring *every* method on exactly
those gives 0.5690 / 0.5592 / 0.5573 / 0.5571 for Jacobi against SGS/SSOR's
0.2279 / 0.2249 / 0.2270 / 0.2270 — the hard set is hard for Jacobi and the
ordering is unchanged on it, at every contrast.

The spatial question the specification asks — are the difficult error structures
related to the low-$\kappa$ regions? — has a measured answer of **no, and
slightly the other way**. Weighting by the surface area element (the DoF lattice
under-weights the outer equator by up to $3\times$, so a raw DoF count would bias
this), the lowest-$\kappa$ quartile of the surface holds 21.8%, 23.3%, 23.3% and
23.4% of the hard errors' energy against 25.9%, 30.7%, 32.8% and 33.0% of the
whole test set's. The enrichment ratio is 0.84, 0.76, 0.71, 0.71: the hard errors
are *depleted* in the low-$\kappa$ regions, and increasingly so as contrast grows
(the low-$\kappa$ quartile is 25.1% of the surface area, so even the 0.84 at
$\varepsilon = 10^{-1}$ is below parity). The selected errors are dominated by
`multiscale` (8–9 of 20) and `mixed` (5 of 20) at every contrast — rough errors,
which is what a point smoother struggles with, rather than errors localized where
$\kappa$ is small.

### 4.4 The two-grid test: the ranking survives, the gap does not close

$\rho_{TG}$ — one smoothing step followed by an exact Galerkin coarse solve, on
the same test errors:

| method | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
|---|---|---|---|---|
| symmetric GS / SSOR | **0.1316** | **0.1263** | **0.1266** | **0.1268** |
| DeepONet (fourier trunk, + skip) | 0.2251 | 0.2086 | 0.2120 | 0.2123 |
| DeepONet (trig trunk, + skip) | 0.2465 | 0.2385 | 0.2395 | 0.2394 |
| Gauss–Seidel | 0.2191 | 0.2123 | 0.2153 | 0.2163 |
| damped Jacobi | 0.2657 | 0.2593 | 0.2613 | 0.2618 |
| DeepONet (plain MLP) | 0.5736 | 0.5520 | 0.5493 | 0.5556 |
| *coarse correction alone* | *0.6095* | *0.5978* | *0.5958* | *0.5956* |

The two-grid ordering is the smoothing ordering, essentially unchanged: the
improved smoothing does translate into improved two-grid convergence, and it
translates for every method including the learned one. But that cuts both ways —
because the smoother ranking is preserved, the learned smoother's deficit is
preserved with it. SGS/SSOR stays $1.7\times$ better than the best learned model
at every contrast.

Two numbers in that table are worth reading together. The coarse correction alone
gives 0.5956–0.6095, and the plain MLP gives 0.5493–0.5736 — so the plain MLP's
entire contribution to a two-grid step is a few percent on top of what the coarse
grid does by itself. That is the quantitative form of "it is not acting as a
smoother".

The last check confirms the machinery rather than the methods: run on the $e_F$
samples, the **coarse correction alone gives $\rho_{TG} = 1.0000$ exactly**, at
every contrast. It has to — $P^{\mathsf T}A_\varepsilon e_F = 0$ forces the coarse
right-hand side to zero, hence $z = 0$, hence $C_Ge_F = e_F$. A measured 1.0000
is the decomposition, the coarse solve and the two-grid driver all agreeing with
the algebra, and it is the strongest single check in the experiment.

### 4.5 Training

Validation $\mathcal L_{\text{smooth}}$ at the end of training (3000 epochs, the
framework's default budget):

| configuration | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ |
|---|---|---|---|---|
| plain MLP | 0.8266 | 0.7907 | 0.7974 | 0.8168 |
| trig trunk + Jacobi skip | 0.1437 | 0.1359 | 0.1333 | 0.1331 |
| fourier trunk + Jacobi skip | 0.1041 | 0.0848 | 0.0834 | 0.0823 |

Two things to note honestly. First, the learned models get *better* as contrast
grows while the classical smoothers stay flat — a real trend in the same
direction as finding (1), though far too small to change any conclusion. Second,
the plain MLP has **not converged**: its validation loss is still falling at a
comparable rate at epoch 3000 to epoch 100, so its poor showing is partly a
budget statement, not purely an architectural one. `experiments/run_budget_check.sh`
re-runs the strongest variant at four times the budget to bound how much of the
remaining gap is budget rather than structure; its output goes to
`results/stage1_budget/` and is deliberately kept out of every table above.

### 4.6 What the answer is

Taking the specification's chain in order — $\varepsilon$ down, contrast up,
$A_\varepsilon$ changes, smoothing changes, identify the poorly reduced $e_F$
components, compare on those, test in the two-grid method:

* $A_\varepsilon$ changes far less than the contrast suggests (condition number
  $44.2 \to 118.6$ against contrast $11\times \to 10001\times$).
* Consequently the smoothing changes very little, for any method, and **no
  coarse-space-complement component emerges as newly difficult**. The hard
  components are rough, multiscale ones, and they are hard at $\varepsilon =
  10^{-1}$ already.
* The learned smoother is *better than damped Jacobi* on exactly those
  components, at every contrast. That part of the specification's hypothesis —
  "the learned operator handles error components that the selected point smoother
  reduces poorly" — is supported.
* It is **not** better than symmetric Gauss–Seidel / SSOR, and the deficit is
  large ($1.6$–$1.8\times$ in $\rho_F$, $1.65$–$1.71\times$ in $\rho_{TG}$) and
  contrast-independent. The specification's further hypothesis — "if this also
  produces smaller $\rho_{TG}$, report that the improved smoothing translates" —
  is *not* supported in the comparative sense: the improvement translates, but
  only relative to Jacobi, not relative to the best classical smoother.
* Where does the shortfall come from? **Smoothing, not the coarse grid, and not
  their interaction.** The two-grid ordering equals the smoothing ordering at
  every contrast, and $\rho_{TG}$ on $e_F$ isolates the same quantity ($\rho_F$)
  because the coarse correction is exactly the identity there. So the learned
  smoother's deficit is entirely a deficit in the smoothing step; there is no
  interaction term hiding it.

## 5. Scope — what is not claimed

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

## 6. Repository layout

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
  refigure.py                 redraw a run's figures from its saved checkpoints,
                              and check that they reproduce its metrics.json
  compare_budget.py           default budget vs 4x budget, for the fairness check
  selftest.py                 84 checks of every identity above
  run_epsilon.py              the whole experiment for one epsilon

experiments/
  run_fem_exports.sh          builds the solver image, runs it per epsilon
  run_conversion.sh           dumps -> NumPy
  run_stage1.sh               all four epsilons + collection
  run_budget_check.sh         the 4x-budget fairness check (optional, slow)

level_data/eps-<E>-L3/        the converted per-epsilon level problem
results/fem_eps-<E>.csv       the manufactured-solution verification tables
results/stage1/eps-<E>/       one run: metrics.json, checkpoints, history,
                              per-sample NPZ, CSVs, plots/
results/stage1/summary.csv    every statistic, tidy long form
results/stage1/readme_tables.md   the same tables as markdown
results/stage1/plots/         the cross-epsilon figures required by §N
results/stage1/fem_verification.csv   every cycle's errors and rates
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

## 7. Reproducing

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

# 3. the experiment: train and evaluate all three configurations per epsilon,
#    then collect the cross-epsilon figures and tables
bash experiments/run_stage1.sh

# 4. optional: redraw the per-run figures from the saved checkpoints. This also
#    re-derives rho_F from the checkpoints and compares it to metrics.json, so it
#    is the check that the published figures belong to the published weights.
python -m hcsm.refigure --results results/stage1

# 5. optional and slow (~26 min): is the learned smoother's deficit a training
#    budget artefact? Re-runs the strongest variant at 4x the budget.
bash experiments/run_budget_check.sh
```

Each of the four epsilon runs takes about 5.5 minutes on 4 threads (three models
x 3000 epochs plus evaluation); the whole of step 3 is about 25 minutes.

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
