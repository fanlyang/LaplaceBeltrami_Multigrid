# Stage I — does the DeepONet learn the *smoothing property*?

This is the Stage I experiment of the three-stage programme in the thesis:

```
Stage I    L_smooth          learning and testing smoothing selectivity
Stage II   L_TG              interaction with coarse-grid correction
Stage III  recursive V-cycle performance in the complete multigrid solver
```

**Only Stage I is on this branch.** Stages II and III are named above because
they are what the "I" refers to and because they explain why nothing here runs a
V-cycle; they are not implemented here and nothing in this document depends on
them.

Stage I asks one question and answers it with one table:

> **Does the DeepONet learn preferential reduction of the Galerkin
> coarse-space complement?**

Everything here is produced by `stage1_smoothing_property.py`; every number in
this document is a value it printed or wrote to `metrics.json`. Nothing is
transcribed from a separate calculation.

## Summary

**Yes. With a branch wide enough to represent the correction, the learned step
is selective _and_ the best single-step method in the table — and two of the
first three things this experiment "found" were artifacts of an under-capacity
branch.**

| | width 256 | width 768 |
|---|---|---|
| $\mu_F$ (complement reduced) | 0.5803 — fifth of six | **0.1456 — first of six** |
| $\mu_C$ (coarse component) | 1.0400 | 1.0928 |
| selective? | yes, $\mu_C/\mu_F = 1.79$ | yes, $\mu_C/\mu_F = 7.50$ |
| rank of what it can express | 256 | 768 (= complement dim) |
| best rank-$k$ linear map on the same data | 0.5860 | 0.0000 |

Both runs are identical in data, loss, schedule, trunk and seed. The only change
is the branch width, and it moves $\mu_F$ by a factor of four and the rank from
fifth to first.

Four findings:

1. **The complement sub-problem is exactly linear** (§2.2), so selectivity is not
   learned so much as *inherited*: any map fitted to the complement data leaves
   the coarse component alone (every row of §5's rank table has
   $\mu_C \approx 1.02$). What the network actually has to do is fit a linear map —
   and, per finding 4, it does not fit it exactly.
2. **A DeepONet has two rank ceilings, and either can become the thing being
   measured** (§4, §5, §6.3). One on the trunk — the correction lies in
   $\operatorname{span}(T)$ whatever the branch returns — and one on the branch, whose
   Jacobian factors through `width` units. Both are invisible in the loss curve.
   At width 256 the network lands within 1.0% of the best rank-256 *linear* map
   (0.5803 against 0.5860): that number measured the width, not the smoother.
3. **A plausible conclusion drawn from one width was wrong** (§6.4). At width 256
   the network looked strong on smooth errors and weak on algebraic ones, which
   suggested combining the learned and classical steps. At width 768 it beats
   Gauss–Seidel on *every* mechanism, including the algebraic noise it appeared
   worst at. There was no complementarity to exploit — there was a branch that
   could not represent the answer. This is the report's own cautionary tale and is
   left visible rather than quietly deleted.
4. **It still misses the linear limit.** The best rank-768 linear map reaches
   $\mu_F = 0.0000$; the network reaches 0.1456. No rank ceiling and no trunk
   bound is binding at width 768 (§4 floor $0.000001$), so the residual is the
   optimiser failing to close the last 0.15 of a problem whose exact solution is a
   linear map the architecture can represent.

And one defect worth stating plainly: $\mathbb{P}(\rho_C > 1) = 1.0$ in both runs.
The network does not leave the coarse component alone, it grows it — by up to
19.7% at width 768, worse than the 8.4% at width 256. Better complement reduction
came with worse coarse behaviour (§6.5).

Two diagnostics make all of this readable, and both are prerequisites for
trusting the headline number: the **trunk-span floor** (§4) and the
**rank-$k$ reference curve** (§5). Without the second, "$\mu_F = 0.58$" and
"$\mu_F = 0.15$" are numbers with nothing to compare them to.

---

## 1. Setup

On level $\ell = 3$ (1024 DoFs on a $32\times32$ DoF lattice), let

$$A := A_3 \in \mathbb{R}^{n\times n}, \qquad P := P_{3\leftarrow 2} \in \mathbb{R}^{n\times n_c}$$

with $n = 1024$, $n_c = 256$. The coarse operator is the Galerkin one,

$$A_c = P^{T} A P ,$$

and the $A$-orthogonal projection onto $\operatorname{range}(P)$ is

$$\Pi_C = P\left(P^{T}AP\right)^{-1}P^{T}A, \qquad \Pi_F = I - \Pi_C .$$

Every error splits as $e = e_C + e_F$ with $e_C = \Pi_C e$ and $e_F = \Pi_F e$.
The inverse of $P^TAP$ is never formed: the coarse system is factorised once
(sparse LU) and solved per sample.

The four identities the brief states are **verified, not assumed** — on the
smoke test they hold at

| identity | relative residual |
|---|---|
| $e_C^{T} A e_F = 0$ | `1.4e-15` |
| $P^{T} A e_F = 0$ | `2.6e-14` |
| $\lVert e\rVert_A^2 = \lVert e_C\rVert_A^2 + \lVert e_F\rVert_A^2$ | `4.4e-15` |
| $\lambda_{\min}(A_c) > 0$ | `0.274290` |

### 1.1 Why this is an extrapolation test, not a memorisation test

$P^{T}$ is surjective, so $\dim\ker(P^{T}) = n - n_c = 768$; $\operatorname{range}(AP)$ has
dimension $n_c = 256$; and $P^{T}APz = 0 \Rightarrow z = 0$ because $A_c$ is SPD.
Therefore

$$\mathbb{R}^{n} = \operatorname{range}(AP) \;\oplus\; \ker(P^{T}) .$$

Now $r_F = Ae_F$ satisfies $P^{T}r_F = P^{T}Ae_F = 0$, so **every training input lies in
$\ker(P^{T})$**, while every coarse residual $r_C = Ae_C$ lies in
$\operatorname{range}(AP)$. Measured on the actual splits:

| quantity | value | meaning |
|---|---|---|
| $\max \lVert P^{T}r_F\rVert/\lVert r_F\rVert$ over training | `2.5e-14` | training inputs are in $\ker(P^T)$ |
| $\min \lVert P^{T}r_C\rVert/\lVert r_C\rVert$ over training | `1.02` | coarse residuals are not |

The two input sets meet only at the origin. Stage I is therefore a genuine
extrapolation question: the network must extend a map it learned on
$\ker(P^{T})$ to a subspace it has never seen a sample from.

---

## 2. Training

Only the complement component is used. The loss is the relative remaining
$A$-energy of $e_F$ after one learned smoothing step:

$$\mathcal{L}_{\mathrm{smooth}}(\theta) = \frac{1}{N}\sum_{i=1}^{N}
\frac{\bigl\lVert e_{F,i} - B_{\theta}(Ae_{F,i}, X_3)\bigr\rVert_A^2}
     {\lVert e_{F,i}\rVert_A^2},
\qquad \lVert v\rVert_A^2 = v^{T}Av .$$

No coarse-grid correction appears anywhere in this stage. Samples with a
vanishing complement fraction are dropped by a tolerance (`--ftol`, relative
rather than absolute, because every error is normalised to unit $A$-energy).

Errors are drawn from the generators already on this branch — `smooth`,
`multiscale`, `localized`, `algebraic`, and their Dirichlet mixtures — each
mechanism on its own RNG stream derived from `(seed, role, mechanism, count)`,
so the train/val/test roles cannot share a draw.

### 2.1 The model, and one choice that is stated out loud

$$B_{\theta}(r, X) = \lVert r\rVert_2 \cdot \hat B_{\theta}\!\left(\frac{r}{\lVert r\rVert_2},\, X\right),
\qquad \delta e = \sum_k b_k(r)\,T_k(x)$$

The wrapping makes the map **homogeneous of degree one in $r$**. This is a
deliberate constraint and it is disclosed because it limits the answer: every
classical smoother in the comparison table is *linear*, and for a linear smoother
$r \mapsto \delta e$ is exactly homogeneous — the exact solve $\delta e = A^{-1}r$
being the extreme case. Imposing homogeneity puts the network in the same class
as the objects it is compared against. A non-homogeneous DeepONet has strictly
more freedom, all of it on the coarse side, which is exactly where the question
is; so the constraint is the conservative choice, not the flattering one.

The branch is bias-free, so $r = 0 \Rightarrow \delta e = 0$: an exact discrete
solution is a fixed point by construction, not merely by training.

### 2.2 The complement sub-problem is exactly linear

This is worth stating before any number, because it determines what Stage I can
and cannot show.

Restricted to the complement, the exact map is $r_F \mapsto A^{-1}r_F$. Every
possible training input is $r_F = Ae_F$ with $e_F \in \Pi_F\mathbb{R}^n$, and
$A^{-1}$ maps $\operatorname{range}(A|_{\Pi_F\mathbb{R}^n})$ back onto
$\Pi_F\mathbb{R}^n$ bijectively. So the target of Stage I is **a linear map**, and a
universal approximator only has to *fit a linear function* to reach
$\mathcal{L}_{\mathrm{smooth}} \to 0$. The network's non-linearity buys nothing here;
it is overhead on this sub-problem.

Two consequences, both of which show up in the results:

1. A linear fit to the training pairs should reach $\mu_F \approx 0$ out of
   sample. The ridge diagnostic tests exactly this, with its regulariser chosen
   on the same validation metric as the network, and it is the reason that
   diagnostic is in the report at all.
2. What is *not* determined by this argument is the behaviour on
   $\operatorname{range}(AP)$. A map fitted only on $\ker(P^{T})$ has complete freedom on
   $\operatorname{range}(AP)$, and the two subspaces span $\mathbb{R}^n$ between them, so
   $\mu_C$ is a pure extrapolation measurement — which is precisely the Stage I
   question.

So the interesting output of this stage is **not** "how small is $\mu_F$" but
"what does a map trained only on the complement do to the coarse component, and
does the network's answer differ from what a linear fit does".

### 2.3 The confound that is measured rather than hidden

Because $\delta e = \sum_k b_k(r)T_k(x)$, the correction lies in $\operatorname{span}(T)$
**whatever the branch returns**. A small $\mu_F$ can therefore be a statement
about the trunk instead of about smoothing. This is not hypothetical — it is
what the first run of this experiment actually measured, and it is why the trunk
geometry is now a decision with a diagnostic behind it (§4).

---

## 3. Evaluation

The learned smoother is scored by the per-sample reduction factor

$$\rho_F(e_F) = \frac{\lVert e_F - B_{\theta}(Ae_F,X_3)\rVert_A}{\lVert e_F\rVert_A},
\qquad \mu_F = \mathbb{E}[\rho_F],$$

and, on the same test errors, for the coarse component

$$\rho_C(e_C) = \frac{\lVert e_C - B_{\theta}(Ae_C,X_3)\rVert_A}{\lVert e_C\rVert_A},
\qquad \mu_C = \mathbb{E}[\rho_C].$$

$\mu_C = 1$ is **not** imposed during training. Selectivity is the statement
$\mu_F \ll \mu_C$. Alongside the means, the median, standard deviation, 95th
percentile, maximum, and the amplified share $\mathbb{P}(\rho>1)$ are reported
for both components.

Every classical smoother is evaluated on **the same test errors, in the same
$A$-energy norm**, and each was checked in `--selftest` against the dense matrix
that defines it (all five agree to $\sim10^{-15}$):

| smoother | operator applied |
|---|---|
| Jacobi | $S = I - D^{-1}A$ |
| Damped Jacobi | $S = I - \omega D^{-1}A$, $\omega$ chosen on **validation** $\mu_F$ |
| Gauss–Seidel | $S = I - (D+L)^{-1}A$ |
| Symmetric GS | $S = \bigl(I-(D+U)^{-1}A\bigr)\bigl(I-(D+L)^{-1}A\bigr)$ |
| SSOR | as above with $\omega$; $\omega=1$ **is** symmetric GS, and the table shows them coinciding |

Both the network and damped Jacobi get their free parameter selected on the same
validation metric and scored on test, so the comparison is symmetric.

---

## 4. The trunk-span diagnostic

`tools/trunk_reach.py` computes, **before training**, the best $\mu_F$ that *any*
branch could achieve against a given trunk:

$$\operatorname{reach}_F(e) = \frac{\lVert \operatorname{proj}_{\operatorname{span}(T)} e_F\rVert_A^2}{\lVert e_F\rVert_A^2},
\qquad \mu_F \ge \mathbb{E}\left[\sqrt{1-\operatorname{reach}_F}\right].$$

For the exported hierarchy the floor is governed by the trunk's highest spatial
frequency, and the value it must reach is the **fine** lattice's Nyquist:

| trunk | dim | rank | floor on $\mu_F$ |
|---|---|---|---|
| Fourier $K=2$ | 48 | 24 | 0.997784 |
| Fourier $K=4$ | 160 | 80 | 0.989724 |
| Fourier $K=6$ | 336 | 168 | 0.965419 |
| Fourier $K=8$ | 576 | 288 | 0.911890 |
| Fourier $K=10$ | 880 | 440 | 0.786705 |
| Fourier $K=12$ | 1248 | 624 | 0.563591 |
| Fourier $K=14$ | 1680 | 840 | 0.286669 |
| **Fourier $K=16$** | 2176 | 1023 | **0.000675** |

The reason is structural. $\operatorname{range}(P)$ is what the level-2 lattice
(for $L3$, a $16\times16$ lattice, Nyquist 8) can represent, so its $A$-orthogonal
complement is very nearly the band **between the coarse and the fine Nyquist** —
here modes $8 \lesssim |k| \le 16$. A Fourier trunk needs modes up to the fine
Nyquist 16 before it can span that band, and nothing below $K=16$ is usable.
That is the structural statement of why the trunk's *frequency content* and not
merely its size decides whether it binds.

A hidden MLP layer narrows this further, because $\operatorname{rank}(T) \le$
trunk width. A trunk of width 256 cannot exceed rank 257 however many modes it is
fed, which holds the floor near 0.75 on its own — a run in that configuration
measures its trunk, not its smoother. Earlier runs of this experiment did exactly
that: with `--trunk-modes 4 --p 256 --width 64` the floor was `0.981` and the
network landed on `µ_F = 0.995`, i.e. on its trunk's ceiling rather than anywhere near
what the branch could have done.

The headline run therefore **freezes** the trunk (`--trunk-freeze fourier-orth`)
to an orthonormalised Fourier basis of the feature map. `Q` has orthonormal
columns spanning the same space as the features, and the features reach the fine
Nyquist, so

$$\operatorname{span}(T) = \mathbb{R}^n, \qquad T^{T}T = I,
\qquad \text{floor} = 0 \ \text{exactly}.$$

Freezing is what makes the trunk *provably* non-binding rather than empirically
so, and orthonormality keeps the branch's coefficients equal to the correction's
own Fourier coefficients, so they carry no conditioning penalty. Nothing in
expressiveness is lost: a learned trunk in this configuration also reached
$\operatorname{rank}(T) = 1024$ of $p = 1024$, so the two parameterise the same class —
but the learned trunk must be re-evaluated and re-differentiated in every
training step (≈70% of the step cost for a matrix that is already full rank), and
it can drift to a lower-rank configuration mid-run, silently re-introducing the
confound this diagnostic exists to remove.

---

## 5. The branch-capacity diagnostic

The trunk is not the only part of the model that can bind. The branch is an MLP
that narrows to `width` units, so its Jacobian factors through
$\mathbb{R}^{\text{width}}$; and the target of this stage is a *linear* map of rank
equal to the complement dimension (768), by §2.2. A branch narrower than the
complement therefore cannot express the exact correction.

`tools/branch_rank_probe.py` computes how much that costs, exactly: the best
linear map of rank $\le k$ fitted to the same training pairs, with the same
**per-sample relative weighting** the network's loss uses, scored on the same test
errors. It is a reduced-rank regression, solved from one SVD, and it is a
property of the **data** — no network is trained.

The weighting matters and is not cosmetic. The network minimises
$\sum_i \lVert\cdot\rVert_A^2/\lVert e_i\rVert_A^2$, which is a weighted least squares
with $w_i = 1/\lVert e_i\rVert_A^2$. Fitting the *unweighted* problem instead optimises
a different objective from the one the table scores, and it is not the best
rank-$k$ map for the reported metric: at $k = 256$ it returns $\mu_F = 0.6803$
against $0.5860$ weighted. That understates what rank 256 can achieve, which
*flatters the network* by about 0.09 — it would make 0.5803 look like a 0.10
margin over the reference when the true margin is 0.006. With the weights applied
the reference is the best rank-$k$ map for the metric being reported.

| rank $k$ | $\mu_F$ | $\mu_C$ | $\mu_C/\mu_F$ |
|---|---|---|---|
| 64 | 0.8057 | 1.0103 | 1.25 |
| 128 | 0.7160 | 1.0150 | 1.42 |
| 192 | 0.6467 | 1.0181 | 1.57 |
| **256** | **0.5860** | 1.0200 | 1.74 |
| 384 | 0.4768 | 1.0224 | 2.14 |
| 512 | 0.3666 | 1.0237 | 2.79 |
| 640 | 0.2375 | 1.0245 | 4.31 |
| **768** | **0.0000** | 1.0249 | ∞ |
| 1024 | 0.0000 | 1.0249 | ∞ |

The training residual set has rank exactly 768, so nothing above $k = 768$ can
help — the curve saturates at the complement dimension, and that is the rank a
branch must have to be able to do the job.

Two things follow, and both matter for reading §6:

* A rank-256 map can do no better than $\mu_F = 0.586$ on these test errors. Any
  run whose branch narrows to 256 is capped there, whatever the optimiser does.
* $\mu_C \approx 1.02$ at **every** rank. Selectivity is not something the
  network has to discover: *any* map fitted to the complement data leaves the
  coarse component essentially alone. The interesting question is therefore not
  whether selectivity appears, but how well the complement is actually reduced.

---

## 6. Results

Two runs, identical in every respect except the width of the branch: same data,
same loss, same schedule, same frozen trunk, same seed. **The width-768 run
answers Stage I's question.** The width-256 run is kept alongside it because it
is the more instructive of the two — it is what a confident, plausible, wrong
answer looks like, and it was the first thing this experiment produced.

### 6.1 The table the brief asks for

1024 test errors, level $\ell = 3$, $n = 1024$, complement dimension 768.
Width-768: best validation $\mu_F = 0.1701$ at epoch 1199 of 1200; wall clock 2092 s.

| Smoother | $\mu_F$ | $\mu_C$ | $\max\rho_F$ | $\mathbb{P}(\rho_F>1)$ | median $\rho_F$ | sd $\rho_F$ | p95 $\rho_F$ | $\mu_C/\mu_F$ |
|---|---|---|---|---|---|---|---|---|
| Jacobi | 0.6996 | 0.6734 | 1.4310 | 0.1387 | 0.6653 | 0.2598 | 1.3035 | 0.96 |
| Damped Jacobi | 0.4248 | 0.7380 | 0.7230 | 0.0000 | 0.4064 | 0.0592 | 0.5522 | 1.74 |
| Gauss–Seidel | 0.3664 | 0.6180 | 0.5658 | 0.0000 | 0.3511 | 0.0476 | 0.4513 | 1.69 |
| Symmetric GS | 0.1889 | 0.4695 | 0.3848 | 0.0000 | 0.1743 | 0.0397 | 0.2606 | 2.49 |
| SSOR | 0.1889 | 0.4695 | 0.3848 | 0.0000 | 0.1743 | 0.0397 | 0.2606 | 2.49 |
| **DeepONet (width 768)** | **0.1456** | **1.0928** | 0.3198 | 0.0000 | 0.1334 | 0.0886 | 0.2695 | **7.50** |
| $A^{-1}$ (reference) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — |
| ridge (diagnostic) | 0.0000 | 1.0249 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ∞ |

and the same table for the width-256 branch (`results/stage1_L3`; best validation
$\mu_F = 0.6302$ at epoch 854 of 1155; wall clock 1015 s). The classical rows are
identical — they do not depend on the model — so only the DeepONet row changes:

| Smoother | $\mu_F$ | $\mu_C$ | $\max\rho_F$ | $\mathbb{P}(\rho_F>1)$ | median $\rho_F$ | sd $\rho_F$ | p95 $\rho_F$ | $\mu_C/\mu_F$ |
|---|---|---|---|---|---|---|---|---|
| **DeepONet (width 256)** | **0.5803** | **1.0400** | 0.9188 | 0.0000 | 0.6800 | 0.2298 | 0.7906 | 1.79 |
| ridge (diagnostic) | 0.0000 | 1.0249 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ∞ |

### 6.2 Answering the boxed question

> **Does DeepONet learn preferential reduction of the Galerkin coarse-space
> complement?**

**Yes — and it is the best single-step method in the table.**

With a branch wide enough to represent the complement correction, the learned
step achieves

$$\mu_F = 0.1456, \qquad \mu_C = 1.0928, \qquad \mu_C/\mu_F = 7.50,$$

which places it **first of the six methods** on complement reduction:

$$\textbf{DeepONet } 0.146 \;<\; \text{SymGS } 0.189 \;<\; \text{GS } 0.366
\;<\; \text{damped Jacobi } 0.425 \;<\; \text{Jacobi } 0.700 .$$

It removes 85% of the complement's $A$-energy in one step, better than symmetric
Gauss–Seidel, and no sample is amplified ($\mathbb{P}(\rho_F>1) = 0$, worst case
$\max\rho_F = 0.320$). Training never mentioned $e_C$, so this is genuine
extrapolation across the direct sum $\operatorname{range}(AP)\oplus\ker(P^{T})$.

Three qualifications, each supported by the rest of §6:

* **Width is doing the work, not non-linearity.** At width 256 the same
  architecture scores 0.5803 — fifth of six. §6.3 gives the reference curve that
  makes this quantitative.
* **It still misses the linear limit.** The best linear map of rank 768 on the
  same data reaches $\mu_F = 0.0000$ (§5). The network's 0.1456 is a residual
  optimisation gap: the parameterisation can represent the answer and SGD does
  not fully find it.
* **Selectivity costs amplification.** $\mu_C = 1.0928$: *every* coarse sample is
  grown, by up to 19.7% (§6.5).

### 6.3 Why width 256 gave the opposite answer

The branch narrows to `width` units before widening again, so its Jacobian
factors through $\mathbb{R}^{\text{width}}$; the complement is 768-dimensional.
§5's rank-$k$ reference says what that costs:

| quantity | $\mu_F$ |
|---|---|
| best rank-256 linear map (§5) | 0.5860 |
| DeepONet, width 256 | 0.5803 |
| best rank-640 linear map (§5) | 0.2375 |
| DeepONet, width 768 | 0.1456 |
| best rank-768 linear map (§5) | 0.0000 |

At width 256 the network lands within 1.0% of the best rank-256 linear map —
**the branch's width, not its non-linearity or its optimiser, is what that number
measured**. Widening the branch to the complement dimension moves $\mu_F$ from
0.5803 to 0.1456, a factor of four, with nothing else changed.

This is the same failure mode as the trunk confound in §4, one level down, and it
is why both bounds are in this report: a DeepONet has *two* rank ceilings, one on
each side of the bilinear form, and either can silently become the thing under
measurement.

The residual gap (0.1456 against 0.0000) is not a rank ceiling — width 768 is
exactly the complement dimension — and not the trunk (§4: floor 0.000001). It is
the optimiser failing to close the last 0.15 of a problem whose exact solution is
a linear map the architecture can represent.

### 6.4 Per-mechanism: "complementary strengths" was an artifact

$\mu_F$ broken down by the generator that produced the error. The width-256 column
is the one that produced an earlier claim of complementary strengths; the
width-768 column is what the same network does with enough capacity:

| mechanism | DeepONet w256 | **DeepONet w768** | Damped Jacobi | Gauss–Seidel |
|---|---|---|---|---|
| smooth | 0.0907 | **0.0129** | 0.4554 | 0.4029 |
| multiscale | 0.6144 | **0.0978** | 0.4345 | 0.3711 |
| localized | 0.4270 | **0.0627** | 0.4461 | 0.4155 |
| algebraic | 0.7675 | **0.2503** | 0.3961 | 0.3325 |
| mixed | 0.6856 | **0.1853** | 0.4166 | 0.3523 |

At width 256 the network looked strong on smooth errors and weak on algebraic
ones, which suggested the learned and classical steps fail in opposite regimes and
ought to be combined. **That reading does not survive the wider run.** At width
768 the network beats Gauss–Seidel on *every* mechanism, including the purely
algebraic coefficient-space noise it appeared worst at — by 0.2503 against
0.3325. There is no complementarity to exploit; there was an under-capacity branch
and a mechanism whose errors it could not represent.

The superseded reading is left visible above rather than quietly deleted, because
the comparison is the point: a single run at the wrong width produced a specific,
plausible, actionable and wrong conclusion about how to build a better smoother.
It is the clearest argument in this report for publishing the capacity bound next
to the result.

### 6.5 A defect worth naming

$\mathbb{P}(\rho_C > 1) = 1.0$ in **both** runs: every coarse sample is amplified.
Better complement reduction comes with worse coarse behaviour — $\mu_C$ rises from
1.0400 (width 256) to 1.0928 (width 768), and $\max\rho_C$ from 1.0837 to 1.1970.

So the network does not leave the coarse component alone so much as grow it, and
the more effective it becomes at its actual job the more it grows it. In a
two-grid cycle that is a real cost, and it is the opposite of what the training
loss asked for: the loss never mentioned $e_C$, and the extrapolation to
$\operatorname{range}(AP)$ is expansive rather than zero. It does not destroy
selectivity ($\mu_C/\mu_F = 7.50$) but it is not nothing, and it is visible only
because the two components are measured separately. Stage II is where this
interaction becomes directly testable.

### 6.6 Figures

Written to `results/stage1_L3/plots/` and `results/stage1_L3_w768/plots/` by the
runs themselves:

| file | what it shows |
|---|---|
| `selectivity_map.png` | every method as a point in the $(\mu_F,\mu_C)$ plane; the selective half-plane is above the diagonal |
| `per_mechanism.png` | the §6.4 table as bars |
| `reduction_histograms.png` | the full $\rho_F$ and $\rho_C$ distributions behind the means |
| `loss_curves.png` | train and validation loss over the run |

---

## 7. Reproducing

```bash
# checks first: projector identities, ker(P^T) dimension, and every classical
# smoother against the dense matrix that defines it
python stage1_smoothing_property.py --selftest

# the two diagnostics (no training)
python tools/trunk_reach.py       --data-dir level_data/L3
python tools/branch_rank_probe.py --data-dir level_data/L3

# THE run -- branch width 768, which is what answers the question (~35 min)
python stage1_smoothing_property.py \
    --data-dir level_data/L3 --out-dir results/stage1_L3_w768 \
    --epochs 1200 --batch 256 --lr 3e-3 --patience 300 \
    --p 1024 --width 768 --depth 4 --trunk-modes 16

# the under-capacity comparison -- branch width 256 (~17 min).  Kept because it
# is what a confident wrong answer looks like; see section 6.3.
python stage1_smoothing_property.py \
    --data-dir level_data/L3 --out-dir results/stage1_L3 \
    --epochs 1200 --batch 256 --lr 3e-3 --patience 300 \
    --p 1024 --width 256 --depth 4 --trunk-modes 16

# redraw the metrics-only figures without retraining
python stage1_smoothing_property.py --replot --out-dir results/stage1_L3
```

Outputs per run: `metrics.json` (every number, with the scope block),
`history.csv`, `run.log` (the table as printed), `plots/`.

**Runs are deterministic.** `torch.manual_seed(seed)` is applied *before* the
model is constructed, not merely inside the training loop: torch seeds its global
RNG nondeterministically at process start, so seeding only inside `train()` leaves
the initial weights varying between runs of identical arguments. Verified — two
independent invocations of the same command produce byte-identical epoch-0 values.
The error draws were already deterministic, each mechanism having its own RNG
stream derived from `(seed, role, mechanism, count)`.

---

## 8. Scope — what this stage does not claim

* **One smoothing step.** No V-cycle is run and no claim is made about the
  complete multigrid solver. That is Stage III.
* **No coarse-grid correction** appears in the loss or the evaluation; the step
  deliberately stops after smoothing.
* **One DoF count, one DoF ordering.** $n$ is baked into the first branch layer,
  so the trained object is tied to this level. No cross-mesh or
  mesh-refinement generalisation is claimed or measured.
* **Coefficient-vector norms only.** Every norm is $v^{T}Av$ on coefficient
  vectors. $M_h$ is not exported, so no continuous $L^2(\Gamma)$ norm is
  available and none is claimed.
* **The complement is not a frequency band.** It is the $A$-orthogonal
  complement of $\operatorname{range}(P)$ — a precisely defined subspace. §4 shows it is
  *nearly* the band between the coarse and fine Nyquist, but that is an
  observation about this hierarchy, not the definition.
* **Both $\mu_F$ values are width-specific.** 0.5803 belongs to a width-256 branch
  and 0.1456 to a width-768 one, and §5's reference curve says so. Neither is a
  statement about DeepONets in general, nor a floor on what a wider branch
  achieves: at width 768 the residual gap to the rank-768 linear limit is an
  optimisation gap, and nothing here establishes where a still wider or
  better-optimised run would stop.
* **The widths chosen are the two that answer the question**, not a sweep. 256 is
  the configuration the first run happened to use, and 768 is the complement
  dimension — the smallest width that *can* represent the exact correction. No
  claim is made about the shape of $\mu_F$ between them, other than the rank-$k$
  linear reference of §5.
* **The ridge and rank-$k$ rows are diagnostics, not competitors.** They are
  linear maps, deliberately outside the brief's table; they are in the report
  because without them there is nothing to say whether $\mu_F = 0.15$ is good,
  bad, or exactly what the architecture permits.
