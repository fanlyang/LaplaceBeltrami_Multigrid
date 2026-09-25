# Laplace–Beltrami Problem on a Torus

A single-file [deal.II](https://www.dealii.org/) program that solves the surface
(Laplace–Beltrami) problem on a torus with high-order Lagrange finite elements,
and solves the resulting linear system with **conjugate gradients preconditioned
by a geometric multigrid V-cycle**.

The program is a substantially extended relative of the
[deal.II step-38](https://www.dealii.org/current/doxygen/deal.II/step_38.html)
tutorial: step-38 explains the surface finite element idea, this code adds a
variable coefficient, a manufactured exact solution, a full geometric multigrid
hierarchy, and a set of matrix/vector dumps for external analysis.

---

## 1. The problem

We solve

$$-\nabla_g \cdot (\kappa \, \nabla_g u) + u = f$$

on the surface of a torus, where $\nabla_g$ and $\nabla_g\cdot$ are the gradient
and divergence induced by the surface metric.

**Geometry.** The torus has major radius $R = 2$ and minor radius $r = 1$ and is
parametrised by the two angles $\xi$ (around the large circle) and $\eta$ (around
the small circle):

$$x = (R + r\cos\eta)\cos\xi, \qquad y = r\sin\eta, \qquad z = (R + r\cos\eta)\sin\xi.$$

The induced metric is diagonal with scale factors

$$h_\xi = R + r\cos\eta, \qquad h_\eta = r .$$

**Coefficient and exact solution.** Both are smooth functions of the angles:

$$\kappa(\xi,\eta) = 1.1 + \sin^2\xi\cos^2\eta, \qquad u(\xi,\eta) = \sin\xi \sin\eta .$$

**Right-hand side.** $f = -\nabla_g\cdot(\kappa\nabla_g u) + u$ is evaluated in
closed form. In the angle coordinates the surface divergence of a field
$(a_\xi, a_\eta)$ is

$$\nabla_g\cdot a = \frac{1}{h_\xi r}\left[\frac{\partial}{\partial \xi}(r a_\xi) + \frac{\partial}{\partial \eta}(h_\xi a_\eta)\right],$$

which the code differentiates analytically. Because $u$ is known exactly, every
run produces a genuine discretisation error, so the program doubles as a
convergence study.

**Boundary conditions.** The torus is a closed surface — there is no boundary and
hence no boundary condition to impose. Only hanging-node constraints from
refinement are eliminated.

---

## 2. Discretisation

| Piece | Choice |
| --- | --- |
| Triangulation | `Triangulation<2, 3>` from `GridGenerator::torus(R, r)` |
| Geometry | `TorusManifold<2>`, so refined vertices land on the true surface and not on the surrounding polyhedron |
| Finite element | `FE_Q<2, 3>(degree)` — the degree is a command-line argument |
| Mapping | `MappingQ<2, 3>(degree)`, so the surface integrals use the exact curved geometry |
| Quadrature | `QGauss<2>(2 * degree + 1)` |
| Assembly | `MeshWorker::mesh_loop`, over active cells for the system and over level cells for the multigrid hierarchy |

Each cell contributes

$$A_{ij} = \int_K \left[\kappa \, \nabla \varphi_i \cdot \nabla \varphi_j + \varphi_i\varphi_j\right] dS,
\qquad
f_i = \int_K f \, \varphi_i \, dS,$$

with the gradients and the Jacobian weights coming from the curved mapping, so
the integral is taken over the surface of the torus rather than over a flat
reference cell. Refinement is uniform, which keeps the mesh hierarchy simple.

---

## 3. The method used to solve the linear system

Which solver is used is chosen **at run time** on the command line, so that
the alternatives can be compared without recompiling:

| `method` | What it does |
| --- | --- |
| `jacobi` | CG preconditioned by Jacobi |
| `ssor` | CG preconditioned by SSOR |
| `mg` | CG preconditioned by one geometric multigrid V-cycle — the default |
| `mg-solver` | the V-cycle used as the solver itself, by defect correction |
| `direct` | sparse direct factorisation (UMFPACK) |

The iterative methods all stop at a relative residual of $10^{-10}$, with a
limit of 1000 steps. `jacobi`, `ssor` and `mg` are CG runs that differ only in
their preconditioner; `mg-solver` iterates the V-cycle on the defect with no
Krylov acceleration, so what it measures is the convergence rate of multigrid
itself; `direct` does not iterate at all. See
[§7](#7-comparing-the-solvers) for what the comparison shows.

The multigrid preconditioner is a V-cycle built on the same finite element
space, restricted to each level:

- **Smoother** — `PreconditionSOR` with 2 symmetric steps per level.
- **Coarse solver** — `MGCoarseGridHouseholder` on the level-0 matrix.
- **Transfer** — `MGTransferPrebuilt`.
- **Interface matrices** — assembled over refinement edges via `is_interface_matrix_entry`, so the V-cycle handles the level interfaces correctly.

Because the level matrices come from the same `cell_worker` as the active
system, the hierarchy is a true geometric multigrid hierarchy for this operator.

**Why a codimension-one deal.II is required.** The object that turns the V-cycle
into a preconditioner for the outer CG iteration is `PreconditionMG`,
instantiated here with an extra `spacedim` argument:

```cpp
PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>, spacedim>
  mg_preconditioner(dof_handler, mg, mg_transfer);
```

That fourth argument is not part of a stock deal.II: upstream `PreconditionMG`
is declared with three template parameters and its constructor takes a
`DoFHandler<dim>` — that is, `DoFHandler<dim, dim>` — which a surface problem
does not have. The
[`fanyoung/dealii-codim`](https://hub.docker.com/r/fanyoung/dealii-codim) build
of deal.II 9.7.1 supplies the `dim != spacedim` variant along with the matching
instantiations of `MGTransferPrebuilt`, `MGConstrainedDoFs` and the level global
transfer. See [Building](#4-building).

---

## 4. Building

### The codimension-one deal.II build

This is a **surface** problem: the mesh is a `Triangulation<2, 3>`, so
`dim = 2` but `spacedim = 3`. A stock deal.II does not instantiate its multigrid
classes for that case, and linking against `dealii/dealii` fails with undefined
symbols such as

```
MGTransferPrebuilt<Vector<double>>::build<2, 3>(DoFHandler<2, 3> const&)
MGConstrainedDoFs::initialize<2, 3>(DoFHandler<2, 3> const&, ...)
MGLevelGlobalTransfer<Vector<double>>::copy_to_mg<2, Vector<double>, 3>(...)
```

The [`fanyoung/dealii-codim`](https://hub.docker.com/r/fanyoung/dealii-codim)
image is a deal.II 9.7.1 build that additionally instantiates them for
`dim != spacedim`, together with the `spacedim` variant of `PreconditionMG`. It
is the supported way to build this code, which is why it is the default in
`CMakeLists.txt`.

> **Note.** The published `fanyoung/dealii-codim:9.7.1` tag is a `linux/arm64`
> image. On an x86_64 machine Docker Desktop runs it under emulation: it works,
> but the build and the runs are several times slower than native.

### Option A — Docker (no local deal.II required)

```bash
docker build -t laplace-beltrami .
docker run --rm laplace-beltrami            # degree 3, 5 refinement cycles
docker run --rm laplace-beltrami 4 4        # degree 4, 4 refinement cycles
```

Output files are written into the working directory. To get them out of the
container, mount a host directory and run there:

```bash
mkdir -p out
docker run --rm -v "$PWD/out:/work" -w /work laplace-beltrami 3 5
```

### Option B — A local deal.II installation

`CMakeLists.txt` defaults `DEAL_II_DIR` to the path used inside the codim image,
`/opt/dealii-9.7.1-codim/lib/cmake/deal.II`. Point it at your own build of the
same, either on the command line

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DDEAL_II_DIR=/path/to/dealii-codim/lib/cmake/deal.II
cmake --build build -j
```

or through the `DEAL_II_DIR` environment variable. The deal.II CMake macros take
care of include paths, compiler flags and the libraries to link.

---

## 5. Running

```bash
./build/solver <degree> <n_refinement_cycles> [method] [--no-dump] [--csv=FILE]
```

For example

```bash
./build/solver 3 5
```

runs a cubic ($Q_3$) discretisation through 5 uniform refinement cycles with the
default solver (CG preconditioned by multigrid), printing for each cycle: the
mesh sizes, the number of degrees of freedom, the iteration count, the measured
convergence rate and residual, the time spent in each phase, and the memory
used. It also prints the $H^1$, $L^2$ and $L^\infty$ errors against the exact
solution.

Choosing a different solver is a third argument:

```bash
./build/solver 3 5 mg-solver    # iterate the V-cycle on its own
./build/solver 3 5 direct       # UMFPACK
```

Two options control the output:

- `--no-dump` skips the `.vtk` and `.coo` files described in [§6](#6-output-files).
  These are large — the `.coo` dump of a deep hierarchy runs to hundreds of
  megabytes — and writing them would dominate the very timings a benchmark run
  is trying to measure.
- `--csv=FILE` appends one row per cycle to `FILE`, with the timings, the
  memory and the errors. Rows are appended rather than rewritten, so a whole
  sweep collects into a single file.

All output files are written into the **current working directory**, so run the
program from a scratch directory if you want to keep them separate.

---

## 6. Output files

Unless `--no-dump` is given, a run with `n_refinement_cycles = N` produces the
following for each cycle `c = 0 … N-1`:

| File | Contents |
| --- | --- |
| `solution-c.vtk` | The computed solution, for ParaView or VisIt |
| `A-global-cycle-c.coo` | The active system matrix $A$ |
| `rhs-global-cycle-c.txt` | The active right-hand side $f$ |
| `A-level-l-cycle-c.coo` | The multigrid level matrix $A_l$, for every level $l$ |
| `mg-level-info-cycle-c.txt` | Per level: number of DoFs, matrix rows and stored nonzeros |

The two custom formats are deliberately plain so that they can be loaded from
MATLAB, Python or Julia:

**`.coo` (sparse coordinate format)**

```
n_rows n_cols
row column value
row column value
...
```

Rows and columns are zero-based and only nonzeros are stored; values are written
in scientific notation with 17 significant digits.

**`.txt` (dense vector format)**

```
size
value_0
value_1
...
```

`mg-level-info-cycle-c.txt` starts with a header line `# level dofs matrix_rows
stored_entries` followed by one row per level.

Note that no per-level right-hand side is written: in a V-cycle the coarse-grid
right-hand side is the restricted residual, not an independently assembled $f_l$.

---

## 7. Comparing the solvers

The five methods of [§3](#3-the-method-used-to-solve-the-linear-system) are not
five ways of doing the same thing — they are the standard alternatives for a
symmetric positive definite system, and running them against each other is what
shows why multigrid is worth its complexity.

```bash
docker build -t laplace-beltrami-bench .
./bench/run_sweep.sh 3 7          # cubic elements, cycles 0..6
./bench/analyse.py bench/results-d3.csv
./bench/plot.py bench/results-d3.csv
```

### 7.1 At 589,824 unknowns

One row per method at the largest problem in the sweep (cubic elements, seven
refinement cycles, 65,536 cells). *Linear algebra* is the preconditioner or
factorisation plus its application, and deliberately excludes assembly: that is
the same $\mathcal{O}(N)$ work for every method and would only dilute the
comparison.

| Method | Iterations | Linear algebra | Total | Memory (objects) | Memory (process) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CG + Jacobi | 1000 — **did not converge** | 179.1 s | 309.2 s | 203.5 MB | 430.5 MB |
| CG + SSOR | 957 | 692.6 s | 835.6 s | 203.5 MB | 451.7 MB |
| MG standalone | 31 | 114.3 s | 335.6 s | 443.5 MB | 833.0 MB |
| **CG + MG** | **15** | **62.1 s** | **329.7 s** | 443.5 MB | 793.1 MB |
| Direct (UMFPACK) | 1 | 652.0 s | 800.4 s | 203.5 MB | **3347.3 MB** |

Three things are worth stating plainly.

**Jacobi does not solve this problem at all.** It exhausts its budget of 1000
iterations with the relative residual still at $9.7 \times 10^{-8}$, three
orders of magnitude short of the $10^{-10}$ every other method reaches. Its
$H^1$ error ($2.570 \times 10^{-7}$) is consequently *worse* than the other
four, which agree with each other to every digit printed
($2.549 \times 10^{-7}$) — the first point in the sweep where the methods no
longer produce the same answer, and the reason the benchmark records
non-convergence instead of treating it as a crash. Jacobi's run also *looks*
cheap in the table (309 s) precisely because it stopped early without solving
anything.

**The direct solver needs 3.3 GB to factor a 204 MB matrix.** That is a factor
of 16.4, and it is the single most important number here.

**Multigrid is now the fastest method as well as the cheapest in memory**, and
it is the only one whose advantage is structural rather than a constant factor.

### 7.2 The scaling is the argument

One problem size proves little: a direct solver has a small constant and wins at
small $N$. What matters is how each method responds to refinement. Fitting
$\log y = a + b \log N$ over the largest half of the *converged* points, where
the asymptotics have taken over — Jacobi's capped run is a lower bound, not a
measurement, and including it would understate the growth it failed to finish:

| Method | Iterations | Linear algebra | Total time | Process memory |
| --- | ---: | ---: | ---: | ---: |
| CG + Jacobi | $N^{0.50}$ | $N^{1.26}$ | $N^{1.08}$ | $N^{0.51}$ |
| CG + SSOR | $N^{0.47}$ | $N^{1.40}$ | $N^{1.28}$ | $N^{0.66}$ |
| MG standalone | $N^{0.00}$ | $N^{0.89}$ | $N^{0.94}$ | $N^{0.74}$ |
| **CG + MG** | $N^{0.02}$ | $N^{0.90}$ | $N^{0.99}$ | $N^{0.74}$ |
| Direct (UMFPACK) | $N^{0.00}$ | $N^{1.40}$ | $N^{1.26}$ | $N^{0.93}$ |

**Only multigrid has a mesh-independent iteration count.** Jacobi needs 15
iterations at 144 unknowns and 673 at 147,456 — the count doubles with every
refinement, $N^{0.50}$, which is the classical $\mathcal{O}(h^{-1})$ growth of an
unpreconditioned Krylov method, and by 589,824 unknowns it has stopped
converging altogether. Multigrid's count moves from 12 to 15 across the entire
range, $N^{0.02}$: it is not that multigrid's iterations are cheaper, it is that
there are always about the same number of them. That single property is why the
method is popular.

**The consequence shows up in the time exponent.** Because Jacobi's iteration
count grows while each iteration costs $\mathcal{O}(N)$, its linear algebra
grows like $N^{1.26}$; SSOR is worse still at $N^{1.40}$. Multigrid grows like
$N^{0.90}$. At 589,824 unknowns that is the difference between 62 s and 693 s,
and the gap widens without bound — the exponents, not the constants, are what a
coarser or finer mesh will change.

**Memory is where the direct solver loses, and it is invisible in the matrix.**
Column 5 counts the matrices, vectors and DoF structures the solver holds, and
there the direct solver looks *economical*: 203.5 MB, the active matrix and
nothing else, exactly what Jacobi and SSOR hold. Column 6 measures the process
instead, and the same method needs **3347.3 MB** — 16.4 times the matrix it was
handed. That gap is the fill-in and its internal index arrays, allocated inside
UMFPACK where no amount of algebraic bookkeeping can see it, and it grows like
$N^{0.93}$ against multigrid's $N^{0.74}$. The consequence is not academic one
refinement level further on: 589,824 unknowns is *the largest problem this study
could run at all*. Over the last refinement the direct solver's memory grew by a
factor of 4.0 for a factor of 4 in unknowns, so the next level projects to well
over 12 GB against the 8 GB available to Docker here. Multigrid's stored
operators, which grow exactly like $N$, project to 1.8 GB for the same step.
Under a 3-D or higher-order discretisation the direct solver's ceiling arrives
several levels sooner, and that — not speed — is the reason it stops being an
option.

Multigrid does use more memory than the simple preconditioners — 443.5 MB
against 203.5 MB, because it stores an operator for every level — and that is an
honest cost of the method, visible in the third panel of
[`bench/scaling.pdf`](bench/scaling.pdf). It is a bounded factor, roughly
$4/3$ of the fine-grid matrix in two dimensions, not a growth rate: the two
multigrid rows sit on a line parallel to the other three, above them by a
constant.

### 7.3 Multigrid as a solver versus multigrid as a preconditioner

It is worth comparing these two, and they must be compared on the *same*
V-cycle — which is how they are implemented here, both going through the same
`prepare_multigrid()`. The comparison isolates one thing: what Krylov
acceleration buys.

| Method | V-cycles at $N = 589{,}824$ | Linear algebra |
| --- | ---: | ---: |
| MG standalone (defect correction) | 31 | 114.3 s |
| CG + MG | 15 | 62.1 s |

Both are mesh-independent — $N^{0.00}$ and $N^{0.02}$ — so **standalone
multigrid already delivers the property that makes multigrid famous**; it does
not need CG to become an optimal method. What CG adds is a factor of about two:
it needs 15 applications of the V-cycle where plain defect correction needs 31,
and the ratio holds at every problem size in the sweep.
That is the expected result and worth stating plainly, because defect correction
applies a fixed-point iteration, converging at the rate $\rho$ of the V-cycle,
while CG minimises over a growing Krylov subspace and converges at a rate
governed by $\sqrt{\kappa}$ instead. CG also buys robustness: it guarantees
monotone decrease of the energy-norm error, which a bare V-cycle iteration does
not, and it degrades gracefully if the smoother or coarse solve is imperfect.

So the honest summary is that CG + MG is roughly twice as fast as MG alone, not
asymptotically better. If a study reports only one of the two, this is the
relationship that is being left out.

### 7.4 A caveat on the timings

These timings were taken inside an `arm64` image emulated on an `x86_64` host
(see [§4](#4-building)), which inflates every wall-clock number and does not
inflate all instruction mixes equally. **The iteration counts, the residuals,
the discretisation errors and the memory in bytes are architecture-independent
and exact.** The time exponents are ratios and largely survive the distortion,
but the absolute seconds should not be quoted; rebuild natively, or run on an
`arm64` host, for numbers fit for publication.

## 8. Results

The table below is the output of a cubic ($Q_3$) discretisation over four
refinement cycles,

```bash
docker run --rm laplace-beltrami 3 4
```

which prints, for every cycle, the mesh size, the number of degrees of freedom,
the number of CG iterations, and the error against the exact solution:

| Cycle | Cells | Levels | DoFs | CG iterations | $H^1$ error | $L^2$ error | $L^\infty$ error |
| ----: | ----: | -----: | ---: | ------------: | ----------: | ----------: | ---------------: |
| 0 | 16 | 1 | 144 | 1 | 5.856e-02 | 1.617e-02 | 5.417e-03 |
| 1 | 64 | 2 | 576 | 12 | 8.054e-03 | 1.114e-03 | 4.698e-04 |
| 2 | 256 | 3 | 2304 | 13 | 1.035e-03 | 7.225e-05 | 3.394e-05 |
| 3 | 1024 | 4 | 9216 | 14 | 1.302e-04 | 4.567e-06 | 2.175e-06 |

Two things are worth reading off it.

**The discretisation converges at the expected rates.** Each refinement halves
the mesh size, and the error is divided by

| Refinement | $H^1$ factor | observed order | $L^2$ factor | observed order |
| ---------- | -----------: | -------------: | -----------: | -------------: |
| 0 → 1 | 7.27 | 2.86 | 14.5 | 3.86 |
| 1 → 2 | 7.79 | 2.96 | 15.4 | 3.95 |
| 2 → 3 | 7.94 | 2.99 | 15.8 | 3.98 |

so the $H^1$ error converges at close to third order and the $L^2$ error at
close to fourth order — the optimal rates $\mathcal{O}(h^p)$ and
$\mathcal{O}(h^{p+1})$ for cubic elements. Since $u$ is smooth, the
discretisation error is dominated by the $h$-dependence rather than by
regularity, and the rates are clean.

**The multigrid preconditioner is doing its job.** The CG iteration count barely
moves — 12, 13, 14 — while the mesh is refined three times (16 → 1024 cells) and
the system grows sixty-fourfold, from 144 to 9216 unknowns. Iterations needed to
reach a relative tolerance of $10^{-10}$ are essentially independent of the mesh
size, which is the property a geometric multigrid preconditioner is supposed to
have. (Cycle 0 has only a single level, so its coarse solve is exact and CG
terminates in one step.)

---

## 9. Code layout

Everything lives in `src/main.cpp`, in namespace `LaplaceBeltrami`, and is
organised in numbered sections:

1. **The problem** — torus geometry (`TorusGeometry`): the angle coordinates,
   the scale factors and $\kappa$.
2. **Exact solution** — `ExactSolution<3>`, including its surface gradient.
3. **Right-hand side** — `RightHandSide<3>`, the analytic $f$.
4. **Scratch and copy data** — `ScratchData` and `CopyData` for the mesh loop.
5. **The problem class** — `LaplaceBeltramiProblem<dim, spacedim>`, plus the
   solver selector (`Method`) and the structures the measurements are
   reported in (`SolveStats`, `Errors`).
6. **DoFs and matrix structures** — `setup_system()`.
7. **Cell worker** — the local integrals, shared by the active and level assembly.
8. **Assembly** — `assemble_system()` and `assemble_multigrid()`.
9. **Solution** — `solve()` and one function per method: `solve_cg_jacobi`,
   `solve_cg_ssor`, `solve_cg_mg`, `solve_mg_standalone` and `solve_direct`,
   with `prepare_multigrid()` building the V-cycle they share.
10. **Error** — `compute_error()`.
11. **Output** — `output_results()`, including the matrix and vector dumps.
12. **Measurement** — `algebraic_memory_mb()`, the peak-RSS reading, and the
    CSV writer.
13. **Driver** — `run()`, and `main()` at the bottom.

### 9.1 The learned smoother

The C++ solver exports its per-level operators (`export_level_data`); the Python
side learns a residual-to-correction map from them.

| file | what it is |
|---|---|
| `deeponet_smoother.py` | the DeepONet smoother: training, evaluation, classical baselines and the V-cycle. Documented in `DEEPONET_SMOOTHER.md` |
| **`stage1_smoothing_property.py`** | **Stage I** — trains on the Galerkin coarse-space *complement* only and measures whether the learned step is selective. Documented in `STAGE1_SMOOTHING_PROPERTY.md` |
| `tools/trunk_reach.py` | how far into the complement a given trunk can reach — the floor on $\mu_F$ that no branch can beat |
| `tools/branch_rank_probe.py` | the same bound for the *branch*: the best rank-$k$ linear map on the same data |
| `tools/convert_level_data.py` | turns the solver's `.coo`/`.txt` dumps into the `.npz`/`.npy` the trainers read |

```bash
python stage1_smoothing_property.py --selftest          # projector + smoother checks
python stage1_smoothing_property.py --data-dir level_data/L3 --out-dir results/stage1_L3
```

---

## 10. Reference

The surface finite element method and the surrounding deal.II machinery are
described in:

- deal.II tutorial **step-38** — *Solving PDEs on curved surfaces*
- deal.II documentation on [multigrid](https://www.dealii.org/current/doxygen/deal.II/group__mg.html)

---

## 11. License

`src/main.cpp` carries the SPDX identifier `LGPL-2.1-or-later`, inherited from
deal.II (which is licensed under the same terms) and from step-38, of which this
program is a derivative. The full license text is in [`LICENSE`](LICENSE).
