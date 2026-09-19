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
reference cell.

**Refinement.** The mesh is refined *adaptively*. Each cycle estimates the error
cell by cell with the Kelly estimator, marks a fraction of the cells, and refines
them; the previous solution is interpolated onto the new mesh and used as the
initial guess for the next solve. Setting `refine_fraction` to 1 turns this back
into uniform refinement (see [Results](#7-results)).

The triangulation carries `limit_level_difference_at_vertices`, the flag that
keeps neighbouring cells within one refinement level of one another. It is what
makes a multigrid hierarchy on a non-uniform mesh well defined, and on a uniform
mesh it costs nothing.

---

## 3. The multigrid preconditioner

The outer iteration is `SolverCG` with a relative tolerance of $10^{-10}$ and a
limit of 1000 iterations. Three preconditioners are implemented, and which one is
used is a **run-time** choice:

```bash
./solver --preconditioner=multigrid     # default
./solver --preconditioner=ssor
./solver --preconditioner=jacobi
```

so the three can be compared without recompiling.

The multigrid preconditioner is a V-cycle built on the same finite element
space, restricted to each level:

- **Smoother** — `PreconditionSOR` with `--smoothing-steps` (default 2) symmetric steps per level.
- **Coarse solver** — `MGCoarseGridHouseholder` on the level-0 matrix.
- **Transfer** — `MGTransferPrebuilt`.
- **Interface matrices** — assembled over refinement edges via `is_interface_matrix_entry`, so the V-cycle handles the level interfaces correctly.

Because the level matrices come from the same `cell_worker` as the active
system, the hierarchy is a true geometric multigrid hierarchy for this operator.

**What an adaptive mesh changes, and one trap in it.** On a uniformly refined
mesh the interface matrices are empty and no degree of freedom sits on a
refinement edge, so none of the adaptive machinery is exercised. On an adaptively
refined mesh both become live, and one further thing has to be right:

> **The initial guess must vanish at the hanging nodes.**

The system handed to CG is the *condensed* one. `AffineConstraints::distribute_local_to_global()`
never writes into the row or column of a constrained degree of freedom — it leaves
only an artificial nonzero diagonal there, with a zero right-hand side entry. A
condensed solution vector is therefore zero at those degrees of freedom, and the
initial guess has to be too.

If it is not, the residual is nonzero at the hanging nodes. The multigrid transfer
and the level smoothers assume it vanishes there, so the V-cycle returns garbage
and CG amplifies it until it overflows. Starting from zero satisfies the assumption
for free, which is why the trap only springs once a solution is interpolated
between meshes — and why it shows up as a *divergence*, not a wrong answer. The
code calls `constraints.set_zero(solution)` after the transfer for exactly this
reason.

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

The adaptive refinement adds a second requirement on that build: it uses
`KellyErrorEstimator<2, 3>` and `SolutionTransfer<2, 3, Vector<double>>`, which
the codim build also provides. Both were checked to compile, link and run before
they were relied on.

> **Note.** The published `fanyoung/dealii-codim:9.7.1` tag is a `linux/arm64`
> image. On an x86_64 machine Docker Desktop runs it under emulation: it works,
> but the build and the runs are several times slower than native.

### Option A — Docker (no local deal.II required)

```bash
docker build -t laplace-beltrami .
docker run --rm laplace-beltrami             # the default adaptive run
docker run --rm laplace-beltrami 4 2 4       # degree 4, 2 initial, 4 adaptive cycles
docker run --rm laplace-beltrami 3 0 4 1.0 0.0   # uniform refinement
```

Output files are written into the working directory. To get them out of the
container, mount a host directory and run there:

```bash
mkdir -p out
docker run --rm -v "$PWD/out:/work" -w /work laplace-beltrami 3 2 4
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
./build/solver [degree] [n_initial_refinements] [n_adaptive_cycles] \
               [refine_fraction] [coarsen_fraction]
```

Every argument is optional, and `./build/solver --help` prints the same list.

| Argument | Default | Meaning |
| --- | ---: | --- |
| `degree` | 3 | polynomial degree of `FE_Q` and of the mapping |
| `n_initial_refinements` | 2 | uniform refinements before the adaptive loop |
| `n_adaptive_cycles` | 4 | number of cycles in the adaptive loop |
| `refine_fraction` | 0.3 | fraction of cells marked for refinement per cycle |
| `coarsen_fraction` | 0.0 | fraction of cells marked for coarsening per cycle |

| Option | Effect |
| --- | --- |
| `--preconditioner=NAME` | `jacobi`, `ssor` or `multigrid` (default) |
| `--max-refinement-level=N` | never refine a cell at or above level `N` |
| `--smoothing-steps=N` | multigrid smoothing steps per level (default 2) |
| `--no-transfer` | start every cycle from zero, instead of interpolating the previous solution |
| `--dump-matrices` | also write the `.coo` matrix dumps (off by default) |

`n_initial_refinements` leaves level 0 alone: level 0 is always the generated
16-cell torus, so this sets how many levels the hierarchy starts with without
making the coarse solve any more expensive.

Each cycle prints the mesh size, the degrees of freedom per level, the number of
CG iterations, and the $H^1$, $L^2$ and $L^\infty$ errors against the exact
solution. A convergence table is printed once at the end, with the observed rate
measured against the number of unknowns — on an adaptively refined mesh $h$ is no
longer a single number, so the number of unknowns is the meaningful measure.

**Uniform refinement is the special case `refine_fraction = 1`**, which is how the
baseline table in [Results](#7-results) is reproduced.

All output files are written into the **current working directory**, so run the
program from a scratch directory if you want to keep them separate.

---

## 6. Output files

For a run with `n_adaptive_cycles = N`, each cycle `c = 0 … N-1` produces:

| File | Contents |
| --- | --- |
| `solution-c.vtk` | The computed solution and the cell-wise error indicator, for ParaView or VisIt |
| `mg-level-info-cycle-c.txt` | Per level: number of DoFs, matrix rows and stored nonzeros |
| `A-global-cycle-c.coo` | The active system matrix $A$ — only with `--dump-matrices` |
| `rhs-global-cycle-c.txt` | The active right-hand side $f$ — only with `--dump-matrices` |
| `A-level-l-cycle-c.coo` | The multigrid level matrix $A_l$, for every level $l$ — only with `--dump-matrices` |

The error indicators ride along in the VTK file as a second cell-wise field, so
the pattern that drove the adaptive refinement can be inspected where it was
decided.

The `.coo` matrices and the right-hand side are off by default: an adaptively
refined mesh produces one level matrix per level per cycle, which is a lot of
megabytes. `mg-level-info-cycle-c.txt` is small and is always written, because it
is the summary that says how the adaptive mesh turned into multigrid levels.

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

## 7. Results

### 7.1 Uniform refinement, as the baseline

This is the special case `refine_fraction = 1` with the transfer switched off,
which reproduces the uniform study the program was originally written around:

```bash
./build/solver 3 0 4 1.0 0.0 --no-transfer
```

It prints, for every cycle, the mesh size, the number of degrees of freedom, the
number of CG iterations, and the error against the exact solution:

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

### 7.2 Adaptive refinement

The default run,

```bash
./build/solver            # degree 3, 2 initial refinements, 4 adaptive cycles
```

starts from a 256-cell mesh and refines the 30 % of cells with the largest error
indicator in each cycle:

| Cycle | Cells | Levels | DoFs | CG iterations | $H^1$ error | $L^2$ error | $H^1$ rate |
| ----: | ----: | -----: | ---: | ------------: | ----------: | ----------: | ---------: |
| 0 | 256 | 3 | 2304 | 13 | 1.035e-03 | 7.225e-05 | — |
| 1 | 496 | 4 | 4744 | 9 | 5.172e-04 | 3.069e-05 | 0.96 |
| 2 | 952 | 5 | 9044 | 10 | 1.582e-04 | 6.510e-06 | 1.84 |
| 3 | 1816 | 5 | 17128 | 9 | 7.984e-05 | 2.673e-06 | 1.07 |

The cell counts are not round numbers, because refining a cell forces its
neighbours along to keep the mesh one-irregular — that is
`limit_level_difference_at_vertices` doing its work.

**Mesh independence survives adaptivity.** The number of unknowns grows 7.4-fold
across the adaptive cycles, from 2304 to 17128, while the CG iteration count stays
between 9 and 13. Mesh independence is a property of the *hierarchy*, not of
uniform refinement: on a non-uniform mesh the level interface matrices stop being
empty, and the V-cycle only stays efficient because they are assembled and passed
to `Multigrid::set_edge_matrices()`. Removing that one call is enough to lose the
property.

**But adaptive refinement has little to exploit here.** The manufactured solution
$u = \sin\xi\sin\eta$ is smooth, so its error is spread evenly over the torus and
there is no localised feature to chase. The Kelly estimator therefore marks a
diffuse set of cells and the mesh stays close to uniform, which is why the
observed $H^1$ rates (0.96, 1.84, 1.07) are noisier and no better than the clean
$h$-rates of the uniform baseline — the numbers are measured against a growing
unknown count on a mesh that is not quasi-uniform in a controlled way. Adaptive
refinement earns its keep on solutions with localised structure; on this problem
it is a fair test of the *machinery*, not of the payoff.

---

## 8. Code layout

Everything lives in `src/main.cpp`, in namespace `LaplaceBeltrami`, and is
organised in numbered sections:

1. **Parameters** — the `Parameters` struct and `parse_command_line()`: every
   knob in one place.
2. **The problem** — torus geometry (`TorusGeometry`): the angle coordinates,
   the scale factors and $\kappa$.
3. **Exact solution** — `ExactSolution<3>`, including its surface gradient.
4. **Right-hand side** — `RightHandSide<3>`, the analytic $f$.
5. **Scratch and copy data** — `ScratchData` and `CopyData` for the mesh loop.
6. **The problem class** — `LaplaceBeltramiProblem<dim, spacedim>`.
7. **The mesh** — `make_grid()`, `refine_mesh()` and `describe_mesh()`: the
   adaptivity driver.
8. **DoFs and matrix structures** — `setup_system()`.
9. **Cell worker** — the local integrals, shared by the active and level assembly.
10. **Assembly** — `assemble_system()` and `assemble_multigrid()`.
11. **Error estimation** — `estimate_error()`, the Kelly estimator.
12. **Solution** — `solve()` and the three preconditioned CG variants.
13. **Post-processing** — `compute_error()`, `record_convergence()` and
    `write_convergence_table()`.
14. **Output** — `output_results()`, including the optional matrix dumps.
15. **Driver** — `run()`, and `main()` at the bottom.

The cycle in `run()` is worth reading in order, because the adaptivity is driven
by state that crosses cycle boundaries:

```cpp
make_grid();
for (cycle ...)
  {
    if (cycle > 0) refine_mesh();   // consumes the indicators of the previous cycle
    setup_system();
    if (solution_transfer) { solution_transfer->interpolate(solution); ... }
    assemble_system();
    assemble_multigrid();
    solve();
    estimate_error();               // produces the indicators for the next cycle
    compute_error();
    record_convergence(cycle);
    output_results(cycle);
  }
```

`error_indicators` is written at the end of a cycle and read at the start of the
next one, so it has to be a member rather than a local. `refine_mesh()` runs
before `setup_system()` so that the interpolation happens once the new degrees of
freedom and constraints exist.

---

## 9. Reference

The surface finite element method and the surrounding deal.II machinery are
described in:

- deal.II tutorial **step-38** — *Solving PDEs on curved surfaces*
- deal.II tutorial **step-6** — *Adaptive mesh refinement*, for the Kelly
  estimator and the mark-and-refine loop
- deal.II tutorial **step-16** — *Multigrid on adaptively refined meshes*, for
  the level interface matrices and the symmetric V-cycle
- deal.II documentation on [multigrid](https://www.dealii.org/current/doxygen/deal.II/group__mg.html)

---

## 10. License

`src/main.cpp` carries the SPDX identifier `LGPL-2.1-or-later`, inherited from
deal.II (which is licensed under the same terms) and from step-38, of which this
program is a derivative. The full license text is in [`LICENSE`](LICENSE).
