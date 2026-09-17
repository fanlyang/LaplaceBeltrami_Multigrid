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

## 3. The multigrid preconditioner

The outer iteration is `SolverCG` with a relative tolerance of $10^{-10}$ and a
limit of 1000 iterations. Three preconditioners are implemented; the active one
is chosen by a **compile-time constant** near the top of `src/main.cpp`:

```cpp
enum class CgPreconditioner { jacobi, ssor, multigrid };

constexpr CgPreconditioner cg_preconditioner = CgPreconditioner::multigrid;
```

Changing that one line switches between Jacobi, SSOR and geometric multigrid.

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
./build/solver <degree> <n_refinement_cycles>
```

For example

```bash
./build/solver 3 5
```

runs a cubic ($Q_3$) discretisation through 5 uniform refinement cycles, printing
for each cycle: the mesh sizes, the number of degrees of freedom, the number of
CG iterations, and the $H^1$, $L^2$ and $L^\infty$ errors against the exact
solution.

All output files are written into the **current working directory**, so run the
program from a scratch directory if you want to keep them separate.

---

## 6. Output files

For a run with `n_refinement_cycles = N`, each cycle `c = 0 … N-1` produces:

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

## 7. Results

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

## 8. Code layout

Everything lives in `src/main.cpp`, in namespace `LaplaceBeltrami`, and is
organised in numbered sections:

1. **The problem** — torus geometry (`TorusGeometry`): the angle coordinates,
   the scale factors and $\kappa$.
2. **Exact solution** — `ExactSolution<3>`, including its surface gradient.
3. **Right-hand side** — `RightHandSide<3>`, the analytic $f$.
4. **Scratch and copy data** — `ScratchData` and `CopyData` for the mesh loop.
5. **The problem class** — `LaplaceBeltramiProblem<dim, spacedim>`, plus the
   preconditioner selector.
6. **DoFs and matrix structures** — `setup_system()`.
7. **Cell worker** — the local integrals, shared by the active and level assembly.
8. **Assembly** — `assemble_system()` and `assemble_multigrid()`.
9. **Solution** — `solve()` and the three preconditioned CG variants.
10. **Error** — `compute_error()`.
11. **Output** — `output_results()`, including the matrix and vector dumps.
12. **Driver** — `run()`, and `main()` at the bottom.

---

## 9. Reference

The surface finite element method and the surrounding deal.II machinery are
described in:

- deal.II tutorial **step-38** — *Solving PDEs on curved surfaces*
- deal.II documentation on [multigrid](https://www.dealii.org/current/doxygen/deal.II/group__mg.html)

---

## 10. License

`src/main.cpp` carries the SPDX identifier `LGPL-2.1-or-later`, inherited from
deal.II (which is licensed under the same terms) and from step-38, of which this
program is a derivative. The full license text is in [`LICENSE`](LICENSE).
