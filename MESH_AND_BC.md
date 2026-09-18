# The vessel mesh: format, labels, and boundary conditions

This note records what had to change to move the solver from the generated
torus to the measured vessel surface, and why.

## 1. `main.cpp` had no mesh-reading code

The revision of `main.cpp` in the repository built its geometry with
`GridGenerator::torus(...)` and a `TorusManifold`. There was no reader of any
kind to reuse, so the reader described in section 3 was written for this task.

## 2. The input file is not something deal.II can read

`vessel.vtk` (19 MB) is a **binary** file in the VTK 5.1 legacy style:

```
# vtk DataFile Version 5.1
vtk output
BINARY
DATASET POLYDATA
POINTS 86528 float
<binary float32 x3>
METADATA
...
POLYGONS 173053 519156
OFFSETS vtktypeint64
<binary int64 x 173053>
CONNECTIVITY vtktypeint64
<binary int64 x 519156>
CELL_DATA 173052
NORMALS Normals float
<binary float32 x3>
FIELD FieldData 8
  GlobalBoundaryCells  BadTriangle  FreeEdge  BooleanRegion
  CapID  ActiveCells  ModelFaceID  GlobalElementID
POINT_DATA 86528
FIELD FieldData 2
  GlobalBoundaryPoints  GlobalNodeID
```

deal.II's `GridIn::read_vtk` accepts exactly one shape of file:

* line 3 must be `ASCII`,
* line 4 must be `DATASET UNSTRUCTURED_GRID`,
* connectivity must be in the plain `CELLS <n> <n*4>` form.

So three things are wrong with the input at once: the binary encoding, the
`POLYDATA` dataset type, and the VTK 5.1 `OFFSETS`/`CONNECTIVITY` connectivity
layout. It cannot be read directly under any option.

The `FIELD` arrays are the interesting part. `CapID` and `ModelFaceID` are
per-cell region labels from the CFD mesher; the others (`BadTriangle`,
`FreeEdge`, `BooleanRegion`, `GlobalBoundaryCells`) are all zero here and carry
no information for this mesh.

## 3. The conversion

```
python tools/vtk_binary_to_ascii.py vessel.vtk meshes/vessel_ascii.vtk
```

writes an ASCII `UNSTRUCTURED_GRID` in the layout the reader wants, and maps
the two meaningful label arrays onto the two integer arrays deal.II knows how
to read (`source/grid/grid_in.cc`, the `CELL_DATA` branch accepts exactly
`MaterialID` and `ManifoldID`, both of type `int`):

| label in `vessel.vtk` | written as | read back in `main.cpp` as |
| --- | --- | --- |
| `CapID` | `SCALARS MaterialID int 1` | `cell->material_id()` |
| `ModelFaceID` | `SCALARS ManifoldID int 1` | `cell->manifold_id()` |

Nothing else is touched. `tools/verify_conversion.py` checks the result against
the binary original and reports:

```
vertices : 86528 vs 86528
  bitwise equal as float32 : True          <- the source is float32, so this is exact
  max |dx|                 : 4.998e-08
triangles: 173052 vs 173052
  identical connectivity   : True
MaterialID: {0: 167750, 1: 744, 2: 4558}
  matches CapID: True
ManifoldID: {1: 167750, 2: 744, 3: 1326, 5: 902, 6: 1069, 7: 1261}
  matches ModelFaceID: True
```

Geometry, connectivity and both label arrays survive without loss. (The output
is 7.5 MB of ASCII where the input was 19 MB of binary.)

### The pre-existing `vessel_geometry_out.vtk` is not usable

That file is a deal.II `GridOut::write_vtk` dump of the same 86528 vertices and
173052 triangles — the geometry and connectivity are right — but its labels
were lost on the way through: `MaterialID` is `0` for every cell and
`ManifoldID` is `-1` for every cell. A mesh with no labels cannot say which
triangles are the vessel wall and which are the inlet and outlet openings, so
it cannot be used to assign boundary conditions. It is why this task needed a
new conversion rather than reusing that file.

## 4. What the labels say, and what the mesh is

The mesh is a **closed, watertight, consistently oriented surface**:

* every one of the 259578 edges belongs to exactly two triangles,
* the Euler characteristic is `86528 - 259578 + 173052 = 2`,
* the winding of every triangle agrees with the `NORMALS` array of the input,
* `triangulation.n_boundary_faces() == 0`.

The labels partition it as:

| label | cells | area | what it is |
| --- | --- | --- | --- |
| `CapID = 0` / `ModelFaceID = 1` | 167750 | 217.09 | the vessel wall |
| `CapID = 1` / `ModelFaceID = 2` | 744 | 4.55 | one opening; **planar** (rms deviation 1e-5) |
| `CapID = 2` / `ModelFaceID = 3` | 1326 | 1.44 | opening, not planar |
| `CapID = 2` / `ModelFaceID = 5` | 902 | 0.36 | opening, not planar |
| `CapID = 2` / `ModelFaceID = 6` | 1069 | 0.62 | opening, not planar |
| `CapID = 2` / `ModelFaceID = 7` | 1261 | 2.68 | opening, not planar |

Total surface area 226.740457 ∈ a bounding box of
4.74 x 9.29 x 22.17 — an elongated vessel with five cut faces at its ends and
one flat planar cut in the middle. `CapID` groups the four curved cuts
together; `ModelFaceID` separates them, so `ModelFaceID` is the finer label if
individual openings ever need different data.

Both groupings are preserved, and since the Dirichlet set below is "every cell
that is not the wall", the two agree on the resulting condition.

## 5. Boundary conditions

Because the surface is closed there is no boundary face and therefore no
boundary id to attach a condition to. Boundary conditions are imposed on
**regions of the surface**, which is exactly what the labels describe:

```
openings  (material_id != 0)   Dirichlet, u = 0
wall      (material_id == 0)   natural (Neumann), du/dn = 0 — nothing imposed
```

The Dirichlet condition is realised by constraining every degree of freedom
whose shape function has support on a cell labelled as an opening; see
`dirichlet_dofs_active()` in `main.cpp`. For the Q1 element used here the
degrees of freedom sit on vertices, so this pins all vertices of the cut faces,
including the rim where they meet the wall.

Two consequences worth being explicit about:

* **The opening cells stay in the domain.** The discrete problem is "the
  surface PDE on the closed vessel, with u prescribed on the openings", which
  is well posed (the bilinear form is coercive because `sigma > 0`, and the
  Dirichlet set is non-empty). Deleting the opening cells instead would give
  the open-wall problem with Dirichlet data on the rims — a different, equally
  valid problem, and the one to switch to if the wall is meant to be the whole
  domain.
* **The Dirichlet set is nested under refinement**, because refining a cell
  labelled as an opening produces children with the same label. Every coarse
  degree of freedom that a fine Dirichlet degree of freedom interpolates from
  is therefore itself a Dirichlet degree of freedom, which is what makes a
  plain `MGTransferPrebuilt` correct here: prolongation cannot leak a nonzero
  value onto the Dirichlet set.

## 6. The mesh is made of triangles, and that constrains the discretisation

The torus revision used `GridGenerator::torus`, which produces a **quad**
mesh, so `FE_Q` and `MappingQ` were the right choices there. This mesh is
173052 **triangles**, and three things have to follow it:

| what | torus (quads) | vessel (triangles) |
| --- | --- | --- |
| element | `FE_Q<2,3>` | `FE_SimplexP<2,3>` |
| geometry | `MappingQ<2,3>` | `MappingFE<2,3>` built on `FE_SimplexP<2,3>(1)` |
| quadrature | `QGauss<2>` | `QGaussSimplex<2>` |

Each of these is a silent failure if it is got wrong, which is worth recording:

* **`FE_Q` on a triangle mesh does not throw.** `distribute_dofs()` happily
  returns 86528 degrees of freedom — one per vertex, which looks right — but
  `n_dofs_per_cell()` is 4 (the quad's four vertices) while a triangle only has
  three. The fourth entry of every cell's degree-of-freedom array is never
  assigned and reads back as `numbers::invalid_dof_index` (4294967295).
  Using that value as a vector index is an out-of-bounds write, and it
  segfaults. This is what the first run of the solver did.
  `MappingQ::is_compatible_with(triangle)` returns `false`, so a mapping can be
  checked; an element cannot, so `main.cpp` now asserts on
  `fe.reference_cell()` against the mesh's.

* **`QGauss<2>` is a quadrature for a square**, whose weights sum to 1; a
  triangle's weights sum to 1/2. Used on a triangle mesh the assembled area
  comes out at **exactly twice** the true area. The solution is unaffected
  (both sides of `Au = b` scale by the same factor, so `u` is unchanged), but
  every integral is wrong by 2. `verify_solution()` now compares the assembled
  area against the geometric area and reports the ratio, which must be 1.

## 7. The geometric multigrid of the torus revision does not carry over

The torus revision used a geometric multigrid V-cycle for the outer CG
iteration, and it was the point of that file. It cannot be used on this mesh
as it stands, for two reasons, and `solve()` now refuses it with an
explanation instead of being killed.

* **A single level is nothing to coarsen to.** Until the imported mesh is
  refined there is one level, so the V-cycle would degenerate into an exact
  solve of the whole system. `solve()` falls back to SSOR and says so.

* **The coarse solve factorises densely.** `MGCoarseGridHouseholder`
  factorises the coarsest level matrix, and on an imported mesh the coarsest
  level *is* the imported mesh. A dense factorisation of 86528 unknowns wants
  about 60 GB; a run with one refinement cycle — so that a genuine two-level
  hierarchy exists — was killed by the OOM killer (exit 137) after starting
  the V-cycle. On the torus this never showed up because
  `GridGenerator::torus` starts from a few hundred cells, and the coarse
  level stays genuinely coarse.

  `solve()` now declines the V-cycle when the coarsest level exceeds 5000
  unknowns. A geometric multigrid for a mesh like this one needs an actual
  coarsening of the imported triangulation, which is separate work.

The Dirichlet handling for a V-cycle is nevertheless in place and correct as
far as it goes: `assemble_multigrid()` constrains the opening degrees of
freedom on every level, and they form a nested hierarchy (refining an opening
cell gives opening children), so prolongation cannot leak a nonzero value onto
the Dirichlet set. That part was not the problem; the coarse solve was.

## 8. Assumptions

The task fixed the discretisation (continuous Galerkin) and the Dirichlet data
(zero on the labelled openings) but left the equation and the remaining data
open. What was chosen, and why:

| quantity | value | reason |
| --- | --- | --- |
| `kappa` | 1 | the torus used a manufactured `kappa(xi,eta)` only to give the exact solution something to exercise; on a measured vessel there is no such coefficient and none was supplied |
| `sigma` | 1 | reaction term, as in the torus revision; makes the form coercive so the problem is well posed for any choice of openings |
| `f` | 1 | with Dirichlet data fixed at zero, `f = 0` would make `u = 0` the exact solution and there would be nothing to solve |

The one thing that would change the answer rather than the presentation is `f`:
the "driven" problem — Laplace's equation on the vessel with `u = 1` on one
opening and `u = 0` on the rest — needs `f = 0` and a per-opening Dirichlet
value, which is a change to `dirichlet_dofs_active()` only, not to the
discretisation.
