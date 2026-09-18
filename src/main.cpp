/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Surface (Laplace-Beltrami) problem on an imported vessel surface,
 * solved with continuous Galerkin (CG) finite elements.
 *
 * Reference: deal.II step-38
 *
 * The earlier revision of this file solved the same equation on a torus
 * built by GridGenerator::torus().  It now reads the vessel triangulation
 * from an ASCII VTK file and derives its boundary conditions from the
 * region labels that the mesh carries.  See sections 1 and 4 below.
 *
 * ------------------------------------------------------------------------ */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>


namespace VesselSurface
{
  using namespace dealii;


  /* ==================================================================
   *
   * 1. The problem
   *
   * We solve the surface (Laplace-Beltrami) problem
   *
   *     -div_g(kappa grad_g u) + sigma u = f
   *
   * on the vessel surface, where grad_g and div_g are the gradient and
   * the divergence induced by the surface metric.
   *
   * ------------------------------------------------------------------
   * ASSUMPTIONS - read this before trusting the numbers
   * ------------------------------------------------------------------
   *
   * The task specification fixed the discretisation (continuous
   * Galerkin) and the Dirichlet data (zero on the labelled openings),
   * but left the governing equation and the remaining data open.  The
   * choices made here are:
   *
   *   kappa = 1   constant, isotropic surface diffusivity.  The torus
   *               revision used a manufactured kappa(xi, eta) whose only
   *               purpose was to give the exact solution something to
   *               exercise; on a measured vessel geometry there is no
   *               such manufactured coefficient, and no measured
   *               diffusivity field was supplied, so the coefficient is
   *               one.
   *
   *   sigma = 1   reaction term, as in the torus revision.  It makes the
   *               bilinear form coercive, so the discrete problem stays
   *               well posed no matter how the labelled openings are
   *               chosen.
   *
   *   f     = 1   constant source.  This one matters: with the Dirichlet
   *               data fixed at zero, f = 0 would make u = 0 the exact
   *               solution and there would be nothing to look at.  A
   *               nonzero source is what makes the solve non-trivial.
   *
   * If the intended problem is instead the *driven* one - Laplace's
   * equation on the vessel wall with u = 1 on the inlet opening and
   * u = 0 on the outlets - then kappa and sigma stay as they are, f
   * becomes 0, and only the two lines that set the Dirichlet values in
   * compute_dirichlet_dofs() need to change.  That is a change of data,
   * not of the discretisation.
   *
   * ================================================================== */

  // Coefficient of the second-order term.
  constexpr double kappa_coefficient = 1.0;

  // Coefficient of the zero-order term.
  constexpr double reaction_coefficient = 1.0;

  // The source term f.
  constexpr double source_value = 1.0;


  /* ==================================================================
   *
   * 2. The mesh
   *
   * The vessel is read from an ASCII VTK file, by deal.II's own VTK
   * reader.  The file is produced from the binary POLYDATA export by
   *
   *     python tools/vtk_binary_to_ascii.py vessel.vtk meshes/vessel_ascii.vtk
   *
   * which rewrites it as an ASCII UNSTRUCTURED_GRID (the only layout
   * GridIn::read_vtk accepts) and - this is the part that matters -
   * carries the two per-cell label arrays of the original export over
   * into the two integer arrays the reader knows about:
   *
   *     CapID       -> MaterialID   read back as cell->material_id()
   *     ModelFaceID -> ManifoldID   read back as cell->manifold_id()
   *
   * The mesh is a closed, consistently oriented, watertight surface:
   * every edge is shared by exactly two triangles and the Euler
   * characteristic is 2.  The two label arrays say what the triangles
   * are:
   *
   *     CapID = 0    167750 triangles   the vessel wall
   *     CapID = 1       744 triangles   a flat, planar cut (one opening)
   *     CapID = 2      4558 triangles   the remaining cut faces
   *
   *     ModelFaceID = 1   the wall, and 2 / 3 / 5 / 6 / 7 the individual
   *                       cut faces that make up the openings
   *
   * ================================================================== */

  /* ==================================================================
   *
   * 3. Which labels mean what
   *
   * The wall label is the one region that is *not* an opening.  Every
   * cell carrying another label lies on a cut face, i.e. on an opening
   * of the vessel.
   * ================================================================== */

  constexpr types::material_id wall_material_id = 0;


  /* ==================================================================
   *
   * 4. Boundary conditions
   *
   * This is the part that differs most from the torus revision, so it
   * is worth being explicit about.
   *
   * The mesh is a *closed* surface.  In deal.II a boundary face is a
   * face that belongs to one cell only; on a closed surface there are
   * none, so triangulation.n_boundary_faces() == 0 and there is no
   * boundary id to attach a condition to.  Boundary conditions here are
   * therefore imposed on *regions of the surface*, identified by the
   * labels the mesh carries - which is exactly what those labels are
   * for.
   *
   *     openings (CapID 1 and 2, i.e. material_id != wall_material_id)
   *         Dirichlet:  u = 0
   *
   *     wall (CapID 0)
   *         natural (Neumann):  d u / d n = 0, i.e. nothing is imposed
   *         and the term simply drops out of the weak form
   *
   * The Dirichlet condition is realised by constraining every degree of
   * freedom whose shape function has support on an opening cell.  For
   * the P1 element used here the degrees of freedom sit on vertices, so
   * this pins all vertices of the cut faces, including the rim where
   * they meet the wall.
   *
   * Note the consequence: the cut faces stay part of the domain and
   * keep contributing to the equations of the rim vertices.  The
   * discrete problem is therefore "the surface PDE on the closed vessel
   * with u prescribed on the openings", which is well posed (the
   * bilinear form is coercive thanks to sigma > 0 and the Dirichlet set
   * is non-empty) and is the natural reading of a labelled, closed
   * surface mesh.  Deleting the opening cells instead would give the
   * open-wall problem with Dirichlet data on the rims - a different,
   * also valid, problem.
   *
   * ================================================================== */


  /* ==================================================================
   *
   * 5. Scratch and copy data for the mesh loop
   *
   * ================================================================== */

  template <int dim, int spacedim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim, spacedim>       &mapping,
                const FiniteElement<dim, spacedim> &fe,
                const unsigned int                  quadrature_degree,
                const UpdateFlags                   update_flags)
      : fe_values(mapping,
                  fe,
                  QGaussSimplex<dim>(quadrature_degree),
                  update_flags)
    {}

    ScratchData(const ScratchData<dim, spacedim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
    {}

    FEValues<dim, spacedim> fe_values;
  };


  // Per-cell data handed from the cell worker to the copier.
  struct CopyData
  {
    unsigned int level;

    FullMatrix<double> cell_matrix;
    Vector<double>     cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;


    template <class Iterator>
    void reinit(const Iterator &cell, const unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_active_or_mg_dof_indices(local_dof_indices);

      level = cell->level();
    }
  };


  /* ==================================================================
   *
   * 6. The problem class
   *
   * ================================================================== */

  // Which preconditioner the outer CG iteration runs with.
  // Change this constant to switch between the available choices.
  //
  // The geometric multigrid V-cycle does not apply to an imported mesh as it
  // stands - it needs more than one level, and its coarse solve factorises
  // the coarsest level densely, which on an imported mesh means factorising
  // the whole problem.  solve() checks both and falls back to SSOR with an
  // explanation; see the note there and MESH_AND_BC.md section 7.
  enum class CgPreconditioner
  {
    jacobi,
    ssor,
    multigrid
  };

  constexpr CgPreconditioner cg_preconditioner = CgPreconditioner::ssor;


  template <int dim, int spacedim>
  class VesselSurfaceProblem
  {
  public:
    VesselSurfaceProblem(const std::string &mesh_file,
                         const unsigned int degree,
                         const unsigned int n_refinement_cycles);

    void run();

  private:
    // -- mesh --------------------------------------------------------
    void read_mesh();

    // -- boundary conditions -----------------------------------------
    std::vector<bool> dirichlet_dofs_active() const;
    std::vector<bool> dirichlet_dofs_on_level(const unsigned int level) const;

    void describe_mesh() const;

    // -- assembly ----------------------------------------------------
    template <class Iterator>
    void cell_worker(const Iterator             &cell,
                     ScratchData<dim, spacedim> &scratch_data,
                     CopyData                   &copy_data);

    void setup_system();
    void assemble_system();
    void assemble_multigrid();

    // -- solution ----------------------------------------------------
    void solve();

    void solve_with_jacobi(SolverCG<Vector<double>> &solver);
    void solve_with_ssor(SolverCG<Vector<double>> &solver);
    void solve_with_multigrid(SolverCG<Vector<double>> &solver);

    // -- post-processing ---------------------------------------------
    void verify_solution() const;
    void output_results(const unsigned int cycle) const;

    // -- data --------------------------------------------------------
    const std::string mesh_file;

    Triangulation<dim, spacedim> triangulation;

    // The mesh is a piecewise linear triangle surface, so both the element
    // and the geometry need the simplex reference cell: FE_SimplexP and a
    // P1 MappingFE, which reproduces the flat triangles exactly.  Handing a
    // hypercube element (FE_Q) or a MappingQ to a triangle mesh is not
    // diagnosed by distribute_dofs() - see the check in run().
    const FE_SimplexP<dim, spacedim> geometry_fe;

    const FE_SimplexP<dim, spacedim> fe;
    DoFHandler<dim, spacedim>        dof_handler;
    const MappingFE<dim, spacedim>   mapping;

    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;
    AffineConstraints<double> constraints;

    Vector<double> solution;
    Vector<double> system_rhs;

    const unsigned int degree;
    const unsigned int n_refinement_cycles;

    // Geometric multigrid hierarchy: one operator and one interface
    // operator per level.
    MGLevelObject<SparsityPattern> mg_sparsity_patterns;
    MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;

    MGLevelObject<SparseMatrix<double>> mg_matrices;
    MGLevelObject<SparseMatrix<double>> mg_interface_matrices;

    MGConstrainedDoFs mg_constrained_dofs;
  };


  template <int dim, int spacedim>
  VesselSurfaceProblem<dim, spacedim>::VesselSurfaceProblem(
    const std::string &mesh_file,
    const unsigned int degree,
    const unsigned int n_refinement_cycles)
    : mesh_file(mesh_file)
    , triangulation(
        Triangulation<dim, spacedim>::limit_level_difference_at_vertices)
    , geometry_fe(1)
    , fe(degree)
    , dof_handler(triangulation)
    , mapping(geometry_fe)
    , degree(degree)
    , n_refinement_cycles(n_refinement_cycles)
  {}


  /* ==================================================================
   *
   * 7. Reading the mesh
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::read_mesh()
  {
    std::ifstream input(mesh_file);
    AssertThrow(input,
                ExcMessage("Could not open the mesh file <" + mesh_file + ">"));

    GridIn<dim, spacedim> grid_in;
    grid_in.attach_triangulation(triangulation);
    grid_in.read_vtk(input);

    // The file carries manifold ids (ModelFaceID).  Register a flat
    // manifold for every id that occurs: a flat manifold subdivides a
    // triangle linearly, which is what we want for a mesh whose geometry
    // is a piecewise linear approximation to begin with, and it keeps
    // Triangulation::get_manifold() from being asked for an id that was
    // never registered.
    types::manifold_id max_manifold_id = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
      max_manifold_id = std::max(max_manifold_id, cell->manifold_id());

    for (types::manifold_id id = 0; id <= max_manifold_id; ++id)
      triangulation.set_manifold(id, FlatManifold<dim, spacedim>());
  }


  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::describe_mesh() const
  {
    std::cout << "   Mesh file: " << mesh_file << std::endl;
    // A boundary face of a codimension-one mesh is a line belonging to a
    // single cell.  On a closed surface there are none - which is the
    // reason the boundary conditions cannot be attached to a boundary id.
    unsigned int n_boundary_faces = 0;
    for (const auto &face : triangulation.active_face_iterators())
      if (face->at_boundary())
        ++n_boundary_faces;

    std::cout << "   Active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Vertices: " << triangulation.n_vertices() << std::endl
              << "   Levels: " << triangulation.n_levels() << std::endl
              << "   Boundary faces: " << n_boundary_faces << "  (closed surface)"
              << std::endl;

    double area = 0.0;
    for (const auto &cell : triangulation.active_cell_iterators())
      area += cell->measure();

    std::cout << "   Surface area: " << area << std::endl;

    // Report what the labels say.  Sorted maps keep the output stable.
    std::map<types::material_id, unsigned int> by_material;
    std::map<types::manifold_id, unsigned int> by_manifold;

    for (const auto &cell : triangulation.active_cell_iterators())
      {
        ++by_material[cell->material_id()];
        ++by_manifold[cell->manifold_id()];
      }

    std::cout << "   Labels - MaterialID (CapID): ";
    for (const auto &[id, n] : by_material)
      std::cout << id << ":" << n
                << (id == wall_material_id ? " (wall) " : " (opening) ");
    std::cout << std::endl;

    std::cout << "   Labels - ManifoldID (ModelFaceID): ";
    for (const auto &[id, n] : by_manifold)
      std::cout << id << ":" << n << " ";
    std::cout << std::endl;
  }


  /* ==================================================================
   *
   * 8. Which degrees of freedom are Dirichlet degrees of freedom
   *
   * A degree of freedom is constrained when at least one of the cells it
   * belongs to is labelled as an opening.  The active and the level
   * numbering are asked separately, because the multigrid hierarchy
   * needs the same information level by level.
   *
   * The Dirichlet set is nested: refining a cell that is labelled as an
   * opening produces children with the same label, so every coarse
   * degree of freedom a fine Dirichlet degree of freedom interpolates
   * from is itself a Dirichlet degree of freedom.  That is what makes a
   * plain MGTransferPrebuilt correct here - prolongation cannot leak a
   * nonzero value onto the Dirichlet set.
   *
   * ================================================================== */

  template <int dim, int spacedim>
  std::vector<bool>
  VesselSurfaceProblem<dim, spacedim>::dirichlet_dofs_active() const
  {
    std::vector<bool> selected(dof_handler.n_dofs(), false);

    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->material_id() != wall_material_id)
        {
          cell->get_dof_indices(local_dof_indices);
          for (const types::global_dof_index index : local_dof_indices)
            selected[index] = true;
        }

    return selected;
  }


  template <int dim, int spacedim>
  std::vector<bool> VesselSurfaceProblem<dim, spacedim>::dirichlet_dofs_on_level(
    const unsigned int level) const
  {
    std::vector<bool> selected(dof_handler.n_dofs(level), false);

    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.cell_iterators_on_level(level))
      if (cell->material_id() != wall_material_id)
        {
          cell->get_mg_dof_indices(local_dof_indices);
          for (const types::global_dof_index index : local_dof_indices)
            selected[index] = true;
        }

    return selected;
  }


  /* ==================================================================
   *
   * 9. Degrees of freedom and matrix structures
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::setup_system()
  {
    // -- degrees of freedom ------------------------------------------
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    const unsigned int n_levels = triangulation.n_levels();

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    std::cout << "   DoFs by level: ";
    for (unsigned int level = 0; level < n_levels; ++level)
      {
        std::cout << dof_handler.n_dofs(level);
        if (level + 1 < n_levels)
          std::cout << ", ";
      }
    std::cout << std::endl;


    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());


    // -- constraints on the active system ----------------------------
    //
    // Hanging nodes first.  Refinement here is always uniform, so this
    // set is empty in practice; it is kept so that the code stays
    // correct if adaptive refinement is ever switched on.  (Should that
    // happen, a degree of freedom must not land in both sets, so the
    // hanging-node constraints would have to be built from a
    // Dirichlet-free submesh.)
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Then the boundary conditions, from the mesh labels: see section 4.
    const std::vector<bool> dirichlet = dirichlet_dofs_active();
    constraints.add_lines(dirichlet);
    constraints.close();

    const std::size_t n_dirichlet =
      std::count(dirichlet.begin(), dirichlet.end(), true);
    const double dirichlet_fraction =
      100.0 * static_cast<double>(n_dirichlet) /
      static_cast<double>(dof_handler.n_dofs());

    std::cout << "   Dirichlet DoFs (openings, u = 0): " << n_dirichlet << " ("
              << std::setprecision(3) << dirichlet_fraction << "%)"
              << std::setprecision(6) << std::endl;
    std::cout << "   Unconstrained DoFs (wall, natural BC): "
              << dof_handler.n_dofs() - n_dirichlet << std::endl;


    // -- sparsity pattern of the active system -----------------------
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);


    // -- constraints on the multigrid levels -------------------------
    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);


    // -- one matrix and one sparsity pattern per level ---------------
    mg_matrices.resize(0, n_levels - 1);
    mg_interface_matrices.resize(0, n_levels - 1);
    mg_sparsity_patterns.resize(0, n_levels - 1);
    mg_interface_sparsity_patterns.resize(0, n_levels - 1);

    for (unsigned int level = 0; level < n_levels; ++level)
      {
        // The level matrix A_l.
        DynamicSparsityPattern dsp_level(dof_handler.n_dofs(level),
                                         dof_handler.n_dofs(level));

        MGTools::make_sparsity_pattern(dof_handler, dsp_level, level);

        mg_sparsity_patterns[level].copy_from(dsp_level);
        mg_matrices[level].reinit(mg_sparsity_patterns[level]);


        // The interface matrix, holding the entries that couple degrees
        // of freedom across refinement edges.
        DynamicSparsityPattern dsp_interface(dof_handler.n_dofs(level),
                                             dof_handler.n_dofs(level));

        MGTools::make_interface_sparsity_pattern(dof_handler,
                                                 mg_constrained_dofs,
                                                 dsp_interface,
                                                 level);

        mg_interface_sparsity_patterns[level].copy_from(dsp_interface);
        mg_interface_matrices[level].reinit(
          mg_interface_sparsity_patterns[level]);
      }
  }


  /* ==================================================================
   *
   * 10. Cell worker
   *
   * Computes the local contributions
   *
   *     A_ij = int_K [ kappa grad phi_i . grad phi_j + sigma phi_i phi_j ] dS
   *     f_i  = int_K f phi_i dS
   *
   * of one cell, for the active as well as for the level cells. The
   * integral is over the vessel surface, which is why the gradients of
   * the shape functions and the Jacobian weights come from a mapping
   * that follows the geometry.
   *
   * With kappa constant the surface gradient is all the geometry the
   * second-order term needs: the metric is carried by the mapping, so
   * no explicit scale factors appear, which is the whole point of
   * solving the surface PDE on the surface rather than in a chart.
   *
   * ================================================================== */

  template <int dim, int spacedim>
  template <class Iterator>
  void VesselSurfaceProblem<dim, spacedim>::cell_worker(
    const Iterator             &cell,
    ScratchData<dim, spacedim> &scratch_data,
    CopyData                   &copy_data)
  {
    FEValues<dim, spacedim> &fe_values = scratch_data.fe_values;

    fe_values.reinit(cell);

    const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();

    copy_data.reinit(cell, dofs_per_cell);

    const std::vector<double> &JxW = fe_values.get_JxW_values();


    // -- quadrature loop ---------------------------------------------
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                // kappa grad phi_i . grad phi_j + sigma phi_i phi_j
                const double diffusion =
                  kappa_coefficient * (fe_values.shape_grad(i, q) *
                                       fe_values.shape_grad(j, q));

                const double reaction = reaction_coefficient *
                                        fe_values.shape_value(i, q) *
                                        fe_values.shape_value(j, q);

                copy_data.cell_matrix(i, j) +=
                  (diffusion + reaction) * JxW[q];
              }


            // f_i = int_K f phi_i dS
            copy_data.cell_rhs(i) +=
              source_value * fe_values.shape_value(i, q) * JxW[q];
          }
      }
  }


  /* ==================================================================
   *
   * 11. Assembly
   *
   * Both the active system and the multigrid hierarchy are assembled
   * from the same cell worker, driven by MeshWorker::mesh_loop. In the
   * first case the loop runs over the active cells, in the second over
   * the level cells.
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::assemble_system()
  {
    system_matrix = 0;
    system_rhs    = 0;

    using ActiveCellIterator =
      typename DoFHandler<dim, spacedim>::active_cell_iterator;

    const auto cell_worker = [this](const ActiveCellIterator     &cell,
                                    ScratchData<dim, spacedim> &scratch_data,
                                    CopyData                   &copy_data) {
      this->cell_worker(cell, scratch_data, copy_data);
    };

    // Scatter the local contributions into the global system. Going
    // through the constraints here means that constrained rows and
    // columns are eliminated while assembling, rather than afterwards:
    // the Dirichlet rows become identity rows with the prescribed value
    // on the right-hand side.
    const auto copier = [this](const CopyData &copy_data) {
      this->constraints.distribute_local_to_global(
        copy_data.cell_matrix,
        copy_data.cell_rhs,
        copy_data.local_dof_indices,
        system_matrix,
        system_rhs);
    };

    ScratchData<dim, spacedim> scratch_data(
      mapping,
      fe,
      2 * degree + 1,
      update_values | update_gradients | update_quadrature_points |
        update_JxW_values);

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          CopyData(),
                          MeshWorker::assemble_own_cells);
  }


  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::assemble_multigrid()
  {
    const unsigned int n_levels = triangulation.n_levels();

    // -- one constraint object per level -----------------------------
    //
    // Two kinds of degree of freedom are eliminated on a level: the ones
    // sitting on refinement edges, which carry no information on their
    // own level, and the Dirichlet ones from section 8, which carry the
    // prescribed data.
    std::vector<AffineConstraints<double>> level_constraints(n_levels);

    for (unsigned int level = 0; level < n_levels; ++level)
      {
        level_constraints[level].reinit(
          dof_handler.locally_owned_mg_dofs(level),
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

        for (const types::global_dof_index dof_index :
             mg_constrained_dofs.get_refinement_edge_indices(level))
          level_constraints[level].constrain_dof_to_zero(dof_index);

        level_constraints[level].add_lines(dirichlet_dofs_on_level(level));

        level_constraints[level].close();
      }


    using LevelCellIterator =
      typename DoFHandler<dim, spacedim>::level_cell_iterator;

    const auto cell_worker = [this](const LevelCellIterator      &cell,
                                    ScratchData<dim, spacedim> &scratch_data,
                                    CopyData                   &copy_data) {
      this->cell_worker(cell, scratch_data, copy_data);
    };


    const auto copier = [this, &level_constraints](const CopyData &copy_data) {
      const unsigned int level = copy_data.level;

      // The level matrix A_l.
      level_constraints[level].distribute_local_to_global(
        copy_data.cell_matrix, copy_data.local_dof_indices, mg_matrices[level]);

      // The interface matrix, which only keeps the entries coupling
      // degrees of freedom across refinement edges.
      const unsigned int dofs_per_cell = copy_data.local_dof_indices.size();

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          if (mg_constrained_dofs.is_interface_matrix_entry(
                level,
                copy_data.local_dof_indices[i],
                copy_data.local_dof_indices[j]))
            mg_interface_matrices[level].add(copy_data.local_dof_indices[i],
                                             copy_data.local_dof_indices[j],
                                             copy_data.cell_matrix(i, j));
    };


    ScratchData<dim, spacedim> scratch_data(
      mapping,
      fe,
      2 * degree + 1,
      update_values | update_gradients | update_quadrature_points |
        update_JxW_values);

    MeshWorker::mesh_loop(dof_handler.begin_mg(),
                          dof_handler.end_mg(),
                          cell_worker,
                          copier,
                          scratch_data,
                          CopyData(),
                          MeshWorker::assemble_own_cells);
  }


  /* ==================================================================
   *
   * 12. Solution
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::solve()
  {
    SolverControl solver_control(10000, 1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);

    solution = 0;

    // Which preconditioner is used is decided by the constant
    // cg_preconditioner at the top of the file, except that the V-cycle has
    // two requirements this problem does not naturally meet.
    //
    // 1. It needs a hierarchy.  Refinement is uniform, so a single level
    //    means there is nothing to coarsen to and the "V-cycle" would
    //    degenerate into an exact solve of the whole system.
    //
    // 2. Its coarse solve, MGCoarseGridHouseholder, factorises the coarsest
    //    level matrix *densely*.  On a generated mesh the coarsest level is
    //    a handful of cells, but on an imported mesh the coarsest level is
    //    the imported mesh itself - and a dense factorisation of 86528
    //    unknowns wants roughly 60 GB, so the process is killed rather than
    //    merely slowed down.  A geometric multigrid for a mesh like this one
    //    needs a genuine coarsening of the imported triangulation, which is
    //    a separate piece of work; until then the V-cycle is refused.
    constexpr types::global_dof_index max_coarse_level_dofs = 5000;

    CgPreconditioner choice = cg_preconditioner;

    if (choice == CgPreconditioner::multigrid)
      {
        if (triangulation.n_levels() < 2)
          {
            std::cout
              << "   Geometric multigrid needs at least two levels, but the "
                 "imported mesh has one.  Falling back to SSOR; pass a "
                 "refinement cycle count > 1 to build a hierarchy."
              << std::endl;
            choice = CgPreconditioner::ssor;
          }
        else if (dof_handler.n_dofs(0) > max_coarse_level_dofs)
          {
            std::cout
              << "   Geometric multigrid declined: the coarsest level holds "
              << dof_handler.n_dofs(0)
              << " unknowns, and the coarse solve factorises it densely.  "
                 "Falling back to SSOR.  A V-cycle for this mesh needs a real "
                 "coarsening of the imported triangulation."
              << std::endl;
            choice = CgPreconditioner::ssor;
          }
      }

    switch (choice)
      {
        case CgPreconditioner::jacobi:
          solve_with_jacobi(solver);
          break;

        case CgPreconditioner::ssor:
          solve_with_ssor(solver);
          break;

        case CgPreconditioner::multigrid:
          solve_with_multigrid(solver);
          break;
      }

    std::cout << "   Number of CG iterations: " << solver_control.last_step()
              << std::endl;
    std::cout << "   Final residual: " << solver_control.last_value()
              << std::endl;

    // Make the constrained degrees of freedom consistent with the values
    // they are pinned to.
    constraints.distribute(solution);
  }


  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::solve_with_jacobi(
    SolverCG<Vector<double>> &solver)
  {
    std::cout << "   CG preconditioner: Jacobi" << std::endl;

    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }


  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::solve_with_ssor(
    SolverCG<Vector<double>> &solver)
  {
    std::cout << "   CG preconditioner: SSOR" << std::endl;

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }


  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::solve_with_multigrid(
    SolverCG<Vector<double>> &solver)
  {
    std::cout << "   CG preconditioner: geometric multigrid" << std::endl;

    // -- transfer between the level and the global numbering ---------
    MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    // -- coarse grid solver ------------------------------------------
    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from(mg_matrices[0]);

    MGCoarseGridHouseholder<double, Vector<double>> coarse_grid_solver;
    coarse_grid_solver.initialize(coarse_matrix);

    // -- smoother ----------------------------------------------------
    using Smoother = PreconditionSOR<SparseMatrix<double>>;

    mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(true);

    // -- level and interface operators -------------------------------
    mg::Matrix<Vector<double>> mg_matrix(mg_matrices);

    mg::Matrix<Vector<double>> mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double>> mg_interface_down(mg_interface_matrices);

    // -- the V-cycle -------------------------------------------------
    Multigrid<Vector<double>> mg(mg_matrix,
                                 coarse_grid_solver,
                                 mg_transfer,
                                 mg_smoother,
                                 mg_smoother);

    mg.set_edge_matrices(mg_interface_down, mg_interface_up);


    PreconditionMG<dim,
                   Vector<double>,
                   MGTransferPrebuilt<Vector<double>>,
                   spacedim>
      mg_preconditioner(dof_handler, mg, mg_transfer);

    // -- outer CG iteration ------------------------------------------
    solver.solve(system_matrix, solution, system_rhs, mg_preconditioner);
  }


  /* ==================================================================
   *
   * 13. Verification
   *
   * There is no exact solution to compare against on a measured vessel,
   * so the solve is checked against properties that hold for any
   * solution of the problem rather than against a formula.
   *
   * (a) The residual of the discrete system must be small.  This is
   *     what the CG tolerance asks for, so it mainly confirms that the
   *     linear algebra and the constraint handling are consistent.
   *
   * (b) A maximum principle holds for
   *
   *         -div_g grad_g u + u = 1,   u = 0 on the Dirichlet set:
   *
   *     at an interior minimum one has -div grad u <= 0, so
   *     u = 1 + div grad u <= 1 there, and at a negative minimum the
   *     left-hand side of the equation would be negative, which it is
   *     not.  Hence 0 <= u <= 1 pointwise.  A discrete solution of a
   *     P1 Galerkin method respects this only up to the obtuseness of
   *     the triangles - the stiffness matrix is an M-matrix when no
   *     angle exceeds 90 degrees - so a small overshoot is possible and
   *     is reported rather than asserted.
   *
   * (c) Integrating the equation over the closed surface kills the
   *     second-order term (the surface has no boundary), leaving
   *
   *         int_S u dS = int_S 1 dS = area(S) = 226.740457.
   *
   *     This is an independent global check on the magnitude of the
   *     solution: a missing factor or a wrong sign would show up as a
   *     large relative difference.  The discrete version only matches up
   *     to the discrete flux through the Dirichlet set, so the agreement
   *     is good rather than exact.
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::verify_solution() const
  {
    // -- (a) residual ------------------------------------------------
    Vector<double> Au(system_rhs.size());
    system_matrix.vmult(Au, solution);

    Vector<double> residual(system_rhs);
    residual -= Au;

    const double relative_residual =
      residual.l2_norm() / system_rhs.l2_norm();

    std::cout << "   ||b - A u|| / ||b||: " << relative_residual << std::endl;


    // -- (b) maximum principle ---------------------------------------
    const double u_min = *std::min_element(solution.begin(), solution.end());
    const double u_max = *std::max_element(solution.begin(), solution.end());
    const double bound = source_value / reaction_coefficient;

    std::cout << "   Solution range: [" << u_min << ", " << u_max
              << "]   (maximum principle: [0, " << bound << "])" << std::endl;

    const double tolerance = 1e-8 + 1e-3 * bound;
    if (u_min < -tolerance || u_max > bound + tolerance)
      std::cout << "   NOTE: the discrete solution leaves the bounds of the "
                   "maximum principle; expected on meshes with obtuse "
                   "triangles."
                << std::endl;


    // -- (c) integral identity ---------------------------------------
    const QGaussSimplex<dim> quadrature(2 * degree + 1);

    FEValues<dim, spacedim> fe_values(mapping,
                                      fe,
                                      quadrature,
                                      update_values | update_JxW_values);

    std::vector<double> values(quadrature.size());

    double integral_of_u = 0.0;
    double area          = 0.0;
    double geometric_area = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, values);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            integral_of_u += values[q] * fe_values.JxW(q);
            area += fe_values.JxW(q);
          }

        geometric_area += cell->measure();
      }

    // The quadrature and the mapping have to agree with the geometry the
    // mesh describes.  If the quadrature were built for the wrong reference
    // cell this ratio would come out at something other than 1 - it is 2 for
    // a hypercube quadrature on a triangle mesh, because a square
    // quadrature's weights sum to 1 where a triangle's sum to 1/2.
    std::cout << "   assembled area / geometric area: " << area / geometric_area
              << "  (must be 1)" << std::endl;

    std::cout << "   int_S u dS = " << std::setprecision(10) << integral_of_u
              << ",  area(S) = " << area << ",  relative difference = "
              << std::abs(integral_of_u - area) / area << std::setprecision(6)
              << std::endl;


    // -- (d) the Dirichlet data is actually in place ------------------
    //
    // Nothing subtle here, but it is the one thing that would be silently
    // wrong if the labels had been mapped onto the wrong id: the pinned
    // degrees of freedom must carry the prescribed value and nothing else.
    const std::vector<bool> dirichlet = dirichlet_dofs_active();

    double max_on_dirichlet = 0.0;
    for (types::global_dof_index i = 0; i < solution.size(); ++i)
      if (dirichlet[i])
        max_on_dirichlet = std::max(max_on_dirichlet, std::abs(solution[i]));

    std::cout << "   max |u| on the Dirichlet set: " << max_on_dirichlet
              << "  (prescribed value is 0)" << std::endl;
  }


  /* ==================================================================
   *
   * 14. Output
   *
   * Besides the solution field, we write out the linear systems that
   * were solved, so that they can be inspected or reused elsewhere:
   * the active system A u = f and every multigrid level operator A_l.
   *
   * ================================================================== */

  namespace
  {
    // Sparse coordinate format:
    //   first line: n_rows n_cols
    //   then:       row column value     (zero-based indices, nonzeros)
    void write_sparse_matrix(const SparseMatrix<double> &matrix,
                             const std::string          &filename)
    {
      std::ofstream out(filename);
      AssertThrow(out, ExcMessage("Could not open " + filename));

      out << std::setprecision(17) << std::scientific;
      out << matrix.m() << " " << matrix.n() << "\n";

      for (unsigned int row = 0; row < matrix.m(); ++row)
        for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
          if (entry->value() != 0.0)
            out << row << " " << entry->column() << " " << entry->value()
                << "\n";
    }


    // Vector format: first line is the dimension, then one value per
    // line.
    void write_vector(const Vector<double> &v, const std::string &filename)
    {
      std::ofstream out(filename);
      AssertThrow(out, ExcMessage("Could not open " + filename));

      out << std::setprecision(17) << std::scientific;
      out << v.size() << "\n";

      for (unsigned int i = 0; i < v.size(); ++i)
        out << v[i] << "\n";
    }
  } // namespace


  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::output_results(
    const unsigned int cycle) const
  {
    const std::string cycle_str = std::to_string(cycle);

    // -- 1. the solution field, for visualisation --------------------
    // The region labels go out with it, so that the openings can be
    // identified in a visualisation without going back to the mesh file.
    Vector<double> material_ids(triangulation.n_active_cells());
    Vector<double> manifold_ids(triangulation.n_active_cells());

    {
      unsigned int i = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          material_ids[i] = cell->material_id();
          manifold_ids[i] = cell->manifold_id();
          ++i;
        }
    }

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(material_ids,
                             "MaterialID",
                             DataOut<dim, spacedim>::type_cell_data);
    data_out.add_data_vector(manifold_ids,
                             "ManifoldID",
                             DataOut<dim, spacedim>::type_cell_data);
    data_out.build_patches(mapping, mapping.get_degree());

    const std::string vtk_name = "vessel-solution-" + cycle_str + ".vtk";
    {
      std::ofstream output(vtk_name);
      data_out.write_vtk(output);
    }
    std::cout << "   Solution written to " << vtk_name << std::endl;


    // -- 2. the active system A u = f --------------------------------
    const std::string global_A = "A-global-cycle-" + cycle_str + ".coo";
    const std::string global_f = "rhs-global-cycle-" + cycle_str + ".txt";
    const std::string global_u = "solution-cycle-" + cycle_str + ".txt";

    write_sparse_matrix(system_matrix, global_A);
    write_vector(system_rhs, global_f);
    write_vector(solution, global_u);


    // -- 3. every multigrid level operator A_l -----------------------
    //
    // No level right-hand side is written: in a V-cycle the coarse grid
    // right-hand side is the restricted residual, not an independently
    // assembled f_l.
    const unsigned int n_levels = triangulation.n_levels();

    for (unsigned int level = 0; level < n_levels; ++level)
      write_sparse_matrix(mg_matrices[level],
                          "A-level-" + std::to_string(level) + "-cycle-" +
                            cycle_str + ".coo");


    // -- 4. a short description of the hierarchy ---------------------
    const std::string info_name = "mg-level-info-cycle-" + cycle_str + ".txt";

    std::ofstream info(info_name);
    AssertThrow(info, ExcMessage("Could not open " + info_name));

    info << "# level dofs matrix_rows stored_entries\n";
    for (unsigned int level = 0; level < n_levels; ++level)
      info << level << " " << dof_handler.n_dofs(level) << " "
           << mg_matrices[level].m() << " "
           << mg_matrices[level].n_nonzero_elements() << "\n";


    std::cout << "   Analysis output: " << global_A << ", " << global_f << ", "
              << global_u << ", A-level-*-cycle-" << cycle << ".coo, "
              << info_name << std::endl;
  }


  /* ==================================================================
   *
   * 15. Driver
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void VesselSurfaceProblem<dim, spacedim>::run()
  {
    // Read the vessel once; from then on it is only refined uniformly.
    read_mesh();

    // The element has to describe the same cell type as the mesh.  This is
    // worth an explicit check because a mismatch is *not* caught by
    // distribute_dofs(): a hypercube element on a triangle mesh silently
    // leaves one entry of every cell's degree-of-freedom array unassigned,
    // and that entry reads back as numbers::invalid_dof_index - which
    // segfaults the moment it is used as a vector index.
    AssertThrow(
      fe.reference_cell() == triangulation.begin_active()->reference_cell(),
      ExcMessage("The finite element uses the " +
                 fe.reference_cell().to_string() +
                 " reference cell but the mesh uses the " +
                 triangulation.begin_active()->reference_cell().to_string() +
                 " reference cell."));

    for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
      {
        std::cout << std::endl
                  << "================================================"
                  << std::endl
                  << "Cycle " << cycle << std::endl
                  << "================================================"
                  << std::endl;


        if (cycle > 0)
          triangulation.refine_global(1);


        describe_mesh();

        setup_system();        // DoFs, constraints, sparsity patterns
        assemble_system();     // the active system A u = f
        assemble_multigrid();  // the level operators of the V-cycle

        solve();
        verify_solution();
        output_results(cycle);
      }
  }

} // namespace VesselSurface


/* ==================================================================
 *
 * Main
 *
 * ================================================================== */

int main(int argc, char *argv[])
{
  try
    {
      using namespace VesselSurface;

      // Accepted forms:
      //   ./solver                                   all defaults
      //   ./solver <degree> <n_refinement_cycles>    default mesh
      //   ./solver <mesh.vtk> <degree> <n_cycles>    everything explicit
      if (argc != 1 && argc != 3 && argc != 4)
        {
          std::cerr
            << "Usage: ./solver [<mesh.vtk>] [<degree>] [<n_refinement_cycles>]"
            << std::endl
            << "Defaults: mesh meshes/vessel_ascii.vtk, degree 1, cycles 1"
            << std::endl;
          return 1;
        }

      const std::string mesh_file =
        (argc == 4) ? argv[1] : "meshes/vessel_ascii.vtk";
      const unsigned int degree =
        (argc >= 3) ? std::stoi(argv[argc - 2]) : 1;
      const unsigned int n_refinement_cycles =
        (argc >= 3) ? std::stoi(argv[argc - 1]) : 1;

      std::cout << "Vessel surface problem, continuous Galerkin (P" << degree
                << " simplex elements)" << std::endl;

      VesselSurfaceProblem<2, 3> vessel_surface(mesh_file,
                                                degree,
                                                n_refinement_cycles);

      vessel_surface.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  return 0;
}
