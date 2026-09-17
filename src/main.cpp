/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Laplace-Beltrami problem on a torus,
 * solved with geometric multigrid preconditioned CG.
 *
 * Reference: deal.II step-38
 *
 * ------------------------------------------------------------------------ */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


namespace LaplaceBeltrami
{
  using namespace dealii;


  /* ==================================================================
   *
   * 1. The problem
   *
   * We solve the surface (Laplace-Beltrami) problem
   *
   *     -div_g(kappa grad_g u) + u = f
   *
   * on the torus, where grad_g and div_g are the gradient and the
   * divergence induced by the surface metric.
   *
   * The torus of major radius R and minor radius r is parametrised by
   * the two angles
   *
   *     xi  : angle around the large circle,
   *     eta : angle around the small circle,
   *
   * with
   *
   *     x = (R + r cos(eta)) cos(xi)
   *     y = r sin(eta)
   *     z = (R + r cos(eta)) sin(xi)
   *
   * and has the metric scale factors
   *
   *     h_xi  = R + r cos(eta),      h_eta = r.
   *
   * The coefficient and the exact solution are
   *
   *     kappa(xi, eta) = 1.1 + sin^2(xi) cos^2(eta)
   *     u(xi, eta)     = sin(xi) sin(eta).
   *
   * ================================================================== */

  namespace TorusGeometry
  {
    // Major and minor radius of the torus.
    constexpr double R = 2.0;
    constexpr double r = 1.0;

    // Offset in kappa(xi, eta) = kappa_0 + sin^2(xi) cos^2(eta).
    constexpr double kappa_0 = 1.1;


    // The two angle coordinates of a point on the torus. Note that they
    // are only defined modulo 2*pi: the torus is closed, and the branch
    // cut at eta = +-pi is a coordinate artifact, not a boundary.
    struct Angles
    {
      double xi;
      double eta;
    };


    // Map a Cartesian point on the torus to its angle coordinates.
    inline Angles angles_of(const Point<3> &p)
    {
      const double rho = std::sqrt(p[0] * p[0] + p[2] * p[2]);

      Angles angles;
      angles.xi  = std::atan2(p[2], p[0]);
      angles.eta = std::atan2(p[1], rho - R);

      return angles;
    }


    // Metric scale factor in the xi direction. The one in the eta
    // direction is the constant r.
    inline double h_xi(const double eta)
    {
      return R + r * std::cos(eta);
    }


    // kappa(xi, eta) = kappa_0 + sin^2(xi) cos^2(eta)
    inline double kappa(const double xi, const double eta)
    {
      const double sin_xi  = std::sin(xi);
      const double cos_eta = std::cos(eta);

      return kappa_0 + sin_xi * sin_xi * cos_eta * cos_eta;
    }
  } // namespace TorusGeometry


  // Coefficient of the zero-order term of the PDE.
  constexpr double reaction_coefficient = 1.0;


  /* ==================================================================
   *
   * 2. Exact solution
   *
   *     u(xi, eta) = sin(xi) sin(eta)
   *
   * Only the case of a two-dimensional surface in three-dimensional
   * space (spacedim == 3) is implemented.
   *
   * ================================================================== */

  template <int spacedim>
  class ExactSolution : public Function<spacedim>
  {
  public:
    ExactSolution()
      : Function<spacedim>(1)
    {}

    virtual double value(const Point<spacedim> &p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1, spacedim>
    gradient(const Point<spacedim> &p,
             const unsigned int component = 0) const override;
  };


  template <>
  double ExactSolution<3>::value(const Point<3> &p, const unsigned int) const
  {
    const TorusGeometry::Angles angles = TorusGeometry::angles_of(p);

    return std::sin(angles.xi) * std::sin(angles.eta);
  }


  template <>
  Tensor<1, 3> ExactSolution<3>::gradient(const Point<3> &p,
                                          const unsigned int) const
  {
    const TorusGeometry::Angles angles = TorusGeometry::angles_of(p);

    const double sin_xi  = std::sin(angles.xi);
    const double cos_xi  = std::cos(angles.xi);
    const double sin_eta = std::sin(angles.eta);
    const double cos_eta = std::cos(angles.eta);

    // Derivatives of u with respect to the angle coordinates.
    const double u_xi  = cos_xi * sin_eta;
    const double u_eta = sin_xi * cos_eta;

    // Orthonormal tangent vectors d/dxi and d/deta, written in
    // Cartesian coordinates.
    Tensor<1, 3> e_xi;
    e_xi[0] = -sin_xi;
    e_xi[1] = 0.0;
    e_xi[2] = cos_xi;

    Tensor<1, 3> e_eta;
    e_eta[0] = -sin_eta * cos_xi;
    e_eta[1] = cos_eta;
    e_eta[2] = -sin_eta * sin_xi;

    // grad_g u = (1/h_xi) du/dxi e_xi + (1/r) du/deta e_eta
    Tensor<1, 3> grad_u;
    for (unsigned int d = 0; d < 3; ++d)
      grad_u[d] = (u_xi / TorusGeometry::h_xi(angles.eta)) * e_xi[d] +
                  (u_eta / TorusGeometry::r) * e_eta[d];

    return grad_u;
  }


  /* ==================================================================
   *
   * 3. Right-hand side
   *
   *     f = -div_g(kappa grad_g u) + u
   *
   * In the angle coordinates, the surface divergence of a field
   * (a_xi, a_eta) reads
   *
   *     div_g(a) = 1/(h_xi r) [ d/dxi (r a_xi) + d/deta (h_xi a_eta) ],
   *
   * which gives
   *
   *     -div_g(kappa grad_g u)
   *
   *       = -1/(h_xi r) [ d/dxi ( (r/h_xi) kappa u_xi )
   *                       + d/deta ( (h_xi/r) kappa u_eta ) ].
   *
   * ================================================================== */

  template <int spacedim>
  class RightHandSide : public Function<spacedim>
  {
  public:
    RightHandSide()
      : Function<spacedim>(1)
    {}

    virtual double value(const Point<spacedim> &p,
                         const unsigned int component = 0) const override;
  };


  template <>
  double RightHandSide<3>::value(const Point<3> &p, const unsigned int) const
  {
    const TorusGeometry::Angles angles = TorusGeometry::angles_of(p);

    const double sin_xi  = std::sin(angles.xi);
    const double cos_xi  = std::cos(angles.xi);
    const double sin_eta = std::sin(angles.eta);
    const double cos_eta = std::cos(angles.eta);

    // Metric scale factor and its eta derivative.
    const double h = TorusGeometry::h_xi(angles.eta);
    const double dh_deta = -TorusGeometry::r * sin_eta;

    // kappa and its angle derivatives.
    const double kappa = TorusGeometry::kappa(angles.xi, angles.eta);

    const double kappa_xi = 2.0 * sin_xi * cos_xi * cos_eta * cos_eta;

    const double kappa_eta = -2.0 * sin_xi * sin_xi * cos_eta * sin_eta;

    // u and its second angle derivatives.
    const double u_xi     = cos_xi * sin_eta;
    const double u_eta    = sin_xi * cos_eta;
    const double u_xixi   = -sin_xi * sin_eta;
    const double u_etaeta = -sin_xi * sin_eta;

    const double xi_part = (kappa_xi * u_xi + kappa * u_xixi) / (h * h);

    const double eta_part = (dh_deta * kappa * u_eta + h * kappa_eta * u_eta +
                             h * kappa * u_etaeta) /
                            (h * TorusGeometry::r * TorusGeometry::r);

    const double minus_divergence = -(xi_part + eta_part);

    const double reaction = sin_xi * sin_eta;

    return minus_divergence + reaction;
  }


  /* ==================================================================
   *
   * 4. Scratch and copy data for the mesh loop
   *
   * ================================================================== */

  template <int dim, int spacedim>
  struct ScratchData
  {
    ScratchData(const MappingQ<dim, spacedim>       &mapping,
                const FiniteElement<dim, spacedim> &fe,
                const unsigned int                  quadrature_degree,
                const UpdateFlags                   update_flags)
      : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
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
   * 5. The problem class
   *
   * ================================================================== */

  // Which preconditioner the outer CG iteration runs with.
  // Change this constant to switch between the available choices.
  enum class CgPreconditioner
  {
    jacobi,
    ssor,
    multigrid
  };

  constexpr CgPreconditioner cg_preconditioner = CgPreconditioner::multigrid;


  template <int dim, int spacedim>
  class LaplaceBeltramiProblem
  {
  public:
    LaplaceBeltramiProblem(const unsigned int degree,
                           const unsigned int n_refinement_cycles);

    void run();

  private:
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
    void compute_error() const;
    void output_results(const unsigned int cycle) const;

    // -- data --------------------------------------------------------
    Triangulation<dim, spacedim> triangulation;

    const FE_Q<dim, spacedim>     fe;
    DoFHandler<dim, spacedim>     dof_handler;
    const MappingQ<dim, spacedim> mapping;

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
  LaplaceBeltramiProblem<dim, spacedim>::LaplaceBeltramiProblem(
    const unsigned int degree,
    const unsigned int n_refinement_cycles)
    : triangulation(
        Triangulation<dim, spacedim>::limit_level_difference_at_vertices)
    , fe(degree)
    , dof_handler(triangulation)
    , mapping(degree)
    , degree(degree)
    , n_refinement_cycles(n_refinement_cycles)
  {}


  /* ==================================================================
   *
   * 6. Degrees of freedom and matrix structures
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::setup_system()
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
    // The torus is closed, so there are no physical boundary conditions
    // to impose. Only hanging nodes from adaptive refinement have to be
    // eliminated, and here refinement is uniform, so in practice this
    // set is empty.
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();


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
   * 7. Cell worker
   *
   * Computes the local contributions
   *
   *     A_ij = int_K [ kappa grad phi_i . grad phi_j + phi_i phi_j ] dS
   *     f_i  = int_K f phi_i dS
   *
   * of one cell, for the active as well as for the level cells. The
   * integral is over the surface of the torus, which is why the
   * gradients of the shape functions and the Jacobian weights come from
   * a mapping that follows the curved geometry.
   *
   * ================================================================== */

  template <int dim, int spacedim>
  template <class Iterator>
  void LaplaceBeltramiProblem<dim, spacedim>::cell_worker(
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


    // -- right-hand side at the quadrature points --------------------
    RightHandSide<spacedim> rhs;
    std::vector<double>     rhs_values(n_q_points);

    rhs.value_list(fe_values.get_quadrature_points(), rhs_values);


    // -- quadrature loop ---------------------------------------------
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const Point<spacedim> &p = fe_values.quadrature_point(q);
        const TorusGeometry::Angles angles = TorusGeometry::angles_of(p);

        const double kappa_value = TorusGeometry::kappa(angles.xi, angles.eta);


        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                // kappa grad phi_i . grad phi_j + phi_i phi_j
                const double diffusion =
                  kappa_value * (fe_values.shape_grad(i, q) *
                                 fe_values.shape_grad(j, q));

                const double reaction = reaction_coefficient *
                                        fe_values.shape_value(i, q) *
                                        fe_values.shape_value(j, q);

                copy_data.cell_matrix(i, j) +=
                  (diffusion + reaction) * JxW[q];
              }


            // f_i = int_K f phi_i dS
            copy_data.cell_rhs(i) +=
              fe_values.shape_value(i, q) * rhs_values[q] * JxW[q];
          }
      }
  }


  /* ==================================================================
   *
   * 8. Assembly
   *
   * Both the active system and the multigrid hierarchy are assembled
   * from the same cell worker, driven by MeshWorker::mesh_loop. In the
   * first case the loop runs over the active cells, in the second over
   * the level cells.
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::assemble_system()
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
    // columns are eliminated while assembling, rather than afterwards.
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
  void LaplaceBeltramiProblem<dim, spacedim>::assemble_multigrid()
  {
    const unsigned int n_levels = triangulation.n_levels();

    // -- one constraint object per level -----------------------------
    //
    // Degrees of freedom sitting on refinement edges are eliminated:
    // they carry no information on their own level.
    std::vector<AffineConstraints<double>> level_constraints(n_levels);

    for (unsigned int level = 0; level < n_levels; ++level)
      {
        level_constraints[level].reinit(
          dof_handler.locally_owned_mg_dofs(level),
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

        for (const types::global_dof_index dof_index :
             mg_constrained_dofs.get_refinement_edge_indices(level))
          level_constraints[level].constrain_dof_to_zero(dof_index);

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
   * 9. Solution
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::solve()
  {
    SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);

    solution = 0;

    // Which preconditioner is used is decided by the constant
    // cg_preconditioner at the top of the file.
    switch (cg_preconditioner)
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

    // Make the constrained degrees of freedom consistent with the ones
    // they are hanging from.
    constraints.distribute(solution);
  }


  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::solve_with_jacobi(
    SolverCG<Vector<double>> &solver)
  {
    std::cout << "   CG preconditioner: Jacobi" << std::endl;

    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }


  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::solve_with_ssor(
    SolverCG<Vector<double>> &solver)
  {
    std::cout << "   CG preconditioner: SSOR" << std::endl;

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }


  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::solve_with_multigrid(
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
   * 10. Error against the exact solution
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::compute_error() const
  {
    const QGauss<dim> quadrature(2 * degree + 1);

    // Integrate the difference between the discrete and the exact
    // solution cell by cell, then reduce it to one global number.
    const auto error_in_norm = [&](const VectorTools::NormType norm) {
      Vector<float> difference_per_cell(triangulation.n_active_cells());

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution<spacedim>(),
                                        difference_per_cell,
                                        quadrature,
                                        norm);

      return VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               norm);
    };

    std::cout << "   H1 error = " << error_in_norm(VectorTools::H1_seminorm)
              << ",  L2 error = " << error_in_norm(VectorTools::L2_norm)
              << ",  Linfty = " << error_in_norm(VectorTools::Linfty_norm)
              << std::endl;
  }


  /* ==================================================================
   *
   * 11. Output
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
  void LaplaceBeltramiProblem<dim, spacedim>::output_results(
    const unsigned int cycle) const
  {
    const std::string cycle_str = std::to_string(cycle);

    // -- 1. the solution field, for visualisation --------------------
    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping, mapping.get_degree());

    const std::string vtk_name = "solution-" + cycle_str + ".vtk";
    {
      std::ofstream output(vtk_name);
      data_out.write_vtk(output);
    }
    std::cout << "   Solution written to " << vtk_name << std::endl;


    // -- 2. the active system A u = f --------------------------------
    const std::string global_A = "A-global-cycle-" + cycle_str + ".coo";
    const std::string global_f = "rhs-global-cycle-" + cycle_str + ".txt";

    write_sparse_matrix(system_matrix, global_A);
    write_vector(system_rhs, global_f);


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
              << "A-level-*-cycle-" << cycle << ".coo, " << info_name
              << std::endl;
  }


  /* ==================================================================
   *
   * 12. Driver
   *
   * ================================================================== */

  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::run()
  {
    // Build the torus once; from then on it is only refined uniformly.
    GridGenerator::torus(triangulation, TorusGeometry::R, TorusGeometry::r);

    // Tell the triangulation that its geometry is a torus, so that
    // vertices created during refinement are placed on the true surface
    // instead of on the surrounding polyhedron.
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0,
                               TorusManifold<dim>(TorusGeometry::R,
                                                  TorusGeometry::r));

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


        std::cout << "   Active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "   Vertices: " << triangulation.n_vertices() << std::endl
                  << "   Levels: " << triangulation.n_levels() << std::endl;


        setup_system();        // DoFs, constraints, sparsity patterns
        assemble_system();     // the active system A u = f
        assemble_multigrid();  // the level operators of the V-cycle

        solve();
        compute_error();
        output_results(cycle);
      }
  }

} // namespace LaplaceBeltrami


/* ==================================================================
 *
 * Main
 *
 * ================================================================== */

int main(int argc, char *argv[])
{
  try
    {
      using namespace LaplaceBeltrami;

      if (argc != 3)
        {
          std::cerr << "Usage: ./solver <degree> <n_refinement_cycles>"
                    << std::endl;
          return 1;
        }

      const unsigned int degree = std::stoi(argv[1]);

      const unsigned int n_refinement_cycles = std::stoi(argv[2]);

      LaplaceBeltramiProblem<2, 3> laplace_beltrami(degree,
                                                    n_refinement_cycles);

      laplace_beltrami.run();
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
