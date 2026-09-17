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
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
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
#include <memory>
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

  // How the linear system is solved. Selected on the command line so that
  // the alternatives can be compared without recompiling:
  //
  //   jacobi     CG preconditioned by Jacobi
  //   ssor       CG preconditioned by SSOR
  //   mg         CG preconditioned by one geometric multigrid V-cycle
  //   mg-solver  the V-cycle used as the solver itself (defect correction)
  //   direct     sparse direct factorisation (UMFPACK)
  enum class Method
  {
    cg_jacobi,
    cg_ssor,
    cg_mg,
    mg_solver,
    direct
  };

  std::string
  method_name(const Method method)
  {
    switch (method)
      {
        case Method::cg_jacobi:
          return "cg-jacobi";
        case Method::cg_ssor:
          return "cg-ssor";
        case Method::cg_mg:
          return "cg-mg";
        case Method::mg_solver:
          return "mg-solver";
        case Method::direct:
          return "direct-umfpack";
      }

    return "unknown";
  }

  Method
  parse_method(const std::string &name)
  {
    if (name == "jacobi")
      return Method::cg_jacobi;
    if (name == "ssor")
      return Method::cg_ssor;
    if (name == "mg")
      return Method::cg_mg;
    if (name == "mg-solver")
      return Method::mg_solver;
    if (name == "direct")
      return Method::direct;

    AssertThrow(false,
                ExcMessage("Unknown method '" + name +
                           "'. Use one of: jacobi, ssor, mg, mg-solver, "
                           "direct."));
    return Method::cg_mg;
  }


  // The discretisation error of one cycle, so that the accuracy reached by
  // each method can be checked to agree.
  struct Errors
  {
    double h1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
  };


  // Everything one cycle of one method reports: one row of the benchmark
  // table. The timings are wall-clock seconds and deliberately split, since
  // the methods differ in where the work sits rather than only in how much
  // of it there is: a direct factorisation does almost all of its work in
  // t_pc_setup and none in t_solve, whereas an iterative method spreads the
  // work the other way round.
  struct SolveStats
  {
    std::string  method;
    unsigned int cycle = 0;

    unsigned int dofs   = 0;
    unsigned int cells  = 0;
    unsigned int levels = 0;

    // CG iterations, V-cycles, or 1 for a direct solve.
    unsigned int iterations = 0;

    // False if the method ran out of its iteration budget before reaching
    // the tolerance. For a method whose iteration count grows with the mesh
    // this is a result rather than an error -- it is the point at which the
    // method stops being usable -- so it is recorded and the run continues
    // instead of aborting and losing the smaller problems too.
    bool converged = true;

    // Measured average residual reduction per step: (r_final/r_0)^(1/n).
    double rate = 0.0;

    // Final relative residual, ||f - A u|| / ||f||.
    double residual = 0.0;

    double t_setup    = 0.0; // DoFs, constraints, sparsity patterns
    double t_assemble = 0.0; // system and level matrices
    double t_pc_setup = 0.0; // preconditioner / hierarchy / factorisation
    double t_solve    = 0.0; // the iteration itself
    double t_error    = 0.0;
    double t_output   = 0.0;

    // Memory held by the deal.II objects we can account for explicitly.
    double algebraic_mb = 0.0;

    // Process peak resident set size. This is the number that also catches
    // memory allocated inside a library, which is where a direct solver's
    // fill-in lives and which algebraic_mb cannot see.
    double peak_rss_mb = 0.0;

    // Peak RSS measured against the RSS at startup, so that the interpreter
    // and runtime overhead every process carries -- hundreds of megabytes
    // for a deal.II binary under emulation -- do not swamp the part that
    // actually depends on the problem. This is the number to plot.
    double rss_growth_mb = 0.0;

    Errors errors;
  };


  template <int dim, int spacedim>
  class LaplaceBeltramiProblem
  {
  public:
    LaplaceBeltramiProblem(const unsigned int degree,
                           const unsigned int n_refinement_cycles,
                           const Method       method,
                           const bool         write_output_files,
                           const std::string &csv_path);

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
    SolveStats solve();

    // One V-cycle on the existing level hierarchy. Built once and shared by
    // the two multigrid-based methods, so that CG+MG and standalone MG are
    // compared on exactly the same operator.
    void prepare_multigrid();

    SolveStats solve_cg_jacobi(double rhs_norm);
    SolveStats solve_cg_ssor(double rhs_norm);
    SolveStats solve_cg_mg(double rhs_norm);
    SolveStats solve_mg_standalone(double rhs_norm);
    SolveStats solve_direct();

    // -- post-processing ---------------------------------------------
    Errors compute_error() const;
    void   output_results(const unsigned int cycle) const;

    // Accounted-for memory of the deal.II objects, in megabytes.
    double algebraic_memory_mb() const;

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

    const Method      method;
    const bool        write_output_files;
    const std::string csv_path;

    // Geometric multigrid hierarchy: one operator and one interface
    // operator per level.
    MGLevelObject<SparsityPattern> mg_sparsity_patterns;
    MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;

    MGLevelObject<SparseMatrix<double>> mg_matrices;
    MGLevelObject<SparseMatrix<double>> mg_interface_matrices;

    MGConstrainedDoFs mg_constrained_dofs;

    // True once assemble_multigrid() has run. The Jacobi, SSOR and direct
    // variants never call it, so nothing may touch the level objects, or
    // account for their memory, unless this is set.
    bool level_operators_built = false;

    // The V-cycle built by prepare_multigrid(). Held as pointers because
    // Multigrid keeps references to the smoother, the coarse solver and the
    // transfer, so all of them have to outlive it.
    using MgSmoother =
      mg::SmootherRelaxation<PreconditionSOR<SparseMatrix<double>>,
                             Vector<double>>;

    std::unique_ptr<MGTransferPrebuilt<Vector<double>>> mg_transfer;
    std::unique_ptr<FullMatrix<double>>                 mg_coarse_matrix;
    std::unique_ptr<MGCoarseGridHouseholder<double, Vector<double>>>
      mg_coarse_solver;
    std::unique_ptr<MgSmoother>                 mg_smoother;
    std::unique_ptr<mg::Matrix<Vector<double>>> mg_matrix;
    std::unique_ptr<mg::Matrix<Vector<double>>> mg_interface_up;
    std::unique_ptr<mg::Matrix<Vector<double>>> mg_interface_down;
    std::unique_ptr<Multigrid<Vector<double>>>  mg;
  };


  template <int dim, int spacedim>
  LaplaceBeltramiProblem<dim, spacedim>::LaplaceBeltramiProblem(
    const unsigned int degree,
    const unsigned int n_refinement_cycles,
    const Method       method,
    const bool         write_output_files,
    const std::string &csv_path)
    : triangulation(
        Triangulation<dim, spacedim>::limit_level_difference_at_vertices)
    , fe(degree)
    , dof_handler(triangulation)
    , mapping(degree)
    , degree(degree)
    , n_refinement_cycles(n_refinement_cycles)
    , method(method)
    , write_output_files(write_output_files)
    , csv_path(csv_path)
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

    // The multigrid level structures are deliberately *not* built here.
    // They are built in assemble_multigrid(), which the Jacobi, SSOR and
    // direct variants skip: a method should only be charged for the
    // machinery it actually uses.
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

    level_operators_built = true;


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
   * Five ways of solving the same linear system, one per entry of the
   * Method enum. Each returns the timings and iteration counts that make
   * up one row of the benchmark table, and leaves `solution` filled in.
   *
   * The split between t_pc_setup and t_solve is the telling one. A Krylov
   * method with a cheap preconditioner puts everything into t_solve.
   * Multigrid puts a noticeable amount into t_pc_setup, because it has to
   * build the level hierarchy before it can invert anything. A direct
   * solver puts almost everything into t_pc_setup and almost nothing into
   * t_solve, which is the whole shape of its cost.
   *
   * ================================================================== */

  template <int dim, int spacedim>
  SolveStats LaplaceBeltramiProblem<dim, spacedim>::solve()
  {
    const double rhs_norm = system_rhs.l2_norm();

    solution = 0;

    SolveStats stats;
    switch (method)
      {
        case Method::cg_jacobi:
          stats = solve_cg_jacobi(rhs_norm);
          break;

        case Method::cg_ssor:
          stats = solve_cg_ssor(rhs_norm);
          break;

        case Method::cg_mg:
          stats = solve_cg_mg(rhs_norm);
          break;

        case Method::mg_solver:
          stats = solve_mg_standalone(rhs_norm);
          break;

        case Method::direct:
          stats = solve_direct();
          break;
      }

    // The residual actually achieved, measured the same way for every
    // method. It is taken before the hanging-node values are filled in,
    // because that is the residual the assembled system and every solver
    // above actually see: the constrained rows are unit rows with a zero
    // right-hand side, so they contribute nothing.
    Vector<double> residual(system_rhs.size());
    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    stats.residual = residual.l2_norm() / rhs_norm;

    // Average residual reduction per step, from the measured final
    // residual. Since every method starts from u = 0, the initial relative
    // residual is 1 by construction.
    if (stats.iterations > 0)
      stats.rate = std::pow(stats.residual, 1.0 / stats.iterations);

    // Make the constrained degrees of freedom consistent with the ones
    // they are hanging from.
    constraints.distribute(solution);

    // The numbers themselves are reported once, by print_stats(), together
    // with the memory and the timings they belong with.
    return stats;
  }


  template <int dim, int spacedim>
  SolveStats LaplaceBeltramiProblem<dim, spacedim>::solve_cg_jacobi(
    const double rhs_norm)
  {
    SolveStats stats;
    stats.method = method_name(Method::cg_jacobi);

    std::cout << "   Method: CG preconditioned by Jacobi" << std::endl;

    SolverControl            solver_control(1000, 1e-10 * rhs_norm);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionJacobi<SparseMatrix<double>> preconditioner;

    Timer timer;
    timer.start();
    preconditioner.initialize(system_matrix);
    stats.t_pc_setup = timer.wall_time();

    timer.restart();
    try
      {
        solver.solve(system_matrix, solution, system_rhs, preconditioner);
      }
    catch (const SolverControl::NoConvergence &)
      {
        // Budget exhausted. Keep the latest iterate: the residual it
        // achieves is reported along with everything else.
        stats.converged = false;
      }
    stats.t_solve = timer.wall_time();

    stats.iterations = solver_control.last_step();

    return stats;
  }


  template <int dim, int spacedim>
  SolveStats LaplaceBeltramiProblem<dim, spacedim>::solve_cg_ssor(
    const double rhs_norm)
  {
    SolveStats stats;
    stats.method = method_name(Method::cg_ssor);

    std::cout << "   Method: CG preconditioned by SSOR" << std::endl;

    SolverControl            solver_control(1000, 1e-10 * rhs_norm);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;

    Timer timer;
    timer.start();
    preconditioner.initialize(system_matrix);
    stats.t_pc_setup = timer.wall_time();

    timer.restart();
    try
      {
        solver.solve(system_matrix, solution, system_rhs, preconditioner);
      }
    catch (const SolverControl::NoConvergence &)
      {
        stats.converged = false;
      }
    stats.t_solve = timer.wall_time();

    stats.iterations = solver_control.last_step();

    return stats;
  }


  template <int dim, int spacedim>
  void LaplaceBeltramiProblem<dim, spacedim>::prepare_multigrid()
  {
    // -- transfer between the level and the global numbering ---------
    mg_transfer = std::make_unique<MGTransferPrebuilt<Vector<double>>>(
      mg_constrained_dofs);
    mg_transfer->build(dof_handler);

    // -- coarse grid solver ------------------------------------------
    mg_coarse_matrix = std::make_unique<FullMatrix<double>>();
    mg_coarse_matrix->copy_from(mg_matrices[0]);

    mg_coarse_solver =
      std::make_unique<MGCoarseGridHouseholder<double, Vector<double>>>();
    mg_coarse_solver->initialize(*mg_coarse_matrix);

    // -- smoother ----------------------------------------------------
    mg_smoother = std::make_unique<MgSmoother>();
    mg_smoother->initialize(mg_matrices);
    mg_smoother->set_steps(2);
    mg_smoother->set_symmetric(true);

    // -- level and interface operators -------------------------------
    mg_matrix = std::make_unique<mg::Matrix<Vector<double>>>(mg_matrices);

    mg_interface_up =
      std::make_unique<mg::Matrix<Vector<double>>>(mg_interface_matrices);
    mg_interface_down =
      std::make_unique<mg::Matrix<Vector<double>>>(mg_interface_matrices);

    // -- the V-cycle -------------------------------------------------
    mg = std::make_unique<Multigrid<Vector<double>>>(*mg_matrix,
                                                     *mg_coarse_solver,
                                                     *mg_transfer,
                                                     *mg_smoother,
                                                     *mg_smoother);

    mg->set_edge_matrices(*mg_interface_down, *mg_interface_up);
  }


  template <int dim, int spacedim>
  SolveStats LaplaceBeltramiProblem<dim, spacedim>::solve_cg_mg(
    const double rhs_norm)
  {
    SolveStats stats;
    stats.method = method_name(Method::cg_mg);

    std::cout << "   Method: CG preconditioned by a multigrid V-cycle"
              << std::endl;

    SolverControl            solver_control(1000, 1e-10 * rhs_norm);
    SolverCG<Vector<double>> solver(solver_control);

    Timer timer;
    timer.start();
    prepare_multigrid();
    stats.t_pc_setup = timer.wall_time();

    PreconditionMG<dim,
                   Vector<double>,
                   MGTransferPrebuilt<Vector<double>>,
                   spacedim>
      mg_preconditioner(dof_handler, *mg, *mg_transfer);

    // -- outer CG iteration ------------------------------------------
    timer.restart();
    try
      {
        solver.solve(system_matrix, solution, system_rhs, mg_preconditioner);
      }
    catch (const SolverControl::NoConvergence &)
      {
        stats.converged = false;
      }
    stats.t_solve = timer.wall_time();

    stats.iterations = solver_control.last_step();

    return stats;
  }


  template <int dim, int spacedim>
  SolveStats LaplaceBeltramiProblem<dim, spacedim>::solve_mg_standalone(
    const double rhs_norm)
  {
    SolveStats stats;
    stats.method = method_name(Method::mg_solver);

    std::cout << "   Method: multigrid as the solver, by defect correction"
              << std::endl;

    Timer timer;
    timer.start();
    prepare_multigrid();
    stats.t_pc_setup = timer.wall_time();

    // The very same operator that CG is handed as a preconditioner. A
    // Multigrid object cannot be applied to a global vector directly -- its
    // cycle() works on its own multilevel vectors -- so the object that
    // carries the level transfer, PreconditionMG, is what applies one
    // V-cycle here. Using it means CG+MG and standalone MG differ only in
    // the outer iteration, which is exactly the comparison being made.
    PreconditionMG<dim,
                   Vector<double>,
                   MGTransferPrebuilt<Vector<double>>,
                   spacedim>
      vcycle(dof_handler, *mg, *mg_transfer);

    // A V-cycle is not a direct solver: it is an approximate inverse, and
    // used on its own it has to be iterated. This is the classical defect
    // correction
    //
    //     u  <-  u + M^{-1} (f - A u),
    //
    // with no Krylov acceleration, so what comes out is the convergence
    // rate of the V-cycle itself.
    Vector<double> residual(system_rhs.size());
    Vector<double> correction(system_rhs.size());

    // r = f - A u. The sign matters: the correction is added to u, so a
    // residual written the other way round turns convergent defect
    // correction into a divergent Richardson iteration.
    const auto compute_residual = [&]() {
      system_matrix.vmult(residual, solution);
      residual -= system_rhs;
      residual *= -1.0;
      return residual.l2_norm();
    };

    const double tolerance = 1e-10 * rhs_norm;

    double       residual_norm = compute_residual();
    unsigned int n_cycles      = 0;

    timer.restart();
    while (residual_norm > tolerance && n_cycles < 1000)
      {
        correction = 0;
        vcycle.vmult(correction, residual);
        solution += correction;
        ++n_cycles;

        residual_norm = compute_residual();
      }
    stats.t_solve    = timer.wall_time();
    stats.iterations = n_cycles;
    stats.converged  = (residual_norm <= tolerance);

    return stats;
  }


  template <int dim, int spacedim>
  SolveStats LaplaceBeltramiProblem<dim, spacedim>::solve_direct()
  {
    SolveStats stats;
    stats.method = method_name(Method::direct);

    std::cout << "   Method: sparse direct factorisation (UMFPACK)"
              << std::endl;

    // Note that there is no need to compress the matrix first:
    // SparseMatrix::compress() is a no-op for a serial matrix, and the
    // assembly has already eliminated the constrained rows and columns.
    SparseDirectUMFPACK direct_solver;

    Timer timer;
    timer.start();
    // The factorisation, fill-in included. All of the work is here.
    direct_solver.initialize(system_matrix);
    stats.t_pc_setup = timer.wall_time();

    timer.restart();
    // Two triangular solves. There is no iteration and no tolerance.
    direct_solver.vmult(solution, system_rhs);
    stats.t_solve = timer.wall_time();

    stats.iterations = 1;

    return stats;
  }


  /* ==================================================================
   *
   * 10. Error against the exact solution
   *
   * ================================================================== */

  template <int dim, int spacedim>
  Errors LaplaceBeltramiProblem<dim, spacedim>::compute_error() const
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

    Errors errors;
    errors.h1   = error_in_norm(VectorTools::H1_seminorm);
    errors.l2   = error_in_norm(VectorTools::L2_norm);
    errors.linf = error_in_norm(VectorTools::Linfty_norm);

    std::cout << "   H1 error = " << errors.h1 << ",  L2 error = " << errors.l2
              << ",  Linfty = " << errors.linf << std::endl;

    return errors;
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
    // In a benchmark run these files are pure overhead: the .coo dumps of a
    // large hierarchy are hundreds of megabytes and would dominate the very
    // timings the benchmark is measuring.
    if (!write_output_files)
      return;

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
   * 12. Measurement
   *
   * Two independent memory numbers, because they answer different
   * questions and a benchmark that reports only one of them can be
   * misleading:
   *
   *  - algebraic_mb counts the deal.II objects we hold ourselves. It is
   *    the number that shows how multigrid's O(N) storage is built up out
   *    of the level operators, and it is reproducible.
   *
   *  - peak_rss_mb is the process high-water mark. It is always >= the
   *    algebraic count, and the gap between the two is memory allocated
   *    inside a library. That gap is the whole story for a direct solver,
   *    whose fill-in and internal index arrays never appear as a deal.II
   *    object.
   *
   * ================================================================== */

  namespace
  {
    double
    to_mb(const std::size_t bytes)
    {
      return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }


    // Peak resident set size of the process in megabytes. It is a high
    // water mark, so it never decreases: in a run that sweeps increasing
    // cycle sizes the value reported for a cycle is the peak up to and
    // including that cycle. As the problems grow monotonically that is the
    // peak for that cycle.
    double
    peak_rss_mb()
    {
      Utilities::System::MemoryStats stats;
      Utilities::System::get_memory_stats(stats);

      return static_cast<double>(stats.VmHWM) / 1024.0;
    }


    // Resident memory right now, as opposed to the high water mark.
    double
    current_rss_mb()
    {
      Utilities::System::MemoryStats stats;
      Utilities::System::get_memory_stats(stats);

      return static_cast<double>(stats.VmRSS) / 1024.0;
    }
  } // namespace


  template <int dim, int spacedim>
  double LaplaceBeltramiProblem<dim, spacedim>::algebraic_memory_mb() const
  {
    std::size_t bytes = 0;

    bytes += triangulation.memory_consumption();
    bytes += dof_handler.memory_consumption();
    bytes += sparsity_pattern.memory_consumption();
    bytes += system_matrix.memory_consumption();
    bytes += solution.memory_consumption();
    bytes += system_rhs.memory_consumption();

    // The level operators of the V-cycle, together with the sparsity
    // patterns they are built on. A SparseMatrix does not own its pattern,
    // so the two counts do not overlap.
    if (level_operators_built)
      for (unsigned int level = mg_matrices.min_level();
           level <= mg_matrices.max_level();
           ++level)
        {
          bytes += mg_sparsity_patterns[level].memory_consumption();
          bytes += mg_matrices[level].memory_consumption();
          bytes += mg_interface_sparsity_patterns[level].memory_consumption();
          bytes += mg_interface_matrices[level].memory_consumption();
        }

    return to_mb(bytes);
  }


  namespace
  {
    double
    total_seconds(const SolveStats &s)
    {
      return s.t_setup + s.t_assemble + s.t_pc_setup + s.t_solve + s.t_error +
             s.t_output;
    }


    void
    print_stats(const SolveStats &s)
    {
      std::cout << "   iterations = " << s.iterations
                << (s.converged ? "" : "  (DID NOT CONVERGE)")
                << ",  reduction per step = " << s.rate
                << ",  residual = " << s.residual << std::endl
                << "   memory: " << s.algebraic_mb << " MB algebraic, "
                << s.rss_growth_mb << " MB above baseline, " << s.peak_rss_mb
                << " MB peak RSS" << std::endl
                << "   time: setup " << s.t_setup << " s, assemble "
                << s.t_assemble << " s, preconditioner " << s.t_pc_setup
                << " s, solve " << s.t_solve << " s, total "
                << total_seconds(s) << " s" << std::endl;
    }


    // One row per cycle. Appended rather than rewritten, so that a sweep
    // over several methods and degrees collects into a single file.
    void
    append_csv(const std::string            &path,
               const unsigned int            degree,
               const std::vector<SolveStats> &rows)
    {
      if (path.empty())
        return;

      const bool fresh = !std::ifstream(path).good();

      std::ofstream out(path, std::ios::app);
      AssertThrow(out, ExcMessage("Could not open " + path));

      if (fresh)
        out << "method,degree,cycle,dofs,cells,levels,converged,iterations,"
               "rate,residual,t_setup,t_assemble,t_pc_setup,t_solve,t_total,"
               "algebraic_mb,rss_growth_mb,peak_rss_mb,"
               "h1_error,l2_error,linf_error\n";

      out << std::setprecision(10);

      for (const SolveStats &s : rows)
        out << s.method << ',' << degree << ',' << s.cycle << ',' << s.dofs
            << ',' << s.cells << ',' << s.levels << ','
            << (s.converged ? "yes" : "no") << ',' << s.iterations << ','
            << s.rate << ',' << s.residual << ',' << s.t_setup << ','
            << s.t_assemble << ',' << s.t_pc_setup << ',' << s.t_solve << ','
            << total_seconds(s) << ',' << s.algebraic_mb << ','
            << s.rss_growth_mb << ',' << s.peak_rss_mb << ',' << s.errors.h1
            << ',' << s.errors.l2 << ',' << s.errors.linf << '\n';
    }
  } // namespace


  /* ==================================================================
   *
   * 13. Driver
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

    // The level hierarchy is only built for the two multigrid methods. A
    // method is charged only for the machinery it actually uses, so that
    // the comparison is against the same discretisation rather than against
    // a common but pointless preparation step.
    const bool uses_multigrid =
      (method == Method::cg_mg || method == Method::mg_solver);

    // What the process costs before any of the problem is built, so that
    // the reported growth measures the problem rather than the runtime.
    const double baseline_rss_mb = current_rss_mb();

    std::vector<SolveStats> all_stats;

    for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
      {
        std::cout << std::endl
                  << "================================================"
                  << std::endl
                  << "Cycle " << cycle << "  (" << method_name(method) << ")"
                  << std::endl
                  << "================================================"
                  << std::endl;


        if (cycle > 0)
          triangulation.refine_global(1);


        std::cout << "   Active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "   Vertices: " << triangulation.n_vertices() << std::endl
                  << "   Levels: " << triangulation.n_levels() << std::endl;


        Timer timer;

        timer.start();
        setup_system(); // DoFs, constraints, sparsity patterns
        const double t_setup = timer.wall_time();

        timer.restart();
        assemble_system(); // the active system A u = f
        if (uses_multigrid)
          assemble_multigrid(); // the level operators of the V-cycle
        const double t_assemble = timer.wall_time();

        SolveStats stats = solve();

        timer.restart();
        stats.errors = compute_error();
        stats.t_error = timer.wall_time();

        timer.restart();
        output_results(cycle);
        stats.t_output = timer.wall_time();

        stats.cycle        = cycle;
        stats.dofs         = dof_handler.n_dofs();
        stats.cells        = triangulation.n_active_cells();
        stats.levels       = triangulation.n_levels();
        stats.t_setup      = t_setup;
        stats.t_assemble   = t_assemble;
        stats.algebraic_mb  = algebraic_memory_mb();
        stats.peak_rss_mb   = peak_rss_mb();
        stats.rss_growth_mb = stats.peak_rss_mb - baseline_rss_mb;

        print_stats(stats);
        all_stats.push_back(stats);
      }

    append_csv(csv_path, degree, all_stats);
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

      //   ./solver <degree> <n_refinement_cycles> [method] [options]
      //
      // The first two arguments and the default method are unchanged, so
      // that existing invocations keep working.
      Method      method            = Method::cg_mg;
      bool        write_output_files = true;
      std::string csv_path;

      std::vector<std::string> positional;

      for (int i = 1; i < argc; ++i)
        {
          const std::string arg = argv[i];

          if (arg == "--no-dump")
            write_output_files = false;
          else if (arg.rfind("--csv=", 0) == 0)
            csv_path = arg.substr(6);
          else if (!arg.empty() && arg[0] == '-')
            {
              std::cerr << "Unknown option '" << arg << "'" << std::endl;
              return 1;
            }
          else
            positional.push_back(arg);
        }

      if (positional.size() < 2 || positional.size() > 3)
        {
          std::cerr
            << "Usage: ./solver <degree> <n_refinement_cycles> [method]\n"
            << "                [--no-dump] [--csv=FILE]\n"
            << "\n"
            << "  method: jacobi | ssor | mg | mg-solver | direct\n"
            << "          (default mg, i.e. CG preconditioned by multigrid)\n"
            << "\n"
            << "  --no-dump    skip the .vtk/.coo/.txt dumps, which are\n"
            << "               large and would dominate the timings\n"
            << "  --csv=FILE   append one row per cycle to FILE\n"
            << std::endl;
          return 1;
        }

      const unsigned int degree = std::stoi(positional[0]);

      const unsigned int n_refinement_cycles = std::stoi(positional[1]);

      if (positional.size() == 3)
        method = parse_method(positional[2]);

      LaplaceBeltramiProblem<2, 3> laplace_beltrami(degree,
                                                    n_refinement_cycles,
                                                    method,
                                                    write_output_files,
                                                    csv_path);

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
