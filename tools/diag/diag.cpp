/* Miniature version of the whole solve, using the element and mapping that
 * suit a triangle mesh, so that the approach can be validated before the
 * full solver is rebuilt.
 *
 * Checks, in order:
 *   - the element's reference cell matches the mesh's
 *   - the Dirichlet degree-of-freedom set is found without a stray index
 *   - the assembled surface area reproduces the mesh area
 *   - CG + SSOR converges
 *   - the discrete solution respects the maximum principle 0 <= u <= 1
 *   - int_S u dS against the surface area
 *   - the Dirichlet degrees of freedom really are zero
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace dealii;

int main(int argc, char **argv)
{
  const std::string  filename = (argc > 1) ? argv[1] : "meshes/vessel_ascii.vtk";
  const unsigned int degree   = (argc > 2) ? std::stoi(argv[2]) : 1;

  try
    {
      Triangulation<2, 3> triangulation(
        Triangulation<2, 3>::limit_level_difference_at_vertices);

      std::ifstream input(filename);
      GridIn<2, 3>  grid_in;
      grid_in.attach_triangulation(triangulation);
      grid_in.read_vtk(input);

      types::manifold_id max_id = 0;
      for (const auto &cell : triangulation.active_cell_iterators())
        max_id = std::max(max_id, cell->manifold_id());
      for (types::manifold_id id = 0; id <= max_id; ++id)
        triangulation.set_manifold(id, FlatManifold<2, 3>());

      const ReferenceCell mesh_cell =
        triangulation.begin_active()->reference_cell();
      std::cout << "mesh reference cell: " << mesh_cell.to_string() << std::endl;

      const FE_SimplexP<2, 3> fe(degree);
      std::cout << "FE_SimplexP(" << degree
                << "): per cell = " << fe.n_dofs_per_cell()
                << ", reference cell = " << fe.reference_cell().to_string()
                << std::endl;

      // The mapping has to describe the same cell type as the mesh.  A P1
      // simplex mapping is exactly the piecewise linear geometry the file
      // describes.
      const FE_SimplexP<2, 3>      mapping_fe(1);
      const MappingFE<2, 3>        mapping(mapping_fe);
      const MappingQ<2, 3>         mapping_q(1);

      std::cout << "MappingFE compatible with mesh cell: "
                << mapping.is_compatible_with(mesh_cell) << std::endl;
      std::cout << "MappingQ  compatible with mesh cell: "
                << mapping_q.is_compatible_with(mesh_cell) << std::endl;

      DoFHandler<2, 3> dof_handler(triangulation);
      dof_handler.distribute_dofs(fe);
      std::cout << "n_dofs = " << dof_handler.n_dofs() << std::endl;

      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      std::vector<bool> selected(dof_handler.n_dofs(), false);
      std::vector<types::global_dof_index> local(fe.n_dofs_per_cell());

      unsigned int bad = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->material_id() != 0)
          {
            cell->get_dof_indices(local);
            for (const types::global_dof_index index : local)
              {
                if (index >= dof_handler.n_dofs())
                  {
                    ++bad;
                    continue;
                  }
                selected[index] = true;
              }
          }
      std::cout << "out-of-range dof indices: " << bad << std::endl;

      constraints.add_lines(selected);
      constraints.close();

      const std::size_t n_dirichlet =
        std::count(selected.begin(), selected.end(), true);
      std::cout << "dirichlet dofs = " << n_dirichlet << " of "
                << dof_handler.n_dofs() << std::endl;

      DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

      SparsityPattern      sparsity;
      SparseMatrix<double> A;
      sparsity.copy_from(dsp);
      A.reinit(sparsity);

      Vector<double> u(dof_handler.n_dofs());
      Vector<double> b(dof_handler.n_dofs());

      const QGaussSimplex<2>  quadrature(2 * degree + 1);
      FEValues<2, 3>          fe_values(mapping,
                               fe,
                               quadrature,
                               update_values | update_gradients |
                                 update_JxW_values);
      const unsigned int      dpc  = fe.n_dofs_per_cell();
      const unsigned int      n_q  = quadrature.size();
      FullMatrix<double>      cell_matrix(dpc, dpc);
      Vector<double>          cell_rhs(dpc);
      std::vector<types::global_dof_index> cell_dofs(dpc);

      double area = 0.0;

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;

          for (unsigned int q = 0; q < n_q; ++q)
            {
              area += fe_values.JxW(q);
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  for (unsigned int j = 0; j < dpc; ++j)
                    cell_matrix(i, j) +=
                      (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
                       fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) *
                      fe_values.JxW(q);
                  cell_rhs(i) += 1.0 * fe_values.shape_value(i, q) * fe_values.JxW(q);
                }
            }

          cell->get_dof_indices(cell_dofs);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 cell_dofs,
                                                 A,
                                                 b);
        }

      std::cout << std::setprecision(10);
      std::cout << "assembled area = " << area << "  (expected 226.740457)"
                << std::endl;

      SolverControl            control(10000, 1e-12 * b.l2_norm());
      SolverCG<Vector<double>> solver(control);
      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(A);
      u = 0;
      solver.solve(A, u, b, preconditioner);
      std::cout << "CG iterations = " << control.last_step()
                << ", final residual = " << control.last_value() << std::endl;

      constraints.distribute(u);

      Vector<double> Au(b.size());
      A.vmult(Au, u);
      Vector<double> r(b);
      r -= Au;
      std::cout << "relative residual = " << r.l2_norm() / b.l2_norm()
                << std::endl;

      std::cout << "u range = [" << *std::min_element(u.begin(), u.end()) << ", "
                << *std::max_element(u.begin(), u.end()) << "]  (expect [0, 1])"
                << std::endl;

      double max_on_dirichlet = 0.0;
      for (types::global_dof_index i = 0; i < u.size(); ++i)
        if (selected[i])
          max_on_dirichlet = std::max(max_on_dirichlet, std::abs(u[i]));
      std::cout << "max |u| on dirichlet set = " << max_on_dirichlet << std::endl;

      std::vector<double> values(n_q);
      double              integral = 0.0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          fe_values.get_function_values(u, values);
          for (unsigned int q = 0; q < n_q; ++q)
            integral += values[q] * fe_values.JxW(q);
        }
      std::cout << "int_S u dS = " << integral
                << ", relative difference from area = "
                << std::abs(integral - area) / area << std::endl;

      std::cout << "DIAG_OK" << std::endl;
    }
  catch (std::exception &exc)
    {
      std::cerr << "EXCEPTION: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
