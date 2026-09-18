/* Standalone check that the converted ASCII vessel mesh can be read by
 * deal.II and that the region labels survive the round trip.
 *
 * Deliberately a small translation unit so it compiles quickly.
 */

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>

using namespace dealii;

int main(int argc, char **argv)
{
  const std::string filename =
    (argc > 1) ? argv[1] : "meshes/vessel_ascii.vtk";
  const unsigned int n_refinements = (argc > 2) ? std::stoi(argv[2]) : 0;

  try
    {
      Triangulation<2, 3> triangulation(
        Triangulation<2, 3>::limit_level_difference_at_vertices);

      std::ifstream input(filename);
      AssertThrow(input, ExcMessage("cannot open " + filename));

      GridIn<2, 3> grid_in;
      grid_in.attach_triangulation(triangulation);
      grid_in.read_vtk(input);

      std::cout << "read: cells=" << triangulation.n_cells()
                << " active=" << triangulation.n_active_cells()
                << " vertices=" << triangulation.n_vertices()
                << " levels=" << triangulation.n_levels() << std::endl;
      // A boundary face of a codimension-one mesh is a line that belongs
      // to a single cell.  A closed surface has none.
      unsigned int n_boundary_faces = 0;
      for (const auto &face : triangulation.active_face_iterators())
        if (face->at_boundary())
          ++n_boundary_faces;
      std::cout << "boundary faces=" << n_boundary_faces << std::endl;

      // Every manifold id carried by the file must have a Manifold object,
      // otherwise get_manifold() asserts in debug builds.
      types::manifold_id max_id = 0;
      for (const auto &cell : triangulation.active_cell_iterators())
        max_id = std::max(max_id, cell->manifold_id());
      std::cout << "max manifold id=" << max_id << std::endl;
      for (types::manifold_id id = 0; id <= max_id; ++id)
        triangulation.set_manifold(id, FlatManifold<2, 3>());

      std::map<types::material_id, unsigned int> by_material;
      std::map<types::manifold_id, unsigned int> by_manifold;
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          ++by_material[cell->material_id()];
          ++by_manifold[cell->manifold_id()];
        }

      std::cout << "material ids (CapID): ";
      for (const auto &[id, n] : by_material)
        std::cout << id << "->" << n << " ";
      std::cout << "\nmanifold ids (ModelFaceID): ";
      for (const auto &[id, n] : by_manifold)
        std::cout << id << "->" << n << " ";
      std::cout << std::endl;

      // Surface area: sum of the flat triangle areas.
      double area = 0.0;
      for (const auto &cell : triangulation.active_cell_iterators())
        area += cell->measure();
      std::cout << "surface area=" << area << std::endl;

      if (n_refinements > 0)
        {
          triangulation.refine_global(n_refinements);
          std::cout << "after " << n_refinements
                    << " refinement(s): active=" << triangulation.n_active_cells()
                    << " vertices=" << triangulation.n_vertices()
                    << " levels=" << triangulation.n_levels() << std::endl;
          std::map<types::material_id, unsigned int> refined;
          for (const auto &cell : triangulation.active_cell_iterators())
            ++refined[cell->material_id()];
          std::cout << "refined material ids: ";
          for (const auto &[id, n] : refined)
            std::cout << id << "->" << n << " ";
          std::cout << std::endl;
        }

      std::cout << "MESH_READ_OK" << std::endl;
    }
  catch (std::exception &exc)
    {
      std::cerr << "EXCEPTION: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
