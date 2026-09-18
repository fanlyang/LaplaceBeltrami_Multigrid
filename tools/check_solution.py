#!/usr/bin/env python3
"""Independent check of the delivered solution vector.

Reads the ASCII mesh and the solution vector straight from disk, works out
which vertices lie on an opening from the labels alone, and checks that the
solution is zero there and inside the maximum-principle bounds elsewhere.
Nothing here shares code with the solver.
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_conversion import parse_ascii  # noqa: E402


def main():
    mesh, solution_txt = sys.argv[1], sys.argv[2]

    _, triangles, arrays = parse_ascii(mesh)
    material = arrays["MaterialID"]

    lines = open(solution_txt).read().split()
    n = int(lines[0])
    u = np.array(lines[1 : 1 + n], dtype=np.float64)

    print(f"solution vector length : {n}")

    # Vertices that belong to at least one opening cell.
    opening_cells = triangles[material != 0]
    on_opening = np.zeros(int(triangles.max()) + 1, dtype=bool)
    on_opening[opening_cells.ravel()] = True

    n_opening_vertices = int(on_opening.sum())
    print(f"vertices on an opening  : {n_opening_vertices}")

    if n != len(on_opening):
        print(f"NOTE: vector length {n} != vertex count {len(on_opening)}; "
              "cannot map by index")
        return 1

    print()
    print(f"max |u| on opening vertices : {np.abs(u[on_opening]).max():.6e}")
    print(f"min u on wall vertices      : {u[~on_opening].min():.6e}")
    print(f"max u on wall vertices      : {u[~on_opening].max():.6e}")
    print(f"number of NaNs              : {int(np.isnan(u).sum())}")

    ok = (
        np.abs(u[on_opening]).max() == 0.0
        and u.min() >= 0.0
        and u.max() <= 1.0
        and not np.isnan(u).any()
    )
    print()
    print("SOLUTION_CHECK_OK" if ok else "SOLUTION_CHECK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
