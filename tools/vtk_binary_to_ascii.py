#!/usr/bin/env python3
"""Convert the binary VTK 5.1 POLYDATA vessel surface into an ASCII VTK
UNSTRUCTURED_GRID that deal.II's GridIn::read_vtk can read.

Why the conversion is needed
----------------------------
The input `vessel.vtk` is a *binary* `DATASET POLYDATA` file written in the
VTK 5.1 style, where the polygon connectivity is stored as a separate
`OFFSETS` / `CONNECTIVITY` pair of 64-bit arrays.  deal.II's VTK reader
(deal.II 9.7, source/grid/grid_in.cc) accepts only

    ASCII
    DATASET UNSTRUCTURED_GRID
    POINTS / CELLS / CELL_TYPES

so neither the binary encoding nor the POLYDATA layout nor the OFFSETS
connectivity is readable as-is.

What is preserved
-----------------
* geometry    - the 86528 vertex coordinates, written with 9 significant
                digits, which round-trips the original float32 values exactly
* connectivity- the 173052 triangles, vertex order and orientation untouched
* labels      - the two per-cell label arrays of the CFD export are mapped onto
                the two per-cell integer arrays deal.II understands:

                    CapID       -> MaterialID  (read as cell->material_id())
                    ModelFaceID -> ManifoldID  (read as cell->manifold_id())

Additionally the per-cell NORMALS are used to verify that the triangle
winding is globally consistent, which deal.II requires for a codimension-one
surface triangulation.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np


class Reader:
    """Minimal cursor over the ASCII part of a legacy VTK file."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def line(self, skip_blank: bool = True) -> str:
        while True:
            nl = self.data.index(b"\n", self.pos)
            text = self.data[self.pos : nl].decode("ascii", "replace")
            self.pos = nl + 1
            if text.strip() or not skip_blank:
                return text

    def array(self, dtype: str, count: int) -> np.ndarray:
        a = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.pos)
        self.pos += count * np.dtype(dtype).itemsize
        return a


def read_binary_polydata(path: str):
    data = open(path, "rb").read()

    r = Reader(data)
    version = r.line()
    title = r.line()
    encoding = r.line()
    dataset = r.line()
    if encoding.strip() != "BINARY":
        raise SystemExit(f"expected a BINARY file, found {encoding!r}")
    if dataset.strip() != "DATASET POLYDATA":
        raise SystemExit(f"expected DATASET POLYDATA, found {dataset!r}")

    points_line = r.line()
    n_points = int(points_line.split()[1])
    points = r.array(">f4", n_points * 3).reshape(-1, 3).astype(np.float64)

    # Skip the METADATA block that precedes the connectivity section.
    while True:
        header = r.line()
        if header.startswith(("POLYGONS", "CELLS")):
            break
    n_offsets, n_conn = (int(v) for v in header.split()[1:3])

    offsets = r.array(">i8" if "int64" in r.line() else ">i4", n_offsets).astype(np.int64)
    conn = r.array(">i8" if "int64" in r.line() else ">i4", n_conn).astype(np.int64)

    if not np.all(np.diff(offsets) == 3):
        raise SystemExit("only triangle meshes are supported")
    triangles = conn.reshape(len(offsets) - 1, 3)

    # CELL_DATA: a NORMALS vector array, then a FIELD block of label arrays.
    cell_data_header = r.line()
    n_cells = int(cell_data_header.split()[1])
    # "NORMALS <name> <type>" - a vector array, always three components.
    r.line()
    n_comp = 3
    normals = r.array(">f4", n_cells * n_comp).reshape(-1, n_comp).astype(np.float64)

    field_header = r.line()
    n_fields = int(field_header.split()[2])
    fields: dict[str, np.ndarray] = {}
    for _ in range(n_fields):
        parts = r.line().split()
        name, ncomp, n, vtk_type = parts[0], int(parts[1]), int(parts[2]), parts[3]
        dtype = ">i8" if "int64" in vtk_type else ">i4"
        arr = r.array(dtype, n * ncomp).astype(np.int64)
        fields[name] = arr.reshape(n, ncomp) if ncomp > 1 else arr

    print(f"read {version!r} / {title!r}")
    print(f"  points      : {points.shape}")
    print(f"  triangles   : {triangles.shape}")
    print(f"  cell fields : {sorted(fields)}")
    return points, triangles, normals, fields


def check_mesh(points, triangles, normals, material, manifold):
    """Sanity checks: orientation, manifoldness, watertightness, labels."""
    a, b, c = (points[triangles[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)

    print("\n--- mesh checks ---")
    if (area == 0).any():
        raise SystemExit(f"{int((area == 0).sum())} degenerate triangles")
    print(f"  total area            : {area.sum():.6f}")

    # deal.II needs a globally consistent orientation for codim-1 meshes.
    unit = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    dot = np.einsum("ij,ij->i", unit, normals)
    n_bad = int((dot <= 0).sum())
    print(f"  winding vs. VTK normal: min {dot.min():.6f}, inverted {n_bad}")
    if n_bad:
        raise SystemExit(
            "triangle winding disagrees with the stored normals; the mesh would "
            "need re-orientation before deal.II can read it"
        )

    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
        axis=1,
    )
    key = edges[:, 0].astype(np.int64) * len(points) + edges[:, 1]
    _, counts = np.unique(key, return_counts=True)
    hist = dict(zip(*[x.tolist() for x in np.unique(counts, return_counts=True)]))
    print(f"  edge incidence        : {hist}")
    if set(hist) != {2}:
        raise SystemExit("mesh is not a closed 2-manifold")
    euler = len(points) - len(np.unique(key)) + len(triangles)
    print(f"  Euler characteristic  : {euler}  (closed surface, no boundary)")
    print(f"  boundary edges        : 0  -> no deal.II boundary faces exist")

    print("\n--- labels ---")
    for name, arr in (("MaterialID (CapID)", material), ("ManifoldID (ModelFaceID)", manifold)):
        u, n = np.unique(arr, return_counts=True)
        print(f"  {name}: {dict(zip(u.tolist(), n.tolist()))}")


def write_ascii(path, points, triangles, material, manifold, title):
    n_points = len(points)
    n_cells = len(triangles)

    with open(path, "w", newline="\n") as out:
        out.write("# vtk DataFile Version 3.0\n")
        out.write(f"{title}\n")
        out.write("ASCII\n")
        out.write("DATASET UNSTRUCTURED_GRID\n")

        out.write(f"POINTS {n_points} double\n")
        # 9 significant digits round-trip the source float32 coordinates exactly.
        np.savetxt(out, points, fmt="%.9g")

        out.write(f"CELLS {n_cells} {n_cells * 4}\n")
        cells = np.hstack([np.full((n_cells, 1), 3, dtype=np.int64), triangles])
        np.savetxt(out, cells, fmt="%d")

        out.write(f"CELL_TYPES {n_cells}\n")
        # VTK_TRIANGLE
        np.savetxt(out, np.full(n_cells, 5, dtype=np.int64), fmt="%d")

        out.write(f"CELL_DATA {n_cells}\n")
        # deal.II reads exactly these two names, both with type 'int'.
        out.write("SCALARS MaterialID int 1\n")
        out.write("LOOKUP_TABLE default\n")
        np.savetxt(out, material, fmt="%d")

        out.write("SCALARS ManifoldID int 1\n")
        out.write("LOOKUP_TABLE default\n")
        np.savetxt(out, manifold, fmt="%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="binary POLYDATA .vtk file")
    ap.add_argument("target", help="ASCII UNSTRUCTURED_GRID .vtk file to write")
    ap.add_argument(
        "--title", default="vessel surface mesh (ASCII, labels preserved)"
    )
    args = ap.parse_args()

    points, triangles, normals, fields = read_binary_polydata(args.source)

    for needed in ("CapID", "ModelFaceID"):
        if needed not in fields:
            raise SystemExit(f"label array {needed!r} missing from the input")

    material = fields["CapID"]
    manifold = fields["ModelFaceID"]

    # deal.II casts MaterialID/ManifoldID to types::material_id/manifold_id,
    # which are unsigned: negative label values would wrap around.
    for name, arr in (("CapID", material), ("ModelFaceID", manifold)):
        if (arr < 0).any():
            raise SystemExit(f"{name} contains negative values")

    check_mesh(points, triangles, normals, material, manifold)

    write_ascii(args.target, points, triangles, material, manifold, args.title)

    import os

    print(f"\nwrote {args.target} ({os.path.getsize(args.target) / 1e6:.1f} MB)")
    print(f"  MaterialID = CapID       {sorted(set(material.tolist()))}")
    print(f"  ManifoldID = ModelFaceID {sorted(set(manifold.tolist()))}")


if __name__ == "__main__":
    sys.exit(main())
