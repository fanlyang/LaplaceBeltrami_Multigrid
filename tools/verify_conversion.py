#!/usr/bin/env python3
"""Independently verify that meshes/vessel_ascii.vtk reproduces the original
binary mesh exactly: same vertices, same triangles, same labels.

Parses the ASCII file the same way deal.II's GridIn::read_vtk walks it, so it
also checks that the section order and the CELL_DATA layout are what the reader
expects.
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vtk_binary_to_ascii import read_binary_polydata  # noqa: E402


def tokens(lines, start, count):
    out = []
    i = start
    while len(out) < count:
        out.extend(lines[i].split())
        i += 1
    return out[:count], i


def parse_ascii(path):
    lines = open(path).read().split("\n")
    assert lines[0].startswith("# vtk DataFile"), lines[0]
    assert lines[2] == "ASCII", lines[2]
    assert lines[3] == "DATASET UNSTRUCTURED_GRID", lines[3]

    def find(prefix, start=0):
        for i in range(start, len(lines)):
            if lines[i].startswith(prefix):
                return i
        raise KeyError(prefix)

    i = find("POINTS")
    n_pts = int(lines[i].split()[1])
    assert lines[i].split()[2] == "double"
    flat, after = tokens(lines, i + 1, n_pts * 3)
    points = np.array(flat, dtype=np.float64).reshape(-1, 3)

    i = find("CELLS", after - 2)
    n_cells, n_ints = (int(v) for v in lines[i].split()[1:3])
    assert n_ints == n_cells * 4, "expected all-triangle cells"
    flat, after = tokens(lines, i + 1, n_ints)
    cells = np.array(flat, dtype=np.int64).reshape(n_cells, 4)
    assert np.all(cells[:, 0] == 3), "expected VTK_TRIANGLE entries"
    triangles = cells[:, 1:]

    i = find("CELL_TYPES", after - 2)
    n_types = int(lines[i].split()[1])
    assert n_types == n_cells
    flat, after = tokens(lines, i + 1, n_types)
    assert np.all(np.array(flat, dtype=np.int64) == 5), "expected VTK_TRIANGLE"

    i = find("CELL_DATA", after - 2)
    n_data = int(lines[i].split()[1])
    assert n_data == n_cells, (n_data, n_cells)

    arrays = {}
    i += 1
    for _ in range(2):
        scalars = lines[i]
        parts = scalars.split()
        assert parts[0] == "SCALARS", scalars
        name = parts[1]
        assert parts[2] == "int", "deal.II requires MaterialID/ManifoldID to be int"
        assert lines[i + 1] == "LOOKUP_TABLE default", lines[i + 1]
        flat, after = tokens(lines, i + 2, n_cells)
        arrays[name] = np.array(flat, dtype=np.int64)
        i = after
        while not lines[i].startswith("SCALARS"):
            if i + 1 >= len(lines):
                break
            i += 1

    return points, triangles, arrays


def main():
    src, dst = sys.argv[1], sys.argv[2]

    p_ref, t_ref, _, _ = read_binary_polydata(src)
    p_new, t_new, arrays = parse_ascii(dst)

    ok = True

    print(f"vertices : {len(p_new)} vs {len(p_ref)}")
    if len(p_new) != len(p_ref):
        ok = False
    else:
        # The source is float32; the ASCII file writes 9 significant digits,
        # which recovers those values exactly.
        same = np.array_equal(p_new.astype(np.float32), p_ref.astype(np.float32))
        exact = np.array_equal(p_new, p_ref)
        print(f"  bitwise equal as float32 : {same}")
        print(f"  bitwise equal as float64 : {exact}")
        print(f"  max |dx|                 : {np.abs(p_new - p_ref).max():.3e}")
        ok &= bool(same)

    print(f"triangles: {len(t_new)} vs {len(t_ref)}")
    if t_new.shape != t_ref.shape:
        ok = False
    else:
        print(f"  identical connectivity   : {np.array_equal(t_new, t_ref)}")
        ok &= bool(np.array_equal(t_new, t_ref))

    # Re-derive the labels from the binary file and compare.
    _, _, _, fields = read_binary_polydata(src)
    for vtk_name, field_name in (("MaterialID", "CapID"), ("ManifoldID", "ModelFaceID")):
        ref = fields[field_name]
        print(f"{vtk_name:<10}: {dict(zip(*[x.tolist() for x in np.unique(arrays[vtk_name], return_counts=True)]))}")
        print(f"  matches {field_name}: {np.array_equal(arrays[vtk_name], ref)}")
        ok &= bool(np.array_equal(arrays[vtk_name], ref))

    print()
    print("CONVERSION_LOSSLESS" if ok else "CONVERSION_MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
