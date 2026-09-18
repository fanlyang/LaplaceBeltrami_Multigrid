#!/bin/bash
set -u
cd /work
cmake -S tools/diag -B build-diag -DCMAKE_BUILD_TYPE=Release > /tmp/d_cmake.log 2>&1 \
  || { echo "CMAKE_FAIL"; tail -20 /tmp/d_cmake.log; exit 1; }
cmake --build build-diag -j"$(nproc)" > /tmp/d_build.log 2>&1 \
  || { echo "BUILD_FAIL"; tail -60 /tmp/d_build.log; exit 1; }
echo "BUILD_OK"
./build-diag/diag meshes/vessel_ascii.vtk "${1:-1}"
echo "diag_exit=$?"
