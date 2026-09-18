#!/bin/bash
# Build and run the standalone mesh-reading test.
set -u
cd /work

cmake -S tools/mesh_test -B build-mesh-test -DCMAKE_BUILD_TYPE=Release > /tmp/mt_cmake.log 2>&1 \
  || { echo "CMAKE_FAIL"; tail -20 /tmp/mt_cmake.log; exit 1; }

cmake --build build-mesh-test -j"$(nproc)" > /tmp/mt_build.log 2>&1 \
  || { echo "BUILD_FAIL"; tail -60 /tmp/mt_build.log; exit 1; }

echo "BUILD_OK"
./build-mesh-test/mesh_test meshes/vessel_ascii.vtk 1
echo "exit=$?"
