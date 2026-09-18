#!/bin/bash
# Build and run the vessel surface solver.
#   run_solver.sh [build-dir] [degree] [n_refinement_cycles]
set -u
cd /work

BUILD_DIR="${1:-build}"
DEGREE="${2:-1}"
CYCLES="${3:-1}"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /tmp/s_cmake.log 2>&1 \
  || { echo "CMAKE_FAIL"; tail -20 /tmp/s_cmake.log; exit 1; }

cmake --build "$BUILD_DIR" -j"$(nproc)" > /tmp/s_build.log 2>&1 \
  || { echo "BUILD_FAIL"; tail -80 /tmp/s_build.log; exit 1; }

echo "BUILD_OK"

OUTDIR=/work/out
mkdir -p "$OUTDIR" 2>/dev/null || OUTDIR=/tmp/out
mkdir -p "$OUTDIR"
cd "$OUTDIR"

t0=$(date +%s)
"/work/$BUILD_DIR/solver" /work/meshes/vessel_ascii.vtk "$DEGREE" "$CYCLES"
rc=$?
echo "solver_exit=$rc elapsed_seconds=$(( $(date +%s) - t0 ))"
echo "output_files_in=$OUTDIR"
ls -la "$OUTDIR"
exit 0
