#!/usr/bin/env bash
#
# Benchmark sweep: every solution method over every refinement cycle,
# collected into a single CSV.
#
#   ./bench/run_sweep.sh [degree] [cycles]
#
# e.g.
#   ./bench/run_sweep.sh 3 6      # Q3 through 6 refinement cycles
#
# Requires the image, built from the repository root:
#
#   docker build -t laplace-beltrami-bench .
#
# Each method is a separate container run, so the memory numbers are not
# contaminated by an earlier method's allocations and every row is measured
# in a fresh process.

set -euo pipefail

# Under Git Bash on Windows, MSYS rewrites anything that looks like a Unix
# path into a Windows one -- including the container-side half of -v and the
# CSV path -- which breaks the mount. Turn that off.
export MSYS_NO_PATHCONV=1

IMAGE=${IMAGE:-laplace-beltrami-bench}
DEGREE=${1:-3}
CYCLES=${2:-6}

HERE=$(cd "$(dirname "$0")" && pwd)
CSV="results-d${DEGREE}.csv"

rm -f "$HERE/$CSV"

for method in jacobi ssor mg mg-solver direct; do
  echo "=============================================================="
  echo "  degree $DEGREE   method $method   cycles 0..$((CYCLES - 1))"
  echo "=============================================================="

  # The results directory is mounted at /work and the CSV is named by its
  # absolute path, so that the image's own working directory -- which the
  # relative entrypoint ./build/solver depends on -- is left alone.
  docker run --rm \
    -v "$HERE:/work" \
    "$IMAGE" "$DEGREE" "$CYCLES" "$method" --no-dump --csv="/work/$CSV"
done

echo
echo "All results written to $HERE/$CSV"
