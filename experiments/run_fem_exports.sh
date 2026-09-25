#!/usr/bin/env bash
# Step 1 of the high-contrast experiment: the finite-element side.
#
# For each epsilon in {1e-1, 1e-2, 1e-3, 1e-4} this script
#
#   * builds the solver image from src/main.cpp,
#   * runs it for kappa_epsilon(xi,eta) = epsilon + sin^2(xi) cos^2(eta)
#     over six uniform refinements (16 ... 16384 DoFs), appending one CSV row
#     per cycle -- that table is the manufactured-solution verification
#     (L2 / H1-seminorm / Linfinity against u_exact = sin(xi) sin(eta), with
#     f recomputed in closed form from the same kappa_epsilon), and
#   * keeps the .coo/.txt dumps, from which tools/convert_level_data.py
#     extracts the level-3 operator (1024 DoFs) and its prolongation.
#
# Everything the experiment needs is written under level_dumps/ and results/.
#
# Usage:  bash experiments/run_fem_exports.sh
#
# Windows note: MSYS_NO_PATHCONV=1 stops Git Bash from rewriting the container
# paths /work and /home/... into C:/... . The entrypoint is given as an
# absolute path inside the image because the image's own entrypoint is
# relative (./build/solver), which `-w /work` would break.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=lb-hc
DEGREE=1          # linear tent functions: one DoF per lattice vertex
CYCLES=6          # cycles 0..5 -> 16, 64, 256, 1024, 4096, 16384 DoFs
EPSILONS=(1e-1 1e-2 1e-3 1e-4)

export MSYS_NO_PATHCONV=1
HERE="$(pwd -W 2>/dev/null || pwd)"

echo "== building $IMAGE from src/main.cpp =="
docker build -t "$IMAGE" .

mkdir -p results
CSV=results/fem_verification.csv
if [ -f "$CSV" ]; then
  echo "== removing the previous $CSV so the run is not appended to =="
  rm -f "$CSV"
fi

for eps in "${EPSILONS[@]}"; do
  tag="eps-${eps}"
  mkdir -p "level_dumps/$tag"
  echo
  echo "== epsilon = $eps  ->  level_dumps/$tag =="
  docker run --rm \
    --entrypoint /home/dealii/solver/build/solver \
    -v "$HERE/level_dumps/$tag:/work" \
    -w /work \
    "$IMAGE" "$DEGREE" "$CYCLES" --epsilon="$eps" --csv="fem_${tag}.csv"
  # The CSV above lands inside the container's /work, i.e. in the dump
  # directory; move it into results/ and tag it with the epsilon.
  mv "level_dumps/$tag/fem_${tag}.csv" "results/fem_${tag}.csv"
done

echo
echo "== verification tables =="
ls -l results/fem_eps-*.csv
echo
echo "== now convert the level data =="
echo "   bash experiments/run_conversion.sh"
