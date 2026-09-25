#!/usr/bin/env bash
# Step 1: the finite-element side of the ANISOTROPIC TENSOR experiment.
#
# For every epsilon this
#   * builds the solver image from src/main.cpp,
#   * runs -div(D_eps grad u) + u = f_eps on the torus over six uniform
#     refinements (16 ... 16384 DoFs), appending one CSV row per cycle -- the
#     manufactured-solution verification, with f_eps recomputed in closed form
#     from D_eps at every epsilon,
#   * dumps the level operators and prolongations, and
#   * writes transfer-check.txt, the comparison of the exported P against
#     deal.II's MGTransferPrebuilt.
#
# Usage:  bash experiments/run_tensor_exports.sh
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=lb-aniso
DEGREE=1
CYCLES=6
# 1 is the isotropic control (D = I); the rest are the anisotropy sweep.
EPSILONS=(1 1e-1 1e-2 1e-3 1e-4)

export MSYS_NO_PATHCONV=1
HERE="$(pwd -W 2>/dev/null || pwd)"

echo "== building $IMAGE from src/main.cpp =="
docker build -t "$IMAGE" .

mkdir -p results
CSV=results/tensor_verification.csv
rm -f "$CSV"

for eps in "${EPSILONS[@]}"; do
  tag=$(printf '%.0e' "$eps")
  mkdir -p "level_dumps/eps-${tag}"
  echo
  echo "== epsilon = $eps  ->  level_dumps/eps-${tag} =="
  docker run --rm \
    --entrypoint /home/dealii/solver/build/solver \
    -v "$HERE/level_dumps/eps-${tag}:/work" \
    -w /work \
    "$IMAGE" "$DEGREE" "$CYCLES" --coefficient=tensor --epsilon="$eps" \
      --csv="verif_${tag}.csv"
  mv "level_dumps/eps-${tag}/verif_${tag}.csv" "results/tensor_verif_${tag}.csv"
  mv "level_dumps/eps-${tag}/transfer-check.txt" \
     "results/transfer-check-${tag}.txt"
done

echo
echo "== verification tables =="
ls -l results/tensor_verif_*.csv
echo
echo "== transfer check (exported P vs MGTransferPrebuilt) =="
for eps in "${EPSILONS[@]}"; do
  tag=$(printf '%.0e' "$eps")
  echo "  epsilon $eps:"
  awk 'NR>1 { if ($4+0 > pm) pm=$4+0; if ($5+0 > rm) rm=$5+0 } END { printf "    worst |P_mg - P| = %.3e, worst |R_mg - P^T| = %.3e\n", pm, rm }' \
      "results/transfer-check-${tag}.txt"
done
