#!/usr/bin/env bash
# Step 3: the Stage-I experiment itself, one run per epsilon, then the
# cross-epsilon collection.
#
# Each run trains two DeepONet models (the plain MLP, and the same network with
# the existing framework's learned Jacobi skip) on L_smooth alone, then measures
# rho_F, rho_C, the controlled modes, the Jacobi-hard errors and the ideal
# Galerkin two-grid reduction on a test set none of them was selected on.
#
# Usage:  bash experiments/run_stage1.sh [PYTHON]
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${1:-python}"
if [ "$PY" = "python" ]; then
  COMMON="$(git rev-parse --git-common-dir 2>/dev/null || echo .)"
  MAIN="$(cd "$COMMON/.." 2>/dev/null && pwd || echo .)"
  for cand in "../.venv-deeponet" ".venv-deeponet" "$MAIN/.venv-deeponet"; do
    if [ -x "$cand/Scripts/python.exe" ]; then PY="$cand/Scripts/python.exe"; break; fi
    if [ -x "$cand/bin/python" ]; then PY="$cand/bin/python"; break; fi
  done
fi
echo "using interpreter: $PY"

EPSILONS=(1e-1 1e-2 1e-3 1e-4)
OUT=results/stage1

for eps in "${EPSILONS[@]}"; do
  tag=$(printf '%.0e' "$eps")
  echo
  echo "################ epsilon = $eps ################"
  "$PY" -m hcsm.run_epsilon \
      --eps "$eps" \
      --data-root level_data \
      --out "$OUT/eps-${tag}" \
      --epochs 3000 \
      --seed 0
done

echo
echo "################ cross-epsilon collection ################"
"$PY" -m hcsm.collect --results "$OUT" --data-root level_data \
    --fem-csv results --out "$OUT"

echo
echo "done."
