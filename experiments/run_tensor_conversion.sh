#!/usr/bin/env bash
# Step 2: raw solver dumps -> the NumPy level problems the experiment reads.
#
# For each epsilon this writes level_data/tensor-eps-<E>-L0 .. L3. Level 3 is
# the training level (1024 DoFs on a 32x32 lattice); L0..L3 together are the
# full hierarchy the V-cycle runs on, coarsest first.
#
# Each directory also gets coarse_matrix.npz, the INDEPENDENTLY REDISCRETIZED
# operator of the level below. The Galerkin diagnostic Q_epsilon uses
# P^T A P, while the solver's V-cycle uses these rediscretized operators; the
# two differ on a curved surface and the difference is recorded rather than
# assumed away.
#
# Usage:  bash experiments/run_tensor_conversion.sh [PYTHON]
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

CYCLE=5
EPSILONS=(1 1e-1 1e-2 1e-3 1e-4)
LEVELS=(0 1 2 3)

for eps in "${EPSILONS[@]}"; do
  tag=$(printf '%.0e' "$eps")
  for L in "${LEVELS[@]}"; do
    out="level_data/tensor-eps-${tag}-L${L}"
    "$PY" tools/convert_level_data.py \
        --data-dir "level_dumps/eps-${tag}" \
        --level "$L" --cycle "$CYCLE" \
        --also-coarse \
        --out-dir "$out" > /dev/null
  done
  echo "  epsilon $eps -> level_data/tensor-eps-${tag}-L0..L3"
done

echo
echo "== converted =="
ls -1 level_data/ | head -30
