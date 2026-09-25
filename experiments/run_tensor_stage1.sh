#!/usr/bin/env bash
# Step 3: the anisotropic Stage-I experiment, one run per epsilon.
#
# Each run draws the four error families, measures the classical smoothers on
# every family, checks the eta(e) bound, selects lambda on validation errors,
# trains the DeepONet on the penalised loss, and runs the real V-cycle on the
# exported hierarchy for both the classical and the learned smoother.
#
# Usage:  bash experiments/run_tensor_stage1.sh [PYTHON] [EPSILONS...]
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${1:-python}"
shift || true
if [ "$PY" = "python" ]; then
  COMMON="$(git rev-parse --git-common-dir 2>/dev/null || echo .)"
  MAIN="$(cd "$COMMON/.." 2>/dev/null && pwd || echo .)"
  for cand in "../.venv-deeponet" ".venv-deeponet" "$MAIN/.venv-deeponet"; do
    if [ -x "$cand/Scripts/python.exe" ]; then PY="$cand/Scripts/python.exe"; break; fi
    if [ -x "$cand/bin/python" ]; then PY="$cand/bin/python"; break; fi
  done
fi
echo "using interpreter: $PY"

if [ "$#" -gt 0 ]; then
  EPSILONS=("$@")
else
  EPSILONS=(1 1e-1 1e-2 1e-3 1e-4)
fi

OUT=results/tensor
for eps in "${EPSILONS[@]}"; do
  tag=$(printf '%.0e' "$eps")
  echo
  echo "################ epsilon = $eps ################"
  "$PY" -m hcsm.run_tensor \
      --eps "$eps" \
      --data-root level_data \
      --out "$OUT/eps-${tag}" \
      --families 96 \
      --epochs 3000 \
      --lambda-epochs 600 \
      --seed 0
done

echo
echo "################ cross-epsilon tables ################"
"$PY" -m hcsm.collect_tensor --results "$OUT" --out "$OUT"
echo "done."
