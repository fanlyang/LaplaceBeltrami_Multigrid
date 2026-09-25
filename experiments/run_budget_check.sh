#!/usr/bin/env bash
# Step 4 (optional): is the plain MLP's showing a training-budget artefact?
#
# The main run uses the existing framework's default budget (3000 epochs). If a
# DeepONet variant is still descending at the last epoch, "it is worse than
# damped Jacobi" could be an under-training artefact rather than a property of
# the architecture. This script answers that directly by re-running ONE
# architecture at four times the budget, at every epsilon, into a separate
# directory so the primary results are untouched.
#
# The architecture checked is deliberately the STRONGEST variant, not the
# weakest: the fairness question is whether the best configuration the
# architecture can offer, given four times the training, closes the gap. A
# budget check on a variant nobody would deploy answers nothing.
#
# It is a bound, not a method result: nothing in the main comparison uses it.
#
# Usage:  bash experiments/run_budget_check.sh [PYTHON] [ARCH]
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${1:-python}"
ARCH="${2:-skip_fourier}"
if [ "$PY" = "python" ]; then
  COMMON="$(git rev-parse --git-common-dir 2>/dev/null || echo .)"
  MAIN="$(cd "$COMMON/.." 2>/dev/null && pwd || echo .)"
  for cand in "../.venv-deeponet" ".venv-deeponet" "$MAIN/.venv-deeponet"; do
    if [ -x "$cand/Scripts/python.exe" ]; then PY="$cand/Scripts/python.exe"; break; fi
    if [ -x "$cand/bin/python" ]; then PY="$cand/bin/python"; break; fi
  done
fi

EPSILONS=(1e-1 1e-2 1e-3 1e-4)
OUT=results/stage1_budget

for eps in "${EPSILONS[@]}"; do
  tag=$(printf '%.0e' "$eps")
  echo
  echo "######## DeepONet[$ARCH] at 4x budget, epsilon = $eps ########"
  "$PY" -m hcsm.run_epsilon \
      --eps "$eps" \
      --data-root level_data \
      --out "$OUT/eps-${tag}" \
      --architectures "$ARCH" \
      --epochs 12000 \
      --seed 0
done

echo
echo "done. Compare against results/stage1/eps-*/metrics.json:"
"$PY" -m hcsm.compare_budget --main results/stage1 --budget "$OUT" --arch "$ARCH"
