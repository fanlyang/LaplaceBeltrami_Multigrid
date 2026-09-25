#!/usr/bin/env bash
# Step 2: turn the raw solver dumps into the NumPy files the experiment reads.
#
# For each epsilon this writes level_data/eps-<E>-L3/, holding the level-3
# operator (1024 DoFs on a 32x32 periodic DoF lattice), the physical support
# points of those DoFs, kappa_epsilon at those points, the prolongation from
# level 2 to level 3, and -- for the record only -- the independently assembled
# level-2 operator.
#
# Level 3 and cycle 5 are used together on purpose. A level matrix is assembled
# on that level's own mesh, and uniform refinement never changes a coarser
# level's mesh, so A-level-3-cycle-5.coo is byte-identical to
# A-level-3-cycle-3.coo; the cycle only decides how deep the hierarchy goes.
# Verified by md5sum, not assumed.
#
# Usage:  bash experiments/run_conversion.sh [PYTHON]
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${1:-python}"
LEVEL=3          # 1024 DoFs -- the fixed fine level of the whole experiment
CYCLE=5
EPSILONS=(1e-1 1e-2 1e-3 1e-4)

# Find an interpreter that has numpy and scipy. This checkout may be a git
# worktree, in which case the venv lives next to the *main* checkout, not here --
# so ask git where the common .git directory is and look beside it.
if [ "$PY" = "python" ]; then
  COMMON="$(git rev-parse --git-common-dir 2>/dev/null || echo .)"
  MAIN="$(cd "$COMMON/.." 2>/dev/null && pwd || echo .)"
  for cand in "../.venv-deeponet" ".venv-deeponet" "$MAIN/.venv-deeponet"; do
    if [ -x "$cand/Scripts/python.exe" ]; then PY="$cand/Scripts/python.exe"; break; fi
    if [ -x "$cand/bin/python" ]; then PY="$cand/bin/python"; break; fi
  done
fi
echo "using interpreter: $PY"
"$PY" -c "import numpy, scipy; print('  numpy', numpy.__version__, '/ scipy', scipy.__version__)"

for eps in "${EPSILONS[@]}"; do
  tag=$(printf '%.0e' "$eps")
  out="level_data/eps-${tag}-L${LEVEL}"
  echo "== epsilon $eps -> $out =="
  "$PY" tools/convert_level_data.py \
      --data-dir "level_dumps/eps-${eps}" \
      --level "$LEVEL" --cycle "$CYCLE" \
      --also-coarse \
      --out-dir "$out"
done

echo
echo "== md5 check that the level-3 operator really is cycle-independent =="
for eps in "${EPSILONS[@]}"; do
  a="level_dumps/eps-${eps}/A-level-${LEVEL}-cycle-3.coo"
  b="level_dumps/eps-${eps}/A-level-${LEVEL}-cycle-${CYCLE}.coo"
  if [ -f "$a" ] && [ -f "$b" ]; then
    printf '  %-6s %s\n' "$eps" "$(cmp -s "$a" "$b" && echo identical || echo DIFFERENT)"
  fi
done

echo
echo "== converted level data =="
ls -1 level_data/
