#!/usr/bin/env bash
# One matched-budget pilot, with an archived and unmodified baseline.
# Usage: bash experiments/run_corrected_pilot.sh /path/to/python
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
PY="$("$PY" -c 'import sys; print(sys.executable)')"
BASE=be2eaee8446f04c027f6374b0bba36408b1d5179
OUT=results/corrected-pilot
SCRATCH=$(mktemp -d)
trap 'rm -rf -- "$SCRATCH"' EXIT
git archive "$BASE" | tar -x -C "$SCRATCH"
mkdir -p "$OUT/baseline"
OUT_ABS="$(cd "$OUT" && pwd)"
"$PY" -m unittest hcsm.test_corrected_pilot -v
"$PY" -m hcsm.selftest_tensor
(cd "$SCRATCH" && "$PY" -m hcsm.run_tensor --eps 1e-4 --families 96 \
  --epochs 3000 --lambda-epochs 600 --seed 0 --threads 4 --out "$OUT_ABS/baseline")
# Original range(epochs+1): 3001 updates and 601 updates per lambda probe.
"$PY" -m hcsm.run_tensor --eps 1e-4 --samples-per-family 24 \
  --max-steps 3001 --lambda-steps 601 --seed 0 --threads 4 --out "$OUT/corrected"
"$PY" -m hcsm.pilot_compare --baseline "$OUT/baseline" \
  --corrected "$OUT/corrected" --out "$OUT"
