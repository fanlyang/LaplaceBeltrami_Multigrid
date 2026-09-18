#!/bin/bash
# Run one of the helper scripts and echo its log at the end, so a piped
# docker run still shows the output.
set -u
MODE="${1:-solver}"
shift || true

mkdir -p /work/out 2>/dev/null || true

case "$MODE" in
  solver)
    bash /work/tools/run_solver.sh "$@" > /tmp/runlog.txt 2>&1
    ;;
  diag)
    bash /work/tools/run_diag.sh "$@" > /tmp/runlog.txt 2>&1
    ;;
  meshtest)
    bash /work/tools/run_mesh_test.sh "$@" > /tmp/runlog.txt 2>&1
    ;;
  *)
    echo "unknown mode $MODE"
    exit 2
    ;;
esac

rc=$?
cat /tmp/runlog.txt
echo "wrapper_rc=$rc"
