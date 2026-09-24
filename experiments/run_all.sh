#!/bin/bash
# The exact experiment matrix behind results/. Run from the repository root.
#
#   bash experiments/run_all.sh quick     # tiny smoke configuration (~1 min)
#   bash experiments/run_all.sh check     # numerical self-test only
#   bash experiments/run_all.sh           # full matrix
#
# Prerequisites
#   - level_data/L0..L5 exist (see DEEPONET_SMOOTHER.md section 3)
#   - the environment from requirements-deeponet.txt is active, or set PY=<python>
#
# Objective
#   Every run below trains on  L = L_CGC + lambda * L_E  (--loss-mode cgc, the
#   default), where L_CGC is the A-energy left after one smoothing step AND one
#   coarse-grid correction.  The coarse operator inside the loss is the assembled
#   one, i.e. the operator the C++ V-cycle uses (--coarse-operator assembled, the
#   default); --coarse-operator galerkin is the alternative in which the corrector
#   is an exact A-orthogonal projection, and metrics.json reports both so the
#   difference is visible.  Runs from before this change trained on L_E alone
#   ("L = alpha*coarse + beta*energy" in the prototype, or L_E + w*coarse in an
#   intermediate release) and are NOT directly comparable on the loss column.
#
# lambda
#   lambda is an experimental weight, not a derived optimum.  Step 4 runs the same
#   design at lambda = 0.0 and at LAMBDA, and the comparison is what selects the
#   value used for the scaling runs in step 5.  Override with LAMBDA=<value>.
#
# Comparability: the test set is fixed by (seed=0, n_test=50) and does not depend
# on the model, so every configuration below is scored on exactly the same 50
# test errors, 10 per mechanism. n_val=256 throughout so validation curves are
# comparable too. Training and inference timings are only meaningful if the runs
# are not competing for CPU -- the matrix is sized for a 16-core machine at
# --threads 4; run them one at a time on a smaller box.
set -e
PY=${PY:-python}
LAMBDA=${LAMBDA:-0.1}
COMMON="--data-dir level_data/L3 --n-val 256 --n-test 50 --vcycle"
OBJ="--loss-mode cgc"

case "$1" in
  check)
    "$PY" tools/selftest.py --hierarchy-root level_data
    exit 0
    ;;
  quick)
    "$PY" deeponet_smoother.py $COMMON $OBJ --out-dir results/smoke --n-train 64 \
        --n-val 32 --n-test 20 --epochs 200 --log-every 50 --p 128 \
        --trunk-features trig
    exit 0
    ;;
esac

# 1. The architecture as literally specified: raw 3D coordinates as trunk input,
#    modest trunk width. Expected to be weak for two structural reasons -- the
#    output is rank-limited to p, and the coordinates give the trunk no
#    high-spatial-frequency basis. This run quantifies both.
"$PY" deeponet_smoother.py $COMMON $OBJ --out-dir results/spec_xyz_p64 \
    --n-train 1024 --epochs 4000 --log-every 1000 --p 64 --trunk-features xyz

# 2. Rank bound in isolation: same smooth trunk, full trunk width p = n_dof.
"$PY" deeponet_smoother.py $COMMON $OBJ --out-dir results/trunk_trig_p1024 \
    --n-train 1024 --epochs 4000 --log-every 1000 --p 1024 --trunk-features trig

# 3. Frequency content of the trunk basis: Fourier features out to Nyquist.
"$PY" deeponet_smoother.py $COMMON $OBJ --out-dir results/trunk_fourier16 \
    --n-train 1024 --epochs 6000 --log-every 1500 --p 1024 \
    --trunk-features fourier --trunk-modes 16

# 4. The increment over a classical step:
#        delta_e = omega D^{-1} r + B_theta(r)
#    The network then only supplies what damped Jacobi leaves behind, which is
#    smooth -- so a small smooth trunk is reasonable again, and the run answers
#    "how much does the learned part actually add over Jacobi?".
#    Run twice: once with the coarse-grid term switched off entirely (lambda = 0,
#    a pure two-grid objective) and once with the auxiliary energy term at LAMBDA.
#    Their metrics.json carry both terms separately, so the comparison is made on
#    L_CGC and on the measured V-cycle, not on the total alone.
"$PY" deeponet_smoother.py $COMMON $OBJ --out-dir results/L3_cgc_l0.0 \
    --n-train 1024 --epochs 6000 --log-every 1500 --p 128 \
    --trunk-features trig --base-smoother jacobi --lambda-energy 0.0

"$PY" deeponet_smoother.py $COMMON $OBJ --out-dir results/L3_cgc_l${LAMBDA} \
    --n-train 1024 --epochs 6000 --log-every 1500 --p 128 \
    --trunk-features trig --base-smoother jacobi --lambda-energy ${LAMBDA}

# 5. Scaling: the same design at level 4 (4096 DoFs) and level 5 (16384 DoFs), at
#    the lambda the step-4 comparison selected.  p does not grow with n here on
#    purpose -- that is the variable under test, and it is what the level-4 result
#    turns on.
"$PY" deeponet_smoother.py $OBJ --data-dir level_data/L4 --out-dir results/L4_cgc_l${LAMBDA} \
    --n-train 512 --n-val 128 --n-test 50 --batch-size 16 --epochs 4000 \
    --log-every 1000 --p 256 --trunk-features trig --base-smoother jacobi --vcycle \
    --lambda-energy ${LAMBDA}

"$PY" deeponet_smoother.py $OBJ --data-dir level_data/L5 --out-dir results/L5_cgc_l${LAMBDA} \
    --n-train 256 --n-val 128 --n-test 50 --batch-size 8 --epochs 3000 \
    --log-every 750 --p 256 --trunk-features trig --base-smoother jacobi --vcycle \
    --lambda-energy ${LAMBDA}

"$PY" tools/summarize_runs.py --glob 'results/*/metrics.json' \
    --csv results/summary.csv | tee results/summary.md
