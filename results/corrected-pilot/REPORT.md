# Corrected anisotropic DeepONet pilot

At epsilon = 1e-4 on the 1024-DoF level, correcting training and evaluation does not produce a meaningful improvement over the original learned smoother. Symmetric GS remains better on every tested family. This is a one-seed pilot, not evidence of statistical superiority.

## Common evaluation

Every checkpoint is evaluated on the same 24 held-out errors per family. Stress contains 23 accepted analytic modes. Values below are mean complement **norm** ratios, not squared-energy ratios. Lower is better.

| Family | Jacobi | Forward GS | Symmetric GS | DeepONet baseline | DeepONet corrected |
|---|---:|---:|---:|---:|---:|
| weak | 0.960947 | 0.915977 | 0.849540 | 0.951411 | 0.951502 |
| gaussian | 0.488299 | 0.361784 | 0.231061 | 0.392723 | 0.392480 |
| fourier | 0.425380 | 0.337950 | 0.221992 | 0.428012 | 0.426631 |
| survivor | 0.840555 | 0.661020 | 0.541244 | 0.680234 | 0.682120 |
| stress | 0.771879 | 0.664229 | 0.588638 | 0.638207 | 0.638332 |

Full-error ratios, p95/p99, worst observed reductions, amplification fractions, and separately measured times are in [comparison.csv](comparison.csv).

| Method | Stationary cycle rate | FGMRES iterations | True relative residual | Outer solve seconds |
|---|---:|---:|---:|---:|
| jacobi | 0.973256 | 103 | 9.056e-11 | 0.227 |
| gs | 0.947978 | 104 | 8.421e-11 | 0.955 |
| sgs | 0.922221 | 44 | 7.336e-11 | 0.662 |
| baseline | 0.978405 | 115 | 9.711e-11 | 2.578 |
| corrected | 0.978322 | 114 | 9.983e-11 | 3.489 |

All common outer solves converged at relative residual tolerance 1e-10 using the same PyAMG FGMRES algorithm and right-hand side. The baseline originally used a mislabeled PCG recurrence; its historical 124 iterations must not be compared with the corrected 114 as a training gain. On the common FGMRES evaluator the comparison is 115 versus 114. Timings are local single-run observations, not statistically robust performance benchmarks.

## What the corrections changed

- Baseline source: `be2eaee8446f04c027f6374b0bba36408b1d5179`. The rerun reproduced the committed epsilon=1e-4 training-loss history exactly (excluding wall-clock times).
- Both runs: 3001 optimizer updates, batch size 32, 96 fixed training errors, 96 validation errors, same seed and architecture. Each of four lambda probes used 601 updates; both selected lambda=1.
- The corrected model restores its best checkpoint at update 2800. The baseline selected its final weights, labeled epoch 3000 (update 3001).
- Learning rate stayed 0.001 in both runs because validation improvements prevented a plateau long enough to trigger the corrected schedule. Regression tests explicitly verify LR decay and early stopping on a controlled plateau.
- The complete-validation fix prevents family exclusion when datasets exceed 128 samples; it changes no validation coverage in this 96-sample pilot.
- No larger dataset, Fourier trunk, larger network, additional training budget, or cycle-aware fine-tuning was introduced. Tiny mixed changes here do not measure those proposed improvements.

## Verification and limits

- Nine focused regression tests passed: complete/partial-batch validation, scheduler and early-stop units, checkpoint restoration, exact update count, projected Jacobi bound, unchanged correction semantics under timing, nonlinear FGMRES, forward-GS solver routing, and residual/scaling behavior (some tests cover multiple checks).
- The existing tensor self-test passed all 33 checks against the stored artifacts. The C++ solver was not rebuilt; manufactured-solution and transfer reports were read, not regenerated.
- The Galerkin projector is used for complement diagnostics; the actual cycles retain independently rediscretized level operators.
- Stationary rates summarize the last six residuals of 30 cycles over four right-hand sides. They are not certified asymptotic contraction factors.
- The common FGMRES comparison uses one right-hand side. Multiple seeds, meshes, and right-hand sides are needed for general performance claims.
- Historical README stress numbers differ from its stored metrics. This report uses the stored/recomputed arrays; it does not copy the historical prose.

## Reproduction

See [the protocol](../../docs/CORRECTED_PILOT.md), [comparison.json](comparison.json), the baseline/corrected checkpoints and histories, and [environment-freeze.txt](environment-freeze.txt). The comparison records SHA-256 fingerprints for the error sets and model files.

Next experiment: increase independent error coverage after the validation fix, then test Fourier features or cycle-aware fine-tuning one change at a time. The present pilot does not support claiming DeepONet beats symmetric GS.
