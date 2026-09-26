# Corrected DeepONet pilot

This pilot separates correctness fixes from architecture or data improvements.
The baseline is the unmodified commit `be2eaee8446f04c027f6374b0bba36408b1d5179`.
Both runs use epsilon 1e-4, level 3 (1024 DoFs), seed 0, four CPU threads, and
the same 24 samples in each of four families for each of train/validation/test.
The trunk remains trig, latent dimension 64, width 128, depth 3, with the
scale-equivariant wrapper and learned Jacobi skip. No network enlargement,
Fourier-feature change, new training samples, or cycle-aware loss was introduced.

## Corrections

- Both trainers validate the full fixed set in bounded batches, including the
  partial last batch. Equal family counts give equal family weight. The old
  first-128 prefix would omit later families on larger datasets.
- Scheduler patience is 8 validation checks; early-stop patience is 25 checks.
  `log_every` counts optimizer updates. The legacy `TrainConfig.epochs` field
  also counts updates, not complete dataset passes. `--max-steps` states this
  explicitly. Scheduler and checkpoint improvement use absolute thresholds.
- Both trainers restore the best validation checkpoint, including lambda probes.
- `--samples-per-family` means exactly that. The old `--families` alias is
  explicitly a total count. `--epochs` is a compatibility alias for exact update
  count; the old extra inclusive update is no longer performed.
- Nonlinear learned and nonsymmetric forward-GS cycles use PyAMG FGMRES.
  The old hand-written "FCG" used an ordinary PCG recurrence. Symmetric classical
  configurations may use PCG; the common pilot comparison uses FGMRES for all.
  The final true residual is recomputed for every solve.
- Each smoother is timed separately after warmup, using the median of three
  batched applications. Residual formation is excluded. Full solver timings
  are measured separately and should not be inferred from these batch timings.
- The projected Jacobi bound is an assertion with a complement-membership check
  and numerical tolerance, not an inequality allowed to fail in principle.
- The collector's V-cycle table now maps values to the correct column labels.

## Matched budgets and interpretation

The original command's 3000 epochs actually performs 3001 updates; each
600-epoch lambda probe performs 601. The corrected pilot therefore requests
3001 and 601 updates respectively. The baseline selects the final checkpoint
at this epsilon, so using its saved checkpoint reproduces the model it evaluated.
The corrected run restores its best checkpoint at update 2800.

Validation continues to improve often enough that the corrected scheduler does
not reduce the learning rate in this pilot. The learning rate is 0.001 throughout
both runs, and both select lambda=1. The scheduler's reduction and early-stop
behavior are exercised by the focused regression tests. Full-validation coverage
also does not change this 96-sample pilot because all samples already fit below
the old 128-sample cap. Those fixes matter for larger datasets but cannot be
claimed as causes of a performance improvement here.

The common evaluator applies both checkpoints to the same independently
generated test samples. SHA-256 fingerprints identify each role's error array
and each checkpoint. Smoothing reports include full/complement norm ratios,
quantiles, worst observed values, and timings. Stationary rates use the last
six residuals of 30 cycles for each of four random right-hand sides; they are
finite-run measurements, not certified asymptotic spectral radii. FGMRES uses
one identical random right-hand side, tolerance 1e-10, no restart, and a maximum
of 500 iterations. This is one seed and one mesh, not a statistical claim.

## Reproduce

Install `requirements-deeponet.txt` in an isolated Python environment. For an
exact environment, use the saved freeze file and the PyTorch CPU wheel index
as needed. From the repository root:

```bash
python -m unittest hcsm.test_corrected_pilot -v
python -m hcsm.selftest_tensor
bash experiments/run_corrected_pilot.sh /path/to/python
```

The driver archives the original baseline commit into a temporary directory,
runs it unchanged, runs the corrected implementation, then evaluates both
checkpoints with `hcsm.pilot_compare`. Its output directory is
`results/corrected-pilot`; use a separate checkout if retaining the saved results.

To reproduce only the common evaluation of the committed checkpoints:

```bash
python -m hcsm.pilot_compare \
  --baseline results/corrected-pilot/baseline \
  --corrected results/corrected-pilot/corrected \
  --out results/corrected-pilot
```

The stored FEM matrices and transfer-check reports were reused. The C++ solver
was not recompiled and no new FEM export was performed. `selftest_tensor`
rechecks matrices/projectors and reads the existing manufactured-solution and
transfer reports; it does not rerun those C++ experiments.

See [the numerical report](../results/corrected-pilot/REPORT.md),
[raw comparison](../results/corrected-pilot/comparison.json), and
[LaTeX analysis](../results/corrected-pilot/analysis.tex).
