# Run comparison

| run | trunk | p | params | log pts | best epoch | val energy | train s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L4_skip_jacobi_trig256 | trig+jacobi | 256 | 656513 | 5 | 2000 | 0.30194 | 1151 |
| L5_skip_jacobi_trig256 | trig+jacobi | 256 | 2229377 | 5 | 3000 | 0.29376 | 1398 |
| skip_jacobi_trig128 | trig+jacobi | 128 | 230401 | 5 | 6000 | 0.19446 | 880 |
| spec_xyz_p64 | xyz | 64 | 213824 | 5 | 4000 | 0.86143 | 349 |
| trunk_fourier16 | fourier16 | 1024 | 738688 | 5 | 6000 | 0.76064 | 1561 |
| trunk_trig_p1024 | trig | 1024 | 460672 | 5 | 4000 | 0.86209 | 937 |

## Mean test energy ratio  ||e - de||_A / ||e||_A  (lower is better)

| run | ALL | smooth | multiscale | localized | algebraic |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4_skip_jacobi_trig256 | 0.6409 | 0.9394 | 0.4543 | 0.8427 | 0.4249 |
| L5_skip_jacobi_trig256 | 0.6616 | 0.9944 | 0.4397 | 0.8451 | 0.4153 |
| skip_jacobi_trig128 | 0.3856 | 0.1968 | 0.3969 | 0.5019 | 0.4305 |
| spec_xyz_p64 | 0.7765 | 0.3679 | 0.9729 | 0.6685 | 1.0150 |
| trunk_fourier16 | 0.7616 | 0.4739 | 0.8786 | 0.6877 | 0.9369 |
| trunk_trig_p1024 | 0.7655 | 0.2952 | 0.9683 | 0.6940 | 1.0108 |

## Per mechanism, best classical reference for the same runs

| run | mechanism | DeepONet | damped Jacobi | Gauss-Seidel | sym. Gauss-Seidel | SSOR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L4_skip_jacobi_trig256 | ALL | 0.6409 | 0.6433 | 0.5878 | 0.4476 | 0.4476 |
| L4_skip_jacobi_trig256 | smooth | 0.9394 | 0.9825 | 0.9567 | 0.9169 | 0.9169 |
| L4_skip_jacobi_trig256 | multiscale | 0.4543 | 0.4426 | 0.4017 | 0.2291 | 0.2291 |
| L4_skip_jacobi_trig256 | localized | 0.8427 | 0.8471 | 0.7525 | 0.6178 | 0.6178 |
| L4_skip_jacobi_trig256 | algebraic | 0.4249 | 0.4066 | 0.3453 | 0.1647 | 0.1647 |
| L5_skip_jacobi_trig256 | ALL | 0.6616 | 0.6615 | 0.6061 | 0.4699 | 0.4699 |
| L5_skip_jacobi_trig256 | smooth | 0.9944 | 0.9956 | 0.9883 | 0.9770 | 0.9770 |
| L5_skip_jacobi_trig256 | multiscale | 0.4397 | 0.4412 | 0.4061 | 0.2293 | 0.2293 |
| L5_skip_jacobi_trig256 | localized | 0.8451 | 0.8473 | 0.7535 | 0.6138 | 0.6138 |
| L5_skip_jacobi_trig256 | algebraic | 0.4153 | 0.4099 | 0.3490 | 0.1641 | 0.1641 |
| skip_jacobi_trig128 | ALL | 0.3856 | 0.6433 | 0.5729 | 0.4297 | 0.4297 |
| skip_jacobi_trig128 | smooth | 0.1968 | 0.9320 | 0.8563 | 0.7435 | 0.7435 |
| skip_jacobi_trig128 | multiscale | 0.3969 | 0.4375 | 0.3889 | 0.2237 | 0.2237 |
| skip_jacobi_trig128 | localized | 0.5019 | 0.8465 | 0.7533 | 0.6242 | 0.6242 |
| skip_jacobi_trig128 | algebraic | 0.4305 | 0.4103 | 0.3368 | 0.1642 | 0.1642 |
| spec_xyz_p64 | ALL | 0.7765 | 0.6433 | 0.5729 | 0.4297 | 0.4297 |
| spec_xyz_p64 | smooth | 0.3679 | 0.9320 | 0.8563 | 0.7435 | 0.7435 |
| spec_xyz_p64 | multiscale | 0.9729 | 0.4375 | 0.3889 | 0.2237 | 0.2237 |
| spec_xyz_p64 | localized | 0.6685 | 0.8465 | 0.7533 | 0.6242 | 0.6242 |
| spec_xyz_p64 | algebraic | 1.0150 | 0.4103 | 0.3368 | 0.1642 | 0.1642 |
| trunk_fourier16 | ALL | 0.7616 | 0.6433 | 0.5729 | 0.4297 | 0.4297 |
| trunk_fourier16 | smooth | 0.4739 | 0.9320 | 0.8563 | 0.7435 | 0.7435 |
| trunk_fourier16 | multiscale | 0.8786 | 0.4375 | 0.3889 | 0.2237 | 0.2237 |
| trunk_fourier16 | localized | 0.6877 | 0.8465 | 0.7533 | 0.6242 | 0.6242 |
| trunk_fourier16 | algebraic | 0.9369 | 0.4103 | 0.3368 | 0.1642 | 0.1642 |
| trunk_trig_p1024 | ALL | 0.7655 | 0.6433 | 0.5729 | 0.4297 | 0.4297 |
| trunk_trig_p1024 | smooth | 0.2952 | 0.9320 | 0.8563 | 0.7435 | 0.7435 |
| trunk_trig_p1024 | multiscale | 0.9683 | 0.4375 | 0.3889 | 0.2237 | 0.2237 |
| trunk_trig_p1024 | localized | 0.6940 | 0.8465 | 0.7533 | 0.6242 | 0.6242 |
| trunk_trig_p1024 | algebraic | 1.0108 | 0.4103 | 0.3368 | 0.1642 | 0.1642 |

## Two-grid / V-cycle (real P and coarse operator)

| run | smoother | smoother only | coarse only | two-grid |
| --- | --- | ---: | ---: | ---: |
| L4_skip_jacobi_trig256 | DeepONet | 0.6409 | 0.5616 | 0.2491 |
| L4_skip_jacobi_trig256 | damped Jacobi | 0.6433 | 0.5616 | 0.2491 |
| L4_skip_jacobi_trig256 | sym. Gauss-Seidel | 0.4476 | 0.5616 | 0.1372 |
| L5_skip_jacobi_trig256 | DeepONet | 0.6616 | 0.5465 | 0.2452 |
| L5_skip_jacobi_trig256 | damped Jacobi | 0.6615 | 0.5465 | 0.2442 |
| L5_skip_jacobi_trig256 | sym. Gauss-Seidel | 0.4699 | 0.5465 | 0.1267 |
| skip_jacobi_trig128 | DeepONet | 0.3856 | 0.5812 | 0.2227 |
| skip_jacobi_trig128 | damped Jacobi | 0.6433 | 0.5812 | 0.2588 |
| skip_jacobi_trig128 | sym. Gauss-Seidel | 0.4297 | 0.5812 | 0.1507 |
| spec_xyz_p64 | DeepONet | 0.7765 | 0.5812 | 0.5535 |
| spec_xyz_p64 | damped Jacobi | 0.6433 | 0.5812 | 0.2588 |
| spec_xyz_p64 | sym. Gauss-Seidel | 0.4297 | 0.5812 | 0.1507 |
| trunk_fourier16 | DeepONet | 0.7616 | 0.5812 | 0.5857 |
| trunk_fourier16 | damped Jacobi | 0.6433 | 0.5812 | 0.2588 |
| trunk_fourier16 | sym. Gauss-Seidel | 0.4297 | 0.5812 | 0.1507 |
| trunk_trig_p1024 | DeepONet | 0.7655 | 0.5812 | 0.5488 |
| trunk_trig_p1024 | damped Jacobi | 0.6433 | 0.5812 | 0.2588 |
| trunk_trig_p1024 | sym. Gauss-Seidel | 0.4297 | 0.5812 | 0.1507 |
| L4_skip_jacobi_trig256 | V-cycle learned@L4 + Jacobi below | - | - | 0.1755 |
| L4_skip_jacobi_trig256 | V-cycle damped Jacobi everywhere | - | - | 0.1751 |
| L5_skip_jacobi_trig256 | V-cycle learned@L5 + Jacobi below | - | - | 0.1751 |
| L5_skip_jacobi_trig256 | V-cycle damped Jacobi everywhere | - | - | 0.1743 |
| skip_jacobi_trig128 | V-cycle learned@L3 + Jacobi below | - | - | 0.1084 |
| skip_jacobi_trig128 | V-cycle damped Jacobi everywhere | - | - | 0.1753 |
| spec_xyz_p64 | V-cycle learned@L3 + Jacobi below | - | - | 0.4283 |
| spec_xyz_p64 | V-cycle damped Jacobi everywhere | - | - | 0.1753 |
| trunk_fourier16 | V-cycle learned@L3 + Jacobi below | - | - | 0.4719 |
| trunk_fourier16 | V-cycle damped Jacobi everywhere | - | - | 0.1753 |
| trunk_trig_p1024 | V-cycle learned@L3 + Jacobi below | - | - | 0.4218 |
| trunk_trig_p1024 | V-cycle damped Jacobi everywhere | - | - | 0.1753 |

## Hierarchy Galerkin check (P^T A P vs the assembled coarse operator)

- L4_skip_jacobi_trig256: relative max|P^T A P - A_H| = 1.899e-03
- L5_skip_jacobi_trig256: relative max|P^T A P - A_H| = 4.660e-04
- skip_jacobi_trig128: relative max|P^T A P - A_H| = 8.326e-03
- spec_xyz_p64: relative max|P^T A P - A_H| = 8.326e-03
- trunk_fourier16: relative max|P^T A P - A_H| = 8.326e-03
- trunk_trig_p1024: relative max|P^T A P - A_H| = 8.326e-03

## Split audit (cross-split signature overlaps; 0 required)

- L4_skip_jacobi_trig256: 0
- L5_skip_jacobi_trig256: 0
- skip_jacobi_trig128: 0
- spec_xyz_p64: 0
- trunk_fourier16: 0
- trunk_trig_p1024: 0
Traceback (most recent call last):
  File "D:\projects\MS\.claude\worktrees\deeponet-smoother\tools\summarize_runs.py", line 154, in <module>
    main()
  File "D:\projects\MS\.claude\worktrees\deeponet-smoother\tools\summarize_runs.py", line 142, in main
    row = [r["run"], r["trunk"], r["p"], r["params"], r["epochs"],
KeyError: 'epochs'
