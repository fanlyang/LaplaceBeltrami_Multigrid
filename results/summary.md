# Run comparison

| run | trunk | p | params | objective | log pts | best epoch | val energy | train s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| L3_cgc_l0.0 | trig+jacobi | 128 | 230401 | L_CGC + 0 * L_E | 5 | 6000 | 1.48193 | 272 |
| L3_cgc_l0.1 | trig+jacobi | 128 | 230401 | L_CGC + 0.1 * L_E | 5 | 6000 | 0.21449 | 268 |
| L4_cgc_l0.1 | trig+jacobi | 256 | 656513 | L_CGC + 0.1 * L_E | 5 | 0 | 0.30239 | 596 |
| L4_skip_jacobi_trig256 | trig+jacobi | 256 | 656513 | - | 5 | 2000 | 0.30194 | 1151 |
| L5_cgc_l0.1 | trig+jacobi | 256 | 2229377 | L_CGC + 0.1 * L_E | 5 | 1500 | 0.29743 | 1830 |
| L5_skip_jacobi_trig256 | trig+jacobi | 256 | 2229377 | - | 5 | 3000 | 0.29376 | 1398 |
| skip_jacobi_trig128 | trig+jacobi | 128 | 230401 | - | 5 | 6000 | 0.19446 | 880 |
| spec_xyz_p64 | xyz | 64 | 213824 | - | 5 | 4000 | 0.86143 | 349 |
| trunk_fourier16 | fourier16 | 1024 | 738688 | - | 5 | 6000 | 0.76064 | 1561 |
| trunk_trig_p1024 | trig | 1024 | 460672 | - | 5 | 4000 | 0.86209 | 937 |

## Mean test energy ratio  ||e - de||_A / ||e||_A  (lower is better)

| run | ALL | smooth | multiscale | localized | algebraic |
| --- | ---: | ---: | ---: | ---: | ---: |
| L3_cgc_l0.0 | 0.9024 | 0.5105 | 0.6298 | 0.5754 | 1.5681 |
| L3_cgc_l0.1 | 0.3849 | 0.1940 | 0.4097 | 0.4316 | 0.4717 |
| L4_cgc_l0.1 | 0.6484 | 0.9835 | 0.4560 | 0.8542 | 0.4051 |
| L4_skip_jacobi_trig256 | 0.6409 | 0.9394 | 0.4543 | 0.8427 | 0.4249 |
| L5_cgc_l0.1 | 0.6631 | 0.9948 | 0.4491 | 0.8546 | 0.4004 |
| L5_skip_jacobi_trig256 | 0.6616 | 0.9944 | 0.4397 | 0.8451 | 0.4153 |
| skip_jacobi_trig128 | 0.3856 | 0.1968 | 0.3969 | 0.5019 | 0.4305 |
| spec_xyz_p64 | 0.7765 | 0.3679 | 0.9729 | 0.6685 | 1.0150 |
| trunk_fourier16 | 0.7616 | 0.4739 | 0.8786 | 0.6877 | 0.9369 |
| trunk_trig_p1024 | 0.7655 | 0.2952 | 0.9683 | 0.6940 | 1.0108 |

## Test-set objective: L_E and L_CGC (energies; lower is better)

| run | L_E | L_CGC assembled | L_CGC galerkin | coarse op in loss |
| --- | ---: | ---: | ---: | --- |
| L3_cgc_l0.0 | 1.06913 | 0.04942 | 0.04929 | assembled |
| L3_cgc_l0.1 | 0.16603 | 0.05821 | 0.05820 | assembled |
| L4_cgc_l0.1 | 0.48313 | 0.07136 | 0.07136 | assembled |
| L4_skip_jacobi_trig256 | - | - | - | - |
| L5_cgc_l0.1 | 0.50244 | 0.07211 | 0.07211 | assembled |
| L5_skip_jacobi_trig256 | - | - | - | - |
| skip_jacobi_trig128 | - | - | - | - |
| spec_xyz_p64 | - | - | - | - |
| trunk_fourier16 | - | - | - | - |
| trunk_trig_p1024 | - | - | - | - |

## Per mechanism, best classical reference for the same runs

| run | mechanism | DeepONet | damped Jacobi | Gauss-Seidel | sym. Gauss-Seidel | SSOR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L3_cgc_l0.0 | ALL | 0.9024 | 0.6433 | 0.5729 | 0.4297 | 0.4297 |
| L3_cgc_l0.0 | smooth | 0.5105 | 0.9320 | 0.8563 | 0.7435 | 0.7435 |
| L3_cgc_l0.0 | multiscale | 0.6298 | 0.4375 | 0.3889 | 0.2237 | 0.2237 |
| L3_cgc_l0.0 | localized | 0.5754 | 0.8465 | 0.7533 | 0.6242 | 0.6242 |
| L3_cgc_l0.0 | algebraic | 1.5681 | 0.4103 | 0.3368 | 0.1642 | 0.1642 |
| L3_cgc_l0.1 | ALL | 0.3849 | 0.6433 | 0.5729 | 0.4297 | 0.4297 |
| L3_cgc_l0.1 | smooth | 0.1940 | 0.9320 | 0.8563 | 0.7435 | 0.7435 |
| L3_cgc_l0.1 | multiscale | 0.4097 | 0.4375 | 0.3889 | 0.2237 | 0.2237 |
| L3_cgc_l0.1 | localized | 0.4316 | 0.8465 | 0.7533 | 0.6242 | 0.6242 |
| L3_cgc_l0.1 | algebraic | 0.4717 | 0.4103 | 0.3368 | 0.1642 | 0.1642 |
| L4_cgc_l0.1 | ALL | 0.6484 | 0.6433 | 0.5878 | 0.4476 | 0.4476 |
| L4_cgc_l0.1 | smooth | 0.9835 | 0.9825 | 0.9567 | 0.9169 | 0.9169 |
| L4_cgc_l0.1 | multiscale | 0.4560 | 0.4426 | 0.4017 | 0.2291 | 0.2291 |
| L4_cgc_l0.1 | localized | 0.8542 | 0.8471 | 0.7525 | 0.6178 | 0.6178 |
| L4_cgc_l0.1 | algebraic | 0.4051 | 0.4066 | 0.3453 | 0.1647 | 0.1647 |
| L4_skip_jacobi_trig256 | ALL | 0.6409 | 0.6433 | 0.5878 | 0.4476 | 0.4476 |
| L4_skip_jacobi_trig256 | smooth | 0.9394 | 0.9825 | 0.9567 | 0.9169 | 0.9169 |
| L4_skip_jacobi_trig256 | multiscale | 0.4543 | 0.4426 | 0.4017 | 0.2291 | 0.2291 |
| L4_skip_jacobi_trig256 | localized | 0.8427 | 0.8471 | 0.7525 | 0.6178 | 0.6178 |
| L4_skip_jacobi_trig256 | algebraic | 0.4249 | 0.4066 | 0.3453 | 0.1647 | 0.1647 |
| L5_cgc_l0.1 | ALL | 0.6631 | 0.6615 | 0.6061 | 0.4699 | 0.4699 |
| L5_cgc_l0.1 | smooth | 0.9948 | 0.9956 | 0.9883 | 0.9770 | 0.9770 |
| L5_cgc_l0.1 | multiscale | 0.4491 | 0.4412 | 0.4061 | 0.2293 | 0.2293 |
| L5_cgc_l0.1 | localized | 0.8546 | 0.8473 | 0.7535 | 0.6138 | 0.6138 |
| L5_cgc_l0.1 | algebraic | 0.4004 | 0.4099 | 0.3490 | 0.1641 | 0.1641 |
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

## Worst case per mechanism (a mean hides the samples a smoother worsens)

| run | mechanism | worst ratio | fraction amplified |
| --- | --- | ---: | ---: |
| L3_cgc_l0.0 | ALL | 1.9779 | 0.3400 |
| L3_cgc_l0.0 | smooth | 0.9788 | 0.0000 |
| L3_cgc_l0.0 | multiscale | 0.9085 | 0.0000 |
| L3_cgc_l0.0 | localized | 0.9412 | 0.0000 |
| L3_cgc_l0.0 | algebraic | 1.9779 | 1.0000 |
| L3_cgc_l0.1 | ALL | 0.6707 | 0.0000 |
| L3_cgc_l0.1 | smooth | 0.3697 | 0.0000 |
| L3_cgc_l0.1 | multiscale | 0.6234 | 0.0000 |
| L3_cgc_l0.1 | localized | 0.6707 | 0.0000 |
| L3_cgc_l0.1 | algebraic | 0.5369 | 0.0000 |
| L4_cgc_l0.1 | ALL | 0.9983 | 0.0000 |
| L4_cgc_l0.1 | smooth | 0.9983 | 0.0000 |
| L4_cgc_l0.1 | multiscale | 0.8162 | 0.0000 |
| L4_cgc_l0.1 | localized | 0.9345 | 0.0000 |
| L4_cgc_l0.1 | algebraic | 0.4280 | 0.0000 |
| L5_cgc_l0.1 | ALL | 0.9991 | 0.0000 |
| L5_cgc_l0.1 | smooth | 0.9991 | 0.0000 |
| L5_cgc_l0.1 | multiscale | 0.8003 | 0.0000 |
| L5_cgc_l0.1 | localized | 0.9345 | 0.0000 |
| L5_cgc_l0.1 | algebraic | 0.4040 | 0.0000 |

## Repeated application  e <- e - B(A e):  mean energy ratio

| run | it. 1 | it. 2 | it. 3 | it. 4 | it. 5 | monotone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| L3_cgc_l0.0 | 0.9024 | 0.6446 | 0.6013 | 0.5153 | 0.4786 | True |
| L3_cgc_l0.1 | 0.3849 | 0.2367 | 0.1722 | 0.1346 | 0.1101 | True |
| L4_cgc_l0.1 | 0.6484 | 0.5359 | 0.4801 | 0.4442 | 0.4194 | True |
| L5_cgc_l0.1 | 0.6631 | 0.5516 | 0.4920 | 0.4530 | 0.4246 | True |

## Two-grid / V-cycle (real P and coarse operator)

| run | smoother | smoother only | coarse only | two-grid |
| --- | --- | ---: | ---: | ---: |
| L3_cgc_l0.0 | DeepONet | 0.9024 | 0.5812 | 0.1985 |
| L3_cgc_l0.0 | damped Jacobi | 0.6433 | 0.5812 | 0.2588 |
| L3_cgc_l0.0 | sym. Gauss-Seidel | 0.4297 | 0.5812 | 0.1507 |
| L3_cgc_l0.1 | DeepONet | 0.3849 | 0.5812 | 0.2088 |
| L3_cgc_l0.1 | damped Jacobi | 0.6433 | 0.5812 | 0.2588 |
| L3_cgc_l0.1 | sym. Gauss-Seidel | 0.4297 | 0.5812 | 0.1507 |
| L4_cgc_l0.1 | DeepONet | 0.6484 | 0.5616 | 0.2476 |
| L4_cgc_l0.1 | damped Jacobi | 0.6433 | 0.5616 | 0.2491 |
| L4_cgc_l0.1 | sym. Gauss-Seidel | 0.4476 | 0.5616 | 0.1372 |
| L4_skip_jacobi_trig256 | DeepONet | 0.6409 | 0.5616 | 0.2491 |
| L4_skip_jacobi_trig256 | damped Jacobi | 0.6433 | 0.5616 | 0.2491 |
| L4_skip_jacobi_trig256 | sym. Gauss-Seidel | 0.4476 | 0.5616 | 0.1372 |
| L5_cgc_l0.1 | DeepONet | 0.6631 | 0.5465 | 0.2430 |
| L5_cgc_l0.1 | damped Jacobi | 0.6615 | 0.5465 | 0.2442 |
| L5_cgc_l0.1 | sym. Gauss-Seidel | 0.4699 | 0.5465 | 0.1267 |
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
| L3_cgc_l0.0 | V-cycle learned@L3 + Jacobi below | - | - | 0.2316 (after 5: 0.01161, worst sample 0.0441) |
| L3_cgc_l0.0 | V-cycle damped Jacobi everywhere | - | - | 0.1753 (after 5: 0.01487, worst sample 0.0404) |
| L3_cgc_l0.1 | V-cycle learned@L3 + Jacobi below | - | - | 0.1076 (after 5: 0.00897, worst sample 0.0325) |
| L3_cgc_l0.1 | V-cycle damped Jacobi everywhere | - | - | 0.1753 (after 5: 0.01487, worst sample 0.0404) |
| L4_cgc_l0.1 | V-cycle learned@L4 + Jacobi below | - | - | 0.1751 (after 5: 0.01426, worst sample 0.0482) |
| L4_cgc_l0.1 | V-cycle damped Jacobi everywhere | - | - | 0.1751 (after 5: 0.01699, worst sample 0.0490) |
| L4_skip_jacobi_trig256 | V-cycle learned@L4 + Jacobi below | - | - | 0.1755 |
| L4_skip_jacobi_trig256 | V-cycle damped Jacobi everywhere | - | - | 0.1751 |
| L5_cgc_l0.1 | V-cycle learned@L5 + Jacobi below | - | - | 0.1736 (after 5: 0.01406, worst sample 0.0530) |
| L5_cgc_l0.1 | V-cycle damped Jacobi everywhere | - | - | 0.1743 (after 5: 0.01588, worst sample 0.0527) |
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

## On the errors a real V-cycle leaves behind (not generated)

| run | error after | DeepONet | damped Jacobi | sym. Gauss-Seidel |
| --- | --- | ---: | ---: | ---: |
| L3_cgc_l0.0 | after 0 vcycles | 0.6197 | 0.7173 | 0.5072 |
| L3_cgc_l0.0 | after 1 vcycles | 2.1297 | 0.7796 | 0.4969 |
| L3_cgc_l0.0 | after 2 vcycles | 3.1528 | 0.7927 | 0.4474 |
| L3_cgc_l0.0 | after 3 vcycles | 3.7364 | 0.8016 | 0.4216 |
| L3_cgc_l0.0 | after 4 vcycles | 4.1207 | 0.8114 | 0.4158 |
| L3_cgc_l0.1 | after 0 vcycles | 0.3513 | 0.7173 | 0.5072 |
| L3_cgc_l0.1 | after 1 vcycles | 0.6537 | 0.7796 | 0.4969 |
| L3_cgc_l0.1 | after 2 vcycles | 0.7823 | 0.7927 | 0.4474 |
| L3_cgc_l0.1 | after 3 vcycles | 0.8437 | 0.8016 | 0.4216 |
| L3_cgc_l0.1 | after 4 vcycles | 0.8636 | 0.8114 | 0.4158 |
| L4_cgc_l0.1 | after 0 vcycles | 0.7421 | 0.7357 | 0.5615 |
| L4_cgc_l0.1 | after 1 vcycles | 0.8067 | 0.8081 | 0.5764 |
| L4_cgc_l0.1 | after 2 vcycles | 0.7998 | 0.8155 | 0.5042 |
| L4_cgc_l0.1 | after 3 vcycles | 0.8024 | 0.8236 | 0.4651 |
| L4_cgc_l0.1 | after 4 vcycles | 0.8109 | 0.8346 | 0.4589 |
| L5_cgc_l0.1 | after 0 vcycles | 0.7433 | 0.7393 | 0.5790 |
| L5_cgc_l0.1 | after 1 vcycles | 0.8159 | 0.8188 | 0.6135 |
| L5_cgc_l0.1 | after 2 vcycles | 0.8164 | 0.8304 | 0.5682 |
| L5_cgc_l0.1 | after 3 vcycles | 0.8123 | 0.8344 | 0.5135 |
| L5_cgc_l0.1 | after 4 vcycles | 0.8215 | 0.8462 | 0.5014 |

## Loss versus measured cycle (same samples; must agree)

- L3_cgc_l0.0: L_CGC 0.0494235120 vs measured two-grid energy 0.0494235120 (difference 0.00e+00)
- L3_cgc_l0.1: L_CGC 0.0582113599 vs measured two-grid energy 0.0582113599 (difference 6.94e-18)
- L4_cgc_l0.1: L_CGC 0.0713587805 vs measured two-grid energy 0.0713587805 (difference 0.00e+00)
- L5_cgc_l0.1: L_CGC 0.0721061550 vs measured two-grid energy 0.0721061550 (difference 2.78e-17)

## Hierarchy Galerkin check (P^T A P vs the assembled coarse operator)

- L3_cgc_l0.0: relative max|P^T A P - A_H| = 8.326e-03
- L3_cgc_l0.1: relative max|P^T A P - A_H| = 8.326e-03
- L4_cgc_l0.1: relative max|P^T A P - A_H| = 1.899e-03
- L4_skip_jacobi_trig256: relative max|P^T A P - A_H| = 1.899e-03
- L5_cgc_l0.1: relative max|P^T A P - A_H| = 4.660e-04
- L5_skip_jacobi_trig256: relative max|P^T A P - A_H| = 4.660e-04
- skip_jacobi_trig128: relative max|P^T A P - A_H| = 8.326e-03
- spec_xyz_p64: relative max|P^T A P - A_H| = 8.326e-03
- trunk_fourier16: relative max|P^T A P - A_H| = 8.326e-03
- trunk_trig_p1024: relative max|P^T A P - A_H| = 8.326e-03

## Split audit (cross-split signature overlaps; 0 required)

- L3_cgc_l0.0: 0
- L3_cgc_l0.1: 0
- L4_cgc_l0.1: 0
- L4_skip_jacobi_trig256: 0
- L5_cgc_l0.1: 0
- L5_skip_jacobi_trig256: 0
- skip_jacobi_trig128: 0
- spec_xyz_p64: 0
- trunk_fourier16: 0
- trunk_trig_p1024: 0

wrote results/summary.csv
