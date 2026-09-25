#### Mean rho_F (one smoothing step, unseen test set)

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.4187 | 0.4154 | 0.4153 | 0.4153 |
| Gauss-Seidel | 0.3525 | 0.3461 | 0.3464 | 0.3465 |
| symmetric GS / SSOR | 0.1757 | 0.1727 | 0.1733 | 0.1734 |
| DeepONet (plain MLP) | 0.8994 | 0.8735 | 0.8775 | 0.8973 |
| DeepONet (trig trunk, + Jacobi skip) | 0.3715 | 0.3615 | 0.3559 | 0.3582 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.3100 | 0.2812 | 0.2789 | 0.2780 |

#### 95th-percentile rho_F

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.5305 | 0.5272 | 0.5263 | 0.5262 |
| Gauss-Seidel | 0.4430 | 0.4361 | 0.4357 | 0.4356 |
| symmetric GS / SSOR | 0.2398 | 0.2371 | 0.2379 | 0.2380 |
| DeepONet (plain MLP) | 0.9986 | 0.9974 | 0.9971 | 0.9991 |
| DeepONet (trig trunk, + Jacobi skip) | 0.4900 | 0.4817 | 0.4770 | 0.4739 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.4060 | 0.3732 | 0.3612 | 0.3613 |

#### Worst-case rho_F over the test set

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.6953 | 0.6773 | 0.6733 | 0.6728 |
| Gauss-Seidel | 0.5122 | 0.5104 | 0.5103 | 0.5103 |
| symmetric GS / SSOR | 0.3305 | 0.3295 | 0.3296 | 0.3296 |
| DeepONet (plain MLP) | 1.0016 | 1.0015 | 1.0016 | 1.0016 |
| DeepONet (trig trunk, + Jacobi skip) | 0.6646 | 0.6407 | 0.6390 | 0.6330 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.5804 | 0.5326 | 0.5033 | 0.5183 |

#### Fraction of test errors AMPLIFIED (rho_F > 1)

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Gauss-Seidel | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| symmetric GS / SSOR | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DeepONet (plain MLP) | 0.0352 | 0.0234 | 0.0117 | 0.0312 |
| DeepONet (trig trunk, + Jacobi skip) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Mean rho_C (the smoother on the coarse component)

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.6883 | 0.6524 | 0.6406 | 0.6398 |
| Gauss-Seidel | 0.5476 | 0.5043 | 0.5034 | 0.5040 |
| symmetric GS / SSOR | 0.3761 | 0.3416 | 0.3399 | 0.3400 |
| DeepONet (plain MLP) | 1.0095 | 1.0098 | 1.0099 | 1.0089 |
| DeepONet (trig trunk, + Jacobi skip) | 0.6723 | 0.6314 | 0.6220 | 0.6191 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.6535 | 0.6053 | 0.5989 | 0.5992 |

#### Mean rho_TG (ideal Galerkin two-grid)

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.2657 | 0.2593 | 0.2613 | 0.2618 |
| Gauss-Seidel | 0.2191 | 0.2123 | 0.2153 | 0.2163 |
| symmetric GS / SSOR | 0.1316 | 0.1263 | 0.1266 | 0.1268 |
| DeepONet (plain MLP) | 0.5736 | 0.5520 | 0.5493 | 0.5556 |
| DeepONet (trig trunk, + Jacobi skip) | 0.2465 | 0.2385 | 0.2395 | 0.2394 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.2251 | 0.2086 | 0.2120 | 0.2123 |

#### 95th-percentile rho_TG

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.3889 | 0.3845 | 0.3831 | 0.3830 |
| Gauss-Seidel | 0.2997 | 0.2921 | 0.2920 | 0.2921 |
| symmetric GS / SSOR | 0.1682 | 0.1648 | 0.1647 | 0.1647 |
| DeepONet (plain MLP) | 0.8869 | 0.8728 | 0.8705 | 0.8769 |
| DeepONet (trig trunk, + Jacobi skip) | 0.3468 | 0.3388 | 0.3376 | 0.3335 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.3066 | 0.2794 | 0.2775 | 0.2767 |

#### Mean rho_TG run on the e_F samples

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.3842 | 0.3813 | 0.3809 | 0.3808 |
| Gauss-Seidel | 0.2714 | 0.2652 | 0.2648 | 0.2648 |
| symmetric GS / SSOR | 0.1553 | 0.1523 | 0.1527 | 0.1528 |
| DeepONet (plain MLP) | 0.8942 | 0.8671 | 0.8670 | 0.8860 |
| DeepONet (trig trunk, + Jacobi skip) | 0.3396 | 0.3288 | 0.3246 | 0.3255 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.2803 | 0.2527 | 0.2502 | 0.2494 |

#### Mean rho_F after 3 smoothing steps

| method | $\varepsilon=10^{-1}$ | $\varepsilon=10^{-2}$ | $\varepsilon=10^{-3}$ | $\varepsilon=10^{-4}$ |
|---|---|---|---|---|
| damped Jacobi | 0.1699 | 0.1661 | 0.1660 | 0.1660 |
| Gauss-Seidel | 0.0826 | 0.0808 | 0.0809 | 0.0809 |
| symmetric GS / SSOR | 0.0427 | 0.0434 | 0.0436 | 0.0437 |
| DeepONet (plain MLP) | 0.9026 | 0.8749 | 0.8841 | 0.9099 |
| DeepONet (trig trunk, + Jacobi skip) | 0.1475 | 0.1396 | 0.1377 | 0.1390 |
| DeepONet (fourier trunk, + Jacobi skip) | 0.1078 | 0.0907 | 0.0897 | 0.0889 |
