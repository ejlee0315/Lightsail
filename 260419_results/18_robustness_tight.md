# Phase 4-C (tight re-run) — 8 critical robustness cases

Settings: rtol=1e-6, n_radial 20/8/48, t=5s (paper-grade).

| Case | PASS | max |tilt| [°] | max |xy| [m] |
|---|:---:|---:|---:|
| nominal | FAIL | 329.01 | 0.071 |
| large_pert | FAIL | 669.02 | 0.141 |
| narrow_beam | FAIL | 516.80 | 0.071 |
| hi_intensity | FAIL | 1617.41 | 0.071 |
| period+10 | FAIL | 374.58 | 44.333 |
| Rc-20 | PASS | 4.23 | 0.071 |
| small_pert | PASS | 1.61 | 0.028 |
| period-10 | FAIL | 324.82 | 0.071 |
