# Phase 4-C — Robustness sweeps

Nominal design: Design A center + BO best ring (P=1424nm, duty=0.47, t=280nm) + R_c=30m parabolic curvature.

Initial: x0=y0=50mm, θ_x0=θ_y0=-2°, spin=120Hz. Threshold: |xy|<1.8m, |tilt|<10°.


## beam_waist

| variable | PASS | max |xy| [m] | max |tilt| [°] |
|---|---|---:|---:|
| w=3.0m | FAIL | 0.071 | 306.59 |
| w=4.0m | FAIL | 0.071 | 186.12 |
| w=5.0m | PASS | 0.071 | 3.59 |

## initial_pert

| variable | PASS | max |xy| [m] | max |tilt| [°] |
|---|---|---:|---:|
| x0=20mm,th=-1.0 | PASS | 0.028 | 1.62 |
| x0=50mm,th=-2.0 | FAIL | 0.071 | 186.12 |
| x0=100mm,th=-2.0 | FAIL | 0.141 | 399.97 |
| x0=50mm,th=-4.0 | FAIL | 0.071 | 196.00 |

## intensity

| variable | PASS | max |xy| [m] | max |tilt| [°] |
|---|---|---:|---:|
| I=0.1xStarshot | FAIL | 0.071 | 18.67 |
| I=1.0xStarshot | FAIL | 0.071 | 186.12 |
| I=5.0xStarshot | FAIL | 0.071 | 956.85 |

## ring_period

| variable | PASS | max |xy| [m] | max |tilt| [°] |
|---|---|---:|---:|
| P=1282nm (-10%) | FAIL | 0.071 | 182.34 |
| P=1424nm (+0%) | FAIL | 0.071 | 186.12 |
| P=1566nm (+10%) | FAIL | 26.573 | 373.25 |

## Rc_tolerance

| variable | PASS | max |xy| [m] | max |tilt| [°] |
|---|---|---:|---:|
| R_c=24.0m (-20%) | PASS | 0.071 | 4.23 |
| R_c=30.0m (+0%) | FAIL | 0.071 | 186.12 |
| R_c=36.0m (+20%) | FAIL | 0.071 | 184.69 |
