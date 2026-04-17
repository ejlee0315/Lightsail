# Lightsail Photonic Crystal Optimization

SiN-based lightsail photonic crystal reflector optimization for interstellar propulsion (Breakthrough Starshot mission profile). Multi-objective Bayesian optimization + RCWA + adjoint gradient descent.

## Key Result

**Simple triangular lattice + circular hole outperforms neural-topology-optimized pentagonal designs by 7–16% in acceleration time** under identical fabrication constraints.

| Condition | Our T (min) | [Norder et al. 2025](https://www.nature.com/articles/s41467-025-57749-y) T | Δ |
|---|---:|---:|---:|
| Unconstrained | **20.73** | 24.6 | **−15.7%** |
| MFS ≥ 500 nm (i-line photolith) | 21.73 | 24.6 | **−11.7%** |
| Paper identical constraints | 22.11 | 24.6 | **−10.1%** |

Three independent methods (BO, pixel-level adjoint gradient, brute-force grid scan) all converged to the same simple circular-hole design, suggesting **freeform topology optimization is unnecessary** for this problem class.

## Architecture

```
src/lightsail/
├── geometry/          ParametricGeometry ABC, PhCReflector, FreeformPhCReflector,
│                      DualHolePhCReflector, DisorderedPhCReflector
├── constraints/       FabConstraints (hard / penalty mode)
├── materials/         SiNDispersion (Luke NIR + Kischkat MIR)
├── simulation/        ElectromagneticSolver ABC, MockSolver, RCWASolver (grcwa)
├── optimization/
│   ├── objectives.py  NIRReflectivityObjective, MIREmissivityObjective,
│   │                  SailArealDensityObjective, AccelerationTimeObjective,
│   │                  FabricationPenaltyObjective
│   ├── mobo_runner.py BoTorch qLogNEHVI multi-objective BO
│   └── adjoint_opt.py Pixel-level adjoint gradient descent (grcwa autograd)
├── experiments/       High-level runners (run_experiment, run_stage1, run_stage2)
└── visualization/     Pareto scatter, spectrum, structure topview
```

## Installation

```bash
python3 -m pip install -e ".[dev,mobo,rcwa]"
python3 -m pytest tests/      # 114 tests
```

Requires Python 3.9+, torch 2.8, botorch 0.10, grcwa 0.1.2, autograd.

## Quickstart

```bash
# Fast mock-solver demo
python3 scripts/demo.py

# RCWA demo
python3 scripts/demo_rcwa.py

# Production runs
python3 scripts/run_stage1_production.py --lattice triangular
python3 scripts/run_mfs500_sweep.py --seed 42 --launch 1550

# Adjoint freeform optimization
python3 -m lightsail.optimization.adjoint_opt --launch 1550 --niter 200 --init multi
```

## Best Designs

**Design A — absolute best T (DUV-compatible)**
- t = 280 nm, P = 1580 nm, circular hole radius = 600 nm
- T = 20.73 min to β = 0.2, D = 45.9 Gm
- Mass 5.1 g (10 m² sail + 1 g payload)

**Design B — i-line photolithography compatible**
- t = 220 nm, P = 1580 nm, circular hole radius = 537 nm
- Wall (MFS) = 506 nm
- T = 21.73 min, D = 48.6 Gm, Mass 5.0 g

Both launched at 1550 nm (Starshot standard). PhC suspended membrane, LPCVD or PECVD SiN.

## Methods

Three independent optimization approaches, all converged to the same simple circular-hole optimum:

1. **Multi-objective Bayesian optimization** (7–11 parameters, BoTorch qLogNEHVI)
2. **Adjoint gradient descent** (96×96 pixel grid, grcwa autograd backend, multi-start)
3. **Grid scan** (thickness × period × fill fraction brute force)

Systematic exploration of alternatives (Fourier freeform, dual-hole supercell, disordered lattice) all rejected by BO — confirming that single guided-mode resonance in a simple circular hole is the physical optimum in this regime.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — Full project state, onboarding, experiment log (53 KB)
- [`docs/discussion_final.md`](docs/discussion_final.md) — Paper-ready discussion, narrative
- [`docs/연구종합보고서.docx`](docs/연구종합보고서.docx) — Comprehensive Korean research report
- [`docs/next_10h_plan.md`](docs/next_10h_plan.md) — Future experiment plan (a-Si, graphene, bilayer)

## References

- Norder, L. et al. "Pentagonal photonic crystal mirrors: scalable lightsails with enhanced acceleration via neural topology optimization." *Nature Communications* **16**, 2753 (2025).
- Brewer, J. et al. "Broadband, High-Reflectivity Dielectric Mirrors at Wafer Scale." *Nano Letters* (2024).
- Atwater, H. A. et al. "Materials challenges for the Starshot lightsail." *Nature Materials* **17**, 861–867 (2018).
- Luke, K. et al. *Optics Letters* **40**, 4823 (2015).
- Kischkat, J. et al. *Applied Optics* **51**, 6789 (2012).

## License

MIT
