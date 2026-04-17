"""Freeform hole shape optimization with fixed thickness/period.

Fix t and P near the best thin design, then optimize ONLY the hole
boundary using high-order Fourier harmonics (n=2..8). This focuses
all BO budget on shape exploration instead of wasting it on t/P.

Parameterization:
  - base_radius_frac ∈ [0, 1]  → maps to valid hole radius
  - rotation_deg ∈ [0, 180]
  - amp_n ∈ [0, 0.25] for n = 2, 3, 4, 5, 6, 7, 8  (7 amplitudes)
  - phase_n ∈ [0, 2π] for n = 2, 3, 4, 5, 6, 7, 8  (7 phases)
  Total: 16 params (all shape-related)

Fixed: thickness=240 nm, period=1580 nm (triangular)

Usage:
    python3 scripts/run_freeform_shape_only.py [--seed 42]
    python3 scripts/run_freeform_shape_only.py --seed 123
"""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

import lightsail.geometry.phc_reflector as _pr
_pr._THICKNESS_MIN_NM = 5.0
_pr._THICKNESS_MAX_NM = 300.0

from lightsail.geometry.base import (
    HoleShape, Hole, LatticeFamily, Material, Structure,
    ParametricGeometry,
)
from lightsail.geometry.lattices import make_lattice
from lightsail.optimization.evaluator import ObjectiveEvaluator
from lightsail.optimization.mobo_runner import MOBOConfig, MOBORunner, save_run_result
from lightsail.optimization.objectives import make_stage1_objectives
from lightsail.optimization.search_space import SearchSpace
from lightsail.constraints.fabrication import FabConstraints, ConstraintMode
from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver

from dataclasses import dataclass, field


# Fixed design parameters (from thin baseline trial 23 neighbourhood)
FIXED_THICKNESS_NM = 240.0
FIXED_PERIOD_NM = 1580.0
FIXED_N_RINGS = 6

# Fourier harmonics range
N_HARMONICS_MIN = 2
N_HARMONICS_MAX = 8
N_HARMONICS = N_HARMONICS_MAX - N_HARMONICS_MIN + 1  # 7

AMP_MAX = 0.25
RADIUS_MIN_NM = 50.0
WALL_MIN_NM = 100.0


@dataclass
class FreeformShapeGeometry(ParametricGeometry):
    """Freeform hole shape with fixed thickness/period.

    16 optimization parameters:
      - base_radius_frac ∈ [0, 1]
      - rotation_deg ∈ [0, 180]
      - amp_2..amp_8 ∈ [0, 0.25]  (7 amplitudes)
      - phase_2..phase_8 ∈ [0, 2π]  (7 phases)
    """

    base_radius_frac: float = 0.5
    rotation_deg: float = 0.0
    amplitudes: list = field(default_factory=lambda: [0.0] * N_HARMONICS)
    phases: list = field(default_factory=lambda: [0.0] * N_HARMONICS)

    _lattice: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._lattice = make_lattice(LatticeFamily.TRIANGULAR, FIXED_PERIOD_NM)
        self.amplitudes = list(self.amplitudes)
        self.phases = list(self.phases)

    @property
    def _radius_nm(self) -> float:
        """Map frac to valid radius."""
        r_max = (FIXED_PERIOD_NM - WALL_MIN_NM) / 2.0
        r_min = RADIUS_MIN_NM
        return r_min + self.base_radius_frac * (r_max - r_min)

    def param_names(self) -> list[str]:
        names = ["base_radius_frac", "rotation_deg"]
        for n in range(N_HARMONICS_MIN, N_HARMONICS_MAX + 1):
            names.append(f"amp_{n}")
        for n in range(N_HARMONICS_MIN, N_HARMONICS_MAX + 1):
            names.append(f"phase_{n}")
        return names

    def param_bounds(self) -> list[tuple[float, float]]:
        bounds = [
            (0.0, 1.0),      # base_radius_frac
            (0.0, 180.0),    # rotation_deg
        ]
        for _ in range(N_HARMONICS):
            bounds.append((0.0, AMP_MAX))    # amp_n
        for _ in range(N_HARMONICS):
            bounds.append((0.0, 2.0 * np.pi))  # phase_n
        return bounds

    def to_param_vector(self) -> np.ndarray:
        v = [self.base_radius_frac, self.rotation_deg]
        v.extend(self.amplitudes)
        v.extend(self.phases)
        return np.array(v, dtype=float)

    def from_param_vector(self, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=float).ravel()
        assert v.size == 2 + 2 * N_HARMONICS, f"Expected {2 + 2*N_HARMONICS}, got {v.size}"
        self.base_radius_frac = float(np.clip(v[0], 0.0, 1.0))
        self.rotation_deg = float(v[1] % 180.0)
        for i in range(N_HARMONICS):
            self.amplitudes[i] = float(np.clip(v[2 + i], 0.0, AMP_MAX))
        for i in range(N_HARMONICS):
            self.phases[i] = float(v[2 + N_HARMONICS + i] % (2.0 * np.pi))

    def hole_shape(self) -> HoleShape:
        r = self._radius_nm
        return HoleShape(
            a_nm=r,
            b_nm=r,  # circular base (asymmetry via Fourier)
            n_sides=6,
            rotation_deg=self.rotation_deg,
            corner_rounding=1.0,  # pure circle base
            fourier_amplitudes=tuple(self.amplitudes),
            fourier_phases=tuple(self.phases),
        )

    def to_structure(self) -> Structure:
        shape = self.hole_shape()
        lattice = self._lattice
        extent = 2.0 * FIXED_N_RINGS * FIXED_PERIOD_NM
        sites = lattice.generate_sites(extent)
        holes = [Hole(x_nm=x, y_nm=y, shape=shape) for (x, y) in sites]
        return Structure(
            material=Material.SIN,
            thickness_nm=FIXED_THICKNESS_NM,
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=FIXED_PERIOD_NM,
            period_x_nm=FIXED_PERIOD_NM,
            period_y_nm=FIXED_PERIOD_NM,
            holes=holes,
            extent_nm=extent,
            metadata={
                "lattice_family": "triangular",
                "n_sides": 6,
                "n_holes": len(holes),
                "nearest_neighbor_nm": float(lattice.nearest_neighbor_distance()),
                "unit_cell_area_nm2": float(lattice.unit_cell_area()),
                "hole_a_rel": self._radius_nm / FIXED_PERIOD_NM,
                "hole_b_rel": self._radius_nm / FIXED_PERIOD_NM,
                "lattice_period_x_nm": FIXED_PERIOD_NM,
                "lattice_period_y_nm": FIXED_PERIOD_NM,
                "lattice_aspect_ratio": 1.0,
                "base_radius_nm": self._radius_nm,
                "fourier_amps": list(self.amplitudes),
                "fourier_phases": list(self.phases),
                "fixed_thickness": True,
                "fixed_period": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--launch", type=float, default=1550.0,
                        help="Launch wavelength (1550 or 1320)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    beta_f = 0.2
    lam_max = args.launch * np.sqrt((1 + beta_f) / (1 - beta_f))

    solver = RCWASolver(config=RCWAConfig(
        nG=41, grid_nx=96, grid_ny=96, polarization="average",
    ))
    constraints = FabConstraints(
        mode=ConstraintMode.PENALTY,
        min_feature_nm=100, min_gap_nm=100,
        fill_fraction_range=(0.001, 0.999),
        thickness_range_nm=(5, 300),
    )

    objectives_cfg = {
        "nir_reflectance": {
            "band_nm": [args.launch, lam_max],
            "n_points": 30,
            "mean_weight": 0.7, "min_weight": 0.3, "weight": 1.0,
        },
        "mir_emissivity": {
            "band_nm": [8000, 14000], "n_points": 30, "weight": 1.0,
        },
        "fabrication_penalty": {"weight": 0.3},
        "sail_areal_density": {
            "material_density_kg_m3": 3100.0, "weight": 1.0,
        },
    }

    geom = FreeformShapeGeometry()
    evaluator = ObjectiveEvaluator(
        geometry=geom,
        solver=solver,
        constraints=constraints,
        objectives=make_stage1_objectives(objectives_cfg),
    )
    search_space = SearchSpace.from_geometry(geom)

    mobo_config = MOBOConfig(
        n_init=80,
        n_iterations=220,
        batch_size=1,
        seed=args.seed,
        sampling_method="sobol",
        acqf_num_restarts=8,
        acqf_raw_samples=128,
        acqf_mc_samples=256,
        ref_point_margin=0.1,
    )

    print("=" * 72)
    print(f"FREEFORM SHAPE-ONLY — launch {args.launch:.0f} nm, seed={args.seed}")
    print(f"  fixed: t={FIXED_THICKNESS_NM:.0f} nm, P={FIXED_PERIOD_NM:.0f} nm")
    print(f"  Fourier: n=2..8 (7 harmonics), 16 params total")
    print(f"  BO: 80 Sobol + 220 iterations = 300 trials")
    print(f"  Doppler range: {args.launch:.0f}-{lam_max:.0f} nm")
    print("=" * 72)

    t0 = time.time()
    runner = MOBORunner(evaluator, search_space, mobo_config)
    result = runner.run()

    out_dir = Path("results") / f"freeform_shape_{args.launch:.0f}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_result(result, out_dir)

    if result.trials:
        best = result.best_by("nir_reflectance")
        if best:
            print(f"\nBest NIR: {best['objective_values']['nir_reflectance']:.4f}")
            print(f"  MIR: {best['objective_values']['mir_emissivity']:.4f}")
            print(f"  density: {best['objective_values']['sail_areal_density']:.4f}")

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed/60:.2f} min → {out_dir}")


if __name__ == "__main__":
    main()
