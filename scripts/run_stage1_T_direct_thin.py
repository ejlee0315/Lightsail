"""Stage 1 BO — T-direct objective (minimize acceleration time), thin, 1550 launch.

REPLACES nir_reflectance with AccelerationTimeObjective that directly minimizes T.
This teaches BO that R at launch wavelength matters more than broadband R.

3 objectives: acceleration_time (↓), mir_emissivity (↑), fabrication_penalty (↓)
+ optional sail_areal_density (↓).

Usage:
    python3 scripts/run_stage1_T_direct_thin.py [--seed 42]
"""
from __future__ import annotations
import argparse, copy, logging, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import lightsail.geometry.phc_reflector as _pr
_pr._THICKNESS_MIN_NM = 5.0
_pr._THICKNESS_MAX_NM = 300.0

from lightsail.experiments.main import run_experiment

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

BASE_OVERRIDES: dict = {
    "name": "stage1_T_direct_thin",
    "thickness_range_nm": [5, 300],
    "solver": {
        "kind": "rcwa", "nG": 41,
        "grid_nx": 96, "grid_ny": 96, "polarization": "average",
    },
    "fabrication": {
        "mode": "mode_a", "min_feature_nm": 100, "min_gap_nm": 100,
        "fill_fraction_range": [0.001, 0.999],
        "thickness_range_nm": [5, 300],
    },
    "mobo": {
        "n_init": 40, "n_iterations": 60, "batch_size": 1,
        "acqf_num_restarts": 5, "acqf_raw_samples": 64,
        "acqf_mc_samples": 128, "seed": 42, "ref_point_margin": 0.1,
    },
    "primary_objective": "acceleration_time",
    "objectives": {
        # T-direct replaces nir_reflectance
        "acceleration_time": {
            "launch_wavelength_nm": 1550.0,
            "beta_final": 0.2,
            "laser_intensity_W_m2": 1.0e10,
            "sail_area_m2": 10.0,
            "payload_mass_kg": 1.0e-3,
            "material_density_kg_m3": 3100.0,
            "n_points": 30,
            "weight": 1.0,
        },
        # Keep NIR for reference scoring (won't be used as objective since
        # acceleration_time replaces it in the factory)
        "nir_reflectance": {
            "band_nm": [1550, 1850], "n_points": 30,
            "mean_weight": 0.7, "min_weight": 0.3, "weight": 1.0,
        },
        "mir_emissivity": {
            "band_nm": [8000, 14000], "n_points": 30, "weight": 1.0,
        },
        "fabrication_penalty": {"weight": 0.3},
        "sail_areal_density": {
            "material_density_kg_m3": 3100.0, "weight": 1.0,
        },
    },
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    overrides = copy.deepcopy(BASE_OVERRIDES)
    overrides["mobo"]["seed"] = args.seed
    overrides["name"] = f"stage1_T_direct_thin_s{args.seed}"
    print("=" * 72)
    print(f"Stage 1 T-DIRECT — minimize T to β=0.2, thin 5-300nm, seed={args.seed}")
    print("  objectives: acceleration_time ↓, MIR ↑, fab ↓, density ↓")
    print("  launch λ = 1550 nm, I = 10 GW/m², A = 10 m²")
    print("=" * 72)
    t0 = time.time()
    output_dir = run_experiment(
        config_path=CONFIGS_DIR / "stage1_triangular.yaml",
        output_root=Path("results"), overrides=overrides,
    )
    print(f"\ndone in {(time.time()-t0)/60:.2f} min  -->  {output_dir}")

if __name__ == "__main__":
    main()
