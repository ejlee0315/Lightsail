"""Stage 1 BO — 1320 nm launch, thin film (5-300 nm), triangular.

Doppler range: 1320–1617 nm (297 nm width, narrower than 1550's 348 nm).

Usage:
    python3 scripts/run_stage1_1320_thin.py [--seed 42]
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
    "name": "stage1_1320_thin",
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
    "objectives": {
        "nir_reflectance": {
            "band_nm": [1320, 1617],   # Doppler range for 1320 launch, beta=0.2
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
    },
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    overrides = copy.deepcopy(BASE_OVERRIDES)
    overrides["mobo"]["seed"] = args.seed
    overrides["name"] = f"stage1_1320_thin_s{args.seed}"
    print("=" * 72)
    print(f"Stage 1 — 1320 nm launch, thin 5-300nm, Doppler 1320-1617, seed={args.seed}")
    print("=" * 72)
    t0 = time.time()
    output_dir = run_experiment(
        config_path=CONFIGS_DIR / "stage1_triangular.yaml",
        output_root=Path("results"), overrides=overrides,
    )
    print(f"\ndone in {(time.time()-t0)/60:.2f} min  -->  {output_dir}")

if __name__ == "__main__":
    main()
