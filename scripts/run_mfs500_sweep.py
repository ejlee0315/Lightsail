"""MFS >= 500 nm constraint BO run (photolithography compatible).

Uses frac bounds + seed-controlled MOBO to find best design compatible
with i-line photolithography (wall >= 500 nm). Direct comparison with
Norder et al. 2025 pentagonal.

Usage:
    python3 scripts/run_mfs500_sweep.py --seed 42 --launch 1550
"""
from __future__ import annotations
import argparse, copy, logging, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

import lightsail.geometry.phc_reflector as _pr
_pr._THICKNESS_MIN_NM = 5.0
_pr._THICKNESS_MAX_NM = 500.0

from lightsail.experiments.main import run_experiment

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

def build_overrides(seed: int, launch: float) -> dict:
    lam_max = launch * np.sqrt(1.2 / 0.8)
    return {
        "name": f"mfs500_thin_{int(launch)}_s{seed}",
        "thickness_range_nm": [5, 500],
        "solver": {
            "kind": "rcwa", "nG": 41,
            "grid_nx": 96, "grid_ny": 96, "polarization": "average",
        },
        "fabrication": {
            "mode": "mode_a",
            "min_feature_nm": 500,  # photolithography compatible
            "min_gap_nm": 500,
            "fill_fraction_range": [0.001, 0.999],
            "thickness_range_nm": [5, 500],
        },
        "mobo": {
            "n_init": 40, "n_iterations": 60, "batch_size": 1,
            "acqf_num_restarts": 5, "acqf_raw_samples": 64,
            "acqf_mc_samples": 128, "seed": seed, "ref_point_margin": 0.1,
        },
        "objectives": {
            "nir_reflectance": {
                "band_nm": [launch, lam_max], "n_points": 30,
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
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--launch", type=float, default=1550.0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    overrides = build_overrides(a.seed, a.launch)
    print("=" * 72)
    print(f"MFS >= 500 BO — launch {a.launch:.0f} nm, seed={a.seed}")
    print("=" * 72)
    t0 = time.time()
    output_dir = run_experiment(
        config_path=CONFIGS_DIR / "stage1_triangular.yaml",
        output_root=Path("results"), overrides=overrides,
    )
    print(f"\ndone in {(time.time()-t0)/60:.2f} min -> {output_dir}")

if __name__ == "__main__":
    main()
