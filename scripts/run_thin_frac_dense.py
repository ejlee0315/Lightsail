"""Dense BO scan with improved frac parameterization.

Runs both 1550 and 1320 launch conditions with:
- New frac-based hole bounds (0% infeasible waste)
- Dense sampling: 80 Sobol + 120 BO = 200 trials (2x previous)
- Thin regime: 5-300 nm

Usage:
    python3 scripts/run_thin_frac_dense.py
"""
from __future__ import annotations
import copy, logging, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import lightsail.geometry.phc_reflector as _pr
_pr._THICKNESS_MIN_NM = 5.0
_pr._THICKNESS_MAX_NM = 300.0

from lightsail.experiments.main import run_experiment

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

SHARED: dict = {
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
        "n_init": 80, "n_iterations": 120, "batch_size": 1,
        "acqf_num_restarts": 8, "acqf_raw_samples": 128,
        "acqf_mc_samples": 256, "seed": 42, "ref_point_margin": 0.1,
    },
    "objectives": {
        "mir_emissivity": {
            "band_nm": [8000, 14000], "n_points": 30, "weight": 1.0,
        },
        "fabrication_penalty": {"weight": 0.3},
        "sail_areal_density": {
            "material_density_kg_m3": 3100.0, "weight": 1.0,
        },
    },
}

RUNS = [
    {
        "name": "thin_frac_1550_dense_s42",
        "nir_band": [1550, 1898],    # full Doppler range β=0→0.2
        "label": "1550 nm launch (Doppler 1550-1898)",
    },
    {
        "name": "thin_frac_1320_dense_s42",
        "nir_band": [1320, 1617],    # full Doppler range β=0→0.2
        "label": "1320 nm launch (Doppler 1320-1617)",
    },
]

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    outputs = []
    for run in RUNS:
        overrides = copy.deepcopy(SHARED)
        overrides["name"] = run["name"]
        overrides["objectives"]["nir_reflectance"] = {
            "band_nm": run["nir_band"], "n_points": 30,
            "mean_weight": 0.7, "min_weight": 0.3, "weight": 1.0,
        }
        print("=" * 72)
        print(f"DENSE FRAC — {run['label']}, 200 trials (80 Sobol + 120 BO)")
        print("=" * 72)
        t0 = time.time()
        out = run_experiment(
            config_path=CONFIGS_DIR / "stage1_triangular.yaml",
            output_root=Path("results"), overrides=overrides,
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed/60:.2f} min → {out}\n")
        outputs.append((run["label"], out, elapsed))

    print("\n" + "=" * 72)
    print("ALL RUNS COMPLETE")
    for label, out, el in outputs:
        print(f"  {label} → {out}  ({el/60:.1f} min)")

if __name__ == "__main__":
    main()
