"""Stage 1 BO run with NIR target band 1550–1850 nm.

Pivot from the 1320–1620 plateau (non-standard 1320 nm launch laser)
to a band that is compatible with Starshot's 1550 nm telecom laser
infrastructure. The 1550–1850 nm range covers the sail-frame Doppler
shift for launch at 1550 nm and a final velocity of approximately
β = 0.175 (less than the canonical 0.2c target, but a cleaner 300 nm
window match and still useful for staged acceleration).

Preserves the other mass-aware settings from
``run_stage1_mass_aware.py`` — 4 objectives (NIR/MIR/fab/areal_density),
RCWA nG=41, 40 Sobol + 60 qLogNEHVI iterations.

Usage:
    python3 scripts/run_stage1_1550_1850.py [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from lightsail.experiments.main import run_experiment


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


BASE_OVERRIDES: dict = {
    "name": "stage1_triangular_1550_1850",
    "solver": {
        "kind": "rcwa",
        "nG": 41,
        "grid_nx": 96,
        "grid_ny": 96,
        "polarization": "average",
    },
    "fabrication": {
        "mode": "mode_a",
        "min_feature_nm": 100,
        "min_gap_nm": 100,
        "fill_fraction_range": [0.001, 0.999],
    },
    "mobo": {
        "n_init": 40,
        "n_iterations": 60,
        "batch_size": 1,
        "acqf_num_restarts": 5,
        "acqf_raw_samples": 64,
        "acqf_mc_samples": 128,
        "seed": 42,
        "ref_point_margin": 0.1,
    },
    "objectives": {
        "nir_reflectance": {
            "band_nm": [1550, 1850],   # <-- pivot target band
            "n_points": 30,
            "mean_weight": 0.7,
            "min_weight": 0.3,
            "weight": 1.0,
        },
        "mir_emissivity": {
            "band_nm": [8000, 14000],
            "n_points": 30,
            "weight": 1.0,
        },
        "fabrication_penalty": {
            "weight": 0.3,
        },
        "sail_areal_density": {
            "material_density_kg_m3": 3100.0,
            "weight": 1.0,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 triangular BO run with NIR band 1550-1850 nm."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    overrides = copy.deepcopy(BASE_OVERRIDES)
    overrides["mobo"]["seed"] = args.seed
    overrides["name"] = f"stage1_triangular_1550_1850_s{args.seed}"

    print("=" * 72)
    print(f"Stage 1 run — NIR target band 1550-1850 nm, seed={args.seed}")
    print("=" * 72)
    print("  objectives: NIR ↑, MIR ↑, fab ↓, sail_areal_density ↓")
    print("  launch λ  : 1550 nm (Starshot telecom standard)")
    print("  Doppler   : sail frame β=0 -> β≈0.175")
    print("  solver    : RCWA nG=41, 96x96")
    print("  BO        : 40 Sobol + 60 qLogNEHVI (100 trials)")
    print()

    t0 = time.time()
    output_dir = run_experiment(
        config_path=CONFIGS_DIR / "stage1_triangular.yaml",
        output_root=Path("results"),
        overrides=overrides,
    )
    elapsed = time.time() - t0
    print()
    print(f"done in {elapsed/60:.2f} min  -->  {output_dir}")


if __name__ == "__main__":
    main()
