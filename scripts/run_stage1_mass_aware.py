"""Mass-aware Stage 1 BO run (triangular only).

Adds sail areal density (g/m^2) as a **4th objective** on top of the
default NIR/MIR/fab set. The physically correct formulation is
``rho_s = rho_SiN * thickness * material_fraction`` where
``material_fraction = 1 - hole_area / unit_cell_area``, so both
thickness and hole fill factor are exposed to BO as mass levers.

Motivation: the previous thick triangular best (t=688 nm, Af=0.475)
has sail areal density ~1.0 g/m^2, which is ~10x the aspirational
Starshot target (~0.1 g/m^2 for a 1 g sail on 10 m^2). We want to see
whether a 4-objective NEHVI front contains designs with NIR R > 0.7
at areal density < 0.3 g/m^2 (i.e. 3x Starshot target but feasible for
our structure space).

Usage:

    python3 scripts/run_stage1_mass_aware.py [--seed 42]

Budget: same 40 Sobol + 60 BO = 100 trials, RCWA nG=41, ~5 min on CPU.
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
    "name": "stage1_triangular_mass_aware",
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
            "band_nm": [1320, 1620],    # updated NIR plateau band
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
        # NEW: 4th objective — minimize sail areal density (g/m^2).
        # rho_SiN = 3100 kg/m^3 is the Kischkat/Luke stoichiometric
        # value already hard-coded in the materials module.
        "sail_areal_density": {
            "material_density_kg_m3": 3100.0,
            "weight": 1.0,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mass-aware Stage 1 triangular BO run (4 objectives)."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    overrides = copy.deepcopy(BASE_OVERRIDES)
    overrides["mobo"]["seed"] = args.seed
    overrides["name"] = f"stage1_triangular_mass_aware_s{args.seed}"

    print("=" * 72)
    print(f"Stage 1 MASS-AWARE run — triangular, seed={args.seed}")
    print("=" * 72)
    print("  objectives: NIR ↑, MIR ↑, fab ↓, sail_areal_density ↓ (g/m^2)")
    print("  solver    : RCWA nG=41, 96x96, polarization averaged")
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
