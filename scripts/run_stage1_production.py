"""Production Stage 1 run (any lattice family), RCWA solver, 100 trials.

Settings picked to balance cost and BO signal for a 7-dimensional search
space:

- solver: grcwa-backed RCWASolver, nG=41, 96×96 grid, polarization averaged
  (see scripts/nG_convergence_study.py — NIR error < 1.3% vs nG=81)
- wavelengths: 30 points in the NIR band + 30 points in the MIR band
- sampling: 40 Sobol init + 60 qLogNEHVI BO iterations (100 trials total)
- seed=42 so runs across lattice families can be compared with the same
  initial Sobol grid

Usage:

    # single family
    python3 scripts/run_stage1_production.py --lattice triangular
    python3 scripts/run_stage1_production.py --lattice hexagonal
    python3 scripts/run_stage1_production.py --lattice pentagonal

    # multiple families in sequence (shares the same seed across runs)
    python3 scripts/run_stage1_production.py --lattice hexagonal,pentagonal
    python3 scripts/run_stage1_production.py --lattice all

Expected cost on a typical laptop CPU: ~5 minutes per family.
Artifacts land in ``results/<timestamp>_stage1_<family>_production/``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless runs

from lightsail.experiments.main import run_experiment


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# Shared override stack — everything except the lattice-specific config
# file and the run name.
BASE_OVERRIDES: dict = {
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
            "band_nm": [1350, 1650],
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
    },
}


# Lattice family → (config file, run name).
LATTICE_CONFIGS: dict = {
    "triangular": ("stage1_triangular.yaml", "stage1_triangular_production"),
    "hexagonal": ("stage1_hexagonal.yaml", "stage1_hexagonal_production"),
    "rectangular": ("stage1_rectangular.yaml", "stage1_rectangular_production"),
    "pentagonal": ("stage1_pentagonal.yaml", "stage1_pentagonal_production"),
}


# Families that need a bigger BO budget (higher-dim search spaces).
_EXTENDED_BUDGET = {
    "rectangular": {"n_init": 50, "n_iterations": 80},  # 8-dim search
}


def run_family(lattice: str) -> Path:
    """Run one lattice family. Returns the output directory."""
    if lattice not in LATTICE_CONFIGS:
        raise ValueError(
            f"Unknown lattice {lattice!r}. Valid: {list(LATTICE_CONFIGS)}"
        )
    config_name, run_name = LATTICE_CONFIGS[lattice]
    config_path = CONFIGS_DIR / config_name

    overrides = copy.deepcopy(BASE_OVERRIDES)
    overrides["name"] = run_name

    # Family-specific BO budget overrides (e.g. rectangular needs more).
    if lattice in _EXTENDED_BUDGET:
        overrides["mobo"].update(_EXTENDED_BUDGET[lattice])

    print("=" * 72)
    print(f"Stage 1 PRODUCTION run — {lattice} lattice")
    print("=" * 72)
    print(f"  config   : {config_path.name}")
    print( "  solver   : RCWA (nG=41, 96x96, polarization averaged)")
    print( "  wl       : 30 NIR + 30 MIR = 60 per trial")
    print( "  BO       : 40 Sobol init + 60 qLogNEHVI iterations (100 trials)")
    print( "  seed     : 42")
    print( "  expected : ~5 minutes on CPU")
    print()

    t0 = time.time()
    output_dir = run_experiment(
        config_path=config_path,
        output_root=Path("results"),
        overrides=overrides,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 72)
    print(f"{lattice} complete in {elapsed/60:.2f} min")
    print(f"  artifacts: {output_dir}")
    print("=" * 72)
    return output_dir


def _parse_lattice_arg(raw: str) -> list:
    if raw == "all":
        return list(LATTICE_CONFIGS.keys())
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 production optimization for one or more lattice families."
    )
    parser.add_argument(
        "--lattice",
        type=str,
        default="triangular",
        help=(
            "Lattice family to run. Single name (triangular / hexagonal / "
            "pentagonal), comma-separated list (e.g. hexagonal,pentagonal), "
            "or 'all' for all three."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    lattices = _parse_lattice_arg(args.lattice)
    print(f"Will run {len(lattices)} family(ies): {lattices}\n")

    outputs = []
    for lat in lattices:
        out = run_family(lat)
        outputs.append((lat, out))
        print()

    print("=" * 72)
    print("All runs complete")
    print("=" * 72)
    for lat, out in outputs:
        print(f"  {lat:12s} → {out}")


if __name__ == "__main__":
    main()
