"""Run ALL 4 lattice families at thin regime (5-300 nm), 1550-1850 band.

Comprehensive scan to find the true global optimum across lattice types
when thickness is constrained to the thin-film regime.

Usage:
    python3 scripts/run_thin_all_families.py
"""
from __future__ import annotations
import copy, logging, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# Patch thickness bounds BEFORE importing
import lightsail.geometry.phc_reflector as _pr
_pr._THICKNESS_MIN_NM = 5.0
_pr._THICKNESS_MAX_NM = 300.0

from lightsail.experiments.main import run_experiment

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

SHARED_OVERRIDES: dict = {
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

FAMILIES = {
    "triangular":  "stage1_triangular.yaml",
    "hexagonal":   "stage1_hexagonal.yaml",
    "rectangular":  "stage1_rectangular.yaml",
    "pentagonal":  "stage1_pentagonal.yaml",
}

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    outputs = []
    for family, config_file in FAMILIES.items():
        overrides = copy.deepcopy(SHARED_OVERRIDES)
        overrides["name"] = f"thin_{family}_1550_s42"
        print("=" * 72)
        print(f"THIN {family.upper()} — 1550-1850 nm, t=5-300 nm, seed=42")
        print("=" * 72)
        t0 = time.time()
        out = run_experiment(
            config_path=CONFIGS_DIR / config_file,
            output_root=Path("results"),
            overrides=overrides,
        )
        elapsed = time.time() - t0
        print(f"  {family} done in {elapsed/60:.2f} min → {out}\n")
        outputs.append((family, out, elapsed))

    print("\n" + "=" * 72)
    print("ALL FAMILIES COMPLETE")
    print("=" * 72)
    for fam, out, el in outputs:
        print(f"  {fam:15s} → {out}  ({el/60:.1f} min)")

if __name__ == "__main__":
    main()
