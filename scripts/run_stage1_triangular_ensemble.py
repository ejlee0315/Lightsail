"""Triangular Stage 1 ensemble — extended BO budget, multiple seeds.

Runs three independent Stage 1 optimizations on the triangular lattice
with the same relaxed constraints as the baseline production run but
with:

- ``n_init = 60`` (up from 40)
- ``n_iterations = 120`` (up from 60)
- 3 different random seeds

This gives us 540 total evaluations and the ability to check whether
the previous triangular best (NIR R = 0.726 at seed=42) is near the
true frontier or if more BO budget / different seeds open up new
territory.

Run:

    python3 scripts/run_stage1_triangular_ensemble.py

Expected runtime: ~35–45 minutes on CPU.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting

from lightsail.experiments.main import run_experiment


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "stage1_triangular.yaml"
)

SEEDS: list = [42, 123, 456]


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
        "n_init": 60,
        "n_iterations": 120,
        "batch_size": 1,
        "acqf_num_restarts": 5,
        "acqf_raw_samples": 64,
        "acqf_mc_samples": 128,
        "seed": 42,  # overwritten per seed below
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 72)
    print("Stage 1 TRIANGULAR ensemble — extended BO budget, 3 seeds")
    print("=" * 72)
    print(f"  seeds    : {SEEDS}")
    print("  BO       : 60 Sobol init + 120 qLogNEHVI iters (180 trials/seed)")
    print(f"  total    : {180 * len(SEEDS)} evaluations")
    print("  expected : ~35–45 minutes on CPU")
    print()

    total_start = time.time()
    outputs: list = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n--- ({i}/{len(SEEDS)}) seed = {seed} ---")
        overrides = copy.deepcopy(BASE_OVERRIDES)
        overrides["name"] = f"stage1_triangular_ensemble_s{seed}"
        overrides["mobo"]["seed"] = seed

        t0 = time.time()
        out = run_experiment(
            config_path=CONFIG_PATH,
            output_root=Path("results"),
            overrides=overrides,
        )
        dt = time.time() - t0
        print(f"  seed {seed} done in {dt/60:.2f} min → {out.name}")
        outputs.append((seed, out))

    total = time.time() - total_start
    print("\n" + "=" * 72)
    print(f"ENSEMBLE complete in {total/60:.2f} min")
    print("=" * 72)
    for seed, out in outputs:
        print(f"  seed {seed}: {out}")
    print()
    print("Next step:")
    print("  python3 scripts/analyze_triangular_ensemble.py")


if __name__ == "__main__":
    main()
