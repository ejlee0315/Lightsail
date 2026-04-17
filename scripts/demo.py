"""Minimal runnable demo for the SiN lightsail pipeline.

Runs a shortened Stage 1 optimization (triangular lattice, MockSolver,
Sobol init + qLogNEHVI) and writes a complete artifact bundle into
``results/<timestamp>_demo_triangular/``:

- ``run.log``                — console log
- ``config.yaml``            — effective config
- ``overrides.yaml``         — runtime overrides applied to the config
- ``trials.json``            — full trial history
- ``params.npy``, ``objectives.npy``, ``pareto_indices.npy``
- ``best_design.yaml``       — best params + objectives
- ``summary.txt``            — top-K ranking per objective
- ``plots/``
    - ``pareto_nir_reflectance_vs_mir_emissivity.png``
    - ``optimization_history.png``
    - ``spectrum_best.png``
    - ``structure_topview.png``

Expected runtime: ~30 seconds on CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so the script runs headless

from lightsail.experiments.main import run_experiment


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "stage1_triangular.yaml"
OUTPUT_ROOT = Path("results")


# Shrink the production config to a 10-iteration, 8-init demo.
DEMO_OVERRIDES = {
    "name": "demo_triangular",
    "mobo": {
        "n_init": 8,
        "n_iterations": 10,
        "acqf_num_restarts": 3,
        "acqf_raw_samples": 32,
        "seed": 42,
    },
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("Running lightsail demo (Stage 1, triangular lattice, MockSolver)")
    print(f"  config   : {CONFIG_PATH}")
    print(f"  overrides: {DEMO_OVERRIDES}")

    output_dir = run_experiment(
        config_path=CONFIG_PATH,
        output_root=OUTPUT_ROOT,
        overrides=DEMO_OVERRIDES,
    )

    print("\n" + "=" * 72)
    print("Demo complete. Inspect the artifacts here:")
    print(f"  {output_dir}")
    print("Key files:")
    for rel in [
        "summary.txt",
        "best_design.yaml",
        "plots/pareto_nir_reflectance_vs_mir_emissivity.png",
        "plots/optimization_history.png",
        "plots/spectrum_best.png",
        "plots/structure_topview.png",
    ]:
        target = output_dir / rel
        marker = "✓" if target.exists() else "✗"
        print(f"  {marker} {rel}")


if __name__ == "__main__":
    main()
