"""Unified CLI entrypoint for lightsail experiments.

Dispatches to Stage 1 or Stage 2 based on the ``stage`` field in the
supplied YAML config.

Usage
-----

    # Stage 1 with triangular lattice
    python3 scripts/run_experiment.py --config configs/stage1_triangular.yaml

    # Stage 1 with a pentagonal supercell
    python3 scripts/run_experiment.py --config configs/stage1_pentagonal.yaml

    # Stage 2 — uses the frozen_phc section of the config
    python3 scripts/run_experiment.py --config configs/stage2_outer_grating.yaml

    # Quick-debug run: shrink iteration counts via CLI overrides
    python3 scripts/run_experiment.py \\
        --config configs/stage1_triangular.yaml \\
        --n-init 6 --n-iter 8 --seed 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CLI runs

from lightsail.experiments.main import run_experiment


def _build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    mobo: dict = {}
    if args.n_init is not None:
        mobo["n_init"] = args.n_init
    if args.n_iter is not None:
        mobo["n_iterations"] = args.n_iter
    if args.seed is not None:
        mobo["seed"] = args.seed
    if mobo:
        overrides["mobo"] = mobo
    if args.name:
        overrides["name"] = args.name
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SiN lightsail optimization experiment from a YAML config."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="Parent directory for timestamped run folders.",
    )
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override the run name used in the output directory.",
    )
    args = parser.parse_args()

    # Logging will be reinitialized inside run_experiment once the output
    # directory exists; until then we can use a minimal configuration.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    overrides = _build_overrides(args)
    output_dir = run_experiment(
        config_path=args.config,
        output_root=args.output,
        overrides=overrides or None,
    )

    print(f"\n✓ Run complete. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
