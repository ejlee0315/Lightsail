"""Run the full 2-stage lightsail optimization pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from lightsail.constraints.fabrication import FabConstraints
from lightsail.experiments.pipeline import TwoStagePipeline
from lightsail.geometry.base import LatticeFamily
from lightsail.simulation.mock import MockSolver


def main():
    parser = argparse.ArgumentParser(
        description="SiN lightsail 2-stage optimization"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/run"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)

    constraints = FabConstraints(
        min_feature_nm=config["fabrication"]["min_feature_nm"],
        min_gap_nm=config["fabrication"]["min_gap_nm"],
        thickness_range_nm=tuple(config["thickness_range_nm"]),
    )

    solver = MockSolver()  # swap for RCWASolver later

    lattice_family = LatticeFamily(
        config["stage1"].get("lattice_family", "triangular")
    )

    pipeline = TwoStagePipeline(
        solver=solver,
        constraints=constraints,
        lattice_family=lattice_family,
        phc_n_rings=config["stage1"].get("n_rings", 6),
        stage1_iterations=config["stage1"]["n_iterations"],
        stage2_iterations=config["stage2"]["n_iterations"],
        seed=seed,
        stage1_objectives_cfg=config["stage1"].get("objectives", {}),
        stage2_objectives_cfg=config["stage2"].get("objectives", {}),
    )

    result = pipeline.run(output_dir=args.output)

    print(f"\nStage 1 best: {result.stage1.best_objectives}")
    print(f"Stage 2 best: {result.stage2.best_objectives}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
