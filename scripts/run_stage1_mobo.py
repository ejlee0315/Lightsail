"""Run Stage 1 (PhC reflector) multi-objective BO with BoTorch.

Example:

    python3 scripts/run_stage1_mobo.py \
        --config configs/default.yaml \
        --output results/stage1_mobo \
        --n-init 16 --n-iter 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import yaml

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.experiments.stage_runner import run_stage1
from lightsail.geometry.base import LatticeFamily
from lightsail.optimization.mobo_runner import MOBOConfig
from lightsail.simulation.mock import MockSolver
from lightsail.visualization.mobo_plots import (
    plot_optimization_history,
    plot_pareto_scatter,
    summarize_best,
)


def _build_mobo_config(cfg: dict, seed: int, overrides: argparse.Namespace) -> MOBOConfig:
    mcfg = (cfg.get("mobo") or {}).get("stage1") or {}
    return MOBOConfig(
        n_init=overrides.n_init or mcfg.get("n_init", 16),
        n_iterations=overrides.n_iter or mcfg.get("n_iterations", 20),
        batch_size=mcfg.get("batch_size", 1),
        seed=seed,
        sampling_method=mcfg.get("sampling", "sobol"),
        acqf_num_restarts=mcfg.get("acqf_num_restarts", 5),
        acqf_raw_samples=mcfg.get("acqf_raw_samples", 64),
        acqf_mc_samples=mcfg.get("acqf_mc_samples", 128),
        device=mcfg.get("device", "cpu"),
        dtype=mcfg.get("dtype", "double"),
        ref_point_margin=mcfg.get("ref_point_margin", 0.1),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 multi-objective BO for the PhC reflector"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage1_mobo"))
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    constraints = FabConstraints(
        min_feature_nm=cfg["fabrication"]["min_feature_nm"],
        min_gap_nm=cfg["fabrication"]["min_gap_nm"],
        thickness_range_nm=tuple(cfg["thickness_range_nm"]),
        mode=ConstraintMode.PENALTY,
    )
    mobo_cfg = _build_mobo_config(cfg, seed, args)
    lattice_family = LatticeFamily(cfg["stage1"].get("lattice_family", "triangular"))

    result, phc = run_stage1(
        solver=MockSolver(),
        constraints=constraints,
        lattice_family=lattice_family,
        phc_n_rings=cfg["stage1"].get("n_rings", 6),
        objectives_cfg=cfg["stage1"].get("objectives", {}),
        mobo_config=mobo_cfg,
        output_dir=args.output,
    )

    print(summarize_best(result))

    plot_pareto_scatter(
        result,
        x_objective="nir_reflectance",
        y_objective="mir_emissivity",
        save_path=args.output / "pareto_nir_vs_mir.png",
    )
    plot_optimization_history(
        result,
        save_path=args.output / "history.png",
    )

    print(f"\nArtifacts written to: {args.output.resolve()}")
    print(
        "Best PhC params: thickness={:.0f} nm, period={:.0f} nm, n_sides={}".format(
            phc.thickness_nm, phc.lattice_period_nm, phc.n_sides
        )
    )


if __name__ == "__main__":
    main()
