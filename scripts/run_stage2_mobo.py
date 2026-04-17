"""Run Stage 2 (metagrating) multi-objective BO with BoTorch.

Stage 2 assumes a frozen PhC reflector coming from Stage 1. For a
standalone Stage 2 run this script synthesizes a default PhC (useful
for testing the Stage 2 loop in isolation). Downstream orchestration
should call ``run_stage1`` first and pass its ``phc`` result to
``run_stage2`` directly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import yaml

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.experiments.stage_runner import run_stage2
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.optimization.mobo_runner import MOBOConfig
from lightsail.simulation.mock import MockSolver
from lightsail.visualization.mobo_plots import (
    plot_optimization_history,
    plot_pareto_scatter,
    summarize_best,
)


def _build_mobo_config(cfg: dict, seed: int, overrides: argparse.Namespace) -> MOBOConfig:
    mcfg = (cfg.get("mobo") or {}).get("stage2") or {}
    return MOBOConfig(
        n_init=overrides.n_init or mcfg.get("n_init", 12),
        n_iterations=overrides.n_iter or mcfg.get("n_iterations", 15),
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
        description="Stage 2 multi-objective BO for the metagrating"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_mobo"))
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

    # Synthesize a reasonable default PhC so Stage 2 can run standalone.
    phc = PhCReflector(
        lattice_family=LatticeFamily(
            cfg["stage1"].get("lattice_family", "triangular")
        ),
        n_rings=cfg["stage1"].get("n_rings", 6),
        thickness_nm=500.0,
        lattice_period_nm=1500.0,
        hole_a_rel=400.0 / 1500.0,
        hole_b_rel=400.0 / 1500.0,
        corner_rounding=0.8,
        shape_parameter=6,
    )

    result, mg = run_stage2(
        phc=phc,
        solver=MockSolver(),
        constraints=constraints,
        objectives_cfg=cfg["stage2"].get("objectives", {}),
        mobo_config=mobo_cfg,
        output_dir=args.output,
    )

    print(summarize_best(result))

    plot_pareto_scatter(
        result,
        x_objective="stabilization",
        y_objective="fabrication_penalty",
        save_path=args.output / "pareto_stab_vs_fab.png",
    )
    plot_optimization_history(result, save_path=args.output / "history.png")

    print(f"\nArtifacts written to: {args.output.resolve()}")
    print(
        "Best MG params: period={:.0f} nm, duty={:.2f}, curvature={:.2f}, "
        "asymmetry={:.2f}, ring_width={:.1f} um".format(
            mg.grating_period_nm,
            mg.duty_cycle,
            mg.curvature,
            mg.asymmetry,
            mg.ring_width_um,
        )
    )


if __name__ == "__main__":
    main()
