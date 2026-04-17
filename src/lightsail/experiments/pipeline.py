"""Two-stage optimization pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lightsail.constraints.fabrication import FabConstraints
from lightsail.experiments.runner import ExperimentRunner, StageResult
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.optimization.objectives import (
    make_stage1_objectives,
    make_stage2_objectives,
)
from lightsail.simulation.base import ElectromagneticSolver

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Combined result from the 2-stage pipeline."""

    stage1: StageResult
    stage2: StageResult
    final_phc: PhCReflector
    final_metagrating: MetaGrating


class TwoStagePipeline:
    """Orchestrates the full Stage 1 → Stage 2 optimization."""

    def __init__(
        self,
        solver: ElectromagneticSolver,
        constraints: FabConstraints | None = None,
        lattice_family: LatticeFamily = LatticeFamily.TRIANGULAR,
        phc_n_rings: int = 6,
        stage1_iterations: int = 50,
        stage2_iterations: int = 50,
        seed: int = 42,
        stage1_objectives_cfg: Optional[dict] = None,
        stage2_objectives_cfg: Optional[dict] = None,
    ):
        self.solver = solver
        self.constraints = constraints or FabConstraints()
        self.lattice_family = lattice_family
        self.phc_n_rings = phc_n_rings
        self.stage1_iterations = stage1_iterations
        self.stage2_iterations = stage2_iterations
        self.seed = seed
        self.stage1_objectives_cfg = stage1_objectives_cfg or {}
        self.stage2_objectives_cfg = stage2_objectives_cfg or {}

    def run(self, output_dir: Path | None = None) -> PipelineResult:
        # --- Stage 1: PhC Reflector ---
        logger.info(
            "=== Stage 1: PhC Reflector Optimization (lattice=%s) ===",
            self.lattice_family.value,
        )
        phc = PhCReflector(
            lattice_family=self.lattice_family,
            n_rings=self.phc_n_rings,
        )

        stage1_runner = ExperimentRunner(
            geometry=phc,
            solver=self.solver,
            objectives=make_stage1_objectives(self.stage1_objectives_cfg),
            constraints=self.constraints,
            n_iterations=self.stage1_iterations,
            seed=self.seed,
            pareto_primary="nir_reflectance",
        )
        stage1_result = stage1_runner.run()
        phc.from_param_vector(stage1_result.best_params)
        logger.info("Stage 1 best: %s", stage1_result.best_objectives)
        if output_dir:
            stage1_runner.save_results(stage1_result, output_dir / "stage1")

        # --- Stage 2: MetaGrating ---
        logger.info("=== Stage 2: MetaGrating Optimization ===")
        metagrating = MetaGrating(
            inner_radius_nm=phc.outer_radius_nm,
            thickness_nm=phc.thickness_nm,
        )

        stage2_runner = ExperimentRunner(
            geometry=metagrating,
            solver=self.solver,
            objectives=make_stage2_objectives(self.stage2_objectives_cfg),
            constraints=self.constraints,
            n_iterations=self.stage2_iterations,
            seed=self.seed + 1,
            pareto_primary="stabilization",
        )
        stage2_result = stage2_runner.run()
        metagrating.from_param_vector(stage2_result.best_params)
        logger.info("Stage 2 best: %s", stage2_result.best_objectives)
        if output_dir:
            stage2_runner.save_results(stage2_result, output_dir / "stage2")

        return PipelineResult(
            stage1=stage1_result,
            stage2=stage2_result,
            final_phc=phc,
            final_metagrating=metagrating,
        )
