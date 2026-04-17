"""High-level Stage 1 and Stage 2 runners driven by :class:`MOBORunner`.

These helpers wire the geometry (PhC reflector or metagrating), the
solver, the constraint checker, the objective factories, and the
:class:`MOBORunner` into a single callable per stage. They are the
intended entry point for the CLI scripts under ``scripts/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import (
    DisorderedPhCReflector,
    DualHolePhCReflector,
    FreeformPhCReflector,
    PhCReflector,
)
from lightsail.optimization.evaluator import ObjectiveEvaluator
from lightsail.optimization.mobo_runner import (
    MOBOConfig,
    MOBORunner,
    RunResult,
    save_run_result,
)
from lightsail.optimization.objectives import (
    make_stage1_objectives,
    make_stage2_objectives,
)
from lightsail.optimization.search_space import SearchSpace
from lightsail.simulation.base import ElectromagneticSolver
from lightsail.simulation.mock import MockSolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def run_stage1(
    *,
    solver: Optional[ElectromagneticSolver] = None,
    constraints: Optional[FabConstraints] = None,
    lattice_family: LatticeFamily = LatticeFamily.TRIANGULAR,
    phc_n_rings: int = 6,
    objectives_cfg: Optional[dict] = None,
    mobo_config: Optional[MOBOConfig] = None,
    output_dir: Optional[Path] = None,
    primary_objective: str = "nir_reflectance",
) -> tuple[RunResult, PhCReflector]:
    """Run Stage 1 (PhC reflector) multi-objective BO.

    Returns ``(run_result, best_phc)`` where ``best_phc`` has been
    updated in-place with the parameters of the trial that maximizes
    ``primary_objective``.

    The returned ``run_result`` contains the full history and the
    Pareto indices. Use ``best_by`` on it to pick any other
    objective, or inspect ``run_result.pareto_trials`` for the
    full non-dominated set.
    """
    solver = solver or MockSolver()
    constraints = constraints or FabConstraints(mode=ConstraintMode.PENALTY)
    mobo_config = mobo_config or MOBOConfig()

    # Select geometry variant based on objectives_cfg flags.
    use_freeform = bool((objectives_cfg or {}).get("_freeform", False))
    use_dual_hole = bool((objectives_cfg or {}).get("_dual_hole", False))
    use_disordered = bool((objectives_cfg or {}).get("_disordered", False))
    if use_disordered:
        phc = DisorderedPhCReflector(n_rings=phc_n_rings)
        logger.info("Using DisorderedPhCReflector (2x2 jittered supercell)")
    elif use_dual_hole:
        phc = DualHolePhCReflector(n_rings=phc_n_rings)
        logger.info(
            "Using DualHolePhCReflector (DUAL_TRIANGULAR supercell, 2 distinct hole sizes)"
        )
    elif use_freeform:
        phc = FreeformPhCReflector(lattice_family=lattice_family, n_rings=phc_n_rings)
        logger.info("Using FreeformPhCReflector (Fourier harmonics n=2,3)")
    else:
        phc = PhCReflector(lattice_family=lattice_family, n_rings=phc_n_rings)

    evaluator = ObjectiveEvaluator(
        geometry=phc,
        solver=solver,
        constraints=constraints,
        objectives=make_stage1_objectives(objectives_cfg or {}),
    )
    search_space = SearchSpace.from_geometry(phc)

    logger.info(
        "=== Stage 1 (PhC reflector / lattice=%s) ===",
        lattice_family.value,
    )
    runner = MOBORunner(evaluator, search_space, mobo_config)
    result = runner.run()

    if result.trials:
        best = result.best_by(primary_objective)
        phc.from_param_vector(best.params)
        logger.info(
            "Stage 1 best by %s: %s",
            primary_objective,
            {k: round(v, 4) for k, v in best.objective_values.items()},
        )

    if output_dir is not None:
        save_run_result(result, Path(output_dir))

    return result, phc


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def run_stage2(
    *,
    phc: PhCReflector,
    solver: Optional[ElectromagneticSolver] = None,
    constraints: Optional[FabConstraints] = None,
    objectives_cfg: Optional[dict] = None,
    mobo_config: Optional[MOBOConfig] = None,
    output_dir: Optional[Path] = None,
    primary_objective: str = "stabilization",
) -> tuple[RunResult, MetaGrating]:
    """Run Stage 2 (metagrating) MOBO anchored to ``phc.outer_radius_nm``.

    The returned ``MetaGrating`` is updated in-place with the parameters
    of the trial that maximizes ``primary_objective``.
    """
    solver = solver or MockSolver()
    constraints = constraints or FabConstraints(mode=ConstraintMode.PENALTY)
    mobo_config = mobo_config or MOBOConfig()

    metagrating = MetaGrating(
        inner_radius_nm=phc.outer_radius_nm,
        thickness_nm=phc.thickness_nm,
    )

    evaluator = ObjectiveEvaluator(
        geometry=metagrating,
        solver=solver,
        constraints=constraints,
        objectives=make_stage2_objectives(objectives_cfg or {}),
    )
    search_space = SearchSpace.from_geometry(metagrating)

    logger.info("=== Stage 2 (MetaGrating) ===")
    runner = MOBORunner(evaluator, search_space, mobo_config)
    result = runner.run()

    if result.trials:
        best = result.best_by(primary_objective)
        metagrating.from_param_vector(best.params)
        logger.info(
            "Stage 2 best by %s: %s",
            primary_objective,
            {k: round(v, 4) for k, v in best.objective_values.items()},
        )

    if output_dir is not None:
        save_run_result(result, Path(output_dir))

    return result, metagrating
