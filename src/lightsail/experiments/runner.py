"""Experiment runner: orchestrates a single optimization stage.

The runner is deliberately thin. All the heavy lifting is in
:class:`ObjectiveEvaluator`, which owns the solver, constraints,
geometry, and objective list. The runner only handles the BO loop
and result bookkeeping.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lightsail.constraints.fabrication import FabConstraints
from lightsail.geometry.base import ParametricGeometry
from lightsail.optimization.evaluator import (
    EvaluationResult,
    ObjectiveEvaluator,
)
from lightsail.optimization.objectives import Objective
from lightsail.optimization.optimizer import BayesianOptimizer, ParetoFront
from lightsail.optimization.search_space import SearchSpace
from lightsail.simulation.base import ElectromagneticSolver

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Output of a single optimization stage."""

    best_params: np.ndarray
    best_objectives: dict[str, float]
    pareto_front: ParetoFront
    all_trials: list[dict]


class ExperimentRunner:
    """Runs one optimization stage (Stage 1 or Stage 2)."""

    def __init__(
        self,
        geometry: ParametricGeometry,
        solver: ElectromagneticSolver,
        objectives: list[Objective],
        constraints: FabConstraints,
        n_iterations: int = 50,
        batch_size: int = 1,
        seed: int = 42,
        pareto_primary: str = "nir_reflectance",
    ):
        self.geometry = geometry
        self.solver = solver
        self.objectives = list(objectives)
        self.constraints = constraints
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.pareto_primary = pareto_primary

        self.search_space = SearchSpace.from_geometry(geometry)
        self.optimizer = BayesianOptimizer(
            search_space=self.search_space,
            objectives=self.objectives,
            seed=seed,
        )
        self.evaluator = ObjectiveEvaluator(
            geometry=geometry,
            solver=solver,
            constraints=constraints,
            objectives=self.objectives,
        )

        self._detailed_history: list[EvaluationResult] = []

    # ------------------------------------------------------------------

    def run(self) -> StageResult:
        logger.info(
            "Starting optimization: %d iterations, %d params, %d objectives",
            self.n_iterations,
            self.search_space.n_dims,
            len(self.objectives),
        )

        for iteration in range(self.n_iterations):
            candidates = self.optimizer.suggest_next(self.batch_size)
            for params in candidates:
                eval_result = self.evaluator.evaluate(params)
                self._detailed_history.append(eval_result)
                self.optimizer.report_result(params, eval_result.scalar_values())

            if (iteration + 1) % 10 == 0:
                best = self._current_best()
                logger.info(
                    "Iteration %d/%d — best: %s",
                    iteration + 1,
                    self.n_iterations,
                    best,
                )

        pareto = self.optimizer.get_pareto_front()
        if pareto.trials:
            best_trial = pareto.best_by(self.pareto_primary, maximize=True)
        else:
            best_trial = self.optimizer.trials[-1]

        return StageResult(
            best_params=best_trial.params,
            best_objectives=best_trial.objective_values,
            pareto_front=pareto,
            all_trials=[
                {
                    "trial_id": t.trial_id,
                    "params": t.params.tolist(),
                    "objectives": t.objective_values,
                }
                for t in self.optimizer.trials
            ],
        )

    # ------------------------------------------------------------------

    def _current_best(self) -> dict[str, float]:
        if not self.optimizer.trials:
            return {}
        pareto = self.optimizer.get_pareto_front()
        if pareto.trials:
            return pareto.best_by(
                self.pareto_primary, maximize=True
            ).objective_values
        return self.optimizer.trials[-1].objective_values

    # ------------------------------------------------------------------

    def save_results(self, result: StageResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "best_params.npy", result.best_params)

        with open(output_dir / "results.json", "w") as f:
            json.dump(
                {
                    "best_objectives": result.best_objectives,
                    "n_pareto": result.pareto_front.n_solutions,
                    "n_trials": len(result.all_trials),
                    "trials": result.all_trials,
                },
                f,
                indent=2,
            )
        logger.info("Results saved to %s", output_dir)
