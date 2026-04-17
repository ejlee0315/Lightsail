"""Bayesian optimization wrapper.

This module defines the optimizer interface. The actual BO backend
(BoTorch/Ax) is not implemented yet — a random-search fallback
is provided for pipeline testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lightsail.optimization.objectives import Objective
from lightsail.optimization.search_space import SearchSpace


@dataclass
class TrialResult:
    """Record of a single optimization trial."""

    trial_id: int
    params: np.ndarray
    objective_values: dict[str, float]


@dataclass
class ParetoFront:
    """Collection of Pareto-optimal solutions."""

    trials: list[TrialResult]

    @property
    def n_solutions(self) -> int:
        return len(self.trials)

    def best_by(self, objective_name: str, maximize: bool = True) -> TrialResult:
        """Return the trial with the best value for a given objective."""
        key = lambda t: t.objective_values.get(objective_name, float("-inf"))
        return max(self.trials, key=key) if maximize else min(self.trials, key=key)


class BayesianOptimizer:
    """Multi-objective Bayesian optimization interface.

    Currently uses random search as a fallback. Replace with
    BoTorch/Ax integration when the optimization backend is ready.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objectives: list[Objective],
        seed: int = 42,
    ):
        self.search_space = search_space
        self.objectives = objectives
        self.rng = np.random.default_rng(seed)
        self.trials: list[TrialResult] = []
        self._trial_counter = 0

    def suggest_next(self, n_candidates: int = 1) -> list[np.ndarray]:
        """Suggest next parameter vectors to evaluate.

        TODO: Replace random sampling with BO acquisition function.
        """
        candidates = []
        for _ in range(n_candidates):
            params = self.search_space.random_sample(self.rng)
            candidates.append(params)
        return candidates

    def report_result(
        self,
        params: np.ndarray,
        objective_values: dict[str, float],
    ) -> TrialResult:
        """Record the result of evaluating a parameter vector."""
        trial = TrialResult(
            trial_id=self._trial_counter,
            params=params.copy(),
            objective_values=dict(objective_values),
        )
        self.trials.append(trial)
        self._trial_counter += 1
        return trial

    def get_pareto_front(self) -> ParetoFront:
        """Extract Pareto-optimal solutions from all trials.

        Uses simple non-dominated sorting over maximize objectives.
        """
        if not self.trials:
            return ParetoFront(trials=[])

        obj_names = [o.name for o in self.objectives]
        signs = [1.0 if o.target == "maximize" else -1.0 for o in self.objectives]

        # Build objective matrix (higher = better after sign flip)
        n = len(self.trials)
        obj_matrix = np.zeros((n, len(obj_names)))
        for i, trial in enumerate(self.trials):
            for j, (name, sign) in enumerate(zip(obj_names, signs)):
                obj_matrix[i, j] = sign * trial.objective_values.get(name, 0.0)

        # Non-dominated sorting
        pareto_indices = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        return ParetoFront(trials=[self.trials[i] for i in pareto_indices])
