"""Central evaluator that runs a set of objectives against one design.

The :class:`ObjectiveEvaluator` is the glue between the optimizer and
the lower-level geometry / solver / constraint stack. Given a
parameter vector:

1. Apply it to the geometry (``geometry.from_param_vector``).
2. Build a :class:`Structure`.
3. Run the :class:`FabConstraints` checker.
4. Construct an :class:`ObjectiveContext` that holds a spectrum cache.
5. Evaluate each :class:`Objective` in the supplied list.
6. Return an :class:`EvaluationResult` with scalar values and
   per-objective metadata.

This decouples the optimization loop from the internal bookkeeping
and makes ``runner.py`` trivial to implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lightsail.constraints.fabrication import FabConstraints
from lightsail.geometry.base import ParametricGeometry
from lightsail.optimization.objectives import (
    Objective,
    ObjectiveContext,
    ObjectiveValue,
)
from lightsail.simulation.base import ElectromagneticSolver


@dataclass
class EvaluationResult:
    """Bundle of everything produced by one design evaluation."""

    params: np.ndarray
    objective_values: dict[str, ObjectiveValue]
    constraint_penalty: float = 0.0
    constraint_violations: list = field(default_factory=list)
    feasible: bool = True

    def scalar_values(self) -> dict[str, float]:
        """Flatten to ``{name: float}`` for the optimizer / logs."""
        return {name: v.value for name, v in self.objective_values.items()}

    def directed_values(self) -> dict[str, float]:
        """Flatten to maximize-convention values (flip sign for minimize)."""
        return {name: v.directed_value for name, v in self.objective_values.items()}


class ObjectiveEvaluator:
    """Runs a list of objectives on a parametric geometry."""

    def __init__(
        self,
        geometry: ParametricGeometry,
        solver: ElectromagneticSolver,
        constraints: FabConstraints,
        objectives: list[Objective],
    ):
        self.geometry = geometry
        self.solver = solver
        self.constraints = constraints
        self.objectives = list(objectives)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, params: np.ndarray) -> EvaluationResult:
        self.geometry.from_param_vector(np.asarray(params, dtype=float))
        structure = self.geometry.to_structure()

        cr = self.constraints.validate(structure)

        ctx = ObjectiveContext(
            structure=structure,
            geometry=self.geometry,
            solver=self.solver,
            constraint_result=cr,
        )

        values: dict[str, ObjectiveValue] = {}
        for obj in self.objectives:
            values[obj.name] = obj.evaluate(ctx)

        return EvaluationResult(
            params=np.asarray(params, dtype=float).copy(),
            objective_values=values,
            constraint_penalty=float(cr.penalty),
            constraint_violations=list(cr.violations),
            feasible=bool(cr.feasible),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def objective_names(self) -> list[str]:
        return [o.name for o in self.objectives]

    @property
    def objective_targets(self) -> dict[str, str]:
        return {o.name: o.target for o in self.objectives}
