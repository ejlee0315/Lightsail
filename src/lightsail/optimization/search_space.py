"""Search space management for Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lightsail.geometry.base import ParametricGeometry


@dataclass
class SearchSpace:
    """Defines the parameter search space from a ParametricGeometry.

    Handles normalization to [0, 1] for optimizer compatibility
    and denormalization back to physical units.
    """

    names: list[str]
    bounds: list[tuple[float, float]]
    integer_params: set[str] = field(default_factory=set)

    @classmethod
    def from_geometry(
        cls,
        geometry: ParametricGeometry,
        integer_params: set[str] | None = None,
    ) -> SearchSpace:
        """Create SearchSpace from a ParametricGeometry instance."""
        return cls(
            names=geometry.param_names(),
            bounds=geometry.param_bounds(),
            integer_params=integer_params or set(),
        )

    @property
    def n_dims(self) -> int:
        return len(self.names)

    def normalize(self, params: np.ndarray) -> np.ndarray:
        """Map physical parameters to [0, 1] range."""
        normalized = np.zeros_like(params, dtype=float)
        for i, (lo, hi) in enumerate(self.bounds):
            if hi > lo:
                normalized[i] = (params[i] - lo) / (hi - lo)
            else:
                normalized[i] = 0.5
        return normalized

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Map [0, 1] values back to physical parameters."""
        params = np.zeros_like(normalized, dtype=float)
        for i, (lo, hi) in enumerate(self.bounds):
            params[i] = lo + normalized[i] * (hi - lo)
            if self.names[i] in self.integer_params:
                params[i] = round(params[i])
        return params

    def random_sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random parameter vector within bounds."""
        rng = rng or np.random.default_rng()
        normalized = rng.uniform(0, 1, size=self.n_dims)
        return self.denormalize(normalized)
