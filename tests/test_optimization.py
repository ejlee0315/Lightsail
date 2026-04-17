"""Tests for optimization modules."""

from __future__ import annotations

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.optimization.objectives import NIRReflectivityObjective
from lightsail.optimization.optimizer import BayesianOptimizer
from lightsail.optimization.search_space import SearchSpace


def _phc() -> PhCReflector:
    return PhCReflector(lattice_family=LatticeFamily.TRIANGULAR)


class TestSearchSpace:
    def test_from_geometry(self):
        space = SearchSpace.from_geometry(_phc())
        assert space.n_dims == 7
        assert space.names == [
            "thickness_nm",
            "lattice_period_nm",
            "hole_a_frac",
            "hole_b_frac",
            "hole_rotation_deg",
            "corner_rounding",
            "shape_parameter",
        ]

    def test_normalize_denormalize(self):
        phc = _phc()
        space = SearchSpace.from_geometry(phc)
        params = phc.to_param_vector()

        normalized = space.normalize(params)
        assert np.all(normalized >= 0) and np.all(normalized <= 1)

        recovered = space.denormalize(normalized)
        np.testing.assert_array_almost_equal(recovered, params)

    def test_random_sample_within_bounds(self):
        space = SearchSpace.from_geometry(_phc())
        sample = space.random_sample()
        assert len(sample) == space.n_dims
        for val, (lo, hi) in zip(sample, space.bounds):
            assert lo <= val <= hi


class TestBayesianOptimizer:
    def test_suggest_and_report(self):
        space = SearchSpace.from_geometry(_phc())
        obj = NIRReflectivityObjective()
        optimizer = BayesianOptimizer(space, [obj])

        candidates = optimizer.suggest_next(3)
        assert len(candidates) == 3

        for params in candidates:
            optimizer.report_result(params, {"nir_reflectance": 0.5})

        assert len(optimizer.trials) == 3

    def test_pareto_front(self):
        space = SearchSpace.from_geometry(_phc())
        obj = NIRReflectivityObjective()
        optimizer = BayesianOptimizer(space, [obj])

        for i in range(10):
            params = space.random_sample()
            optimizer.report_result(params, {"nir_reflectance": float(i) / 10})

        pareto = optimizer.get_pareto_front()
        assert pareto.n_solutions >= 1
        best = pareto.best_by("nir_reflectance")
        assert best.objective_values["nir_reflectance"] == 0.9
