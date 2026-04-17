"""Tests for the initial sampling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lightsail.optimization.initial_sampling import (
    initial_samples,
    latin_hypercube,
    sobol_samples,
)


class TestLatinHypercube:
    def test_shape_and_range(self):
        X = latin_hypercube(32, 5, seed=42)
        assert X.shape == (32, 5)
        assert (X >= 0.0).all() and (X <= 1.0).all()

    def test_reproducible(self):
        X1 = latin_hypercube(16, 4, seed=7)
        X2 = latin_hypercube(16, 4, seed=7)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_diverge(self):
        X1 = latin_hypercube(16, 4, seed=1)
        X2 = latin_hypercube(16, 4, seed=2)
        assert not np.array_equal(X1, X2)

    def test_stratification(self):
        """Every 1/n cell contains exactly one sample per dimension."""
        n, d = 20, 3
        X = latin_hypercube(n, d, seed=99)
        for j in range(d):
            bins = np.floor(X[:, j] * n).astype(int)
            bins = np.clip(bins, 0, n - 1)
            assert len(np.unique(bins)) == n


class TestSobol:
    def test_shape_and_range(self):
        try:
            X = sobol_samples(16, 4, seed=7)
        except ImportError:
            pytest.skip("torch not installed")
        assert X.shape == (16, 4)
        assert (X >= 0.0).all() and (X <= 1.0).all()

    def test_reproducible(self):
        try:
            X1 = sobol_samples(32, 5, seed=3)
            X2 = sobol_samples(32, 5, seed=3)
        except ImportError:
            pytest.skip("torch not installed")
        np.testing.assert_array_equal(X1, X2)


class TestDispatcher:
    def test_lhs(self):
        X = initial_samples(16, 3, method="lhs", seed=1)
        assert X.shape == (16, 3)

    def test_latin_hypercube_alias(self):
        X = initial_samples(16, 3, method="latin_hypercube", seed=1)
        assert X.shape == (16, 3)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            initial_samples(16, 3, method="not_a_method")
