"""Tests for the SiN dispersion model."""

from __future__ import annotations

import numpy as np
import pytest

from lightsail.materials import SiNDispersion, sin_permittivity, sin_refractive_index


class TestSiNDispersion:
    def test_nir_is_lossless(self):
        d = SiNDispersion()
        nk = d.nk(1500.0)
        assert nk.imag == pytest.approx(0.0, abs=1e-9)
        assert 1.95 < nk.real < 2.05

    def test_mir_has_absorption_peak(self):
        d = SiNDispersion()
        # Si-N stretch peak near 10.5-11 µm — k should be large.
        k = d.k(11000.0)
        assert k > 1.0

    def test_nir_continuity_across_crossover(self):
        """Luke at ~1500 nm should be close to Kischkat at 1540 nm."""
        d = SiNDispersion()
        n_luke = d.n(1500.0)
        n_kisch = d.n(1540.0)
        assert abs(float(n_luke) - float(n_kisch)) < 0.05

    def test_array_input(self):
        wls = np.array([500.0, 1500.0, 5000.0, 10500.0, 14000.0])
        out = sin_refractive_index(wls)
        assert out.shape == wls.shape
        assert out.dtype == np.complex128
        assert (out.real > 0).all()
        assert (out.imag >= 0).all()

    def test_permittivity_is_square_of_n(self):
        wls = np.array([1500.0, 10500.0])
        nk = sin_refractive_index(wls)
        eps = sin_permittivity(wls)
        np.testing.assert_allclose(eps, nk ** 2, rtol=1e-12)

    def test_scalar_passthrough(self):
        d = SiNDispersion()
        n1 = d.n(1500.0)
        assert np.shape(n1) == ()  # scalar input → scalar-like output
