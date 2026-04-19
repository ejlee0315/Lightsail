"""Smoke tests for SiC Reststrahlen dispersion."""
from __future__ import annotations

import numpy as np
import pytest

from lightsail.materials.sic import SiCDispersion, sic_epsilon


def test_eps_inf_at_short_wavelength():
    """At λ ≪ Reststrahlen band (e.g. 0.5 µm), ε → ε_∞ ≈ 6.7."""
    sic = SiCDispersion()
    eps = sic.epsilon(500.0)
    assert 6.5 < eps.real < 7.0
    assert abs(eps.imag) < 1e-2


def test_negative_eps_in_reststrahlen():
    """Between ω_TO and ω_LO (10.3–12.6 µm), Re(ε) < 0."""
    sic = SiCDispersion()
    eps_11um = sic.epsilon(11_000.0)
    assert eps_11um.real < 0.0


def test_large_imaginary_at_TO_phonon():
    """Im(ε) peaks at the TO phonon (≈ 12.55 µm) due to narrow γ."""
    sic = SiCDispersion()
    eps_TO = sic.epsilon(12_550.0)
    # With γ=4.76 cm⁻¹, the resonant Im(ε) is on the order of hundreds.
    assert eps_TO.imag > 50.0


def test_callable_signature_matches_layer_spec():
    sic = SiCDispersion()
    fn = sic.epsilon_callable()
    val = fn(11_000.0)
    assert isinstance(val, complex)


def test_free_function_matches_class():
    e1 = SiCDispersion().epsilon(11_000.0)
    e2 = sic_epsilon(11_000.0)
    assert e1 == e2
