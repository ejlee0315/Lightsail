"""Smoke tests for h-BN multi-oscillator dispersion."""
from __future__ import annotations

import numpy as np

from lightsail.materials.hbn import HBNDispersion, hbn_epsilon


def test_eps_inf_at_short_wavelength():
    """At λ = 1 µm (well below both phonon bands), ε ≈ ε_∞ ≈ 4.5."""
    hbn = HBNDispersion()
    eps = hbn.epsilon(1000.0)
    assert 4.0 < eps.real < 5.5
    assert abs(eps.imag) < 1e-2


def test_negative_eps_in_in_plane_reststrahlen():
    """Between 6.2 and 7.3 µm Re(ε) < 0 (in-plane phonon Reststrahlen)."""
    hbn = HBNDispersion()
    eps = hbn.epsilon(6800.0)  # ≈ middle of 6.2–7.3 µm band
    assert eps.real < 0.0


def test_negative_eps_in_out_of_plane_reststrahlen():
    """Between 12.1 and 13.2 µm Re(ε) < 0 (out-of-plane band)."""
    hbn = HBNDispersion()
    eps = hbn.epsilon(12700.0)
    assert eps.real < 0.0


def test_two_distinct_resonances():
    """Im(ε) has two distinct peaks across MIR (6.5 µm and 12.7 µm)."""
    hbn = HBNDispersion()
    wls = np.linspace(2000, 16000, 200)
    eps = np.array([hbn.epsilon(float(w)) for w in wls])
    im = np.imag(eps)
    # Find local maxima above a moderate threshold
    high_pts = wls[im > 50.0]
    assert len(high_pts) > 0
    # Both bands should produce high Im(ε)
    in_plane_band = np.any((wls >= 6200) & (wls <= 7300) & (im > 50.0))
    oop_band = np.any((wls >= 12100) & (wls <= 13200) & (im > 50.0))
    assert in_plane_band, "no high-Im(ε) point in 6.2–7.3 µm band"
    assert oop_band, "no high-Im(ε) point in 12.1–13.2 µm band"


def test_callable_signature():
    fn = HBNDispersion().epsilon_callable()
    assert isinstance(fn(7000.0), complex)


def test_free_function_matches_class():
    e1 = HBNDispersion().epsilon(7000.0)
    e2 = hbn_epsilon(7000.0)
    assert e1 == e2
