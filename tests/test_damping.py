"""Smoke tests for relativistic damping force (docx Eq. 4.8)."""
from __future__ import annotations

import numpy as np
import pytest

grcwa = pytest.importorskip("grcwa")

from lightsail.geometry.metagrating import MetaGrating
from lightsail.simulation.damping import (
    compute_damping_force,
    doppler_factor,
    lorentz_gamma,
    sail_frame_wavelength_nm,
)
from lightsail.simulation.grating_fmm import FMMGratingConfig


_FAST_CFG = FMMGratingConfig(nG=11, nx=64, ny=4)


def _sample_mg():
    return MetaGrating(
        inner_radius_nm=5_000_000.0,
        thickness_nm=240.0,
        grating_period_nm=2000.0,
        duty_cycle=0.5,
        curvature=0.05,
        asymmetry=0.05,
        ring_width_um=2000.0,
    )


def test_doppler_factor_at_beta_zero_is_one():
    assert abs(doppler_factor(0.0) - 1.0) < 1e-12


def test_lorentz_gamma_at_beta_zero_is_one():
    assert abs(lorentz_gamma(0.0) - 1.0) < 1e-12


def test_sail_frame_wavelength_redshifts_with_beta():
    wl_lab = 1550.0
    wl_05 = sail_frame_wavelength_nm(wl_lab, 0.5)
    # D = √((1+0.5)/(1-0.5)) = √3 ≈ 1.732
    np.testing.assert_allclose(wl_05, wl_lab * np.sqrt(3.0), rtol=1e-9)


def test_compute_damping_force_keys():
    mg = _sample_mg()
    out = compute_damping_force(
        mg, beta=0.1, v_y_per_c=1.0e-4, lab_wavelength_nm=1550.0,
        n_radial_bins=2, config=_FAST_CFG,
    )
    for key in (
        "p_dot_y_Pa",
        "term_static_Pa",
        "term_aberration_Pa",
        "term_metasurface_Pa",
        "alpha_damp_Pa_per_mps",
        "doppler_factor_D",
        "lorentz_gamma",
        "C_pr_1",
        "C_pr_2",
        "dC_pr_2_dtheta",
    ):
        assert key in out
        assert np.isfinite(out[key])


def test_metasurface_term_zero_at_beta_zero():
    """At β=0, the (1/D − 1) factor → 0 exactly."""
    mg = _sample_mg()
    out = compute_damping_force(
        mg, beta=0.0, v_y_per_c=1.0e-4, lab_wavelength_nm=1550.0,
        n_radial_bins=2, config=_FAST_CFG,
    )
    assert out["term_metasurface_Pa"] == 0.0
    assert out["alpha_metasurface_Pa_per_mps"] == 0.0
