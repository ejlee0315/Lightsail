"""Smoke tests for stiffness matrix computation."""
from __future__ import annotations

import numpy as np
import pytest

grcwa = pytest.importorskip("grcwa")

from lightsail.geometry.metagrating import MetaGrating
from lightsail.simulation.grating_fmm import FMMGratingConfig
from lightsail.simulation.stiffness import (
    DEFAULT_INTENSITY_W_PER_M2,
    StiffnessResult,
    compute_stiffness_matrix,
)


_FAST_CFG = FMMGratingConfig(nG=11, nx=64, ny=4)


def _sample_metagrating():
    return MetaGrating(
        inner_radius_nm=5_000_000.0,        # 5 mm inner radius
        thickness_nm=240.0,
        grating_period_nm=2000.0,           # > λ_max so m=±1 propagates
        duty_cycle=0.5,
        curvature=0.05,
        asymmetry=0.05,
        ring_width_um=2000.0,               # 2 mm radial extent
    )


def test_compute_stiffness_matrix_returns_matrix():
    mg = _sample_metagrating()
    res = compute_stiffness_matrix(
        mg,
        nir_band_nm=(1550.0, 1850.0),
        nir_n_points=2,
        n_radial_bins=2,
        config=_FAST_CFG,
    )
    assert isinstance(res, StiffnessResult)
    M = res.as_matrix()
    assert M.shape == (2, 2)
    # Plane-wave + axisymmetric: k_xx = k_θx = 0 exactly.
    assert M[0, 0] == 0.0
    assert M[1, 0] == 0.0


def test_compute_stiffness_matrix_metadata_consistent():
    mg = _sample_metagrating()
    res = compute_stiffness_matrix(
        mg, nir_band_nm=(1550.0, 1850.0),
        nir_n_points=2, n_radial_bins=2, config=_FAST_CFG,
    )
    # Ring area must equal π(r_outer² − r_inner²)
    r_in = mg.inner_radius_nm * 1e-9
    r_out = mg.outer_radius_nm * 1e-9
    expected_area = float(np.pi * (r_out ** 2 - r_in ** 2))
    np.testing.assert_allclose(res.ring_area_m2, expected_area, rtol=1e-9)
    # k_θθ = k_xθ × r_mean exactly
    expected_k_thth = res.k_xtheta_N_per_rad * res.ring_mean_radius_m
    np.testing.assert_allclose(res.k_thetatheta_Nm_per_rad, expected_k_thth, rtol=1e-9)


def test_intensity_scaling_is_linear():
    mg = _sample_metagrating()
    cfg = _FAST_CFG
    r1 = compute_stiffness_matrix(
        mg, intensity_W_per_m2=1.0e10, nir_n_points=2, n_radial_bins=2, config=cfg,
    )
    r2 = compute_stiffness_matrix(
        mg, intensity_W_per_m2=2.0e10, nir_n_points=2, n_radial_bins=2, config=cfg,
    )
    # σ_xθ scales linearly with intensity → so does k_xθ
    np.testing.assert_allclose(
        r2.k_xtheta_N_per_rad, 2.0 * r1.k_xtheta_N_per_rad, rtol=1e-6,
    )
