"""Smoke tests for the 1D-grating FMM module."""
from __future__ import annotations

import numpy as np
import pytest

grcwa = pytest.importorskip("grcwa")

from lightsail.simulation.grating_fmm import (
    FMMGratingConfig,
    aggregate_metagrating_response,
    compute_dC_pr_dtheta,
    compute_lateral_coefficients,
    evaluate_1d_grating,
)


_FAST_CFG = FMMGratingConfig(nG=11, nx=64, ny=4)


def test_evaluate_1d_grating_power_balance():
    """For lossless SiN in NIR, R_total + T_total ≈ 1."""
    res = evaluate_1d_grating(
        period_nm=1500.0,
        duty_cycle=0.5,
        thickness_nm=240.0,
        wavelength_nm=1550.0,
        theta_deg=0.0,
        config=_FAST_CFG,
    )
    total = res.R_total + res.T_total
    assert 0.95 <= total <= 1.05, f"R+T = {total}"


def test_compute_lateral_coefficients_keys():
    out = compute_lateral_coefficients(
        period_nm=1500.0,
        duty_cycle=0.5,
        thickness_nm=240.0,
        wavelength_nm=1550.0,
        config=_FAST_CFG,
    )
    for key in ("C_pr_0", "C_pr_1", "C_pr_2", "R_total", "T_total"):
        assert key in out
    assert out["C_pr_0"] >= 0


def test_C_pr_1_small_for_symmetric_grating_at_normal_incidence():
    """Symmetric duty=0.5 grating at θ=0 → ⟨sin θ_m⟩(R+T) ≈ 0 by ±m symmetry."""
    out = compute_lateral_coefficients(
        period_nm=1500.0,
        duty_cycle=0.5,
        thickness_nm=240.0,
        wavelength_nm=1550.0,
        theta_deg=0.0,
        config=_FAST_CFG,
    )
    assert abs(out["C_pr_1"]) < 5e-3, f"C_pr_1 = {out['C_pr_1']}"


def test_dC_pr_dtheta_returns_finite_derivatives():
    d = compute_dC_pr_dtheta(
        period_nm=1500.0,
        duty_cycle=0.5,
        thickness_nm=240.0,
        wavelength_nm=1550.0,
        theta_center_deg=0.0,
        dtheta_deg=2.0,
        config=_FAST_CFG,
    )
    for key in ("dC_pr_0_dtheta", "dC_pr_1_dtheta", "dC_pr_2_dtheta"):
        assert key in d
        assert np.isfinite(d[key])


def test_aggregate_metagrating_response_shape():
    agg = aggregate_metagrating_response(
        grating_period_nm=1500.0,
        duty_cycle=0.5,
        thickness_nm=240.0,
        wavelengths_nm=np.array([1550.0, 1700.0]),
        curvature=0.05,
        n_radial_bins=2,
        config=_FAST_CFG,
    )
    assert agg["n"] == 4
    assert "mean_dC_pr_1_dtheta" in agg
    assert "mean_dC_pr_2_dtheta" in agg
    assert len(agg["bin_periods_nm"]) == 2
