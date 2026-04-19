"""Smoke tests for LayeredRCWASolver."""
from __future__ import annotations

import numpy as np
import pytest

grcwa = pytest.importorskip("grcwa")

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.simulation.layered_rcwa import LayeredRCWASolver, LayerSpec
from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver


def _make_phc_structure():
    phc = PhCReflector(
        lattice_family=LatticeFamily.TRIANGULAR,
        thickness_nm=240.0,
        lattice_period_nm=1580.0,
        hole_a_rel=0.38,
        hole_b_rel=0.38,
        hole_rotation_deg=0.0,
        corner_rounding=1.0,
        shape_parameter=8.0,
    )
    return phc.to_structure()


def _fast_config():
    return RCWAConfig(nG=15, grid_nx=48, grid_ny=48)


def test_layered_no_extras_matches_base_solver():
    """With empty layer lists, LayeredRCWASolver ≡ RCWASolver."""
    cfg = _fast_config()
    base = RCWASolver(config=cfg)
    layered = LayeredRCWASolver(config=cfg, layers_above=[], layers_below=[])
    structure = _make_phc_structure()
    wl = np.array([1550.0, 1700.0])
    R_base = base.evaluate_reflectivity(structure, wl)
    R_layered = layered.evaluate_reflectivity(structure, wl)
    np.testing.assert_allclose(R_base, R_layered, atol=1e-6)


def test_layered_with_lossy_backside_reduces_T():
    """Adding a lossy back layer should reduce T."""
    cfg = _fast_config()
    base = RCWASolver(config=cfg)
    structure = _make_phc_structure()
    wl = np.array([1550.0, 1700.0])
    T_base = base.evaluate_transmission(structure, wl)

    lossy = LayerSpec(
        thickness_nm=20.0,
        eps_constant=complex(2.0, 1.5),
        name="absorber",
    )
    layered = LayeredRCWASolver(config=cfg, layers_below=[lossy])
    T_layered = layered.evaluate_transmission(structure, wl)

    assert (T_layered < T_base).all(), f"T_base={T_base}, T_layered={T_layered}"


def test_layer_spec_eps_callable_overrides_constant():
    """When both are set, eps_callable should win over eps_constant."""
    layer = LayerSpec(
        thickness_nm=10.0,
        eps_callable=lambda wl: complex(4.0, 0.0),
        eps_constant=complex(99.0, 0.0),
    )
    assert layer.epsilon(1550.0) == complex(4.0, 0.0)


def test_layer_spec_requires_one_of_eps():
    layer = LayerSpec(thickness_nm=10.0)
    with pytest.raises(ValueError):
        layer.epsilon(1550.0)


def test_describe_stack_includes_layer_names():
    cfg = _fast_config()
    layered = LayeredRCWASolver(
        config=cfg,
        layers_below=[
            LayerSpec(thickness_nm=3.4, eps_constant=complex(2.0, 0.0), name="g_x10"),
        ],
    )
    desc = layered.describe_stack()
    assert "g_x10" in desc
    assert "PhC" in desc
