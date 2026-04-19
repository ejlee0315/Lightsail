"""Smoke tests for graphene conductivity model."""
from __future__ import annotations

import numpy as np
import pytest

from lightsail.materials.graphene import (
    GRAPHENE_LAYER_THICKNESS_M,
    GrapheneConductivity,
    graphene_layer_eps,
)


def test_universal_absorption_constant_is_2_3_percent():
    """π e² / (ℏ c) ≈ 0.023 — graphene's celebrated universal absorption."""
    alpha = GrapheneConductivity.universal_absorption()
    # Allow ±2% tolerance around the textbook value 0.0229
    assert 0.022 < alpha < 0.024, f"α = {alpha}"


def test_epsilon_complex_at_telecom():
    """At 1550 nm with E_F=0.3 eV: ℏω ≈ 0.8 eV > 2 E_F → strong interband."""
    g = GrapheneConductivity(E_F_eV=0.3)
    eps = g.epsilon(1550.0)
    assert eps.imag > 0.0, f"Im(ε) should be positive (lossy), got {eps}"
    # Real part can be < 1 (graphene is metallic-ish in this band)
    assert isinstance(eps, complex)


def test_intra_dominates_at_mid_ir():
    """At λ = 10 µm (MIR), Drude (intraband) should dominate over interband."""
    g = GrapheneConductivity(E_F_eV=0.3)
    sigma_intra = g.sigma_intra(10_000.0)
    sigma_inter = g.sigma_inter(10_000.0)
    # |Im(σ_intra)| > |σ_inter| at MIR
    assert abs(sigma_intra.imag) > abs(sigma_inter), (
        f"intra={sigma_intra}, inter={sigma_inter}"
    )


def test_callable_signature_matches_layer_spec():
    """epsilon_callable should be λ_nm → complex, suitable for LayerSpec."""
    g = GrapheneConductivity(E_F_eV=0.3)
    fn = g.epsilon_callable()
    val = fn(1550.0)
    assert isinstance(val, complex)


def test_free_function_matches_class():
    g = GrapheneConductivity(E_F_eV=0.4)
    e1 = g.epsilon(1550.0)
    e2 = graphene_layer_eps(1550.0, E_F_eV=0.4)
    assert e1 == e2
