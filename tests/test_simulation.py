"""Tests for the ElectromagneticSolver interface and MockSolver."""

from __future__ import annotations

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.simulation.mock import MockSolver
from lightsail.simulation.results import SimulationResult


def _default_phc() -> PhCReflector:
    return PhCReflector(
        lattice_family=LatticeFamily.TRIANGULAR,
        lattice_period_nm=1500,
        hole_a_rel=400 / 1500,
        hole_b_rel=400 / 1500,
        corner_rounding=1.0,
        thickness_nm=500,
    )


def _default_metagrating() -> MetaGrating:
    return MetaGrating(
        inner_radius_nm=8000.0,
        thickness_nm=500.0,
        grating_period_nm=1500.0,
        duty_cycle=0.5,
        curvature=0.0,
        asymmetry=0.0,
        ring_width_um=10.0,
    )


class TestMockSolverInterface:
    def test_reflectivity_shape_and_range(self):
        solver = MockSolver()
        structure = _default_phc().to_structure()
        wl = np.linspace(1000, 2000, 64)
        r = solver.evaluate_reflectivity(structure, wl)
        assert r.shape == wl.shape
        assert np.all((r >= 0.0) & (r <= 1.0))

    def test_transmission_shape_and_range(self):
        solver = MockSolver()
        structure = _default_phc().to_structure()
        wl = np.linspace(1000, 2000, 64)
        t = solver.evaluate_transmission(structure, wl)
        assert t.shape == wl.shape
        assert np.all((t >= 0.0) & (t <= 1.0))

    def test_emissivity_matches_kirchhoff(self):
        solver = MockSolver()
        structure = _default_phc().to_structure()
        wl = np.linspace(1000, 15000, 128)
        r = solver.evaluate_reflectivity(structure, wl)
        t = solver.evaluate_transmission(structure, wl)
        e = solver.evaluate_emissivity(structure, wl)
        # ε + R + T ≈ 1 everywhere
        np.testing.assert_allclose(e + r + t, 1.0, atol=1e-6)

    def test_energy_conservation_over_wide_band(self):
        solver = MockSolver()
        structure = _default_phc().to_structure()
        wl = np.linspace(500, 20000, 400)
        r = solver.evaluate_reflectivity(structure, wl)
        t = solver.evaluate_transmission(structure, wl)
        e = solver.evaluate_emissivity(structure, wl)
        assert np.all(r + t + e <= 1.0 + 1e-6)
        assert np.all(r + t + e >= 1.0 - 1e-6)

    def test_compute_spectrum_helper(self):
        solver = MockSolver()
        structure = _default_phc().to_structure()
        wl = np.linspace(1000, 2000, 20)
        result = solver.compute_spectrum(structure, wl)
        assert isinstance(result, SimulationResult)
        assert len(result.reflectance) == 20
        assert "solver" in result.metadata


class TestMockSolverPhysicsIntuition:
    """Sanity checks — NOT precise physics, just monotonic trends."""

    def test_thickness_increases_mir_absorption(self):
        solver = MockSolver()
        thin = _default_phc()
        thin.thickness_nm = 250.0
        thick = _default_phc()
        thick.thickness_nm = 900.0
        mir = (8000.0, 14000.0)
        e_thin = solver.band_mean_emissivity(thin.to_structure(), mir)
        e_thick = solver.band_mean_emissivity(thick.to_structure(), mir)
        assert e_thick > e_thin

    def test_metagrating_is_handled(self):
        """A metagrating-only structure is valid input."""
        solver = MockSolver()
        mg_struct = _default_metagrating().to_structure()
        wl = np.linspace(1000, 2000, 40)
        r = solver.evaluate_reflectivity(mg_struct, wl)
        assert np.all(r >= 0.0)
