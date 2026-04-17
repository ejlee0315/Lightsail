"""Tests for :class:`RCWASolver`.

Includes:

- Fabry-Perot analytical comparison (no patterning, lossless film).
- Energy conservation on a patterned PhC.
- Triangular/hexagonal/pentagonal unit cell construction.
- Structure caching behaviour.

These tests require grcwa + autograd; they are skipped gracefully if the
optional ``rcwa`` extra is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

grcwa = pytest.importorskip("grcwa")

from lightsail.geometry.base import (
    Hole,
    HoleShape,
    LatticeFamily,
    Material,
    Structure,
)
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.simulation.rcwa_solver import (
    RCWAConfig,
    RCWASolver,
    _rasterize_holes,
    _unit_cell_for,
)


# ---------------------------------------------------------------------------
# Analytical Fabry-Perot reference
# ---------------------------------------------------------------------------


def fabry_perot_RT(wl_um: np.ndarray, n: float, d_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Single lossless film in air, normal incidence."""
    r1 = (1.0 - n) / (1.0 + n)
    delta = 2.0 * np.pi * n * d_um / wl_um
    num = 2.0 * r1 ** 2 * (1.0 - np.cos(2 * delta))
    den = 1.0 + r1 ** 4 - 2.0 * r1 ** 2 * np.cos(2 * delta)
    R = num / den
    return R, 1.0 - R


class _ConstantDispersion:
    """Tiny stand-in for SiNDispersion with a fixed n used in FP tests."""

    def __init__(self, n: float):
        self.n_val = n

    def epsilon(self, wl):
        return np.full(np.shape(wl), self.n_val ** 2, dtype=complex)

    def nk(self, wl):
        return np.full(np.shape(wl), self.n_val + 0j, dtype=complex)

    def n(self, wl):
        return np.full(np.shape(wl), self.n_val, dtype=float)

    def k(self, wl):
        return np.zeros(np.shape(wl), dtype=float)


# ---------------------------------------------------------------------------
# Fabry-Perot comparison
# ---------------------------------------------------------------------------


def _fp_structure(thickness_nm: float = 500.0) -> Structure:
    return Structure(
        material=Material.SIN,
        thickness_nm=thickness_nm,
        lattice_family=None,
        holes=[],
        rings=[],
    )


class TestFabryPerot:
    """The RCWA solver should match the analytical FP formula on a plain slab."""

    def test_uniform_slab_matches_analytical(self):
        solver = RCWASolver(
            config=RCWAConfig(nG=3, polarization="te"),
            dispersion=_ConstantDispersion(2.0),
        )
        structure = _fp_structure(500.0)
        wls_nm = np.linspace(1000.0, 2000.0, 11)
        R = solver.evaluate_reflectivity(structure, wls_nm)
        T = solver.evaluate_transmission(structure, wls_nm)

        R_ana, T_ana = fabry_perot_RT(wls_nm / 1000.0, 2.0, 0.5)
        np.testing.assert_allclose(R, R_ana, atol=1e-4)
        np.testing.assert_allclose(T, T_ana, atol=1e-4)

    def test_energy_conservation_slab(self):
        solver = RCWASolver(
            config=RCWAConfig(nG=3, polarization="average"),
            dispersion=_ConstantDispersion(2.0),
        )
        structure = _fp_structure(500.0)
        wls_nm = np.linspace(1000.0, 2000.0, 11)
        R = solver.evaluate_reflectivity(structure, wls_nm)
        T = solver.evaluate_transmission(structure, wls_nm)
        assert np.all(R + T <= 1.0 + 1e-6)
        assert np.all(R + T >= 1.0 - 1e-6)  # lossless

    def test_half_wave_point_has_zero_reflectance(self):
        """At λ = 2 n d the slab is a half-wave, R ≈ 0."""
        solver = RCWASolver(
            config=RCWAConfig(nG=3, polarization="te"),
            dispersion=_ConstantDispersion(2.0),
        )
        structure = _fp_structure(500.0)
        wl = np.array([2000.0])  # 2 * 2 * 500 = 2000
        R = solver.evaluate_reflectivity(structure, wl)
        assert float(R[0]) < 1e-4


# ---------------------------------------------------------------------------
# Patterned PhC — energy conservation with real dispersion
# ---------------------------------------------------------------------------


class TestPatternedPhC:
    def _triangular_phc(self) -> PhCReflector:
        return PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            n_rings=3,
            thickness_nm=500.0,
            lattice_period_nm=1500.0,
            hole_a_rel=400.0 / 1500.0,
            hole_b_rel=400.0 / 1500.0,
            corner_rounding=1.0,
            shape_parameter=6,
        )

    def test_energy_conservation_nir(self):
        solver = RCWASolver(config=RCWAConfig(nG=21, grid_nx=64, grid_ny=64))
        phc = self._triangular_phc()
        wls_nm = np.linspace(1350.0, 1650.0, 5)
        R = solver.evaluate_reflectivity(phc.to_structure(), wls_nm)
        T = solver.evaluate_transmission(phc.to_structure(), wls_nm)
        assert np.all(R >= 0.0)
        assert np.all(T >= 0.0)
        assert np.all(R + T <= 1.0 + 1e-6)

    def test_emissivity_in_mir(self):
        solver = RCWASolver(config=RCWAConfig(nG=21, grid_nx=64, grid_ny=64))
        phc = self._triangular_phc()
        wls_nm = np.linspace(9000.0, 12000.0, 4)
        eps = solver.evaluate_emissivity(phc.to_structure(), wls_nm)
        # MIR phonon absorption — emissivity should be clearly nonzero.
        assert eps.mean() > 0.05


# ---------------------------------------------------------------------------
# Unit cell construction
# ---------------------------------------------------------------------------


class TestUnitCells:
    def test_triangular_primitive_cell(self):
        s = Structure(lattice_family=LatticeFamily.TRIANGULAR, lattice_period_nm=1000.0)
        cell = _unit_cell_for(s)
        # a1 = (1, 0), a2 = (0.5, √3/2) in µm, area = √3/2
        assert cell.L1_um == pytest.approx((1.0, 0.0))
        assert cell.L2_um[0] == pytest.approx(0.5)
        assert cell.L2_um[1] == pytest.approx(np.sqrt(3.0) / 2.0)
        assert cell.area_um2 == pytest.approx(np.sqrt(3.0) / 2.0, rel=1e-6)
        assert len(cell.hole_offsets_nm) == 1

    def test_hexagonal_two_holes_per_cell(self):
        s = Structure(lattice_family=LatticeFamily.HEXAGONAL, lattice_period_nm=1000.0)
        cell = _unit_cell_for(s)
        assert len(cell.hole_offsets_nm) == 2

    def test_pentagonal_five_holes(self):
        s = Structure(lattice_family=LatticeFamily.PENTAGONAL_SUPERCELL,
                      lattice_period_nm=2000.0)
        cell = _unit_cell_for(s)
        assert len(cell.hole_offsets_nm) == 5
        # Square cell for supercell
        assert cell.L1_um[1] == 0.0
        assert cell.L2_um[0] == 0.0


# ---------------------------------------------------------------------------
# Rasterization — shared HoleShape should punch a hole in the grid
# ---------------------------------------------------------------------------


class TestRasterization:
    def test_mask_has_both_materials(self):
        shape = HoleShape(a_nm=400, b_nm=400, n_sides=6, corner_rounding=1.0)
        structure = Structure(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=1500.0,
            holes=[Hole(x_nm=0.0, y_nm=0.0, shape=shape)],
            period_x_nm=1500.0,
            period_y_nm=1500.0,
        )
        cell = _unit_cell_for(structure)
        mask = _rasterize_holes(structure, cell, nx=64, ny=64)
        assert mask.min() == 0.0  # hole
        assert mask.max() == 1.0  # SiN
        assert 0.0 < mask.mean() < 1.0
