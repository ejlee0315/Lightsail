"""Tests for geometry modules."""

from __future__ import annotations

import numpy as np
import pytest

from lightsail.geometry.base import (
    HoleShape,
    LatticeFamily,
    Ring,
)
from lightsail.geometry.lattices import (
    HexagonalLattice,
    PentagonalSupercell,
    TriangularLattice,
    make_lattice,
)
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector


# ---------------------------------------------------------------------------
# HoleShape
# ---------------------------------------------------------------------------


class TestHoleShape:
    def test_circle_limit_is_ellipse(self):
        shape = HoleShape(a_nm=300, b_nm=300, n_sides=6, corner_rounding=1.0)
        pts = shape.boundary(n_pts=256)
        radii = np.linalg.norm(pts, axis=1)
        # Fully rounded shape with a == b should be a circle of radius 300
        np.testing.assert_allclose(radii, 300.0, atol=1e-6)

    def test_sharp_square_has_expected_bbox(self):
        # Sharp square inscribed in unit circle has vertex distance 1.
        # With a = b = 500, bounding box width = 2 * (cos(45°) * 1 * 500) * sqrt(2)
        # but more simply, min radius = cos(pi/4) * 500 ≈ 353.55 and
        # max radius = 500.
        shape = HoleShape(a_nm=500, b_nm=500, n_sides=4, corner_rounding=0.0)
        assert shape.max_extent_nm() == pytest.approx(2 * 500.0, rel=1e-3)
        assert shape.min_feature_nm() == pytest.approx(
            2 * np.cos(np.pi / 4) * 500.0, rel=1e-3
        )

    def test_anisotropic_bbox(self):
        shape = HoleShape(a_nm=400, b_nm=200, n_sides=6, corner_rounding=1.0)
        w, h = shape.bounding_box_nm()
        assert w == pytest.approx(2 * 400.0, rel=1e-3)
        assert h == pytest.approx(2 * 200.0, rel=1e-3)

    def test_area_positive(self):
        shape = HoleShape(a_nm=300, b_nm=200, n_sides=5, corner_rounding=0.4)
        assert shape.area_nm2() > 0

    def test_n_sides_validation(self):
        with pytest.raises(ValueError):
            HoleShape(a_nm=100, b_nm=100, n_sides=2)

    def test_negative_a_rejected(self):
        with pytest.raises(ValueError):
            HoleShape(a_nm=-1, b_nm=100, n_sides=4)


# ---------------------------------------------------------------------------
# Lattices
# ---------------------------------------------------------------------------


class TestLattices:
    def test_triangular_periodicity(self):
        lat = TriangularLattice(period_nm=800.0)
        assert lat.nearest_neighbor_distance() == 800.0
        assert lat.unit_cell_area() == pytest.approx(0.5 * np.sqrt(3) * 800**2)

    def test_triangular_site_density(self):
        lat = TriangularLattice(period_nm=800.0)
        sites = lat.generate_sites(extent_nm=10_000.0)
        assert len(sites) > 0
        for x, y in sites:
            assert np.hypot(x, y) <= 5_000.0 + 1e-9

    def test_hexagonal_has_two_sublattices(self):
        lat = HexagonalLattice(period_nm=500.0)
        sites = lat.generate_sites(extent_nm=4_000.0)
        # Honeycomb: the two sublattice sites A=(0, 0) and B=(0, period) are
        # both present in the central patch.
        assert len(sites) > 0
        assert (0.0, 0.0) in [(round(x, 3), round(y, 3)) for (x, y) in sites]
        assert (0.0, 500.0) in [(round(x, 3), round(y, 3)) for (x, y) in sites]
        # NN distance across sublattices equals period_nm by construction
        assert lat.nearest_neighbor_distance() == 500.0

    def test_pentagonal_supercell_motif_count(self):
        lat = PentagonalSupercell(period_nm=2_000.0)
        sites = lat.generate_sites(extent_nm=6_000.0)
        # At least the 5 vertices of a single supercell should land inside
        assert len(sites) >= 5

    def test_make_lattice_factory(self):
        for fam in LatticeFamily:
            lat = make_lattice(fam, period_nm=900.0)
            assert lat.period_nm == 900.0
            assert lat.generate_sites(5_000.0)


# ---------------------------------------------------------------------------
# PhCReflector
# ---------------------------------------------------------------------------


class TestPhCReflector:
    def test_param_roundtrip(self):
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            thickness_nm=500,
            lattice_period_nm=900,
            hole_a_rel=0.35,
            hole_b_rel=0.35,
            hole_rotation_deg=15,
            corner_rounding=0.4,
            shape_parameter=6,
        )
        vec = phc.to_param_vector()
        assert len(vec) == 7

        phc2 = PhCReflector(lattice_family=LatticeFamily.TRIANGULAR)
        phc2.from_param_vector(vec)
        np.testing.assert_array_almost_equal(phc2.to_param_vector(), vec)

    def test_generate_holes_non_empty(self):
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=800,
            hole_a_rel=0.35,
            hole_b_rel=0.35,
        )
        holes = phc.generate_holes()
        assert len(holes) > 0
        # All holes share the same shape object
        shape_ids = {id(h.shape) for h in holes}
        assert len(shape_ids) == 1

    def test_derived_hole_nm_from_relative(self):
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=1000,
            hole_a_rel=0.3,
            hole_b_rel=0.25,
        )
        assert phc.hole_a_nm == pytest.approx(300.0)
        assert phc.hole_b_nm == pytest.approx(250.0)

    def test_to_structure_metadata_contains_lattice_info(self):
        phc = PhCReflector(lattice_family=LatticeFamily.HEXAGONAL)
        structure = phc.to_structure()
        assert structure.has_phc
        assert structure.metadata["lattice_family"] == "hexagonal"
        assert "nearest_neighbor_nm" in structure.metadata
        assert "unit_cell_area_nm2" in structure.metadata

    def test_shape_parameter_rounds_to_int(self):
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            shape_parameter=5.7,
        )
        assert phc.n_sides == 6
        assert phc.hole_shape().n_sides == 6

    def test_shape_parameter_clamped(self):
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            shape_parameter=100.0,
        )
        assert phc.n_sides == 8

    def test_outer_radius_follows_period(self):
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=1000,
            n_rings=5,
        )
        assert phc.outer_radius_nm == 5000


# ---------------------------------------------------------------------------
# MetaGrating
# ---------------------------------------------------------------------------


class TestMetaGrating:
    def test_param_roundtrip(self):
        mg = MetaGrating(
            grating_period_nm=1500,
            duty_cycle=0.5,
            curvature=0.1,
            asymmetry=-0.05,
            ring_width_um=10,
        )
        vec = mg.to_param_vector()
        assert len(vec) == 5

        mg2 = MetaGrating()
        mg2.from_param_vector(vec)
        np.testing.assert_array_almost_equal(mg2.to_param_vector(), vec)

    def test_ring_count_matches_zone_width(self):
        mg = MetaGrating(grating_period_nm=1000.0, duty_cycle=0.5, ring_width_um=10.0)
        rings = mg.generate_rings()
        # 10 µm / 1 µm period = 10 rings
        assert mg.n_rings == 10
        assert len(rings) == 10

    def test_ring_width_from_duty_cycle(self):
        mg = MetaGrating(grating_period_nm=1200.0, duty_cycle=0.5)
        assert mg.ring_width_nm == pytest.approx(600.0)
        assert mg.gap_width_nm == pytest.approx(600.0)

    def test_rings_are_non_overlapping(self):
        mg = MetaGrating(grating_period_nm=1000.0, duty_cycle=0.5)
        rings = mg.generate_rings()
        for i in range(len(rings) - 1):
            assert rings[i + 1].inner_radius_nm >= rings[i].outer_radius_nm

    def test_to_structure(self):
        mg = MetaGrating()
        structure = mg.to_structure()
        assert structure.has_metagrating
        assert not structure.has_phc
        assert structure.metadata["n_rings"] == len(structure.rings)


# ---------------------------------------------------------------------------
# Ring boundary (curvature + asymmetry)
# ---------------------------------------------------------------------------


class TestRingBoundary:
    def test_circle_when_flat(self):
        r = Ring(inner_radius_nm=1000, outer_radius_nm=1500)
        inner, outer = r.boundary(n_pts=128)
        np.testing.assert_allclose(np.linalg.norm(inner, axis=1), 1000.0)
        np.testing.assert_allclose(np.linalg.norm(outer, axis=1), 1500.0)

    def test_curvature_warps_ring(self):
        r = Ring(inner_radius_nm=1000, outer_radius_nm=1500, curvature=0.1)
        inner, _ = r.boundary(n_pts=128)
        radii = np.linalg.norm(inner, axis=1)
        assert radii.max() > 1000.0 * 1.05
        assert radii.min() < 1000.0 * 0.95
