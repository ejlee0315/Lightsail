"""Tests for fabrication constraints."""

from __future__ import annotations

import pytest

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector


# ---------------------------------------------------------------------------
# PhC constraints
# ---------------------------------------------------------------------------


class TestPhCConstraints:
    def _feasible_phc(self) -> PhCReflector:
        """A deliberately feasible PhC design."""
        return PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            thickness_nm=500,
            lattice_period_nm=1600,
            hole_a_rel=0.25,   # 400 nm
            hole_b_rel=0.25,   # 400 nm
            hole_rotation_deg=0,
            corner_rounding=1.0,   # circle — fully rounded
            shape_parameter=6,
        )

    def test_feasible_phc_hard_mode(self):
        constraints = FabConstraints(mode=ConstraintMode.HARD)
        result = constraints.validate(self._feasible_phc().to_structure())
        assert result.feasible
        assert result.penalty == 0.0
        assert result.violations == []

    def test_feasible_phc_penalty_mode(self):
        constraints = FabConstraints(mode=ConstraintMode.PENALTY)
        result = constraints.validate(self._feasible_phc().to_structure())
        assert result.feasible
        assert result.penalty == 0.0

    def test_hole_too_small_rejected_in_hard_mode(self):
        constraints = FabConstraints(mode=ConstraintMode.HARD, min_feature_nm=500)
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=1600,
            hole_a_rel=150 / 1600,  # min feature = 300 nm < 500
            hole_b_rel=150 / 1600,
            corner_rounding=1.0,
        )
        result = constraints.validate(phc.to_structure())
        assert not result.feasible
        assert result.penalty > 0

    def test_hole_too_small_soft_in_penalty_mode(self):
        constraints = FabConstraints(mode=ConstraintMode.PENALTY, min_feature_nm=500)
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=1600,
            hole_a_rel=150 / 1600,
            hole_b_rel=150 / 1600,
            corner_rounding=1.0,
        )
        result = constraints.validate(phc.to_structure())
        assert result.feasible  # penalty mode never rejects
        assert result.penalty > 0
        assert any("min feature" in v for v in result.violations)

    def test_gap_too_small(self):
        constraints = FabConstraints(mode=ConstraintMode.HARD, min_gap_nm=500)
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=800,
            hole_a_rel=350 / 800,  # 2a = 700, gap = 100 nm
            hole_b_rel=350 / 800,
            corner_rounding=1.0,
        )
        result = constraints.validate(phc.to_structure())
        assert not result.feasible
        assert any("gap" in v for v in result.violations)

    def test_disconnected_geometry_detected(self):
        # Hole saturated at the relative upper bound (0.48) with a short
        # period pushes the walls to the minimum thickness allowed by the
        # parameterization. We can still force an outright-disconnected
        # structure by keeping the hole larger than the lattice spacing —
        # here we bypass from_param_vector and set a_rel above the normal
        # cap to simulate what the checker should catch.
        constraints = FabConstraints(mode=ConstraintMode.HARD)
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=800,
            hole_a_rel=0.48,
            hole_b_rel=0.48,
            corner_rounding=1.0,
        )
        # Push slightly past the cap to provoke a disconnection-style fail.
        # Must set frac to 1.0+ (clamped) AND override rel to bypass the
        # normal mapping so the structure sees oversized holes.
        phc.hole_a_frac = 1.0
        phc.hole_b_frac = 1.0
        phc.hole_a_rel = 0.625  # 2a = 1000 nm > 800 nm period
        phc.hole_b_rel = 0.625
        result = constraints.validate(phc.to_structure())
        assert not result.feasible
        assert any("disconnected" in v.lower() for v in result.violations)

    def test_thickness_out_of_range(self):
        constraints = FabConstraints(
            mode=ConstraintMode.HARD,
            thickness_range_nm=(200, 1000),
        )
        phc = self._feasible_phc()
        phc.thickness_nm = 1500
        result = constraints.validate(phc.to_structure())
        assert not result.feasible

    def test_fill_fraction_sanity(self):
        # Very small hole → fill fraction below sanity range
        constraints = FabConstraints(
            mode=ConstraintMode.HARD,
            min_feature_nm=0,  # disable min-feature violation
            min_gap_nm=0,
            fill_fraction_range=(0.05, 0.60),
        )
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=2000,
            hole_a_rel=0.05,   # 100 nm
            hole_b_rel=0.05,
            corner_rounding=1.0,
        )
        result = constraints.validate(phc.to_structure())
        assert any("fill fraction" in v for v in result.violations)


# ---------------------------------------------------------------------------
# MetaGrating constraints
# ---------------------------------------------------------------------------


class TestMetaGratingConstraints:
    def test_feasible_metagrating(self):
        constraints = FabConstraints(mode=ConstraintMode.HARD)
        mg = MetaGrating(
            grating_period_nm=1500,
            duty_cycle=0.5,
            curvature=0.0,
            asymmetry=0.0,
            ring_width_um=10,
        )
        result = constraints.validate(mg.to_structure())
        assert result.feasible

    def test_ring_too_narrow(self):
        constraints = FabConstraints(
            mode=ConstraintMode.HARD,
            min_feature_nm=500,
            min_gap_nm=0,
        )
        mg = MetaGrating(
            grating_period_nm=1000,
            duty_cycle=0.2,   # width = 200 nm < 500
            ring_width_um=10,
        )
        result = constraints.validate(mg.to_structure())
        assert not result.feasible
        assert any("ring" in v.lower() and "width" in v for v in result.violations)

    def test_ring_gap_too_small(self):
        constraints = FabConstraints(
            mode=ConstraintMode.HARD,
            min_feature_nm=0,
            min_gap_nm=500,
        )
        mg = MetaGrating(
            grating_period_nm=1000,
            duty_cycle=0.8,   # gap = 200 nm < 500
            ring_width_um=10,
        )
        result = constraints.validate(mg.to_structure())
        assert not result.feasible

    def test_extreme_curvature_penalized(self):
        constraints = FabConstraints(mode=ConstraintMode.PENALTY)
        mg = MetaGrating(
            grating_period_nm=1500,
            duty_cycle=0.5,
            curvature=0.45,
            asymmetry=0.45,
            ring_width_um=10,
        )
        result = constraints.validate(mg.to_structure())
        # penalty mode is always feasible=True but penalty > 0
        assert result.penalty > 0
