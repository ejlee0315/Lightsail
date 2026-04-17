"""Tests for objectives, stabilization proxies, and the evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.optimization.evaluator import (
    EvaluationResult,
    ObjectiveEvaluator,
)
from lightsail.optimization.objectives import (
    AsymmetryStabilizationProxy,
    FabricationPenaltyObjective,
    MIREmissivityObjective,
    MassAndFabPenaltyObjective,
    NIRReflectivityObjective,
    ObjectiveContext,
    RadialMomentumProxy,
    StabilizationProxyObjective,
    make_stage1_objectives,
    make_stage2_objectives,
)
from lightsail.simulation.mock import MockSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _feasible_phc() -> PhCReflector:
    return PhCReflector(
        lattice_family=LatticeFamily.TRIANGULAR,
        thickness_nm=500.0,
        lattice_period_nm=1500.0,
        hole_a_rel=400.0 / 1500.0,
        hole_b_rel=400.0 / 1500.0,
        hole_rotation_deg=0.0,
        corner_rounding=1.0,
        shape_parameter=6.0,
    )


def _feasible_metagrating() -> MetaGrating:
    return MetaGrating(
        inner_radius_nm=9000.0,
        thickness_nm=500.0,
        grating_period_nm=1500.0,
        duty_cycle=0.5,
        curvature=0.1,
        asymmetry=0.05,
        ring_width_um=10.0,
    )


def _phc_context() -> ObjectiveContext:
    phc = _feasible_phc()
    solver = MockSolver()
    cc = FabConstraints(mode=ConstraintMode.PENALTY)
    return ObjectiveContext(
        structure=phc.to_structure(),
        geometry=phc,
        solver=solver,
        constraint_result=cc.validate(phc.to_structure()),
    )


def _mg_context() -> ObjectiveContext:
    mg = _feasible_metagrating()
    solver = MockSolver()
    cc = FabConstraints(mode=ConstraintMode.PENALTY)
    return ObjectiveContext(
        structure=mg.to_structure(),
        geometry=mg,
        solver=solver,
        constraint_result=cc.validate(mg.to_structure()),
    )


# ---------------------------------------------------------------------------
# Single-objective tests
# ---------------------------------------------------------------------------


class TestNIRReflectivity:
    def test_returns_in_unit_interval(self):
        ctx = _phc_context()
        obj = NIRReflectivityObjective()
        val = obj.evaluate(ctx)
        assert 0.0 <= val.value <= 1.0
        assert val.target == "maximize"
        assert "mean_R" in val.metadata
        assert "min_R" in val.metadata

    def test_pure_min_vs_pure_mean(self):
        ctx = _phc_context()
        mean_only = NIRReflectivityObjective(mean_weight=1.0, min_weight=0.0)
        min_only = NIRReflectivityObjective(mean_weight=0.0, min_weight=1.0)
        v_mean = mean_only.evaluate(ctx)
        v_min = min_only.evaluate(ctx)
        assert v_mean.value >= v_min.value - 1e-9  # mean >= min


class TestMIREmissivity:
    def test_returns_in_unit_interval(self):
        ctx = _phc_context()
        obj = MIREmissivityObjective()
        val = obj.evaluate(ctx)
        assert 0.0 <= val.value <= 1.0
        assert val.target == "maximize"

    def test_thicker_film_has_higher_emissivity(self):
        phc_thin = _feasible_phc()
        phc_thin.thickness_nm = 250.0
        phc_thick = _feasible_phc()
        phc_thick.thickness_nm = 900.0

        solver = MockSolver()
        cc = FabConstraints(mode=ConstraintMode.PENALTY)

        def evaluate(phc):
            ctx = ObjectiveContext(
                structure=phc.to_structure(),
                geometry=phc,
                solver=solver,
                constraint_result=cc.validate(phc.to_structure()),
            )
            return MIREmissivityObjective().evaluate(ctx).value

        assert evaluate(phc_thick) > evaluate(phc_thin)


class TestMassAndFab:
    def test_penalty_minimize(self):
        ctx = _phc_context()
        obj = MassAndFabPenaltyObjective()
        val = obj.evaluate(ctx)
        assert val.target == "minimize"
        assert val.value >= 0.0
        assert "mass_norm" in val.metadata
        assert "fab_penalty" in val.metadata

    def test_zero_fab_penalty_for_feasible(self):
        ctx = _phc_context()
        obj = MassAndFabPenaltyObjective()
        val = obj.evaluate(ctx)
        assert val.metadata["fab_penalty"] == 0.0


class TestFabricationPenalty:
    def test_feasible_phc_has_zero_penalty(self):
        ctx = _phc_context()
        val = FabricationPenaltyObjective().evaluate(ctx)
        assert val.value == 0.0

    def test_infeasible_phc_has_positive_penalty(self):
        phc = PhCReflector(
            lattice_family=LatticeFamily.TRIANGULAR,
            lattice_period_nm=800,
            hole_a_rel=150 / 800,
            hole_b_rel=150 / 800,
            corner_rounding=1.0,
        )
        # Use a tighter constraint than the default 100 nm so that
        # this 300 nm-diameter hole actually violates min_feature.
        cc = FabConstraints(mode=ConstraintMode.PENALTY, min_feature_nm=500)
        solver = MockSolver()
        ctx = ObjectiveContext(
            structure=phc.to_structure(),
            geometry=phc,
            solver=solver,
            constraint_result=cc.validate(phc.to_structure()),
        )
        val = FabricationPenaltyObjective().evaluate(ctx)
        assert val.value > 0.0


# ---------------------------------------------------------------------------
# Stabilization proxies
# ---------------------------------------------------------------------------


class TestStabilizationProxies:
    def test_asymmetry_proxy_scales_with_warp(self):
        proxy = AsymmetryStabilizationProxy()
        flat = _feasible_metagrating()
        flat.curvature = 0.0
        flat.asymmetry = 0.0

        warped = _feasible_metagrating()
        warped.curvature = 0.15
        warped.asymmetry = 0.10

        solver = MockSolver()
        cc = FabConstraints(mode=ConstraintMode.PENALTY)

        def score(mg):
            ctx = ObjectiveContext(
                structure=mg.to_structure(),
                geometry=mg,
                solver=solver,
                constraint_result=cc.validate(mg.to_structure()),
            )
            return proxy.score(ctx)[0]

        assert score(warped) > score(flat)

    def test_proxy_zero_without_rings(self):
        proxy = AsymmetryStabilizationProxy()
        ctx = _phc_context()  # no rings
        score, meta = proxy.score(ctx)
        assert score == 0.0
        assert meta.get("reason") == "no rings"

    def test_radial_momentum_mode(self):
        proxy = RadialMomentumProxy()
        ctx = _mg_context()
        score, meta = proxy.score(ctx)
        assert 0.0 <= score <= 1.0
        assert "diffraction" in meta

    def test_objective_wrapper(self):
        obj = StabilizationProxyObjective(mode="asymmetry")
        val = obj.evaluate(_mg_context())
        assert val.target == "maximize"
        assert 0.0 <= val.value <= 1.0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            StabilizationProxyObjective(mode="not_a_real_mode")


# ---------------------------------------------------------------------------
# Stage factories
# ---------------------------------------------------------------------------


class TestStageFactories:
    def test_stage1_default_produces_three_objectives(self):
        objs = make_stage1_objectives()
        names = [o.name for o in objs]
        assert "nir_reflectance" in names
        assert "mir_emissivity" in names
        assert "fabrication_penalty" in names

    def test_stage2_default_produces_four_objectives(self):
        objs = make_stage2_objectives()
        names = [o.name for o in objs]
        assert "nir_reflectance" in names
        assert "mir_emissivity" in names
        assert "stabilization" in names
        assert "fabrication_penalty" in names

    def test_stage1_config_overrides_weight(self):
        cfg = {"nir_reflectance": {"weight": 2.5, "band_nm": [1400, 1600]}}
        objs = make_stage1_objectives(cfg)
        nir = next(o for o in objs if o.name == "nir_reflectance")
        assert nir.weight == pytest.approx(2.5)
        assert nir.band_nm == (1400.0, 1600.0)

    def test_stage2_config_selects_radial_momentum(self):
        cfg = {"stabilization": {"mode": "radial_momentum"}}
        objs = make_stage2_objectives(cfg)
        stab = next(o for o in objs if o.name == "stabilization")
        assert isinstance(stab.proxy, RadialMomentumProxy)


# ---------------------------------------------------------------------------
# ObjectiveEvaluator
# ---------------------------------------------------------------------------


class TestObjectiveEvaluator:
    def test_end_to_end_phc_evaluation(self):
        phc = _feasible_phc()
        evaluator = ObjectiveEvaluator(
            geometry=phc,
            solver=MockSolver(),
            constraints=FabConstraints(mode=ConstraintMode.PENALTY),
            objectives=make_stage1_objectives(),
        )
        result = evaluator.evaluate(phc.to_param_vector())
        assert isinstance(result, EvaluationResult)
        assert "nir_reflectance" in result.objective_values
        assert "mir_emissivity" in result.objective_values
        assert "fabrication_penalty" in result.objective_values
        scalars = result.scalar_values()
        assert all(isinstance(v, float) for v in scalars.values())

    def test_evaluator_caches_spectrum(self):
        """Two NIR-based objectives on the same band must share one spectrum call."""
        phc = _feasible_phc()
        solver = MockSolver()

        call_count = {"n": 0}
        original = solver.compute_spectrum

        def counting_compute(structure, wavelengths_nm):
            call_count["n"] += 1
            return original(structure, wavelengths_nm)

        solver.compute_spectrum = counting_compute  # type: ignore[assignment]

        evaluator = ObjectiveEvaluator(
            geometry=phc,
            solver=solver,
            constraints=FabConstraints(mode=ConstraintMode.PENALTY),
            objectives=[
                NIRReflectivityObjective(band_nm=(1350, 1650), n_points=30),
                NIRReflectivityObjective(
                    band_nm=(1350, 1650),
                    n_points=30,
                    name="nir_reflectance_copy",
                ),
            ],
        )
        evaluator.evaluate(phc.to_param_vector())
        # Both objectives asked for the same band & n_points -> 1 solver call
        assert call_count["n"] == 1

    def test_evaluator_reports_feasible_on_good_design(self):
        phc = _feasible_phc()
        evaluator = ObjectiveEvaluator(
            geometry=phc,
            solver=MockSolver(),
            constraints=FabConstraints(mode=ConstraintMode.HARD),
            objectives=make_stage1_objectives(),
        )
        result = evaluator.evaluate(phc.to_param_vector())
        assert result.feasible
        assert result.constraint_penalty == 0.0

    def test_metagrating_stage2_evaluation(self):
        mg = _feasible_metagrating()
        evaluator = ObjectiveEvaluator(
            geometry=mg,
            solver=MockSolver(),
            constraints=FabConstraints(mode=ConstraintMode.PENALTY),
            objectives=make_stage2_objectives(),
        )
        result = evaluator.evaluate(mg.to_param_vector())
        for name in (
            "nir_reflectance",
            "mir_emissivity",
            "stabilization",
            "fabrication_penalty",
        ):
            assert name in result.objective_values
