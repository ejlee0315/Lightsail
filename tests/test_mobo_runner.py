"""Tests for :class:`MOBORunner`.

These tests require torch + botorch + gpytorch. They are skipped if
any of those are missing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
botorch = pytest.importorskip("botorch")
gpytorch = pytest.importorskip("gpytorch")

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.experiments.stage_runner import run_stage1, run_stage2
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.optimization.evaluator import ObjectiveEvaluator
from lightsail.optimization.mobo_runner import (
    MOBOConfig,
    MOBORunner,
    RunResult,
    save_run_result,
)
from lightsail.optimization.objectives import make_stage1_objectives
from lightsail.optimization.search_space import SearchSpace
from lightsail.simulation.mock import MockSolver


def _small_config() -> MOBOConfig:
    return MOBOConfig(
        n_init=5,
        n_iterations=3,
        batch_size=1,
        seed=0,
        sampling_method="sobol",
        acqf_num_restarts=2,
        acqf_raw_samples=16,
    )


class TestMOBORunnerSmoke:
    def test_runs_and_collects_trials(self):
        phc = PhCReflector(lattice_family=LatticeFamily.TRIANGULAR)
        evaluator = ObjectiveEvaluator(
            geometry=phc,
            solver=MockSolver(),
            constraints=FabConstraints(mode=ConstraintMode.PENALTY),
            objectives=make_stage1_objectives(),
        )
        space = SearchSpace.from_geometry(phc)
        runner = MOBORunner(evaluator, space, _small_config())
        result = runner.run()

        assert isinstance(result, RunResult)
        assert result.n_trials == 8  # 5 init + 3 BO
        sources = {t.source for t in result.trials}
        assert "init" in sources
        assert any(s in ("bo", "bo_fallback") for s in sources)

    def test_pareto_indices_non_empty(self):
        phc = PhCReflector(lattice_family=LatticeFamily.TRIANGULAR)
        evaluator = ObjectiveEvaluator(
            geometry=phc,
            solver=MockSolver(),
            constraints=FabConstraints(mode=ConstraintMode.PENALTY),
            objectives=make_stage1_objectives(),
        )
        space = SearchSpace.from_geometry(phc)
        result = MOBORunner(evaluator, space, _small_config()).run()
        assert len(result.pareto_indices) >= 1
        for idx in result.pareto_indices:
            assert 0 <= idx < result.n_trials

    def test_best_by_returns_extreme(self):
        phc = PhCReflector(lattice_family=LatticeFamily.TRIANGULAR)
        evaluator = ObjectiveEvaluator(
            geometry=phc,
            solver=MockSolver(),
            constraints=FabConstraints(mode=ConstraintMode.PENALTY),
            objectives=make_stage1_objectives(),
        )
        space = SearchSpace.from_geometry(phc)
        result = MOBORunner(evaluator, space, _small_config()).run()

        best = result.best_by("nir_reflectance")
        all_vals = [t.objective_values["nir_reflectance"] for t in result.trials]
        assert best.objective_values["nir_reflectance"] == max(all_vals)

        # fab is minimize -> best_by should return the minimum
        best_fab = result.best_by("fabrication_penalty")
        all_fab = [t.objective_values["fabrication_penalty"] for t in result.trials]
        assert best_fab.objective_values["fabrication_penalty"] == min(all_fab)

    def test_save_run_result(self, tmp_path):
        phc = PhCReflector(lattice_family=LatticeFamily.TRIANGULAR)
        evaluator = ObjectiveEvaluator(
            geometry=phc,
            solver=MockSolver(),
            constraints=FabConstraints(mode=ConstraintMode.PENALTY),
            objectives=make_stage1_objectives(),
        )
        space = SearchSpace.from_geometry(phc)
        result = MOBORunner(evaluator, space, _small_config()).run()

        save_run_result(result, tmp_path)
        assert (tmp_path / "trials.json").exists()
        assert (tmp_path / "params.npy").exists()
        assert (tmp_path / "objectives.npy").exists()
        assert (tmp_path / "pareto_indices.npy").exists()


class TestStageRunners:
    def test_run_stage1_minimal(self):
        result, phc = run_stage1(
            mobo_config=MOBOConfig(
                n_init=4, n_iterations=2, seed=1,
                acqf_num_restarts=2, acqf_raw_samples=8,
            ),
        )
        assert result.n_trials == 6
        assert phc.n_sides in range(3, 9)

    def test_run_stage2_minimal(self):
        _, phc = run_stage1(
            mobo_config=MOBOConfig(
                n_init=3, n_iterations=1, seed=1,
                acqf_num_restarts=2, acqf_raw_samples=8,
            ),
        )
        result, mg = run_stage2(
            phc=phc,
            mobo_config=MOBOConfig(
                n_init=3, n_iterations=1, seed=2,
                acqf_num_restarts=2, acqf_raw_samples=8,
            ),
        )
        assert result.n_trials == 4
        assert mg.inner_radius_nm == phc.outer_radius_nm
