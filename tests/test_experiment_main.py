"""Tests for the unified :mod:`lightsail.experiments.main` entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest

# BoTorch + torch are needed for the underlying MOBO runner.
pytest.importorskip("torch")
pytest.importorskip("botorch")

import yaml

from lightsail.experiments.main import (
    _build_solver,
    _deep_update,
    _load_config,
    run_experiment,
)
from lightsail.simulation.mock import MockSolver


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_deep_update_merges_nested_keys(self):
        target = {"a": {"b": 1, "c": 2}, "d": 3}
        overrides = {"a": {"b": 10}, "e": 4}
        _deep_update(target, overrides)
        assert target == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}

    def test_deep_update_replaces_non_dict_values(self):
        target = {"a": [1, 2, 3]}
        _deep_update(target, {"a": [9]})
        assert target == {"a": [9]}

    def test_load_config_applies_overrides(self, tmp_path):
        cfg = {"stage": 1, "mobo": {"n_init": 16, "seed": 1}}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(cfg))
        loaded = _load_config(path, {"mobo": {"n_init": 4}})
        assert loaded["mobo"]["n_init"] == 4
        assert loaded["mobo"]["seed"] == 1


# ---------------------------------------------------------------------------
# Config files smoke-test
# ---------------------------------------------------------------------------


class TestConfigFiles:
    @pytest.mark.parametrize(
        "config_name",
        [
            "stage1_triangular.yaml",
            "stage1_hexagonal.yaml",
            "stage1_pentagonal.yaml",
            "stage2_outer_grating.yaml",
        ],
    )
    def test_configs_parse_and_have_required_fields(self, config_name):
        cfg = yaml.safe_load((CONFIGS_DIR / config_name).read_text())
        assert "stage" in cfg
        assert cfg["stage"] in (1, 2)
        assert "mobo" in cfg
        assert "objectives" in cfg
        if cfg["stage"] == 1:
            assert "geometry" in cfg
            assert "lattice_family" in cfg["geometry"]
        if cfg["stage"] == 2:
            assert "frozen_phc" in cfg


# ---------------------------------------------------------------------------
# Full entrypoint smoke-test (tiny iteration counts)
# ---------------------------------------------------------------------------


class TestRunExperiment:
    def test_stage1_triangular_minimal(self, tmp_path):
        out = run_experiment(
            config_path=CONFIGS_DIR / "stage1_triangular.yaml",
            output_root=tmp_path,
            overrides={
                "name": "test_stage1_triangular",
                "mobo": {
                    "n_init": 4,
                    "n_iterations": 2,
                    "acqf_num_restarts": 2,
                    "acqf_raw_samples": 8,
                },
            },
        )
        assert out.exists()
        assert (out / "run.log").exists()
        assert (out / "config.yaml").exists()
        assert (out / "trials.json").exists()
        assert (out / "best_design.yaml").exists()
        assert (out / "summary.txt").exists()
        assert (out / "plots").is_dir()
        # At least Pareto + history + spectrum
        pngs = list((out / "plots").glob("*.png"))
        assert len(pngs) >= 3

    def test_stage2_outer_grating_minimal(self, tmp_path):
        out = run_experiment(
            config_path=CONFIGS_DIR / "stage2_outer_grating.yaml",
            output_root=tmp_path,
            overrides={
                "name": "test_stage2",
                "mobo": {
                    "n_init": 3,
                    "n_iterations": 2,
                    "acqf_num_restarts": 2,
                    "acqf_raw_samples": 8,
                },
            },
        )
        assert (out / "trials.json").exists()
        assert (out / "best_design.yaml").exists()

    def test_build_solver_default_mock(self):
        solver = _build_solver({})
        assert isinstance(solver, MockSolver)

    def test_build_solver_explicit_mock(self):
        solver = _build_solver({"solver": {"kind": "mock"}})
        assert isinstance(solver, MockSolver)

    def test_build_solver_rcwa(self):
        pytest.importorskip("grcwa")
        from lightsail.simulation.rcwa_solver import RCWASolver
        solver = _build_solver({"solver": {"kind": "rcwa", "nG": 21, "grid_nx": 32, "grid_ny": 32}})
        assert isinstance(solver, RCWASolver)
        assert solver.config.nG == 21

    def test_build_solver_unknown_rejected(self):
        with pytest.raises(ValueError):
            _build_solver({"solver": {"kind": "fdtd"}})

    def test_unknown_stage_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.dump({"stage": 99, "name": "bad"}))
        with pytest.raises(ValueError):
            run_experiment(bad, output_root=tmp_path)
