"""Unified entrypoint for running one optimization experiment.

A single function, :func:`run_experiment`, takes a YAML config path and:

1. reads the config and applies any runtime overrides,
2. creates a timestamped output directory under ``results/``,
3. copies the config into that directory for reproducibility,
4. sets up console + file logging,
5. builds fabrication constraints, MOBO hyper-parameters, and the
   geometry from the config,
6. dispatches to Stage 1 or Stage 2 via :mod:`stage_runner`,
7. writes plots (Pareto, history, best spectrum, best structure) and
   a ``best_design.yaml`` summary,
8. returns the path to the output directory.

Only the ``stage`` field in the config decides which stage runs. All
other parameters are config-driven.
"""

from __future__ import annotations

import copy
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from lightsail.constraints.fabrication import ConstraintMode, FabConstraints
from lightsail.experiments.logging_setup import setup_logging
from lightsail.experiments.stage_runner import run_stage1, run_stage2
from lightsail.geometry.base import LatticeFamily, ParametricGeometry
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.optimization.mobo_runner import MOBOConfig, RunResult
from lightsail.simulation.base import ElectromagneticSolver
from lightsail.simulation.mock import MockSolver

try:
    from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver
    _HAS_RCWA = True
except ImportError:  # grcwa not installed
    _HAS_RCWA = False
from lightsail.visualization.mobo_plots import (
    plot_optimization_history,
    plot_pareto_scatter,
    summarize_best,
)
from lightsail.visualization.plots import (
    plot_broadband_spectrum,
    plot_structure_topview,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _deep_update(target: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``target`` (in place)."""
    for key, value in overrides.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_config(path: Path, overrides: Optional[dict] = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a dict")
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg


def _make_output_dir(root: Path, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(root) / f"{timestamp}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def _build_fab_constraints(cfg: dict) -> FabConstraints:
    fab = cfg.get("fabrication", {}) or {}
    return FabConstraints(
        min_feature_nm=float(fab.get("min_feature_nm", 500.0)),
        min_gap_nm=float(fab.get("min_gap_nm", 500.0)),
        thickness_range_nm=tuple(cfg.get("thickness_range_nm", [200.0, 1000.0])),
        fill_fraction_range=tuple(fab.get("fill_fraction_range", [0.05, 0.60])),
        fab_mode=str(fab.get("mode", "mode_a")),
        mode=ConstraintMode.PENALTY,
    )


def _build_mobo_config(cfg: dict) -> MOBOConfig:
    m = cfg.get("mobo", {}) or {}
    return MOBOConfig(
        n_init=int(m.get("n_init", 16)),
        n_iterations=int(m.get("n_iterations", 20)),
        batch_size=int(m.get("batch_size", 1)),
        seed=int(m.get("seed", cfg.get("seed", 42))),
        sampling_method=str(m.get("sampling", "sobol")),
        acqf_num_restarts=int(m.get("acqf_num_restarts", 5)),
        acqf_raw_samples=int(m.get("acqf_raw_samples", 64)),
        acqf_mc_samples=int(m.get("acqf_mc_samples", 128)),
        device=str(m.get("device", "cpu")),
        dtype=str(m.get("dtype", "double")),
        ref_point_margin=float(m.get("ref_point_margin", 0.1)),
    )


def _build_solver(cfg: dict) -> ElectromagneticSolver:
    """Pick the solver based on ``cfg['solver']`` (defaults to MockSolver).

    Supported choices:
      - ``mock`` (default, always available)
      - ``rcwa`` — grcwa-backed RCWASolver (requires ``lightsail[rcwa]``)
    """
    solver_cfg = cfg.get("solver") or {}
    kind = str(solver_cfg.get("kind", "mock")).lower()

    if kind == "mock":
        return MockSolver()
    if kind == "rcwa":
        if not _HAS_RCWA:
            raise ImportError(
                "RCWASolver requested in config but grcwa is not installed. "
                "Install with `pip install 'lightsail[rcwa]'`."
            )
        rcwa_cfg = RCWAConfig(
            nG=int(solver_cfg.get("nG", 41)),
            grid_nx=int(solver_cfg.get("grid_nx", 96)),
            grid_ny=int(solver_cfg.get("grid_ny", 96)),
            polarization=str(solver_cfg.get("polarization", "average")),
            theta_deg=float(solver_cfg.get("theta_deg", 0.0)),
            phi_deg=float(solver_cfg.get("phi_deg", 0.0)),
        )
        return RCWASolver(config=rcwa_cfg)
    raise ValueError(f"Unknown solver kind: {kind!r}")


def _frozen_phc_from_cfg(cfg: dict) -> PhCReflector:
    """Build a PhCReflector from a frozen_phc config section.

    Accepts either relative (``hole_a_rel``/``hole_b_rel``) or absolute
    (``hole_a_nm``/``hole_b_nm``) hole sizes, for convenience when copying
    parameters out of a best_design.yaml that still uses nm, or out of a
    production run that already uses rel.
    """
    frozen = cfg.get("frozen_phc")
    if not frozen:
        raise ValueError("Stage 2 config missing required 'frozen_phc' section")

    period_nm = float(frozen["lattice_period_nm"])

    if "hole_a_rel" in frozen:
        a_rel = float(frozen["hole_a_rel"])
    elif "hole_a_nm" in frozen:
        a_rel = float(frozen["hole_a_nm"]) / period_nm
    else:
        raise ValueError(
            "frozen_phc needs either 'hole_a_rel' or 'hole_a_nm'"
        )

    if "hole_b_rel" in frozen:
        b_rel = float(frozen["hole_b_rel"])
    elif "hole_b_nm" in frozen:
        b_rel = float(frozen["hole_b_nm"]) / period_nm
    else:
        raise ValueError(
            "frozen_phc needs either 'hole_b_rel' or 'hole_b_nm'"
        )

    return PhCReflector(
        lattice_family=LatticeFamily(frozen.get("lattice_family", "triangular")),
        n_rings=int(frozen.get("n_rings", 6)),
        thickness_nm=float(frozen["thickness_nm"]),
        lattice_period_nm=period_nm,
        hole_a_rel=a_rel,
        hole_b_rel=b_rel,
        hole_rotation_deg=float(frozen.get("hole_rotation_deg", 0.0)),
        corner_rounding=float(frozen.get("corner_rounding", 0.8)),
        shape_parameter=float(frozen.get("shape_parameter", 6.0)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(
    config_path: Path,
    output_root: Path = Path("results"),
    overrides: Optional[dict] = None,
    solver: Optional[ElectromagneticSolver] = None,
) -> Path:
    """Run a Stage 1 or Stage 2 experiment from a YAML config.

    Parameters
    ----------
    config_path:
        Path to a YAML config file containing at minimum a ``stage``
        field (1 or 2). See ``configs/stage1_*.yaml`` and
        ``configs/stage2_*.yaml`` for the expected schema.
    output_root:
        Parent directory for per-run timestamped output folders.
    overrides:
        Optional deep-merged overrides applied to the loaded config
        before the run. Useful for demo scripts that want to shrink
        iteration counts without editing the YAML.
    solver:
        EM solver instance. Defaults to :class:`MockSolver`; pass an
        RCWA-backed solver once that backend exists.

    Returns
    -------
    Path to the timestamped output directory.
    """
    config_path = Path(config_path)
    cfg = _load_config(config_path, overrides)

    stage = int(cfg.get("stage", 1))
    name = str(cfg.get("name", f"stage{stage}"))

    output_dir = _make_output_dir(output_root, name)
    log_file = setup_logging(output_dir)
    shutil.copy(config_path, output_dir / "config.yaml")
    if overrides:
        (output_dir / "overrides.yaml").write_text(yaml.dump(overrides))

    logger.info("=" * 72)
    logger.info("Lightsail experiment: %s  (stage %d)", name, stage)
    logger.info("Config: %s", config_path)
    logger.info("Output: %s", output_dir)
    logger.info("Log:    %s", log_file)
    logger.info("=" * 72)

    constraints = _build_fab_constraints(cfg)
    mobo_config = _build_mobo_config(cfg)
    if solver is None:
        solver = _build_solver(cfg)
    logger.info("Solver: %s", type(solver).__name__)

    primary = cfg.get("primary_objective")

    if stage == 1:
        geom_cfg = cfg.get("geometry", {}) or {}
        lattice = LatticeFamily(geom_cfg.get("lattice_family", "triangular"))
        n_rings = int(geom_cfg.get("n_rings", 6))
        logger.info(
            "Stage 1 geometry: lattice=%s, n_rings=%d", lattice.value, n_rings
        )
        result, geometry = run_stage1(
            solver=solver,
            constraints=constraints,
            lattice_family=lattice,
            phc_n_rings=n_rings,
            objectives_cfg=cfg.get("objectives", {}),
            mobo_config=mobo_config,
            output_dir=output_dir,
            primary_objective=primary or "nir_reflectance",
        )
    elif stage == 2:
        phc = _frozen_phc_from_cfg(cfg)
        logger.info(
            "Stage 2 frozen PhC: thickness=%.0fnm period=%.0fnm, outer=%.0fnm",
            phc.thickness_nm, phc.lattice_period_nm, phc.outer_radius_nm,
        )
        result, geometry = run_stage2(
            phc=phc,
            solver=solver,
            constraints=constraints,
            objectives_cfg=cfg.get("objectives", {}),
            mobo_config=mobo_config,
            output_dir=output_dir,
            primary_objective=primary or "stabilization",
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    _produce_artifacts(result, geometry, solver, output_dir, cfg, primary)

    logger.info("Experiment complete. Artifacts in %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Artifact production
# ---------------------------------------------------------------------------


def _produce_artifacts(
    result: RunResult,
    geometry: ParametricGeometry,
    solver: ElectromagneticSolver,
    output_dir: Path,
    cfg: dict,
    primary: Optional[str],
) -> None:
    plots_dir = output_dir / "plots"

    # 1. Text summary
    summary = summarize_best(result)
    (output_dir / "summary.txt").write_text(summary + "\n")
    for line in summary.split("\n"):
        logger.info(line)

    if not result.trials:
        logger.warning("No trials recorded — skipping plots.")
        return

    # 2. Pareto plot — first two objectives on the axes
    obj_names = result.objective_names
    x = obj_names[0]
    y = obj_names[1] if len(obj_names) >= 2 else obj_names[0]
    try:
        plot_pareto_scatter(
            result, x, y, save_path=plots_dir / f"pareto_{x}_vs_{y}.png"
        )
    except Exception as e:  # pragma: no cover
        logger.warning("Pareto plot failed: %s", e)

    # 3. Optimization history
    try:
        plot_optimization_history(
            result, save_path=plots_dir / "optimization_history.png"
        )
    except Exception as e:  # pragma: no cover
        logger.warning("History plot failed: %s", e)

    # 4. Best-design spectrum + top-view + YAML summary
    primary = primary or obj_names[0]
    try:
        best = result.best_by(primary)
        geometry.from_param_vector(best.params)
        structure = geometry.to_structure()

        wl = np.concatenate(
            [
                np.linspace(1000.0, 2500.0, 180),
                np.linspace(5000.0, 15000.0, 180),
            ]
        )
        sim_result = solver.compute_spectrum(structure, wl)

        plot_broadband_spectrum(
            sim_result,
            title=f"Best design spectrum (trial {best.trial_id})",
            save_path=plots_dir / "spectrum_best.png",
        )
        plot_structure_topview(
            structure,
            title=f"Best design top view (trial {best.trial_id})",
            save_path=plots_dir / "structure_topview.png",
        )

        best_yaml = {
            "trial_id": int(best.trial_id),
            "iteration": int(best.iteration),
            "source": best.source,
            "feasible": bool(best.feasible),
            "constraint_penalty": float(best.constraint_penalty),
            "objectives": {k: float(v) for k, v in best.objective_values.items()},
            "params": {
                name: float(best.params[i])
                for i, name in enumerate(result.search_space_names)
            },
        }
        (output_dir / "best_design.yaml").write_text(
            yaml.dump(best_yaml, sort_keys=False)
        )
        logger.info("Best design saved: %s", output_dir / "best_design.yaml")

    except Exception as e:  # pragma: no cover
        logger.warning("Failed to produce best-design artifacts: %s", e)
