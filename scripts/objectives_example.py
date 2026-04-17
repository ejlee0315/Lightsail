"""Minimal end-to-end demo of the objective stack.

Builds a concrete PhC reflector and metagrating, runs the MockSolver,
then evaluates both Stage 1 and Stage 2 objective sets via
:class:`ObjectiveEvaluator`. Prints every objective's scalar value
and metadata.

Run from the project root:

    python3 scripts/objectives_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.optimization.evaluator import ObjectiveEvaluator
from lightsail.optimization.objectives import (
    make_stage1_objectives,
    make_stage2_objectives,
)
from lightsail.simulation.mock import MockSolver


CONFIG = Path("configs/default.yaml")


def load_config() -> dict:
    if CONFIG.exists():
        with open(CONFIG) as f:
            return yaml.safe_load(f)
    return {}


def build_phc() -> PhCReflector:
    return PhCReflector(
        lattice_family=LatticeFamily.TRIANGULAR,
        n_rings=6,
        thickness_nm=500.0,
        lattice_period_nm=1500.0,
        hole_a_rel=400.0 / 1500.0,
        hole_b_rel=400.0 / 1500.0,
        hole_rotation_deg=0.0,
        corner_rounding=0.8,
        shape_parameter=6.0,
    )


def build_metagrating(phc: PhCReflector) -> MetaGrating:
    return MetaGrating(
        inner_radius_nm=phc.outer_radius_nm,
        thickness_nm=phc.thickness_nm,
        grating_period_nm=1500.0,
        duty_cycle=0.5,
        curvature=0.1,
        asymmetry=0.05,
        ring_width_um=12.0,
    )


def print_result(label: str, evaluation) -> None:
    print(f"\n=== {label} ===")
    print(f"  feasible            : {evaluation.feasible}")
    print(f"  constraint_penalty  : {evaluation.constraint_penalty:.4f}")
    print("  objectives:")
    for name, val in evaluation.objective_values.items():
        arrow = "↑" if val.target == "maximize" else "↓"
        print(f"    {name:22s} {arrow} value={val.value:.4f}  weight={val.weight:.2f}")
        for mk, mv in val.metadata.items():
            if isinstance(mv, (int, float)):
                print(f"      · {mk:18s} {mv:.4f}")
            else:
                print(f"      · {mk:18s} {mv}")


def main() -> None:
    cfg = load_config()
    stage1_cfg = (cfg.get("stage1") or {}).get("objectives", {})
    stage2_cfg = (cfg.get("stage2") or {}).get("objectives", {})

    solver = MockSolver()
    constraints = FabConstraints(mode=ConstraintMode.PENALTY)

    # --- Stage 1 ---
    phc = build_phc()
    stage1_eval = ObjectiveEvaluator(
        geometry=phc,
        solver=solver,
        constraints=constraints,
        objectives=make_stage1_objectives(stage1_cfg),
    )
    r1 = stage1_eval.evaluate(phc.to_param_vector())
    print_result("Stage 1 — PhC reflector", r1)

    # --- Stage 2 ---
    mg = build_metagrating(phc)
    stage2_eval = ObjectiveEvaluator(
        geometry=mg,
        solver=solver,
        constraints=constraints,
        objectives=make_stage2_objectives(stage2_cfg),
    )
    r2 = stage2_eval.evaluate(mg.to_param_vector())
    print_result("Stage 2 — MetaGrating", r2)

    # --- Small param perturbation to show responsiveness ---
    print("\n=== Responsiveness check (PhC: thickness 500 -> 900 nm) ===")
    v = phc.to_param_vector()
    baseline = stage1_eval.evaluate(v).scalar_values()
    v_thick = v.copy()
    v_thick[0] = 900.0
    perturbed = stage1_eval.evaluate(v_thick).scalar_values()
    for name in baseline:
        diff = perturbed[name] - baseline[name]
        print(f"  Δ {name:22s} {baseline[name]:.4f} -> {perturbed[name]:.4f}  ({diff:+.4f})")


if __name__ == "__main__":
    main()
