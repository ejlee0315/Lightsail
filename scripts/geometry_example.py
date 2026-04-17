"""Minimal end-to-end usage example for the geometry + constraint stack.

This script:
  1. Builds a PhCReflector with triangular lattice + rounded-polygon holes.
  2. Builds a MetaGrating anchored at the PhC outer radius.
  3. Runs the FabConstraints checker in both HARD and PENALTY modes.
  4. Writes top-view PNGs of the PhC, metagrating, and a single hole shape.

Run from the project root:

    python3 scripts/geometry_example.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # allow headless rendering

from lightsail.constraints.fabrication import (
    ConstraintMode,
    FabConstraints,
)
from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.visualization.plots import (
    plot_hole_shape,
    plot_structure_topview,
)


OUT = Path("results/geometry_example")


def build_phc() -> PhCReflector:
    """A deliberately feasible triangular-lattice PhC."""
    return PhCReflector(
        lattice_family=LatticeFamily.TRIANGULAR,
        n_rings=6,
        thickness_nm=500.0,
        lattice_period_nm=1500.0,
        hole_a_rel=450.0 / 1500.0,   # 450 nm / 1500 nm period
        hole_b_rel=350.0 / 1500.0,
        hole_rotation_deg=15.0,
        corner_rounding=0.6,
        shape_parameter=6,   # rounds to hexagonal hole
    )


def build_metagrating(inner_radius_nm: float, thickness_nm: float) -> MetaGrating:
    return MetaGrating(
        inner_radius_nm=inner_radius_nm,
        thickness_nm=thickness_nm,
        grating_period_nm=1500.0,
        duty_cycle=0.5,
        curvature=0.05,
        asymmetry=0.02,
        ring_width_um=12.0,
    )


def report(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"  feasible : {result.feasible}")
    print(f"  penalty  : {result.penalty:.4f}")
    if result.violations:
        print("  violations:")
        for v in result.violations:
            print(f"    - {v}")
    else:
        print("  violations: (none)")
    if result.metrics:
        print("  metrics:")
        for k, v in result.metrics.items():
            print(f"    {k:28s} {v:.3f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- geometry ---
    phc = build_phc()
    mg = build_metagrating(phc.outer_radius_nm, phc.thickness_nm)

    phc_struct = phc.to_structure()
    mg_struct = mg.to_structure()

    print(f"PhC      : {len(phc_struct.holes)} holes, "
          f"outer radius {phc.outer_radius_nm/1000:.2f} µm, "
          f"n_sides={phc.n_sides}")
    print(f"MetaGrat : {len(mg_struct.rings)} rings, "
          f"outer radius {mg.outer_radius_nm/1000:.2f} µm")

    # --- constraint checking: penalty mode ---
    penalty_cc = FabConstraints(mode=ConstraintMode.PENALTY)
    report("PhC (penalty mode)", penalty_cc.validate(phc_struct))
    report("MetaGrating (penalty mode)", penalty_cc.validate(mg_struct))

    # --- constraint checking: hard mode ---
    hard_cc = FabConstraints(mode=ConstraintMode.HARD)
    report("PhC (hard mode)", hard_cc.validate(phc_struct))
    report("MetaGrating (hard mode)", hard_cc.validate(mg_struct))

    # --- visualization ---
    plot_hole_shape(phc.hole_shape(), save_path=OUT / "hole_shape.png")
    plot_structure_topview(
        phc_struct,
        title=f"PhC reflector ({phc.lattice_family.value})",
        save_path=OUT / "phc_topview.png",
    )
    plot_structure_topview(
        mg_struct,
        title="Concentric curved metagrating",
        save_path=OUT / "metagrating_topview.png",
    )
    print(f"\nFigures written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
