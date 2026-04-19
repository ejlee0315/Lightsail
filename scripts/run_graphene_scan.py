"""P2.3 — Graphene N-layer scan on Design A + thermal Pareto.

Stacks N ∈ {0, 5, 10, 20, 50} graphene monolayers behind the Design A
PhC (t=280, P=1580, hole r=600), evaluates R, T, ε in NIR (1550–1850 nm)
and MIR (8–14 µm), then computes the steady-state sail temperature
under Starshot illumination.

Thermal balance (both faces radiate)::

    P_abs    = α_NIR · I · A_sail            with α_NIR = 1 − R_NIR − T_NIR
    P_emit   = 2 · A_sail · σ_SB · ε_MIR · T_steady^4

    ⇒  T_steady = (α_NIR · I / (2 σ_SB ε_MIR))^(1/4)

Outputs a results/<timestamp>_p2_3_graphene_scan/ folder with a
summary table (CSV + Markdown), per-N spectrum plots, and the Pareto
plot of MIR-ε vs added mass.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.materials import GRAPHENE_LAYER_THICKNESS_M, GrapheneConductivity
from lightsail.simulation import (
    LayeredRCWASolver,
    LayerSpec,
    RCWAConfig,
    RCWASolver,
)


NIR_BAND = (1550.0, 1850.0)
MIR_BAND = (8000.0, 14000.0)
INTENSITY_W_PER_M2 = 1.0e10           # Starshot 10 GW/m²
SIGMA_SB = 5.670374419e-8             # W/m²/K⁴

# Design A (CLAUDE.md, 2026-04-17): T = 20.73 min
DESIGN_A = {
    "thickness_nm": 280.0,
    "lattice_period_nm": 1580.0,
    "hole_a_rel": 600.0 / 1580.0,
    "hole_b_rel": 600.0 / 1580.0,
    "hole_rotation_deg": 0.0,
    "corner_rounding": 1.0,
    "shape_parameter": 8.0,
    "lattice_family": LatticeFamily.TRIANGULAR,
}

GRAPHENE_LAYER_COUNTS = (0, 5, 10, 20, 50)

# Mass: SiN ρ ≈ 3100 kg/m³, graphene ~ 0.77 mg/m² per monolayer
SIN_DENSITY_KG_PER_M3 = 3100.0
GRAPHENE_AREAL_DENSITY_KG_PER_M2 = 0.77e-6   # 0.77 mg/m² per layer


def design_A() -> PhCReflector:
    return PhCReflector(**DESIGN_A)


def graphene_layer_spec(n_layers: int, E_F_eV: float = 0.3) -> LayerSpec:
    """Single bulk slab equivalent to N monolayers (ε independent of N)."""
    g = GrapheneConductivity(E_F_eV=E_F_eV)
    return LayerSpec(
        thickness_nm=n_layers * GRAPHENE_LAYER_THICKNESS_M * 1e9,
        eps_callable=g.epsilon_callable(GRAPHENE_LAYER_THICKNESS_M),
        name=f"graphene_x{n_layers}",
    )


def thermal_steady_K(alpha_nir: float, eps_mir: float) -> float:
    """T_steady from absorbed laser power balanced against MIR emission (both faces)."""
    if eps_mir <= 0.0 or alpha_nir <= 0.0:
        return float("nan")
    return float(
        (alpha_nir * INTENSITY_W_PER_M2 / (2.0 * SIGMA_SB * eps_mir)) ** 0.25
    )


def main() -> None:
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p2_3_graphene_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    rcwa_cfg = RCWAConfig(nG=41, grid_nx=64, grid_ny=64)
    structure = design_A().to_structure()

    nir_wls = np.linspace(NIR_BAND[0], NIR_BAND[1], 9)
    mir_wls = np.linspace(MIR_BAND[0], MIR_BAND[1], 7)

    rows = []
    for n in GRAPHENE_LAYER_COUNTS:
        if n == 0:
            solver = RCWASolver(config=rcwa_cfg)
        else:
            solver = LayeredRCWASolver(
                config=rcwa_cfg, layers_below=[graphene_layer_spec(n)],
            )

        R_nir = solver.evaluate_reflectivity(structure, nir_wls)
        T_nir = solver.evaluate_transmission(structure, nir_wls)
        eps_nir = np.clip(1.0 - R_nir - T_nir, 0.0, 1.0)

        R_mir = solver.evaluate_reflectivity(structure, mir_wls)
        T_mir = solver.evaluate_transmission(structure, mir_wls)
        eps_mir = np.clip(1.0 - R_mir - T_mir, 0.0, 1.0)

        mean_R_nir = float(R_nir.mean())
        mean_T_nir = float(T_nir.mean())
        mean_alpha_nir = float(eps_nir.mean())
        mean_eps_mir = float(eps_mir.mean())

        sin_areal_kg_per_m2 = (
            SIN_DENSITY_KG_PER_M3 * DESIGN_A["thickness_nm"] * 1e-9
            * (1.0 - _hole_fill_fraction())
        )
        graphene_areal_kg_per_m2 = n * GRAPHENE_AREAL_DENSITY_KG_PER_M2
        total_areal_kg_per_m2 = sin_areal_kg_per_m2 + graphene_areal_kg_per_m2
        mass_10m2_g = total_areal_kg_per_m2 * 10.0 * 1000.0

        T_K = thermal_steady_K(mean_alpha_nir, mean_eps_mir)

        row = {
            "n_layers": n,
            "mean_R_NIR": mean_R_nir,
            "mean_T_NIR": mean_T_nir,
            "mean_alpha_NIR": mean_alpha_nir,
            "mean_eps_MIR": mean_eps_mir,
            "areal_density_g_per_m2": total_areal_kg_per_m2 * 1000.0,
            "sail_mass_10m2_g": mass_10m2_g,
            "T_steady_K": T_K,
            "graphene_extra_mass_g_per_m2": graphene_areal_kg_per_m2 * 1000.0,
        }
        rows.append(row)
        print(
            f"  N={n:2d}: R_NIR={mean_R_nir:.3f}  T_NIR={mean_T_nir:.3f}  "
            f"α_NIR={mean_alpha_nir:.4f}  ε_MIR={mean_eps_mir:.3f}  "
            f"T_steady={T_K:.0f} K  mass={mass_10m2_g:.2f}g"
        )

    csv_path = out_dir / "graphene_scan_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {csv_path}")

    md_path = out_dir / "graphene_scan_summary.md"
    with open(md_path, "w") as f:
        f.write("# Graphene N-layer scan on Design A\n\n")
        f.write(
            "| N | R_NIR | T_NIR | α_NIR | ε_MIR | "
            "T_steady [K] | mass [g, 10 m²] |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['n_layers']:>3} "
                f"| {r['mean_R_NIR']:.3f} "
                f"| {r['mean_T_NIR']:.3f} "
                f"| {r['mean_alpha_NIR']:.4f} "
                f"| {r['mean_eps_MIR']:.3f} "
                f"| {r['T_steady_K']:.0f} "
                f"| {r['sail_mass_10m2_g']:.2f} |\n"
            )
    print(f"Markdown written to {md_path}")
    _maybe_plot(out_dir, rows)


def _hole_fill_fraction() -> float:
    """Material fraction = 1 − hole area / unit cell area for Design A."""
    a = DESIGN_A["hole_a_rel"]
    b = DESIGN_A["hole_b_rel"]
    period = DESIGN_A["lattice_period_nm"]
    hole_area = np.pi * (a * period) * (b * period)
    cell_area = period * period * np.sqrt(3.0) / 2.0   # triangular primitive
    return float(min(0.999, hole_area / cell_area))


def _maybe_plot(out_dir: Path, rows: list) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = [r["n_layers"] for r in rows]
    eps_mir = [r["mean_eps_MIR"] for r in rows]
    R_nir = [r["mean_R_NIR"] for r in rows]
    T_K = [r["T_steady_K"] for r in rows]
    mass = [r["sail_mass_10m2_g"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(n, eps_mir, "o-", label="ε_MIR")
    axes[0].plot(n, R_nir, "s-", label="R_NIR")
    axes[0].set_xlabel("# graphene monolayers")
    axes[0].set_ylabel("Band-mean")
    axes[0].set_title("NIR R / MIR ε vs N")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(n, T_K, "o-", color="C3")
    axes[1].set_xlabel("# graphene monolayers")
    axes[1].set_ylabel("T_steady [K]")
    axes[1].set_title("Steady-state sail temperature")
    axes[1].grid(alpha=0.3)

    axes[2].plot(mass, eps_mir, "o-", color="C2")
    for nv, mv, ev in zip(n, mass, eps_mir):
        axes[2].annotate(f"N={nv}", (mv, ev), fontsize=8)
    axes[2].set_xlabel("Total mass (10 m²) [g]")
    axes[2].set_ylabel("MIR ε")
    axes[2].set_title("Pareto: ε_MIR vs mass")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "graphene_scan_plot.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
