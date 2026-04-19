"""Option-4-step1 — Uniform SiC backside underlayer scan.

Stacks a uniform 4H-SiC slab (Reststrahlen 10.3–12.6 µm) behind the
Design A PhC and sweeps thickness ∈ {0, 50, 100, 250, 500, 1000} nm.
Reports R_NIR / α_NIR / ε_MIR (8–14 µm) and the resulting steady-state
sail temperature under Starshot 10 GW/m² illumination.

This is the *uniform* baseline for the multi-resonance metasurface
absorber program. If SiC alone yields ε_MIR > 0.3 averaged across the
band, no patterning is needed; otherwise we move to a patterned
metasurface (Mie/MIM resonator at 8–14 µm).
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.materials import SIC_DENSITY_KG_PER_M3, SiCDispersion
from lightsail.simulation import (
    LayeredRCWASolver,
    LayerSpec,
    RCWAConfig,
    RCWASolver,
)


NIR_BAND = (1550.0, 1850.0)
MIR_BAND = (8000.0, 14000.0)
INTENSITY_W_PER_M2 = 1.0e10
SIGMA_SB = 5.670374419e-8

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

SIC_THICKNESSES_NM = (0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
SIN_DENSITY_KG_PER_M3 = 3100.0


def design_A() -> PhCReflector:
    return PhCReflector(**DESIGN_A)


def sic_layer(thickness_nm: float) -> LayerSpec:
    sic = SiCDispersion()
    return LayerSpec(
        thickness_nm=thickness_nm,
        eps_callable=sic.epsilon_callable(),
        name=f"SiC_{thickness_nm:.0f}nm",
    )


def thermal_steady_K(alpha_nir: float, eps_mir: float) -> float:
    if alpha_nir <= 1.0e-6 or eps_mir <= 1.0e-6:
        return float("nan")
    return float(
        (alpha_nir * INTENSITY_W_PER_M2 / (2.0 * SIGMA_SB * eps_mir)) ** 0.25
    )


def hole_fill_fraction() -> float:
    a = DESIGN_A["hole_a_rel"]
    b = DESIGN_A["hole_b_rel"]
    period = DESIGN_A["lattice_period_nm"]
    hole_area = np.pi * (a * period) * (b * period)
    cell_area = period * period * np.sqrt(3.0) / 2.0
    return float(min(0.999, hole_area / cell_area))


def main() -> None:
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p2_4_sic_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    rcwa_cfg = RCWAConfig(nG=41, grid_nx=64, grid_ny=64)
    structure = design_A().to_structure()

    nir_wls = np.linspace(NIR_BAND[0], NIR_BAND[1], 9)
    mir_wls = np.linspace(MIR_BAND[0], MIR_BAND[1], 13)

    rows = []
    for t_nm in SIC_THICKNESSES_NM:
        if t_nm == 0.0:
            solver = RCWASolver(config=rcwa_cfg)
        else:
            solver = LayeredRCWASolver(
                config=rcwa_cfg, layers_below=[sic_layer(t_nm)],
            )

        R_nir = solver.evaluate_reflectivity(structure, nir_wls)
        T_nir = solver.evaluate_transmission(structure, nir_wls)
        alpha_nir = np.clip(1.0 - R_nir - T_nir, 0.0, 1.0)

        R_mir = solver.evaluate_reflectivity(structure, mir_wls)
        T_mir = solver.evaluate_transmission(structure, mir_wls)
        eps_mir_band = np.clip(1.0 - R_mir - T_mir, 0.0, 1.0)

        sin_areal = (
            SIN_DENSITY_KG_PER_M3 * DESIGN_A["thickness_nm"] * 1e-9
            * (1.0 - hole_fill_fraction())
        )
        sic_areal = SIC_DENSITY_KG_PER_M3 * t_nm * 1e-9
        total_areal = sin_areal + sic_areal
        sail_mass_g = total_areal * 10.0 * 1000.0

        T_K = thermal_steady_K(float(alpha_nir.mean()), float(eps_mir_band.mean()))

        # peak ε in the Reststrahlen window
        peak_idx = int(np.argmax(eps_mir_band))
        peak_wl_um = mir_wls[peak_idx] / 1000.0
        peak_eps = float(eps_mir_band[peak_idx])

        row = {
            "thickness_nm": t_nm,
            "mean_R_NIR": float(R_nir.mean()),
            "mean_T_NIR": float(T_nir.mean()),
            "mean_alpha_NIR": float(alpha_nir.mean()),
            "mean_eps_MIR": float(eps_mir_band.mean()),
            "peak_eps_MIR": peak_eps,
            "peak_wl_um": peak_wl_um,
            "areal_density_g_per_m2": float(total_areal * 1000.0),
            "sail_mass_10m2_g": sail_mass_g,
            "T_steady_K": T_K,
            "extra_mass_g_per_m2": float(sic_areal * 1000.0),
        }
        rows.append(row)
        print(
            f"  t_SiC={t_nm:5.0f}nm: R_NIR={row['mean_R_NIR']:.3f}  "
            f"α_NIR={row['mean_alpha_NIR']:.4f}  ε_MIR_avg={row['mean_eps_MIR']:.3f}  "
            f"peak={peak_eps:.3f} @{peak_wl_um:.1f}µm  "
            f"T={row['T_steady_K'] if np.isfinite(row['T_steady_K']) else float('nan'):.0f}K  "
            f"mass={sail_mass_g:.2f}g"
        )

    csv_path = out_dir / "sic_scan_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "sic_scan_summary.md"
    with open(md_path, "w") as f:
        f.write("# SiC underlayer scan on Design A (Option-4 step 1)\n\n")
        f.write(
            "| t_SiC [nm] | R_NIR | α_NIR | ε_MIR_avg | peak ε / λ | "
            "T_steady [K] | mass [g, 10 m²] |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            t_str = "n/a" if not np.isfinite(r["T_steady_K"]) else f"{r['T_steady_K']:.0f}"
            f.write(
                f"| {r['thickness_nm']:.0f} "
                f"| {r['mean_R_NIR']:.3f} "
                f"| {r['mean_alpha_NIR']:.4f} "
                f"| {r['mean_eps_MIR']:.3f} "
                f"| {r['peak_eps_MIR']:.3f} @ {r['peak_wl_um']:.1f}µm "
                f"| {t_str} "
                f"| {r['sail_mass_10m2_g']:.2f} |\n"
            )

    print(f"\nCSV     → {csv_path}")
    print(f"Summary → {md_path}")
    _maybe_plot(out_dir, rows, mir_wls, structure)


def _maybe_plot(out_dir: Path, rows: list, mir_wls, structure) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    t = [r["thickness_nm"] for r in rows]
    ε = [r["mean_eps_MIR"] for r in rows]
    peak = [r["peak_eps_MIR"] for r in rows]
    R = [r["mean_R_NIR"] for r in rows]
    mass = [r["sail_mass_10m2_g"] for r in rows]
    Tk = [r["T_steady_K"] for r in rows]

    axes[0].plot(t, ε, "o-", label="ε_MIR avg (8–14 µm)")
    axes[0].plot(t, peak, "s--", label="peak ε_MIR")
    axes[0].plot(t, R, "^-", label="R_NIR (preserve)")
    axes[0].set_xlabel("SiC thickness [nm]")
    axes[0].set_ylabel("Band-mean / peak")
    axes[0].set_title("SiC thickness sweep")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, Tk, "o-", color="C3")
    axes[1].set_xlabel("SiC thickness [nm]")
    axes[1].set_ylabel("T_steady [K]")
    axes[1].set_title("Steady-state sail temperature")
    axes[1].grid(alpha=0.3)

    axes[2].plot(mass, ε, "o-", color="C2")
    for tv, mv, ev in zip(t, mass, ε):
        axes[2].annotate(f"{tv:.0f}nm", (mv, ev), fontsize=8)
    axes[2].set_xlabel("Total mass (10 m²) [g]")
    axes[2].set_ylabel("ε_MIR avg")
    axes[2].set_title("Pareto: ε_MIR vs mass")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "sic_scan_plot.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
