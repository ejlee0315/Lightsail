"""Option-4 — Compare backside thermal-functional underlayer materials.

Sweeps several materials at multiple thicknesses on the back of
Design A and reports R_NIR / α_NIR / ε_MIR / mass / T_steady. Goal is
to find a *multi-resonance* MIR absorber whose mass penalty is
acceptable for Starshot-class lightsails.

Materials tested
----------------
* none           (bare PhC reference)
* graphene N=1, 5, 10        (universal absorption — fails at MIR per P2.3)
* SiC 50, 100, 250 nm        (single Reststrahlen 10–13 µm, but heavy)
* h-BN 50, 100, 250 nm       (TWO phonon bands 6–7 + 12–13 µm; lighter)

Output: results/<timestamp>_p4_underlayer_compare/ with CSV + Markdown
table + per-row spectrum plots.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.materials import (
    GRAPHENE_LAYER_THICKNESS_M,
    GrapheneConductivity,
    HBNDispersion,
    HBN_DENSITY_KG_PER_M3,
    SIC_DENSITY_KG_PER_M3,
    SiCDispersion,
)
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
SAIL_AREA_M2 = 10.0
PAYLOAD_KG = 1.0e-3

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
SIN_DENSITY_KG_PER_M3 = 3100.0
GRAPHENE_AREAL_DENSITY_KG_PER_M2 = 0.77e-6


CASES = [
    # (label, material, layers list)  — each entry produces one configuration
    ("none (bare PhC)", None, None),
    ("graphene_x1",  "graphene", 1),
    ("graphene_x5",  "graphene", 5),
    ("graphene_x10", "graphene", 10),
    ("SiC_50nm",   "sic", 50.0),
    ("SiC_100nm",  "sic", 100.0),
    ("SiC_250nm",  "sic", 250.0),
    ("hBN_50nm",   "hbn", 50.0),
    ("hBN_100nm",  "hbn", 100.0),
    ("hBN_250nm",  "hbn", 250.0),
]


def design_A():
    return PhCReflector(**DESIGN_A)


def hole_fill_fraction() -> float:
    a = DESIGN_A["hole_a_rel"]
    b = DESIGN_A["hole_b_rel"]
    period = DESIGN_A["lattice_period_nm"]
    hole_area = np.pi * (a * period) * (b * period)
    cell_area = period * period * np.sqrt(3.0) / 2.0
    return float(min(0.999, hole_area / cell_area))


def make_solver_and_extra_mass(material: str, spec, rcwa_cfg) -> tuple[RCWASolver, float]:
    """Return (solver, extra_areal_density_kg_per_m2)."""
    if material is None:
        return RCWASolver(config=rcwa_cfg), 0.0

    if material == "graphene":
        n_layers = int(spec)
        g = GrapheneConductivity(E_F_eV=0.3)
        layer = LayerSpec(
            thickness_nm=n_layers * GRAPHENE_LAYER_THICKNESS_M * 1e9,
            eps_callable=g.epsilon_callable(GRAPHENE_LAYER_THICKNESS_M),
            name=f"graphene_x{n_layers}",
        )
        extra_areal = n_layers * GRAPHENE_AREAL_DENSITY_KG_PER_M2
    elif material == "sic":
        t_nm = float(spec)
        sic = SiCDispersion()
        layer = LayerSpec(
            thickness_nm=t_nm,
            eps_callable=sic.epsilon_callable(),
            name=f"SiC_{t_nm:.0f}nm",
        )
        extra_areal = SIC_DENSITY_KG_PER_M3 * t_nm * 1e-9
    elif material == "hbn":
        t_nm = float(spec)
        hbn = HBNDispersion()
        layer = LayerSpec(
            thickness_nm=t_nm,
            eps_callable=hbn.epsilon_callable(),
            name=f"hBN_{t_nm:.0f}nm",
        )
        extra_areal = HBN_DENSITY_KG_PER_M3 * t_nm * 1e-9
    else:
        raise ValueError(f"Unknown material: {material}")

    return LayeredRCWASolver(config=rcwa_cfg, layers_below=[layer]), extra_areal


def thermal_steady_K(alpha_nir: float, eps_mir: float) -> float:
    if alpha_nir <= 1.0e-6 or eps_mir <= 1.0e-6:
        return float("nan")
    return float(
        (alpha_nir * INTENSITY_W_PER_M2 / (2.0 * SIGMA_SB * eps_mir)) ** 0.25
    )


def evaluate(label: str, material, spec, rcwa_cfg, structure, nir_wls, mir_wls) -> dict:
    solver, extra_areal = make_solver_and_extra_mass(material, spec, rcwa_cfg)
    R_nir = solver.evaluate_reflectivity(structure, nir_wls)
    T_nir = solver.evaluate_transmission(structure, nir_wls)
    alpha_nir = np.clip(1.0 - R_nir - T_nir, 0.0, 1.0)
    R_mir = solver.evaluate_reflectivity(structure, mir_wls)
    T_mir = solver.evaluate_transmission(structure, mir_wls)
    eps_mir = np.clip(1.0 - R_mir - T_mir, 0.0, 1.0)

    sin_areal = (
        SIN_DENSITY_KG_PER_M3 * DESIGN_A["thickness_nm"] * 1e-9
        * (1.0 - hole_fill_fraction())
    )
    total_areal = sin_areal + extra_areal
    sail_mass_g = total_areal * SAIL_AREA_M2 * 1000.0

    return {
        "label": label,
        "mean_R_NIR": float(R_nir.mean()),
        "mean_T_NIR": float(T_nir.mean()),
        "mean_alpha_NIR": float(alpha_nir.mean()),
        "mean_eps_MIR": float(eps_mir.mean()),
        "peak_eps_MIR": float(eps_mir.max()),
        "peak_lambda_um": float(mir_wls[int(np.argmax(eps_mir))] / 1000.0),
        "areal_density_g_per_m2": float(total_areal * 1000.0),
        "extra_mass_g_per_m2": float(extra_areal * 1000.0),
        "sail_mass_10m2_g": sail_mass_g,
        "T_steady_K": thermal_steady_K(float(alpha_nir.mean()), float(eps_mir.mean())),
    }


def main() -> None:
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_underlayer_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    rcwa_cfg = RCWAConfig(nG=41, grid_nx=64, grid_ny=64)
    structure = design_A().to_structure()
    nir_wls = np.linspace(NIR_BAND[0], NIR_BAND[1], 9)
    mir_wls = np.linspace(MIR_BAND[0], MIR_BAND[1], 13)

    rows = []
    for label, material, spec in CASES:
        print(f"  {label} ...")
        row = evaluate(label, material, spec, rcwa_cfg, structure, nir_wls, mir_wls)
        rows.append(row)
        T_str = "n/a" if not np.isfinite(row["T_steady_K"]) else f"{row['T_steady_K']:.0f}K"
        print(
            f"    R_NIR={row['mean_R_NIR']:.3f}  α_NIR={row['mean_alpha_NIR']:.4f}  "
            f"ε_MIR_avg={row['mean_eps_MIR']:.3f}  peak={row['peak_eps_MIR']:.3f}@{row['peak_lambda_um']:.1f}µm  "
            f"T={T_str}  mass={row['sail_mass_10m2_g']:.2f}g"
        )

    csv_path = out_dir / "underlayer_compare.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "underlayer_compare.md"
    with open(md_path, "w") as f:
        f.write("# Backside thermal-functional underlayer comparison (Option-4)\n\n")
        f.write(
            "| Layer | R_NIR | α_NIR | ε_MIR avg | peak ε / λ | "
            "mass [g, 10 m²] | T_steady [K] |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            T_str = "n/a" if not np.isfinite(r["T_steady_K"]) else f"{r['T_steady_K']:.0f}"
            f.write(
                f"| {r['label']} "
                f"| {r['mean_R_NIR']:.3f} "
                f"| {r['mean_alpha_NIR']:.4f} "
                f"| {r['mean_eps_MIR']:.3f} "
                f"| {r['peak_eps_MIR']:.3f} @ {r['peak_lambda_um']:.1f}µm "
                f"| {r['sail_mass_10m2_g']:.2f} "
                f"| {T_str} |\n"
            )

    print(f"\nCSV     → {csv_path}")
    print(f"Summary → {md_path}")
    _maybe_plot(out_dir, rows, mir_wls, structure, rcwa_cfg)


def _maybe_plot(out_dir, rows, mir_wls, structure, rcwa_cfg) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [r["label"] for r in rows]
    eps_avg = [r["mean_eps_MIR"] for r in rows]
    mass = [r["sail_mass_10m2_g"] for r in rows]

    # Pareto: ε_MIR vs mass
    colors = []
    for lab in labels:
        if "graphene" in lab:
            colors.append("C0")
        elif "SiC" in lab:
            colors.append("C1")
        elif "hBN" in lab:
            colors.append("C2")
        else:
            colors.append("k")
    axes[0].scatter(mass, eps_avg, c=colors, s=80)
    for lab, m, e in zip(labels, mass, eps_avg):
        axes[0].annotate(lab, (m, e), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("Total sail mass (10 m²) [g]")
    axes[0].set_ylabel("ε_MIR average (8–14 µm)")
    axes[0].set_title("Pareto: ε_MIR vs mass")
    axes[0].grid(alpha=0.3)

    # MIR spectrum overlay (for selected materials)
    selected_indices = [0, 2, 5, 8]   # bare, graphene_x5, SiC_100, hBN_100
    for idx in selected_indices:
        if idx >= len(rows):
            continue
        # Re-compute spectrum on demand
        label, material, spec = CASES[idx]
        solver, _ = make_solver_and_extra_mass(material, spec, rcwa_cfg)
        wls_dense = np.linspace(MIR_BAND[0], MIR_BAND[1], 41)
        R = solver.evaluate_reflectivity(structure, wls_dense)
        T = solver.evaluate_transmission(structure, wls_dense)
        eps = np.clip(1.0 - R - T, 0.0, 1.0)
        axes[1].plot(wls_dense / 1000.0, eps, label=label)
    axes[1].set_xlabel("Wavelength [µm]")
    axes[1].set_ylabel("ε(λ)")
    axes[1].set_title("MIR emissivity spectrum")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "underlayer_compare_plot.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
