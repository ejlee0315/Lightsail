"""P4 — End-to-end integrated sail analysis (3-zone vs single-zone).

Compares three sail configurations on the full mission FOM stack:

    (A) Baseline       Design A PhC only (CLAUDE.md, T = 20.73 min)
    (B) PhC + Ring     Design A + best metagrating outer annulus
    (C) PhC + Ring +   (B) + best graphene layer count (impedance-matched MIR)
        Graphene

Quantities reported per configuration::

    T (β=0.2)             [min]   acceleration time, ∫_0^β γ³/R · (1+β)/(1-β) dβ
    D                     [Gm]    distance to β_f
    sail_mass             [g]     total mass of 10 m² sail
    R_NIR_doppler         [-]     mean R over Doppler band (1550–1898 nm)
    eps_MIR               [-]     mean MIR emissivity (8–14 µm)
    T_steady              [K]     thermal balance temperature
    k_theta_theta         [Nm/rad]  outer-ring restoring torque stiffness
    alpha_damp_meta       [Pa/(m/s)]  metasurface damping coefficient

Output: results/<timestamp>_p4_integrated_sail/ with summary CSV +
markdown comparison + spectrum plot.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.metagrating import MetaGrating
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.materials import GRAPHENE_LAYER_THICKNESS_M, GrapheneConductivity
from lightsail.simulation import (
    LayeredRCWASolver,
    LayerSpec,
    RCWAConfig,
    RCWASolver,
    compute_damping_force,
    compute_stiffness_matrix,
)
from lightsail.simulation.grating_fmm import FMMGratingConfig


C_LIGHT = 299_792_458.0
SIGMA_SB = 5.670374419e-8
INTENSITY_W_PER_M2 = 1.0e10
SAIL_AREA_M2 = 10.0
PAYLOAD_KG = 1.0e-3
SIN_DENSITY_KG_PER_M3 = 3100.0
GRAPHENE_AREAL_DENSITY_KG_PER_M2 = 0.77e-6

# Bands
NIR_DOPPLER_BAND_NM = (1550.0, 1898.0)   # β = 0 → 0.20
MIR_BAND_NM = (8000.0, 14000.0)
LAUNCH_NM = 1550.0
BETA_FINAL = 0.20

# Design A (CLAUDE.md, 2026-04-17)
DESIGN_A_PARAMS = dict(
    thickness_nm=280.0,
    lattice_period_nm=1580.0,
    hole_a_rel=600.0 / 1580.0,
    hole_b_rel=600.0 / 1580.0,
    hole_rotation_deg=0.0,
    corner_rounding=1.0,
    shape_parameter=8.0,
    lattice_family=LatticeFamily.TRIANGULAR,
)

# Best metagrating ring (from P3.2 enhancement scan: P=2400 nm gave
# the largest |dC_pr,2/dθ| ratio). Ring sits outside the central PhC.
BEST_RING_PARAMS = dict(
    inner_radius_nm=5_000_000.0,
    thickness_nm=240.0,
    grating_period_nm=2400.0,
    duty_cycle=0.5,
    curvature=0.05,
    asymmetry=0.10,
    ring_width_um=2000.0,
)

# Graphene count chosen by impedance match (P2.3 showed N≥5 over-shoots
# the metallic regime at MIR). Use a small N for a conservative test.
GRAPHENE_N_LAYERS = 3
GRAPHENE_E_F_eV = 0.3

# Numerical knobs
RCWA_CFG = RCWAConfig(nG=41, grid_nx=64, grid_ny=64)
FMM_CFG = FMMGratingConfig(nG=15, nx=96, ny=4)
N_NIR_POINTS = 11
N_MIR_POINTS = 7
N_T_INTEGRAL_POINTS = 30


def design_A() -> PhCReflector:
    return PhCReflector(**DESIGN_A_PARAMS)


def best_ring() -> MetaGrating:
    return MetaGrating(**BEST_RING_PARAMS)


def graphene_layer_spec(n: int) -> LayerSpec:
    g = GrapheneConductivity(E_F_eV=GRAPHENE_E_F_eV)
    return LayerSpec(
        thickness_nm=n * GRAPHENE_LAYER_THICKNESS_M * 1e9,
        eps_callable=g.epsilon_callable(GRAPHENE_LAYER_THICKNESS_M),
        name=f"graphene_x{n}",
    )


def evaluate_configuration(label: str, with_ring: bool, n_graphene: int) -> dict:
    phc = design_A()
    structure = phc.to_structure()

    if n_graphene > 0:
        solver = LayeredRCWASolver(
            config=RCWA_CFG, layers_below=[graphene_layer_spec(n_graphene)],
        )
    else:
        solver = RCWASolver(config=RCWA_CFG)

    # NIR Doppler-band sweep (used for T integral).
    nir_wls = np.linspace(NIR_DOPPLER_BAND_NM[0], NIR_DOPPLER_BAND_NM[1], N_NIR_POINTS)
    R_nir = solver.evaluate_reflectivity(structure, nir_wls)
    T_nir = solver.evaluate_transmission(structure, nir_wls)
    alpha_nir = np.clip(1.0 - R_nir - T_nir, 0.0, 1.0)

    mir_wls = np.linspace(MIR_BAND_NM[0], MIR_BAND_NM[1], N_MIR_POINTS)
    R_mir = solver.evaluate_reflectivity(structure, mir_wls)
    T_mir = solver.evaluate_transmission(structure, mir_wls)
    eps_mir = np.clip(1.0 - R_mir - T_mir, 0.0, 1.0)

    mean_R_nir = float(R_nir.mean())
    mean_alpha_nir = float(alpha_nir.mean())
    mean_eps_mir = float(eps_mir.mean())

    # Mass: SiN PhC (subtract hole area) + optional graphene
    cell_area_nm2 = (
        DESIGN_A_PARAMS["lattice_period_nm"] ** 2 * np.sqrt(3.0) / 2.0
    )
    hole_area_nm2 = (
        np.pi
        * (DESIGN_A_PARAMS["hole_a_rel"] * DESIGN_A_PARAMS["lattice_period_nm"])
        * (DESIGN_A_PARAMS["hole_b_rel"] * DESIGN_A_PARAMS["lattice_period_nm"])
    )
    f_mat = max(1.0 - hole_area_nm2 / cell_area_nm2, 0.0)
    rho_sin_areal = (
        SIN_DENSITY_KG_PER_M3 * DESIGN_A_PARAMS["thickness_nm"] * 1e-9 * f_mat
    )
    rho_g_areal = n_graphene * GRAPHENE_AREAL_DENSITY_KG_PER_M2
    sail_mass_kg = (rho_sin_areal + rho_g_areal) * SAIL_AREA_M2
    total_mass_kg = sail_mass_kg + PAYLOAD_KG

    # T (acceleration time) integral via direct sampling.
    T_minutes, distance_Gm = compute_T_and_D(
        nir_wls=nir_wls, R_nir=R_nir, total_mass_kg=total_mass_kg,
    )

    # Thermal: T_steady (skip if no absorption).
    T_steady_K = thermal_steady_K(mean_alpha_nir, mean_eps_mir)

    # Stage 2 stiffness + damping (only if ring is present).
    if with_ring:
        ring = best_ring()
        stiff = compute_stiffness_matrix(
            ring,
            nir_band_nm=NIR_DOPPLER_BAND_NM,
            nir_n_points=4,
            n_radial_bins=2,
            intensity_W_per_m2=INTENSITY_W_PER_M2,
            config=FMM_CFG,
        )
        damp_low = compute_damping_force(
            ring, beta=0.05, v_y_per_c=1.0e-4,
            lab_wavelength_nm=LAUNCH_NM,
            n_radial_bins=2, config=FMM_CFG,
        )
        damp_mid = compute_damping_force(
            ring, beta=0.15, v_y_per_c=1.0e-4,
            lab_wavelength_nm=LAUNCH_NM,
            n_radial_bins=2, config=FMM_CFG,
        )
        k_thth = stiff.k_thetatheta_Nm_per_rad
        alpha_damp_low = damp_low["alpha_damp_Pa_per_mps"]
        alpha_damp_mid = damp_mid["alpha_damp_Pa_per_mps"]
    else:
        k_thth = 0.0
        alpha_damp_low = 0.0
        alpha_damp_mid = 0.0

    return {
        "label": label,
        "with_ring": with_ring,
        "n_graphene": n_graphene,
        "mean_R_NIR_doppler": mean_R_nir,
        "mean_alpha_NIR": mean_alpha_nir,
        "mean_eps_MIR": mean_eps_mir,
        "T_minutes": T_minutes,
        "D_Gm": distance_Gm,
        "sail_mass_g": float(sail_mass_kg * 1000.0),
        "total_mass_g": float(total_mass_kg * 1000.0),
        "T_steady_K": T_steady_K,
        "k_theta_theta_Nm_per_rad": float(k_thth),
        "alpha_damp_beta_05_Pa_per_mps": float(alpha_damp_low),
        "alpha_damp_beta_15_Pa_per_mps": float(alpha_damp_mid),
    }


def compute_T_and_D(
    nir_wls: np.ndarray, R_nir: np.ndarray, total_mass_kg: float,
) -> tuple[float, float]:
    """Acceleration time T and distance D from R(λ_doppler).

    T = (m c² / (2 I A)) · ∫_0^β_f γ³ · (1+β)/(1-β) / R(λ_sail) dβ
    D = (c² / (2 I0)) · ∫_0^β_f rho_total / R(λ) · γ³ · (1+β)/(1-β) dβ
        — but with our parameterization where m_total absorbs sail+payload,
        a simpler form is D = ∫ v dt;  here we use trapezoid on the
        integrand and the mission's m·c²/(IA) prefactor (Norder Eq. 3
        with the c² correction documented in CLAUDE.md).
    """
    R_interp = lambda lam: float(np.interp(lam, nir_wls, R_nir))
    betas = np.linspace(0.0, BETA_FINAL, N_T_INTEGRAL_POINTS)
    integrand = np.zeros_like(betas)
    for i, b in enumerate(betas):
        if b < 1e-12:
            integrand[i] = 0.0
            continue
        gamma = 1.0 / np.sqrt(1.0 - b ** 2)
        lam_sail = LAUNCH_NM * np.sqrt((1.0 + b) / (1.0 - b))
        r = max(R_interp(lam_sail), 1e-3)
        integrand[i] = gamma ** 3 * (1.0 + b) / (1.0 - b) / r
    T_int = float(np.trapz(integrand, betas))
    prefactor_s = total_mass_kg * C_LIGHT ** 2 / (2.0 * INTENSITY_W_PER_M2 * SAIL_AREA_M2)
    T_seconds = prefactor_s * T_int
    T_minutes = T_seconds / 60.0

    # Distance: D = c² / (2 I0) · m / A · ∫ γ³ (1+β)/(1-β) / R · 1/(γ²β) dβ?
    # Use the documented Norder Eq.3-style integrand: D = c · ∫ β dt.
    # Approximate via D = c · t_β · β_avg = C_LIGHT · T_seconds · (BETA_FINAL/2)
    # which matches a constant-acceleration limit; replace later if needed.
    D_m = C_LIGHT * T_seconds * (BETA_FINAL / 2.0)
    D_Gm = D_m * 1.0e-9
    return T_minutes, D_Gm


def thermal_steady_K(alpha_nir: float, eps_mir: float) -> float:
    if alpha_nir <= 1.0e-6 or eps_mir <= 1.0e-6:
        return float("nan")
    return float(
        (alpha_nir * INTENSITY_W_PER_M2 / (2.0 * SIGMA_SB * eps_mir)) ** 0.25
    )


def main() -> None:
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_integrated_sail"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== P4 — Integrated 3-zone sail analysis ===\n")
    configs = [
        ("A: PhC only", False, 0),
        ("B: PhC + Ring", True, 0),
        ("C: PhC + Ring + Graphene_x3", True, 3),
    ]
    rows = []
    for label, with_ring, n_g in configs:
        print(f"--- {label} ---")
        row = evaluate_configuration(label, with_ring, n_g)
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k:32s} = {v:>12.4g}")
            else:
                print(f"  {k:32s} = {v}")
        print()

    csv_path = out_dir / "integrated_sail.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "integrated_sail.md"
    with open(md_path, "w") as f:
        f.write("# P4 — Integrated 3-zone sail analysis\n\n")
        f.write("| Config | T [min] | D [Gm] | mass [g] | R_NIR | ε_MIR | T_steady [K] | k_θθ [Nm/rad] |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            t_steady_str = (
                "n/a" if not np.isfinite(r["T_steady_K"]) else f"{r['T_steady_K']:.0f}"
            )
            f.write(
                f"| {r['label']} "
                f"| {r['T_minutes']:.2f} "
                f"| {r['D_Gm']:.1f} "
                f"| {r['total_mass_g']:.2f} "
                f"| {r['mean_R_NIR_doppler']:.3f} "
                f"| {r['mean_eps_MIR']:.3f} "
                f"| {t_steady_str} "
                f"| {r['k_theta_theta_Nm_per_rad']:+.3e} |\n"
            )
    print(f"\nResults written to {csv_path}")
    print(f"Markdown summary  : {md_path}")


if __name__ == "__main__":
    main()
