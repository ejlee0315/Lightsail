"""Phase 4-C — Robustness sweeps.

Tests stability of the BO-best ring + R_c=30m curved sail under
variations in:
  (a) beam waist w ∈ {3, 4, 5} m
  (b) initial perturbation: x₀ ∈ {20, 50, 100} mm × θ₀ ∈ {-1, -2, -4}°
  (c) laser intensity I₀ ∈ {1, 5, 10} GW/m²
  (d) ring period tolerance ±10%, sail R_c tolerance ±20%

Output: PASS region map for paper-grade reviewer-proof robustness claim.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.dynamics import (
    GaussianBeam,
    IntegrationConfig,
    SailGeometry,
    SailMass,
    compute_center_lut,
    compute_ring_lut,
    run_trajectory,
)
from lightsail.simulation import RCWAConfig
from lightsail.simulation.grating_fmm import FMMGratingConfig


# ---------------------------------------------------------------------------
# Recommended design (Phase 1+2 final, t=280)
# ---------------------------------------------------------------------------
DESIGN_A = dict(
    lattice_family=LatticeFamily.TRIANGULAR,
    thickness_nm=280.0,
    lattice_period_nm=1580.0,
    hole_a_rel=600.0 / 1580.0,
    hole_b_rel=600.0 / 1580.0,
    hole_rotation_deg=0.0,
    corner_rounding=1.0,
    shape_parameter=8.0,
)
RING_NOMINAL = dict(grating_period_nm=1424.0, duty_cycle=0.47, thickness_nm=280.0)
RC_NOMINAL = 30.0

SAIL_RADIUS_M = 1.5
RING_INNER_M = 1.45
SAIL_MASS_KG = 5.0e-3
SPIN_FREQ_HZ = 120.0
T_END_S = 5.0
N_EVAL = 150
RTOL = 1e-5

MAX_POSITION_M = 1.8
MAX_TILT_DEG = 10.0
MAX_SPIN_DRIFT_FRAC = 0.05


def build_ring_lut_for(period_nm: float, duty: float, thickness_nm: float = 280.0):
    return compute_ring_lut(
        grating_period_nm=period_nm,
        duty_cycle=duty,
        thickness_nm=thickness_nm,
        theta_grid_deg=np.linspace(-5.0, 5.0, 11),
        wavelengths_nm=np.linspace(1550.0, 1898.0, 3),
        fmm_config=FMMGratingConfig(nG=15, nx=64, ny=4),
    )


def build_center_lut_once():
    return compute_center_lut(
        PhCReflector(**DESIGN_A),
        np.linspace(-5.0, 5.0, 9),
        np.array([1700.0]),
        RCWAConfig(nG=21, grid_nx=64, grid_ny=64),
    )


def run_one(
    center_lut, ring_lut, R_c_m: float,
    waist_m: float, I0_W_per_m2: float,
    x0_mm: float, y0_mm: float,
    th_x_deg: float, th_y_deg: float,
):
    geo = SailGeometry(
        R_inner_m=RING_INNER_M, R_outer_m=SAIL_RADIUS_M,
        curvature_radius_m=R_c_m,
    )
    beam = GaussianBeam(I0_W_per_m2=I0_W_per_m2, waist_m=waist_m, wavelength_nm=1700.0)
    mass = SailMass(mass_kg=SAIL_MASS_KG, radius_m=SAIL_RADIUS_M)
    int_cfg = IntegrationConfig(n_radial_center=12, n_radial_ring=4, n_azimuthal=32)
    res = run_trajectory(
        initial_position_m=(x0_mm * 1e-3, y0_mm * 1e-3, 0.0),
        initial_tilt_rad=(np.deg2rad(th_x_deg), np.deg2rad(th_y_deg), 0.0),
        spin_freq_Hz=SPIN_FREQ_HZ,
        geometry=geo, beam=beam, mass=mass,
        center_lut=center_lut, ring_lut=ring_lut,
        integration_config=int_cfg,
        t_end_s=T_END_S, n_eval=N_EVAL, rtol=RTOL,
    )
    max_lat = float(res.lateral_displacement_m().max())
    max_tilt = float(res.tilt_magnitude_deg().max())
    spin_drift = float(np.max(np.abs((res.state[11] - 2 * np.pi * SPIN_FREQ_HZ) / (2 * np.pi * SPIN_FREQ_HZ))))
    return {
        "PASS": (max_lat < MAX_POSITION_M and max_tilt < MAX_TILT_DEG
                 and spin_drift < MAX_SPIN_DRIFT_FRAC),
        "max_lateral_m": max_lat, "max_tilt_deg": max_tilt,
        "spin_drift_frac": spin_drift, "z_m": float(res.z[-1]),
    }


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building LUTs (center + nominal ring) ...")
    center_lut = build_center_lut_once()
    ring_nominal = build_ring_lut_for(RING_NOMINAL["grating_period_nm"], RING_NOMINAL["duty_cycle"])
    print()

    rows = []

    # === (a) Beam waist sweep ===
    print("[a] Beam waist sweep:")
    for w in (3.0, 4.0, 5.0, 6.0):
        v = run_one(center_lut, ring_nominal, RC_NOMINAL, w, 1e10, 50, 50, -2, -2)
        rows.append({"sweep": "beam_waist", "var_label": f"w={w}m",
                     "value": w, **v})
        print(f"  w={w}m: {'PASS' if v['PASS'] else 'FAIL'}  tilt={v['max_tilt_deg']:.2f}° xy={v['max_lateral_m']:.3f}m")

    # === (b) Initial perturbation ===
    print("\n[b] Initial perturbation:")
    for x0 in (20, 50, 100, 200):
        for th in (-1.0, -2.0, -4.0):
            v = run_one(center_lut, ring_nominal, RC_NOMINAL, 4.0, 1e10, x0, x0, th, th)
            rows.append({"sweep": "initial_pert",
                         "var_label": f"x0={x0}mm,th={th}",
                         "value": x0, **v})
            print(f"  x0={x0}mm θ={th}°: {'PASS' if v['PASS'] else 'FAIL'}  tilt={v['max_tilt_deg']:.2f}° xy={v['max_lateral_m']:.3f}m")

    # === (c) Laser intensity ===
    print("\n[c] Laser intensity:")
    for I in (0.1e10, 0.5e10, 1.0e10, 2.0e10, 5.0e10):
        v = run_one(center_lut, ring_nominal, RC_NOMINAL, 4.0, I, 50, 50, -2, -2)
        rows.append({"sweep": "intensity",
                     "var_label": f"I={I/1e10:.1f}xStarshot",
                     "value": I/1e10, **v})
        print(f"  I={I/1e10:.1f}×10GW: {'PASS' if v['PASS'] else 'FAIL'}  tilt={v['max_tilt_deg']:.2f}° xy={v['max_lateral_m']:.3f}m")

    # === (d) Ring period tolerance ===
    print("\n[d] Ring period tolerance (±10%):")
    nominal_P = RING_NOMINAL["grating_period_nm"]
    for frac in (-0.1, -0.05, 0.0, 0.05, 0.1):
        P_test = nominal_P * (1 + frac)
        ring_test = build_ring_lut_for(P_test, RING_NOMINAL["duty_cycle"])
        v = run_one(center_lut, ring_test, RC_NOMINAL, 4.0, 1e10, 50, 50, -2, -2)
        rows.append({"sweep": "ring_period",
                     "var_label": f"P={P_test:.0f}nm ({frac*100:+.0f}%)",
                     "value": P_test, **v})
        print(f"  P={P_test:.0f}nm: {'PASS' if v['PASS'] else 'FAIL'}  tilt={v['max_tilt_deg']:.2f}° xy={v['max_lateral_m']:.3f}m")

    # === (e) Sail R_c tolerance ===
    print("\n[e] Sail curvature R_c tolerance (±20%):")
    for frac in (-0.2, -0.1, 0.0, 0.1, 0.2):
        Rc_test = RC_NOMINAL * (1 + frac)
        v = run_one(center_lut, ring_nominal, Rc_test, 4.0, 1e10, 50, 50, -2, -2)
        rows.append({"sweep": "Rc_tolerance",
                     "var_label": f"R_c={Rc_test:.1f}m ({frac*100:+.0f}%)",
                     "value": Rc_test, **v})
        print(f"  R_c={Rc_test:.1f}m: {'PASS' if v['PASS'] else 'FAIL'}  tilt={v['max_tilt_deg']:.2f}° xy={v['max_lateral_m']:.3f}m")

    csv_path = out_dir / "robustness_sweeps.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "robustness_sweeps.md"
    with open(md_path, "w") as f:
        f.write("# Phase 4-C — Robustness sweeps\n\n")
        f.write("Nominal design: Design A center + BO best ring (P=1424nm, duty=0.47, t=280nm) + R_c=30m parabolic curvature.\n\n")
        f.write("Initial: x0=y0=50mm, θ_x0=θ_y0=-2°, spin=120Hz. Threshold: |xy|<1.8m, |tilt|<10°.\n\n")
        last_sweep = None
        for r in rows:
            if r["sweep"] != last_sweep:
                f.write(f"\n## {r['sweep']}\n\n| variable | PASS | max |xy| [m] | max |tilt| [°] |\n|---|---|---:|---:|\n")
                last_sweep = r["sweep"]
            f.write(f"| {r['var_label']} | {'PASS' if r['PASS'] else 'FAIL'} | {r['max_lateral_m']:.3f} | {r['max_tilt_deg']:.2f} |\n")

    pass_count = sum(1 for r in rows if r["PASS"])
    print(f"\n=== Summary: {pass_count}/{len(rows)} PASS ===")
    print(f"CSV → {csv_path}")
    print(f"MD  → {md_path}")


if __name__ == "__main__":
    main()
