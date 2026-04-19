"""Phase 4-C refined — Tight-settings robustness for 6 core cases.

Re-runs the critical robustness checks with paper-grade integration
(rtol=1e-6, n_radial 20/8, n_azim 48, t=5s) to distinguish numerical
artifacts from true physical FAILs.
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


DESIGN_A = dict(
    lattice_family=LatticeFamily.TRIANGULAR,
    thickness_nm=280.0, lattice_period_nm=1580.0,
    hole_a_rel=600.0 / 1580.0, hole_b_rel=600.0 / 1580.0,
    hole_rotation_deg=0.0, corner_rounding=1.0, shape_parameter=8.0,
)
RING_NOMINAL = dict(grating_period_nm=1424.0, duty_cycle=0.47, thickness_nm=280.0)
RC_NOMINAL = 30.0

SAIL_RADIUS_M = 1.5
RING_INNER_M = 1.45
SAIL_MASS_KG = 5.0e-3
SPIN_FREQ_HZ = 120.0
T_END_S = 5.0
N_EVAL = 150
RTOL = 1e-6
ATOL = 1e-9

MAX_POSITION_M = 1.8
MAX_TILT_DEG = 10.0
MAX_SPIN_DRIFT_FRAC = 0.05


def build_ring_lut(period_nm, duty, thickness_nm=280.0):
    return compute_ring_lut(
        grating_period_nm=period_nm, duty_cycle=duty, thickness_nm=thickness_nm,
        theta_grid_deg=np.linspace(-5, 5, 11),
        wavelengths_nm=np.linspace(1550, 1898, 3),
        fmm_config=FMMGratingConfig(nG=15, nx=64, ny=4),
    )


def build_center_lut():
    return compute_center_lut(
        PhCReflector(**DESIGN_A),
        np.linspace(-5, 5, 9), np.array([1700.0]),
        RCWAConfig(nG=21, grid_nx=64, grid_ny=64),
    )


def run_one(label, center_lut, ring_lut, R_c_m, waist_m, I0, x0_mm, y0_mm, thx_deg, thy_deg):
    geo = SailGeometry(R_inner_m=RING_INNER_M, R_outer_m=SAIL_RADIUS_M, curvature_radius_m=R_c_m)
    beam = GaussianBeam(I0_W_per_m2=I0, waist_m=waist_m, wavelength_nm=1700.0)
    mass = SailMass(mass_kg=SAIL_MASS_KG, radius_m=SAIL_RADIUS_M)
    cfg = IntegrationConfig(n_radial_center=20, n_radial_ring=8, n_azimuthal=48)
    res = run_trajectory(
        initial_position_m=(x0_mm * 1e-3, y0_mm * 1e-3, 0.0),
        initial_tilt_rad=(np.deg2rad(thx_deg), np.deg2rad(thy_deg), 0.0),
        spin_freq_Hz=SPIN_FREQ_HZ, geometry=geo, beam=beam, mass=mass,
        center_lut=center_lut, ring_lut=ring_lut,
        integration_config=cfg,
        t_end_s=T_END_S, n_eval=N_EVAL, rtol=RTOL, atol=ATOL,
    )
    max_lat = float(res.lateral_displacement_m().max())
    max_tilt = float(res.tilt_magnitude_deg().max())
    spin_drift = float(np.max(np.abs((res.state[11] - 2*np.pi*SPIN_FREQ_HZ)/(2*np.pi*SPIN_FREQ_HZ))))
    PASS = max_lat < MAX_POSITION_M and max_tilt < MAX_TILT_DEG and spin_drift < MAX_SPIN_DRIFT_FRAC
    return {
        "label": label, "PASS": PASS,
        "max_lateral_m": max_lat, "max_tilt_deg": max_tilt,
        "spin_drift_frac": spin_drift, "z_m": float(res.z[-1]),
    }


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_robustness_tight"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building LUTs (center + nominal ring)...")
    center_lut = build_center_lut()
    ring_nominal = build_ring_lut(RING_NOMINAL["grating_period_nm"], RING_NOMINAL["duty_cycle"])
    ring_plus10 = build_ring_lut(RING_NOMINAL["grating_period_nm"] * 1.1, RING_NOMINAL["duty_cycle"])
    ring_minus10 = build_ring_lut(RING_NOMINAL["grating_period_nm"] * 0.9, RING_NOMINAL["duty_cycle"])
    print()

    rows = []
    print("=== Tight-settings robustness (6 core cases) ===")
    # Case 1: Nominal (re-verify PASS)
    print("  [1] Nominal: w=4m, x=50mm, θ=-2°, I=1×Starshot, R_c=30m ...", end=" ", flush=True)
    r = run_one("nominal", center_lut, ring_nominal, 30.0, 4.0, 1e10, 50, 50, -2, -2)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    # Case 2: Larger perturbation
    print("  [2] 2× perturbation: x=100mm, θ=-4° ...", end=" ", flush=True)
    r = run_one("large_pert", center_lut, ring_nominal, 30.0, 4.0, 1e10, 100, 100, -4, -4)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    # Case 3: Narrower beam
    print("  [3] Narrow beam: w=3m ...", end=" ", flush=True)
    r = run_one("narrow_beam", center_lut, ring_nominal, 30.0, 3.0, 1e10, 50, 50, -2, -2)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    # Case 4: Higher intensity (thermal unrelated, but dynamics scale)
    print("  [4] 5× intensity: I=5 GW/m² ...", end=" ", flush=True)
    r = run_one("hi_intensity", center_lut, ring_nominal, 30.0, 4.0, 5e10, 50, 50, -2, -2)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    # Case 5: Ring period +10% (fab tolerance)
    print("  [5] Ring period +10% ...", end=" ", flush=True)
    r = run_one("period+10", center_lut, ring_plus10, 30.0, 4.0, 1e10, 50, 50, -2, -2)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    # Case 6: R_c -20% (curvature fab tolerance)
    print("  [6] R_c -20% (=24m) ...", end=" ", flush=True)
    r = run_one("Rc-20", center_lut, ring_nominal, 24.0, 4.0, 1e10, 50, 50, -2, -2)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    # Case 7 bonus: Small perturbation (x=20mm, θ=-1°) — expected comfortable PASS
    print("  [7] Small perturbation: x=20mm, θ=-1° ...", end=" ", flush=True)
    r = run_one("small_pert", center_lut, ring_nominal, 30.0, 4.0, 1e10, 20, 20, -1, -1)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    # Case 8: Ring period -10% (other fab tolerance)
    print("  [8] Ring period -10% ...", end=" ", flush=True)
    r = run_one("period-10", center_lut, ring_minus10, 30.0, 4.0, 1e10, 50, 50, -2, -2)
    rows.append(r); print(f"{'PASS' if r['PASS'] else 'FAIL'}  tilt={r['max_tilt_deg']:.2f}° xy={r['max_lateral_m']:.3f}m")

    csv_path = out_dir / "robustness_tight.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    md_path = out_dir / "robustness_tight.md"
    with open(md_path, "w") as f:
        f.write("# Phase 4-C (tight re-run) — 8 critical robustness cases\n\n")
        f.write("Settings: rtol=1e-6, n_radial 20/8/48, t=5s (paper-grade).\n\n")
        f.write("| Case | PASS | max |tilt| [°] | max |xy| [m] |\n")
        f.write("|---|:---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['label']} | {'PASS' if r['PASS'] else 'FAIL'} | {r['max_tilt_deg']:.2f} | {r['max_lateral_m']:.3f} |\n")

    pc = sum(1 for r in rows if r["PASS"])
    print(f"\n=== Tight robustness: {pc}/{len(rows)} PASS ===")
    print(f"CSV → {csv_path}")


if __name__ == "__main__":
    main()
