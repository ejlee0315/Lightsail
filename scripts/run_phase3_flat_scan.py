"""Phase 3-4 — Flat-sail stability via azimuthal ring modulation scan.

Sweeps (n_petals, mod_amp) with a flat sail (R_c = ∞) to test whether
breaking the ring's SO(2) rotational symmetry alone is enough to
achieve paper-style PASS (without any sail curvature). This is the
"Option C" goal: Gieseler 2024-style flat sail beam-riding from
engineered metasurface alone.

Sweep grid::

    n_petals ∈ {0, 2, 3, 4, 6}
    mod_amp  ∈ {0.0, 0.1, 0.3, 0.5}

(n=0 or ε=0 = axisymmetric baseline, used as control.)
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
    compute_ring_lut_2d,
    run_trajectory,
)
from lightsail.simulation import RCWAConfig
from lightsail.simulation.grating_fmm import FMMGratingConfig


# ---------------------------------------------------------------------------
# Fixed designs (match Phase 1 paper-grade simulation)
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
BASE_RING = dict(grating_period_nm=1424.0, base_duty=0.47, thickness_nm=280.0)

SAIL_RADIUS_M = 1.5
RING_INNER_M = 1.45
SAIL_MASS_KG = 5.0e-3
INTENSITY_W_PER_M2 = 1.0e10
BEAM_WAIST_M = 4.0
SPIN_FREQ_HZ = 120.0
T_END_S = 5.0
N_EVAL = 200

# Paper PASS thresholds (Gieseler 2024)
MAX_POSITION_M = 1.8
MAX_TILT_DEG = 10.0
MAX_SPIN_DRIFT_FRAC = 0.05

INIT_X_M = 0.05
INIT_Y_M = 0.05
INIT_TILT_X_DEG = -2.0
INIT_TILT_Y_DEG = -2.0

# Grid
N_PETALS_SET = (0, 2, 3, 4, 6)
MOD_AMP_SET = (0.0, 0.1, 0.3, 0.5)


def build_luts():
    print("[A1] Building center LUT + 2D ring LUT (θ × duty × λ) ...")
    theta_grid = np.linspace(-5.0, 5.0, 11)
    wls = np.linspace(1550.0, 1898.0, 3)
    duty_grid = np.linspace(0.1, 0.8, 6)   # 6-point duty grid for interpolation
    phc = PhCReflector(**DESIGN_A)
    center = compute_center_lut(
        phc, theta_grid, wls, RCWAConfig(nG=41, grid_nx=96, grid_ny=96),
    )
    ring2d = compute_ring_lut_2d(
        grating_period_nm=BASE_RING["grating_period_nm"],
        duty_grid=duty_grid,
        thickness_nm=BASE_RING["thickness_nm"],
        theta_grid_deg=theta_grid,
        wavelengths_nm=wls,
        fmm_config=FMMGratingConfig(nG=21, nx=128, ny=4),
    )
    return center, ring2d


def run_one(center_lut, ring_lut, n_petals: int, mod_amp: float):
    geo = SailGeometry(
        R_inner_m=RING_INNER_M,
        R_outer_m=SAIL_RADIUS_M,
        curvature_radius_m=float("inf"),     # FLAT sail
    )
    beam = GaussianBeam(
        I0_W_per_m2=INTENSITY_W_PER_M2, waist_m=BEAM_WAIST_M, wavelength_nm=1700.0,
    )
    mass = SailMass(mass_kg=SAIL_MASS_KG, radius_m=SAIL_RADIUS_M)
    int_cfg = IntegrationConfig(
        n_radial_center=20, n_radial_ring=8, n_azimuthal=64,
    )
    res = run_trajectory(
        initial_position_m=(INIT_X_M, INIT_Y_M, 0.0),
        initial_tilt_rad=(np.deg2rad(INIT_TILT_X_DEG), np.deg2rad(INIT_TILT_Y_DEG), 0.0),
        spin_freq_Hz=SPIN_FREQ_HZ,
        geometry=geo, beam=beam, mass=mass,
        center_lut=center_lut, ring_lut=ring_lut,
        integration_config=int_cfg,
        t_end_s=T_END_S, n_eval=N_EVAL,
        mod_amp=mod_amp,
        n_petals=n_petals,
        base_duty=BASE_RING["base_duty"],
    )
    lat = res.lateral_displacement_m()
    tilt = res.tilt_magnitude_deg()
    spin_omega = res.state[11]
    spin_init = 2.0 * np.pi * SPIN_FREQ_HZ
    spin_drift = float(np.max(np.abs((spin_omega - spin_init) / spin_init)))

    max_lat = float(lat.max())
    max_tilt = float(tilt.max())
    pass_lat = max_lat < MAX_POSITION_M
    pass_tilt = max_tilt < MAX_TILT_DEG
    pass_spin = spin_drift < MAX_SPIN_DRIFT_FRAC
    return {
        "PASS": pass_lat and pass_tilt and pass_spin,
        "max_lateral_m": max_lat,
        "max_tilt_deg": max_tilt,
        "spin_drift_frac": spin_drift,
        "z_m": float(res.z[-1]),
        "trajectory": res,
    }


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p5_flat_asym_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    center_lut, ring_lut = build_luts()

    print("\n[Phase 3-4] Flat-sail (n_petals × mod_amp) scan")
    print("=" * 72)
    rows = []
    for n_petals in N_PETALS_SET:
        for mod_amp in MOD_AMP_SET:
            if mod_amp == 0.0 and n_petals != 0:
                continue  # degenerate (no modulation)
            if mod_amp != 0.0 and n_petals == 0:
                continue  # degenerate
            print(f"  n_petals={n_petals:>2d}  mod_amp={mod_amp:.2f} ...", end=" ", flush=True)
            v = run_one(center_lut, ring_lut, n_petals, mod_amp)
            rows.append({
                "n_petals": n_petals,
                "mod_amp": mod_amp,
                "PASS": v["PASS"],
                "max_lateral_m": v["max_lateral_m"],
                "max_tilt_deg": v["max_tilt_deg"],
                "spin_drift_frac": v["spin_drift_frac"],
                "z_m": v["z_m"],
            })
            print(
                f"{'PASS' if v['PASS'] else 'FAIL'}  "
                f"|xy|={v['max_lateral_m']:6.3f}m  |tilt|={v['max_tilt_deg']:7.2f}°"
            )

    csv_path = out_dir / "phase3_flat_scan.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "phase3_flat_scan.md"
    with open(md_path, "w") as f:
        f.write("# Phase 3-4 — Flat-sail stability via azimuthal modulation\n\n")
        f.write("| n_petals | mod_amp | Verdict | max |xy| [m] | max |tilt| [°] |\n")
        f.write("|---:|---:|:---:|---:|---:|\n")
        for r in rows:
            v = "PASS" if r["PASS"] else "FAIL"
            f.write(
                f"| {r['n_petals']} | {r['mod_amp']:.2f} | {v} "
                f"| {r['max_lateral_m']:.3f} | {r['max_tilt_deg']:.2f} |\n"
            )
    print(f"\nCSV     → {csv_path}")
    print(f"Summary → {md_path}")


if __name__ == "__main__":
    main()
