"""Phase 3-5 — BO with trajectory PASS objective (max_tilt minimization).

Sobol sampling + GP refinement over the full 7-param ring space::

    grating_period_nm ∈ [1100, 2800]     (period)
    duty_cycle        ∈ [0.2, 0.8]
    curvature         ∈ [-0.2, 0.2]
    ring_width_um     ∈ [2, 50]
    mod_amp           ∈ [0, 0.5]
    n_petals          ∈ {0, 2, 3, 4, 6}     (integer-rounded)
    curvature_radius_m ∈ [10, ∞]             (global sail R_c)

Objective: composite score
    obj = max_tilt_deg / 10.0  +  max_lateral_m / 1.8
    (lower = better; obj < 2.0 == PASS by both criteria)

Each trajectory ≈ 1 min. We do Sobol-only (no GP) for wall-time
predictability: 40 random samples + 10 top-refined.

Output: results/<ts>_p5_phase3_bo/ with CSV, per-trial NPZ, and the
best trajectory plot.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.dynamics import (
    CenterPhCLUT,
    GaussianBeam,
    IntegrationConfig,
    RingLUT2D,
    SailGeometry,
    SailMass,
    compute_center_lut,
    compute_ring_lut_2d,
    run_trajectory,
)
from lightsail.simulation import RCWAConfig
from lightsail.simulation.grating_fmm import FMMGratingConfig

try:
    from scipy.stats import qmc
    HAS_QMC = True
except ImportError:
    HAS_QMC = False


# ---------------------------------------------------------------------------
# Fixed designs
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

# BO search space (min, max). n_petals is integer; curvature_radius is
# expressed as 1/R_c so that ∞ is the 0 endpoint.
BOUNDS = [
    (1200.0, 2800.0),    # 0 grating_period_nm
    (0.2, 0.8),          # 1 duty_cycle
    (2.0, 50.0),         # 2 ring_width_um
    (0.0, 0.5),          # 3 mod_amp
    (0.0, 6.0),          # 4 n_petals_continuous (rounded to 0/2/3/4/6)
    (0.0, 0.1),          # 5 inverse_R_c [1/m]; 0=flat, 0.1=R_c=10m
]
N_INIT = 30
N_REFINE = 10
SEED = 42

SAIL_RADIUS_M = 1.5
RING_INNER_M = 1.45
SAIL_MASS_KG = 5.0e-3
INTENSITY_W_PER_M2 = 1.0e10
BEAM_WAIST_M = 4.0
SPIN_FREQ_HZ = 120.0
T_END_S = 3.0
N_EVAL = 100
RTOL = 1e-4    # looser = faster for divergent cases

MAX_POSITION_M = 1.8
MAX_TILT_DEG = 10.0
MAX_SPIN_DRIFT_FRAC = 0.05
INIT_X_M = 0.05
INIT_Y_M = 0.05
INIT_TILT_X_DEG = -2.0
INIT_TILT_Y_DEG = -2.0


def round_n_petals(n_cont: float) -> int:
    """Round continuous n_petals value to {0, 2, 3, 4, 6}."""
    allowed = np.array([0, 2, 3, 4, 6])
    n = int(allowed[np.argmin(np.abs(allowed - n_cont))])
    return n


# ---------------------------------------------------------------------------
# LUT cache — built once per (period, duty_grid).
# For BO we precompute a 3D LUT and interpolate.  Since period varies, we
# would need a 4D LUT (period × duty × θ × λ) which is too large.
# Practical solution: rebuild ring LUT per BO trial (≈ 30 s/trial).
# ---------------------------------------------------------------------------


def build_center_lut_once():
    theta_grid = np.linspace(-5.0, 5.0, 9)
    wls = np.array([1700.0])
    phc = PhCReflector(**DESIGN_A)
    return compute_center_lut(
        phc, theta_grid, wls, RCWAConfig(nG=21, grid_nx=64, grid_ny=64),
    )


def build_ring_lut_for(period_nm: float, duty_grid: np.ndarray) -> RingLUT2D:
    theta_grid = np.linspace(-5.0, 5.0, 9)
    wls = np.array([1700.0])
    return compute_ring_lut_2d(
        grating_period_nm=period_nm,
        duty_grid=duty_grid,
        thickness_nm=280.0,
        theta_grid_deg=theta_grid,
        wavelengths_nm=wls,
        fmm_config=FMMGratingConfig(nG=15, nx=64, ny=4),
    )


def evaluate(params: np.ndarray, center_lut: CenterPhCLUT) -> dict:
    """Evaluate one BO point and return composite score."""
    period, duty, ring_w_um, mod_amp, n_cont, inv_Rc = params
    n_petals = round_n_petals(n_cont)
    # Curvature radius: inv_Rc = 0 means flat; else R_c = 1/inv_Rc
    R_c = float("inf") if inv_Rc < 1e-6 else 1.0 / float(inv_Rc)

    # Build ring LUT for this period (rebuild per BO iteration)
    duty_grid = np.linspace(
        max(0.1, duty - 0.3), min(0.9, duty + 0.3), 5,
    )
    ring_lut = build_ring_lut_for(float(period), duty_grid)

    geo = SailGeometry(
        R_inner_m=RING_INNER_M, R_outer_m=SAIL_RADIUS_M,
        curvature_radius_m=R_c,
    )
    beam = GaussianBeam(
        I0_W_per_m2=INTENSITY_W_PER_M2, waist_m=BEAM_WAIST_M, wavelength_nm=1700.0,
    )
    mass = SailMass(mass_kg=SAIL_MASS_KG, radius_m=SAIL_RADIUS_M)
    int_cfg = IntegrationConfig(
        n_radial_center=12, n_radial_ring=4, n_azimuthal=32,
    )
    try:
        res = run_trajectory(
            initial_position_m=(INIT_X_M, INIT_Y_M, 0.0),
            initial_tilt_rad=(np.deg2rad(INIT_TILT_X_DEG), np.deg2rad(INIT_TILT_Y_DEG), 0.0),
            spin_freq_Hz=SPIN_FREQ_HZ,
            geometry=geo, beam=beam, mass=mass,
            center_lut=center_lut, ring_lut=ring_lut,
            integration_config=int_cfg,
            t_end_s=T_END_S, n_eval=N_EVAL,
            rtol=RTOL,
            mod_amp=float(mod_amp), n_petals=n_petals, base_duty=float(duty),
        )
        max_lat = float(res.lateral_displacement_m().max())
        max_tilt = float(res.tilt_magnitude_deg().max())
        spin_drift = float(np.max(np.abs((res.state[11] - 2 * np.pi * SPIN_FREQ_HZ) / (2 * np.pi * SPIN_FREQ_HZ))))
    except Exception as err:
        print(f"    FAIL eval: {err}")
        return {
            "params": list(params), "n_petals": n_petals, "R_c_m": R_c,
            "PASS": False, "max_lateral_m": 1e6, "max_tilt_deg": 1e6,
            "spin_drift_frac": 1.0,
            "obj": 1e6, "error": str(err),
        }

    pass_lat = max_lat < MAX_POSITION_M
    pass_tilt = max_tilt < MAX_TILT_DEG
    pass_spin = spin_drift < MAX_SPIN_DRIFT_FRAC
    # Composite objective (lower = better)
    obj = (
        min(max_tilt / MAX_TILT_DEG, 100.0)
        + min(max_lat / MAX_POSITION_M, 100.0)
    )

    return {
        "params": list(params),
        "period_nm": float(period), "duty": float(duty),
        "ring_w_um": float(ring_w_um), "mod_amp": float(mod_amp),
        "n_petals": n_petals, "R_c_m": R_c,
        "PASS": pass_lat and pass_tilt and pass_spin,
        "max_lateral_m": max_lat, "max_tilt_deg": max_tilt,
        "spin_drift_frac": spin_drift,
        "obj": obj,
    }


def sobol_samples(n: int, dim: int, seed: int) -> np.ndarray:
    if HAS_QMC:
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        return sampler.random_base2(m=int(np.ceil(np.log2(n))))[:n]
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p5_phase3_bo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[A1] Building center LUT (once) ...")
    center_lut = build_center_lut_once()

    dim = len(BOUNDS)
    lb = np.array([b[0] for b in BOUNDS])
    ub = np.array([b[1] for b in BOUNDS])

    print(f"\n[Phase 3-5] BO sweep ({N_INIT} init + {N_REFINE} refine, total {N_INIT + N_REFINE})")
    print("=" * 72)

    # Phase 1: Sobol init
    unit = sobol_samples(N_INIT, dim, SEED)
    all_results = []
    for i, u in enumerate(unit):
        params = lb + u * (ub - lb)
        print(
            f"  [{i+1:2d}/{N_INIT + N_REFINE}] "
            f"P={params[0]:.0f} duty={params[1]:.2f} W={params[2]:.1f}um "
            f"mod={params[3]:.2f} n={round_n_petals(params[4])} "
            f"1/Rc={params[5]:.4f} ...",
            end=" ", flush=True,
        )
        r = evaluate(params, center_lut)
        all_results.append(r)
        verdict = "PASS" if r["PASS"] else "FAIL"
        print(
            f"{verdict}  tilt={r['max_tilt_deg']:6.2f}° "
            f"xy={r['max_lateral_m']:.3f}m  obj={r['obj']:.3f}"
        )

    # Phase 2: refine by perturbing top-3 init points
    all_results.sort(key=lambda r: r["obj"])
    print(f"\n--- refining top 3 ({N_REFINE} perturbations) ---")
    rng = np.random.default_rng(SEED + 100)
    top3 = all_results[:3]
    for i in range(N_REFINE):
        parent = top3[i % 3]
        parent_params = np.array(parent["params"])
        noise = rng.normal(0, 0.08, dim)   # 8% relative noise per dim
        new_u = np.clip(
            (parent_params - lb) / (ub - lb) + noise, 0.0, 1.0
        )
        params = lb + new_u * (ub - lb)
        print(
            f"  [{N_INIT + i + 1:2d}/{N_INIT + N_REFINE}] "
            f"P={params[0]:.0f} duty={params[1]:.2f} W={params[2]:.1f}um "
            f"mod={params[3]:.2f} n={round_n_petals(params[4])} "
            f"1/Rc={params[5]:.4f} ...",
            end=" ", flush=True,
        )
        r = evaluate(params, center_lut)
        all_results.append(r)
        verdict = "PASS" if r["PASS"] else "FAIL"
        print(
            f"{verdict}  tilt={r['max_tilt_deg']:6.2f}° "
            f"xy={r['max_lateral_m']:.3f}m  obj={r['obj']:.3f}"
        )

    # Final sort
    all_results.sort(key=lambda r: r["obj"])
    best = all_results[0]

    # Write CSV
    csv_path = out_dir / "phase3_bo_all.csv"
    with open(csv_path, "w", newline="") as f:
        keys = [k for k in all_results[0].keys() if k != "params"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    # Best design YAML
    with open(out_dir / "best_design.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    # Markdown summary
    md_path = out_dir / "phase3_bo_summary.md"
    with open(md_path, "w") as f:
        f.write("# Phase 3-5 — BO with trajectory PASS objective\n\n")
        f.write(f"**Best design** (obj = {best['obj']:.3f}):\n\n")
        f.write(f"- Period: {best['period_nm']:.0f} nm\n")
        f.write(f"- Duty: {best['duty']:.3f}\n")
        f.write(f"- Ring width: {best['ring_w_um']:.2f} µm\n")
        f.write(f"- mod_amp: {best['mod_amp']:.3f}\n")
        f.write(f"- n_petals: {best['n_petals']}\n")
        f.write(f"- R_c: {best['R_c_m']:.2f} m\n")
        f.write(f"- **max |tilt|: {best['max_tilt_deg']:.3f}°**\n")
        f.write(f"- **max |xy|: {best['max_lateral_m']:.3f} m**\n")
        f.write(f"- PASS: **{best['PASS']}**\n\n")

        f.write("## Top 10 designs\n\n")
        f.write("| rank | P [nm] | duty | W [µm] | mod | n | R_c [m] | tilt [°] | xy [m] | PASS |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for i, r in enumerate(all_results[:10]):
            Rc_str = "flat" if not np.isfinite(r["R_c_m"]) else f"{r['R_c_m']:.1f}"
            f.write(
                f"| {i+1} | {r['period_nm']:.0f} | {r['duty']:.2f} "
                f"| {r['ring_w_um']:.1f} | {r['mod_amp']:.2f} | {r['n_petals']} "
                f"| {Rc_str} | {r['max_tilt_deg']:.2f} "
                f"| {r['max_lateral_m']:.3f} | "
                f"{'PASS' if r['PASS'] else 'FAIL'} |\n"
            )

    print(f"\nBest obj = {best['obj']:.3f} ({'PASS' if best['PASS'] else 'FAIL'})")
    print(f"  params: P={best['period_nm']:.0f} duty={best['duty']:.2f} "
          f"mod={best['mod_amp']:.2f} n={best['n_petals']} R_c={best['R_c_m']:.1f}m")
    print(f"  max |tilt|={best['max_tilt_deg']:.2f}°  max |xy|={best['max_lateral_m']:.3f}m")
    print(f"\nCSV     → {csv_path}")
    print(f"Summary → {md_path}")


if __name__ == "__main__":
    main()
