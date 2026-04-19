"""Phase 1-3 + Phase 2 — Curvature sweep + Ablation study.

Runs trajectory simulation for combinations of:
    * curvature  ∈ {flat, +100m, +30m, +10m, +3m}
    * ring       ∈ {none, reference (P=2400), bo_best (P=1424)}

For each (curvature, ring) combination, integrates 5-s trajectory with
paper IC and reports PASS/FAIL verdict.

Outputs a PASS/FAIL matrix CSV + multi-panel paper figure.
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
# Designs and physics constants (matched to run_paper_trajectory.py)
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
RING_REFERENCE = dict(grating_period_nm=2400.0, duty_cycle=0.5, thickness_nm=280.0)
RING_BO_BEST = dict(grating_period_nm=1424.0, duty_cycle=0.47, thickness_nm=280.0)
RING_NONE = None  # 'none' = use BO best LUT but zero out ring force via geometry trick

SAIL_RADIUS_M = 1.5
RING_INNER_M = 1.45
SAIL_MASS_KG = 5.0e-3
INTENSITY_W_PER_M2 = 1.0e10
BEAM_WAIST_M = 4.0
SPIN_FREQ_HZ = 120.0
T_END_S = 5.0
N_EVAL = 200

# Paper PASS/FAIL thresholds
MAX_POSITION_M = 1.8
MAX_TILT_DEG = 10.0
MAX_SPIN_DRIFT_FRAC = 0.05

# Initial perturbation
INIT_X_M = 0.05
INIT_Y_M = 0.05
INIT_TILT_X_DEG = -2.0
INIT_TILT_Y_DEG = -2.0


CURVATURE_SET = [
    ("flat", float("inf")),
    ("Rc100m", 100.0),
    ("Rc30m",  30.0),
    ("Rc10m",  10.0),
    ("Rc3m",    3.0),
]
RING_SET = [
    ("none",       None),                  # ring effectively "off" → R_inner = R_outer
    ("reference",  RING_REFERENCE),
    ("bo_best",    RING_BO_BEST),
]


def build_luts():
    print("[A1] Building Force LUTs (one center + two rings) ...")
    theta_grid = np.linspace(-5.0, 5.0, 11)
    wls = np.linspace(1550.0, 1898.0, 3)
    phc = PhCReflector(**DESIGN_A)
    rcwa_cfg = RCWAConfig(nG=41, grid_nx=96, grid_ny=96)
    center = compute_center_lut(phc, theta_grid, wls, rcwa_cfg)

    fmm_cfg = FMMGratingConfig(nG=21, nx=128, ny=4)
    ring_ref = compute_ring_lut(
        RING_REFERENCE["grating_period_nm"], RING_REFERENCE["duty_cycle"],
        RING_REFERENCE["thickness_nm"], theta_grid, wls, fmm_cfg,
    )
    ring_bo = compute_ring_lut(
        RING_BO_BEST["grating_period_nm"], RING_BO_BEST["duty_cycle"],
        RING_BO_BEST["thickness_nm"], theta_grid, wls, fmm_cfg,
    )
    return center, ring_ref, ring_bo


def trajectory_one(center_lut, ring_lut, curvature_m: float, has_ring: bool):
    """Run one trajectory and return verdict dict."""
    # Trick to disable ring: collapse annulus to single point at outer radius
    R_inner = SAIL_RADIUS_M - 1e-6 if not has_ring else RING_INNER_M
    geo = SailGeometry(
        R_inner_m=R_inner, R_outer_m=SAIL_RADIUS_M,
        curvature_radius_m=curvature_m,
    )
    beam = GaussianBeam(
        I0_W_per_m2=INTENSITY_W_PER_M2, waist_m=BEAM_WAIST_M, wavelength_nm=1700.0,
    )
    mass = SailMass(mass_kg=SAIL_MASS_KG, radius_m=SAIL_RADIUS_M)
    int_cfg = IntegrationConfig(
        n_radial_center=20, n_radial_ring=8, n_azimuthal=48,
    )
    res = run_trajectory(
        initial_position_m=(INIT_X_M, INIT_Y_M, 0.0),
        initial_tilt_rad=(np.deg2rad(INIT_TILT_X_DEG), np.deg2rad(INIT_TILT_Y_DEG), 0.0),
        spin_freq_Hz=SPIN_FREQ_HZ,
        geometry=geo, beam=beam, mass=mass,
        center_lut=center_lut, ring_lut=ring_lut,
        integration_config=int_cfg,
        t_end_s=T_END_S, n_eval=N_EVAL,
    )
    lat = res.lateral_displacement_m()
    tilt = res.tilt_magnitude_deg()
    spin_omega = res.state[11]
    spin_init = 2.0 * np.pi * SPIN_FREQ_HZ
    spin_drift = (spin_omega - spin_init) / spin_init

    max_lat = float(lat.max())
    max_tilt = float(tilt.max())
    max_spin_drift = float(np.max(np.abs(spin_drift)))

    pass_lateral = max_lat < MAX_POSITION_M
    pass_tilt = max_tilt < MAX_TILT_DEG
    pass_spin = max_spin_drift < MAX_SPIN_DRIFT_FRAC

    return {
        "PASS": pass_lateral and pass_tilt and pass_spin,
        "max_lateral_m": max_lat,
        "max_tilt_deg": max_tilt,
        "max_spin_drift_frac": max_spin_drift,
        "z_distance_m": float(res.z[-1]),
        "trajectory": res,
    }


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_curvature_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    center_lut, ring_ref, ring_bo = build_luts()

    print("\n[Phase 1-3 + Phase 2] Curvature × Ring grid scan")
    print("=" * 72)
    results = []
    for curv_label, curv in CURVATURE_SET:
        for ring_label, ring_params in RING_SET:
            if ring_label == "none":
                ring_lut = ring_bo  # any ring; we disable via geometry
                has_ring = False
            elif ring_label == "reference":
                ring_lut = ring_ref
                has_ring = True
            else:
                ring_lut = ring_bo
                has_ring = True

            print(f"  curvature={curv_label:>8s}  ring={ring_label:>10s} ...", end=" ", flush=True)
            v = trajectory_one(center_lut, ring_lut, curv, has_ring)
            row = {
                "curvature_label": curv_label,
                "curvature_m": curv,
                "ring_label": ring_label,
                "PASS": v["PASS"],
                "max_lateral_m": v["max_lateral_m"],
                "max_tilt_deg": v["max_tilt_deg"],
                "max_spin_drift_frac": v["max_spin_drift_frac"],
                "z_distance_m": v["z_distance_m"],
            }
            results.append((row, v["trajectory"]))
            verdict_emoji = "PASS" if v["PASS"] else "FAIL"
            print(
                f"{verdict_emoji:>4s}   max|xy|={v['max_lateral_m']:6.3f}m   "
                f"max|tilt|={v['max_tilt_deg']:8.2f}°"
            )

    csv_path = out_dir / "curvature_sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0][0].keys()))
        w.writeheader()
        for row, _ in results:
            w.writerow(row)

    md_path = out_dir / "curvature_sweep_summary.md"
    with open(md_path, "w") as f:
        f.write("# Curvature × Ring trajectory PASS/FAIL matrix\n\n")
        f.write("| Curvature | Ring | Verdict | max |xy| [m] | max |tilt| [°] | z [km/5s] |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row, _ in results:
            v = "PASS" if row["PASS"] else "FAIL"
            f.write(
                f"| {row['curvature_label']} | {row['ring_label']} | {v} "
                f"| {row['max_lateral_m']:.3f} | {row['max_tilt_deg']:.2f} "
                f"| {row['z_distance_m']/1000:.0f} |\n"
            )

    print(f"\nCSV     → {csv_path}")
    print(f"Summary → {md_path}")
    _maybe_plot(out_dir, results)


def _maybe_plot(out_dir, results):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(results)
    fig, axes = plt.subplots(len(CURVATURE_SET), len(RING_SET),
                              figsize=(4 * len(RING_SET), 2.5 * len(CURVATURE_SET)),
                              sharex=True)
    for idx, (row, traj) in enumerate(results):
        i = idx // len(RING_SET)
        j = idx % len(RING_SET)
        ax = axes[i, j] if axes.ndim == 2 else axes[idx]
        ax.plot(traj.t_s, traj.tilt_magnitude_deg(), color="C1", label="|tilt|")
        ax.plot(traj.t_s, traj.lateral_displacement_m() * 1000, color="C0", label="|xy| [mm]")
        ax.axhline(MAX_TILT_DEG, color="r", ls="--", alpha=0.5)
        verdict = "PASS" if row["PASS"] else "FAIL"
        color = "green" if row["PASS"] else "red"
        ax.set_title(
            f"{row['curvature_label']} / {row['ring_label']} — {verdict}",
            color=color, fontsize=9,
        )
        if i == len(CURVATURE_SET) - 1:
            ax.set_xlabel("t [s]")
        if j == 0:
            ax.set_ylabel("|tilt| [°] / |xy| [mm]")
        ax.set_yscale("symlog")
        ax.grid(alpha=0.3, which="both")
        if idx == 0:
            ax.legend(fontsize=7)
    fig.suptitle(
        "Phase 1-3 + Phase 2 — Curvature × Ring trajectory matrix\n"
        "(Initial: x=y=50mm, θ_x=θ_y=−2°, spin=120 Hz; 5-s simulation)",
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "curvature_sweep_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
