"""Paper-grade trajectory simulation + PASS/FAIL verdict (A5).

Replicates the Gieseler 2024 (Nat Commun 15:4203) stability test
adapted to our 2-zone (center PhC + outer ring) lightsail.

Procedure
---------
1. Build force LUTs for Design A center + a chosen outer ring.
2. Apply paper's initial perturbation: x₀=y₀=50 mm, θ_x₀=θ_y₀=−2°.
3. Spin sail at 120 Hz around z.
4. Integrate 6-DOF EoM for 5 seconds in a Gaussian laser beam.
5. PASS verdict if::
       max |position(t)| < 1.8 m            (stays in beam)
       max |tilt(t)|     < 10°              (no flip-over)
       max |Δω_z|        < 5%               (spin preserved)

Outputs trajectory plot + per-axis time series + verdict text.
"""
from __future__ import annotations

import argparse
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
# Design A center + reference ring
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

REFERENCE_RING = dict(grating_period_nm=2400.0, duty_cycle=0.5, thickness_nm=280.0)

# BO best (seed 42, after proxy fix, 2026-04-19)
BO_BEST_RING = dict(grating_period_nm=1424.0, duty_cycle=0.47, thickness_nm=280.0)

# Sail / beam parameters (paper-equivalent for our scale)
SAIL_RADIUS_M = 1.5
RING_INNER_M = 1.45               # ring annulus 1.45 → 1.5 m (5 cm wide)
SAIL_MASS_KG = 5.0e-3              # ~Design A mass for 10 m²
INTENSITY_W_PER_M2 = 1.0e10        # Starshot 10 GW/m²
BEAM_WAIST_M = 4.0                 # beam waist
SPIN_FREQ_HZ = 120.0               # paper's choice
T_END_S = 5.0
N_EVAL = 500
DEFAULT_CURVATURE_RADIUS_M = float("inf")    # ∞ = flat sail

# Paper PASS/FAIL thresholds (Gieseler 2024 used 1.8 m for their setup;
# we keep the same to be apples-to-apples).
MAX_POSITION_M = 1.8
MAX_TILT_DEG = 10.0
MAX_SPIN_DRIFT_FRAC = 0.05

# Initial perturbation (paper)
INIT_X_M = 0.05
INIT_Y_M = 0.05
INIT_TILT_X_DEG = -2.0
INIT_TILT_Y_DEG = -2.0


def build_luts(args):
    print(f"[A1] Building Force LUTs (n_theta={args.n_theta}, n_wl={args.n_wl}, "
          f"center nG={args.center_nG}, ring nG={args.ring_nG})...")
    theta_grid = np.linspace(-args.theta_max_deg, args.theta_max_deg, args.n_theta)
    wls = np.linspace(1550.0, 1898.0, args.n_wl)
    phc = PhCReflector(**DESIGN_A)
    rcwa_cfg = RCWAConfig(nG=args.center_nG, grid_nx=args.center_nx, grid_ny=args.center_nx)
    center = compute_center_lut(phc, theta_grid, wls, rcwa_cfg)

    fmm_cfg = FMMGratingConfig(nG=args.ring_nG, nx=args.ring_nx, ny=4)
    ring_params = BO_BEST_RING if args.ring == "bo_best" else REFERENCE_RING
    print(f"  Ring params: {ring_params}")
    ring = compute_ring_lut(
        ring_params["grating_period_nm"],
        ring_params["duty_cycle"],
        ring_params["thickness_nm"],
        theta_grid, wls, fmm_cfg,
    )
    return center, ring


def simulate(center_lut, ring_lut, args):
    geo = SailGeometry(
        R_inner_m=RING_INNER_M,
        R_outer_m=SAIL_RADIUS_M,
        curvature_radius_m=args.curvature_m,
    )
    beam = GaussianBeam(
        I0_W_per_m2=INTENSITY_W_PER_M2,
        waist_m=BEAM_WAIST_M,
        wavelength_nm=1700.0,
    )
    mass = SailMass(mass_kg=SAIL_MASS_KG, radius_m=SAIL_RADIUS_M)
    int_cfg = IntegrationConfig(
        n_radial_center=args.n_radial_center,
        n_radial_ring=args.n_radial_ring,
        n_azimuthal=args.n_azimuthal,
    )

    print("[A3] Integrating 6-DOF EoM (5 s, ~few minutes)...")
    result = run_trajectory(
        initial_position_m=(INIT_X_M, INIT_Y_M, 0.0),
        initial_tilt_rad=(np.deg2rad(INIT_TILT_X_DEG), np.deg2rad(INIT_TILT_Y_DEG), 0.0),
        spin_freq_Hz=SPIN_FREQ_HZ,
        geometry=geo, beam=beam, mass=mass,
        center_lut=center_lut, ring_lut=ring_lut,
        integration_config=int_cfg,
        t_end_s=T_END_S, n_eval=N_EVAL,
    )
    return result


def verdict(result) -> dict:
    lat = result.lateral_displacement_m()
    tilt = result.tilt_magnitude_deg()
    spin_omega = result.state[11]
    spin_init = 2.0 * np.pi * SPIN_FREQ_HZ
    spin_drift = (spin_omega - spin_init) / spin_init

    max_lat = float(lat.max())
    max_tilt = float(tilt.max())
    max_spin_drift = float(np.max(np.abs(spin_drift)))

    pass_lateral = max_lat < MAX_POSITION_M
    pass_tilt = max_tilt < MAX_TILT_DEG
    pass_spin = max_spin_drift < MAX_SPIN_DRIFT_FRAC

    overall = pass_lateral and pass_tilt and pass_spin

    return {
        "PASS": overall,
        "max_lateral_displacement_m": max_lat,
        "max_tilt_deg": max_tilt,
        "max_spin_drift_fraction": max_spin_drift,
        "pass_lateral": pass_lateral,
        "pass_tilt": pass_tilt,
        "pass_spin": pass_spin,
        "z_distance_m": float(result.z[-1]),
    }


def save_outputs(result, verdict_dict, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "trajectory.npz",
        t=result.t_s, state=result.state,
        verdict=str(verdict_dict),
    )
    with open(out_dir / "verdict.txt", "w") as f:
        f.write("# Paper-grade trajectory PASS/FAIL\n\n")
        for k, v in verdict_dict.items():
            f.write(f"{k}: {v}\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(result.t_s, result.x * 1000, label="x")
        axes[0, 0].plot(result.t_s, result.y * 1000, label="y")
        axes[0, 0].axhline(MAX_POSITION_M * 1000, color="r", ls="--", alpha=0.5)
        axes[0, 0].axhline(-MAX_POSITION_M * 1000, color="r", ls="--", alpha=0.5)
        axes[0, 0].set_xlabel("t [s]")
        axes[0, 0].set_ylabel("position [mm]")
        axes[0, 0].set_title("Lateral position")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(result.t_s, result.theta_x_deg, label="θ_x")
        axes[0, 1].plot(result.t_s, result.theta_y_deg, label="θ_y")
        axes[0, 1].axhline(MAX_TILT_DEG, color="r", ls="--", alpha=0.5)
        axes[0, 1].axhline(-MAX_TILT_DEG, color="r", ls="--", alpha=0.5)
        axes[0, 1].set_xlabel("t [s]")
        axes[0, 1].set_ylabel("tilt [deg]")
        axes[0, 1].set_title("Tilt angles")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(result.t_s, result.z, color="C2")
        axes[1, 0].set_xlabel("t [s]")
        axes[1, 0].set_ylabel("z [m]")
        axes[1, 0].set_title("Forward (z) distance")
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(result.x * 1000, result.y * 1000)
        axes[1, 1].plot([INIT_X_M * 1000], [INIT_Y_M * 1000], "go", label="start")
        axes[1, 1].plot([result.x[-1] * 1000], [result.y[-1] * 1000], "rs", label="end")
        circle = plt.Circle((0, 0), MAX_POSITION_M * 1000, fill=False, color="r", ls="--")
        axes[1, 1].add_patch(circle)
        axes[1, 1].set_xlabel("x [mm]")
        axes[1, 1].set_ylabel("y [mm]")
        axes[1, 1].set_title("Lateral trajectory in lab frame")
        axes[1, 1].set_aspect("equal")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        verdict_str = "PASS" if verdict_dict["PASS"] else "FAIL"
        fig.suptitle(
            f"Paper-style trajectory verdict: {verdict_str}  "
            f"(max |xy|={verdict_dict['max_lateral_displacement_m']:.3f}m, "
            f"max |tilt|={verdict_dict['max_tilt_deg']:.2f}°)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory.png", dpi=140)
        plt.close(fig)
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-theta", type=int, default=11)
    p.add_argument("--n-wl", type=int, default=3)
    p.add_argument("--theta-max-deg", type=float, default=5.0)
    p.add_argument("--center-nG", type=int, default=41)
    p.add_argument("--center-nx", type=int, default=96)
    p.add_argument("--ring-nG", type=int, default=21)
    p.add_argument("--ring-nx", type=int, default=128)
    p.add_argument("--n-radial-center", type=int, default=20)
    p.add_argument("--n-radial-ring", type=int, default=8)
    p.add_argument("--n-azimuthal", type=int, default=48)
    p.add_argument("--ring", choices=["reference", "bo_best"], default="bo_best")
    p.add_argument("--curvature-m", type=float, default=DEFAULT_CURVATURE_RADIUS_M,
                   help="Sail curvature radius in m (positive = convex, ∞ = flat)")
    args = p.parse_args()

    Rc_tag = "flat" if not np.isfinite(args.curvature_m) else f"Rc{args.curvature_m:.0f}m"
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_traj_{args.ring}_{Rc_tag}"

    center_lut, ring_lut = build_luts(args)
    result = simulate(center_lut, ring_lut, args)
    v = verdict(result)
    save_outputs(result, v, out_dir)

    print()
    print("=" * 60)
    print(f"Paper-style verdict: {'PASS' if v['PASS'] else 'FAIL'}")
    print("=" * 60)
    for k, val in v.items():
        print(f"  {k:35s} = {val}")
    print(f"\nOutputs → {out_dir}/")


if __name__ == "__main__":
    main()
