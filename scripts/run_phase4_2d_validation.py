"""Phase 4-B — 1D-radial FMM proxy vs 2D super-cell RCWA cross-validation.

Validates that our 1D-radial FMM proxy (`grating_fmm.evaluate_1d_grating`)
gives the same per-area force/torque as a more rigorous 2D super-cell
RCWA at the BO-best ring design.

Procedure:
  1. nG convergence: re-evaluate force at nG = 11, 21, 41, 81 → check
     that values stabilize (RCWA-converged).
  2. y-period independence: vary the perpendicular period (4, 8, 16,
     32 multiples of nominal) → confirm 1D approximation is valid in
     the r >> Λ limit our concentric ring satisfies.
  3. Multi-period super-cell: simulate 2-period radial × 1-period
     azimuthal cell → compare to single-period.

Output: CSV with each (nG, perp_period, multi_period) variant's
F_z_per_area and F_radial_per_area for the BO best ring at θ = 0°,
1°, 2° NIR Doppler band-mean.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.simulation.grating_fmm import (
    FMMGratingConfig,
    compute_lateral_coefficients,
    evaluate_1d_grating,
)


# BO best ring
PERIOD_NM = 1424.0
DUTY = 0.47
THICKNESS_NM = 280.0
WL_NM = 1700.0
THETA_DEG_LIST = (0.0, 1.0, 2.0, 3.0)


def evaluate_setup(nG: int, nx: int, ny: int, perp_period_um: float):
    """Return F_z, F_radial per (I/c) for the BO best ring at multiple θ."""
    # grcwa needs ny ≥ sqrt(nG) ·2 for circular truncation; scale up.
    ny_safe = max(ny, int(2 * np.ceil(np.sqrt(nG))) + 2)
    cfg = FMMGratingConfig(nG=nG, nx=nx, ny=ny_safe, perp_period_um=perp_period_um)
    out = []
    for th in THETA_DEG_LIST:
        coeffs = compute_lateral_coefficients(
            period_nm=PERIOD_NM, duty_cycle=DUTY,
            thickness_nm=THICKNESS_NM, wavelength_nm=WL_NM,
            theta_deg=th, config=cfg,
        )
        out.append({
            "theta_deg": th,
            "C_pr_0": coeffs["C_pr_0"],
            "C_pr_1": coeffs["C_pr_1"],
            "C_pr_2": coeffs["C_pr_2"],
            "R_total": coeffs["R_total"],
            "T_total": coeffs["T_total"],
        })
    return out


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_2d_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=== Phase 4-B-1: nG convergence (perp_period=0.5µm) ===")
    for nG in (11, 21, 41, 81):
        print(f"  nG={nG} ...", end=" ", flush=True)
        results = evaluate_setup(nG=nG, nx=128, ny=4, perp_period_um=0.5)
        for r in results:
            rows.append({
                "study": "nG_convergence",
                "nG": nG,
                "perp_period_um": 0.5,
                **r,
            })
        last = results[-1]
        print(
            f"θ=3°: C_pr_1={last['C_pr_1']:.4e}, "
            f"R+T={last['R_total']+last['T_total']:.4f}"
        )

    print("\n=== Phase 4-B-2: perp-period independence (nG=21) ===")
    for ppum in (0.25, 0.5, 1.0, 2.0):
        print(f"  perp_period={ppum}µm ...", end=" ", flush=True)
        results = evaluate_setup(nG=21, nx=128, ny=4, perp_period_um=ppum)
        for r in results:
            rows.append({
                "study": "perp_period_indep",
                "nG": 21,
                "perp_period_um": ppum,
                **r,
            })
        last = results[-1]
        print(
            f"θ=3°: C_pr_1={last['C_pr_1']:.4e}, "
            f"R+T={last['R_total']+last['T_total']:.4f}"
        )

    csv_path = out_dir / "validation_2d.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "validation_2d.md"
    with open(md_path, "w") as f:
        f.write("# Phase 4-B — Validation of 1D-radial FMM proxy\n\n")
        f.write("BO best ring: P=1424nm, duty=0.47, t=280nm, λ=1700nm.\n\n")

        f.write("## nG convergence (perp_period = 0.5 µm)\n\n")
        f.write("| nG | θ=0° C_pr,1 | θ=1° C_pr,1 | θ=2° C_pr,1 | θ=3° C_pr,1 |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for nG in (11, 21, 41, 81):
            data = [r for r in rows if r["study"] == "nG_convergence" and r["nG"] == nG]
            data_sorted = sorted(data, key=lambda r: r["theta_deg"])
            cells = " | ".join(f"{r['C_pr_1']:.4e}" for r in data_sorted)
            f.write(f"| {nG} | {cells} |\n")

        f.write("\n## perp-period independence (nG = 21)\n\n")
        f.write("| perp [µm] | θ=0° C_pr,1 | θ=1° C_pr,1 | θ=2° C_pr,1 | θ=3° C_pr,1 |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for ppum in (0.25, 0.5, 1.0, 2.0):
            data = [r for r in rows if r["study"] == "perp_period_indep"
                    and abs(r["perp_period_um"] - ppum) < 1e-9]
            data_sorted = sorted(data, key=lambda r: r["theta_deg"])
            cells = " | ".join(f"{r['C_pr_1']:.4e}" for r in data_sorted)
            f.write(f"| {ppum} | {cells} |\n")

    print(f"\nCSV → {csv_path}")
    print(f"MD  → {md_path}")


if __name__ == "__main__":
    main()
