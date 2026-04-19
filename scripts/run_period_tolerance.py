"""P5 — Ring period sensitivity scan.

Mirrors the Stage 1 fab_tolerance sweep (CLAUDE.md, 2026-04-18) but
applied to the outer metagrating: ±10% perturbation on ring period,
duty cycle, curvature, asymmetry, and ring_width. Reports the
fractional change in stiffness ``k_θθ`` and damping coefficient
``α_damp`` to identify the most sensitive ring parameter.

The Stage 1 result was that period is by far the most sensitive
parameter for the central PhC. This script asks the same question
for the ring zone — informs photolithography-tolerance budgeting.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.metagrating import MetaGrating
from lightsail.simulation.damping import compute_damping_force
from lightsail.simulation.grating_fmm import FMMGratingConfig
from lightsail.simulation.stiffness import compute_stiffness_matrix


BASELINE = MetaGrating(
    inner_radius_nm=5_000_000.0,
    thickness_nm=240.0,
    grating_period_nm=2400.0,
    duty_cycle=0.5,
    curvature=0.05,
    asymmetry=0.10,
    ring_width_um=2000.0,
)
PERTURBATIONS = (-0.10, -0.05, +0.05, +0.10)
PARAMS_TO_PERTURB = (
    "grating_period_nm",
    "duty_cycle",
    "curvature",
    "asymmetry",
    "ring_width_um",
)
FMM_CFG = FMMGratingConfig(nG=15, nx=96, ny=4)


def perturb(mg: MetaGrating, attr: str, frac: float) -> MetaGrating:
    new = MetaGrating(**{
        k: getattr(mg, k) for k in (
            "inner_radius_nm", "thickness_nm",
            "grating_period_nm", "duty_cycle", "curvature",
            "asymmetry", "ring_width_um",
        )
    })
    base_val = getattr(mg, attr)
    new_val = base_val * (1.0 + frac) if base_val != 0 else frac * 0.1
    setattr(new, attr, new_val)
    return new


def evaluate(mg: MetaGrating) -> tuple[float, float]:
    stiff = compute_stiffness_matrix(
        mg, nir_n_points=4, n_radial_bins=2, config=FMM_CFG,
    )
    damp = compute_damping_force(
        mg, beta=0.10, v_y_per_c=1.0e-4,
        n_radial_bins=2, config=FMM_CFG,
    )
    return (
        float(stiff.k_thetatheta_Nm_per_rad),
        float(damp["alpha_damp_Pa_per_mps"]),
    )


def main() -> None:
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p5_ring_tolerance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing baseline ...")
    base_kθθ, base_α = evaluate(BASELINE)
    print(f"  baseline: k_θθ = {base_kθθ:+.3e}, α_damp = {base_α:+.3e}")

    rows = []
    for attr in PARAMS_TO_PERTURB:
        for frac in PERTURBATIONS:
            try:
                pert = perturb(BASELINE, attr, frac)
                kθθ, α = evaluate(pert)
                d_kθθ_pct = (kθθ - base_kθθ) / abs(base_kθθ) * 100.0 if base_kθθ != 0 else float("nan")
                d_α_pct = (α - base_α) / abs(base_α) * 100.0 if base_α != 0 else float("nan")
                row = {
                    "param": attr,
                    "frac": frac,
                    "perturbed_value": getattr(pert, attr),
                    "k_theta_theta_Nm_per_rad": kθθ,
                    "alpha_damp_Pa_per_mps": α,
                    "d_k_pct": d_kθθ_pct,
                    "d_alpha_pct": d_α_pct,
                }
            except Exception as err:
                row = {"param": attr, "frac": frac, "error": str(err)}
            rows.append(row)
            print(
                f"  {attr:>20s} ×(1{frac:+.2f}) → "
                f"Δk = {row.get('d_k_pct', float('nan')):+7.1f}%  "
                f"Δα = {row.get('d_alpha_pct', float('nan')):+7.1f}%"
            )

    csv_path = out_dir / "ring_tolerance.csv"
    with open(csv_path, "w", newline="") as f:
        keys = ["param", "frac", "perturbed_value",
                "k_theta_theta_Nm_per_rad", "alpha_damp_Pa_per_mps",
                "d_k_pct", "d_alpha_pct", "error"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
