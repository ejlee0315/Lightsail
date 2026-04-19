"""P3.2 — Verify outer ring enhances ∂C_pr,2/∂θ (docx Eq. 4.8 third term).

Compares C_pr,2 angular sensitivity at three configurations:

    (a) bare PhC propulsion region   — period Λ matched to bare PhC
    (b) outer metagrating sweep      — sweep grating period in [1500, 2800] nm
    (c) Doppler-band sweep           — repeats at multiple β

The key claim of docx §2.3 is that an outer metagrating amplifies the
``∂C_pr,2/∂θ`` term — i.e. larger angular sensitivity of the
diffraction-lobe spread. This script reports the ratio
``(∂C_pr,2/∂θ)_with-ring / (∂C_pr,2/∂θ)_bare-PhC`` per (β, period).
A ratio significantly > 1 supports the metasurface-enhancement claim.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.simulation.damping import (
    compute_damping_force,
    sail_frame_wavelength_nm,
)
from lightsail.simulation.grating_fmm import (
    FMMGratingConfig,
    aggregate_metagrating_response,
)
from lightsail.geometry.metagrating import MetaGrating


# Bare PhC "grating" stand-in: period = lattice period of Design A,
# duty = material fill fraction, thickness = Design A thickness.
BARE_PHC_PERIOD_NM = 1580.0
BARE_PHC_DUTY = 0.5            # nominal triangular hole-bar
BARE_PHC_THICKNESS_NM = 280.0

# Sweep ring designs (period in nm). Asymmetry/curvature held fixed.
RING_PERIODS_NM = (1500.0, 1800.0, 2000.0, 2400.0, 2800.0)
RING_DUTY = 0.5
RING_CURV = 0.05
RING_ASYM = 0.10
RING_THICKNESS_NM = 240.0

# β values to scan (sail-frame Doppler).
BETA_VALUES = (0.0, 0.05, 0.10, 0.15, 0.20)

LAB_WL_NM = 1550.0


def main() -> None:
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p3_2_damping_verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FMMGratingConfig(nG=15, nx=96, ny=4)

    rows = []
    for beta in BETA_VALUES:
        wl_prime = sail_frame_wavelength_nm(LAB_WL_NM, beta)

        # (a) bare PhC stand-in
        bare = aggregate_metagrating_response(
            grating_period_nm=BARE_PHC_PERIOD_NM,
            duty_cycle=BARE_PHC_DUTY,
            thickness_nm=BARE_PHC_THICKNESS_NM,
            wavelengths_nm=np.array([wl_prime]),
            curvature=0.0,
            n_radial_bins=1,
            theta_center_deg=0.0,
            dtheta_deg=1.0,
            config=cfg,
        )
        bare_dC2 = float(bare["mean_dC_pr_2_dtheta"])

        # (b) ring sweep
        for period in RING_PERIODS_NM:
            ring = aggregate_metagrating_response(
                grating_period_nm=period,
                duty_cycle=RING_DUTY,
                thickness_nm=RING_THICKNESS_NM,
                wavelengths_nm=np.array([wl_prime]),
                curvature=RING_CURV,
                n_radial_bins=2,
                theta_center_deg=0.0,
                dtheta_deg=1.0,
                config=cfg,
            )
            ring_dC2 = float(ring["mean_dC_pr_2_dtheta"])
            ratio = (
                ring_dC2 / bare_dC2 if abs(bare_dC2) > 1e-30 else float("inf")
            )
            row = {
                "beta": beta,
                "wl_prime_nm": wl_prime,
                "bare_dC_pr_2_dtheta": bare_dC2,
                "ring_period_nm": period,
                "ring_dC_pr_2_dtheta": ring_dC2,
                "enhancement_ratio": ratio,
            }
            rows.append(row)
            print(
                f"  β={beta:.2f}  λ′={wl_prime:.0f}nm  "
                f"P_ring={period:.0f}  bare={bare_dC2:+.4e}  "
                f"ring={ring_dC2:+.4e}  ratio={ratio:+.2f}"
            )

    # Cross-check via end-to-end damping coefficient at a representative
    # ring design (period = 2000 nm, mid-Doppler β=0.10) using the
    # MetaGrating geometry path (same C_pr,2/θ derivative under the hood).
    mg = MetaGrating(
        inner_radius_nm=5_000_000.0,
        thickness_nm=RING_THICKNESS_NM,
        grating_period_nm=2000.0,
        duty_cycle=RING_DUTY,
        curvature=RING_CURV,
        asymmetry=RING_ASYM,
        ring_width_um=2000.0,
    )
    print()
    print("=== End-to-end damping coefficient (MetaGrating, β=0.10) ===")
    out = compute_damping_force(
        mg, beta=0.10, v_y_per_c=1.0e-4,
        lab_wavelength_nm=LAB_WL_NM,
        n_radial_bins=2, config=cfg,
    )
    print(f"  α_damp        = {out['alpha_damp_Pa_per_mps']:+.4e} Pa/(m/s)")
    print(f"  α_aberration  = {out['alpha_aberration_Pa_per_mps']:+.4e} Pa/(m/s)")
    print(f"  α_metasurface = {out['alpha_metasurface_Pa_per_mps']:+.4e} Pa/(m/s)")
    print(f"  ratio meta/aber = {abs(out['alpha_metasurface_Pa_per_mps']/out['alpha_aberration_Pa_per_mps']):.3f}")

    csv_path = out_dir / "damping_enhancement_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {csv_path}")
    _maybe_plot(out_dir, rows)


def _maybe_plot(out_dir: Path, rows: list) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    by_beta = {}
    for r in rows:
        by_beta.setdefault(r["beta"], []).append(r)
    for beta, group in sorted(by_beta.items()):
        periods = [g["ring_period_nm"] for g in group]
        ratios = [g["enhancement_ratio"] for g in group]
        ax.plot(periods, ratios, "o-", label=f"β = {beta:.2f}")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Ring grating period [nm]")
    ax.set_ylabel("(∂C_pr,2/∂θ)_ring / (∂C_pr,2/∂θ)_bare")
    ax.set_title("Outer-ring damping enhancement (docx Eq. 4.8 third term)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "damping_enhancement_plot.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
