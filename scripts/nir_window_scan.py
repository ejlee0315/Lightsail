"""Scan 1000–2000 nm for the best continuous 300-nm-wide reflectance band
on each family's best Stage 1 design.

Motivation: the default NIR target band 1350–1650 nm was chosen before
any lattice optimization was done. For designs that peak outside that
window (some families do), the fixed-band score underestimates how good
they actually are for lightsail propulsion. This script rescans each
best design on a finer 1000–2000 nm grid and reports the 300-nm window
that maximizes mean reflectance.

Output: text table + matplotlib figure with R(λ) for all designs,
fixed reference band shaded, best 300-nm window marked.

Usage:
    python3 scripts/nir_window_scan.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver


# --- wavelength grid -----------------------------------------------------
LAMBDA_MIN_NM = 1000.0
LAMBDA_MAX_NM = 2000.0
LAMBDA_STEP_NM = 10.0
WINDOW_NM = 300.0

# Fixed reference band for cross-check
REF_BAND_LO = 1350.0
REF_BAND_HI = 1650.0

# RCWA settings
NG = 41     # fast scan; validate the winner at nG=81 afterwards if needed
GRID = 96


@dataclass
class DesignSpec:
    name: str
    lattice_family: LatticeFamily
    thickness_nm: float
    lattice_period_nm: float
    hole_a_rel: float
    hole_b_rel: float
    hole_rotation_deg: float
    corner_rounding: float
    shape_parameter: float
    lattice_aspect_ratio: float = 1.0
    reported_nir: float = float("nan")   # the score at the fixed 1350–1650 band

    def build(self) -> PhCReflector:
        return PhCReflector(
            lattice_family=self.lattice_family,
            n_rings=6,
            thickness_nm=self.thickness_nm,
            lattice_period_nm=self.lattice_period_nm,
            hole_a_rel=self.hole_a_rel,
            hole_b_rel=self.hole_b_rel,
            hole_rotation_deg=self.hole_rotation_deg,
            corner_rounding=self.corner_rounding,
            shape_parameter=self.shape_parameter,
            lattice_aspect_ratio=self.lattice_aspect_ratio,
        )


# Designs to scan (best per lattice family)
DESIGNS: list[DesignSpec] = [
    DesignSpec(
        name="triangular (s123, trial 101)",
        lattice_family=LatticeFamily.TRIANGULAR,
        thickness_nm=687.9939,
        lattice_period_nm=1509.6008,
        hole_a_rel=0.3641862,
        hole_b_rel=0.3974993,
        hole_rotation_deg=4.164857,
        corner_rounding=1.0,
        shape_parameter=6.4902,
        reported_nir=0.8489,
    ),
    DesignSpec(
        name="hexagonal (prod, trial 96)",
        lattice_family=LatticeFamily.HEXAGONAL,
        thickness_nm=567.4051,
        lattice_period_nm=1437.8576,
        hole_a_rel=0.3148169,
        hole_b_rel=0.3073298,
        hole_rotation_deg=131.3830,
        corner_rounding=1.0,
        shape_parameter=8.0,
        reported_nir=0.3941,
    ),
    DesignSpec(
        name="rectangular (prod, trial 118)",
        lattice_family=LatticeFamily.RECTANGULAR,
        thickness_nm=961.0028,
        lattice_period_nm=1535.3906,
        hole_a_rel=0.3839358,
        hole_b_rel=0.4411955,
        hole_rotation_deg=76.3254,
        corner_rounding=1.0,
        shape_parameter=3.0,
        lattice_aspect_ratio=1.0772,
        reported_nir=0.4124,
    ),
    DesignSpec(
        name="pentagonal (prod, trial 98)",
        lattice_family=LatticeFamily.PENTAGONAL_SUPERCELL,
        thickness_nm=563.7839,
        lattice_period_nm=1343.3662,
        hole_a_rel=0.05,
        hole_b_rel=0.07808,
        hole_rotation_deg=130.6550,
        corner_rounding=0.5422,
        shape_parameter=3.6123,
        reported_nir=0.3169,
    ),
]


def best_window(
    wavelengths_nm: np.ndarray,
    R: np.ndarray,
    window_width_nm: float,
) -> tuple[float, float, float, float]:
    """Return (lam_lo, lam_hi, mean_R, min_R) for the 300-nm window
    that maximizes mean(R). ``wavelengths_nm`` is assumed sorted."""
    step = float(wavelengths_nm[1] - wavelengths_nm[0])
    n_in_window = int(round(window_width_nm / step)) + 1  # inclusive count

    if n_in_window > len(wavelengths_nm):
        raise ValueError("Window wider than wavelength grid")

    best_mean = -np.inf
    best_idx = 0
    for start in range(len(wavelengths_nm) - n_in_window + 1):
        end = start + n_in_window
        m = R[start:end].mean()
        if m > best_mean:
            best_mean = m
            best_idx = start

    end = best_idx + n_in_window
    lam_lo = float(wavelengths_nm[best_idx])
    lam_hi = float(wavelengths_nm[end - 1])
    return lam_lo, lam_hi, float(best_mean), float(R[best_idx:end].min())


def main() -> None:
    wavelengths = np.arange(
        LAMBDA_MIN_NM, LAMBDA_MAX_NM + 0.5 * LAMBDA_STEP_NM, LAMBDA_STEP_NM
    )
    print(
        f"Scanning {len(wavelengths)} wavelengths from "
        f"{LAMBDA_MIN_NM:.0f}–{LAMBDA_MAX_NM:.0f} nm at {LAMBDA_STEP_NM:.0f} nm step, "
        f"nG={NG}, grid={GRID}x{GRID}"
    )
    print(
        f"Window width: {WINDOW_NM:.0f} nm "
        f"({int(round(WINDOW_NM / LAMBDA_STEP_NM)) + 1} points per window)"
    )
    print()

    solver = RCWASolver(
        config=RCWAConfig(
            nG=NG, grid_nx=GRID, grid_ny=GRID, polarization="average"
        )
    )

    results: list[tuple[DesignSpec, np.ndarray, tuple]] = []

    hdr = (
        f"{'design':<34} {'ref 1350-1650':>14} {'best window':>18} "
        f"{'mean R':>8} {'min R':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for design in DESIGNS:
        phc = design.build()
        structure = phc.to_structure()
        R = solver.evaluate_reflectivity(structure, wavelengths)

        # Reference fixed-band mean
        ref_mask = (wavelengths >= REF_BAND_LO) & (wavelengths <= REF_BAND_HI)
        ref_mean = float(R[ref_mask].mean())
        ref_min = float(R[ref_mask].min())

        lam_lo, lam_hi, best_mean, best_min = best_window(
            wavelengths, R, WINDOW_NM
        )
        results.append((design, R, (lam_lo, lam_hi, best_mean, best_min)))

        print(
            f"{design.name:<34} "
            f"{ref_mean:6.3f} ({ref_min:.2f}) "
            f" {lam_lo:4.0f}-{lam_hi:4.0f}nm "
            f"{best_mean:8.3f} {best_min:8.3f}"
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for (design, R, window), color in zip(results, colors):
        lam_lo, lam_hi, mean_R, _ = window
        ax.plot(
            wavelengths,
            R,
            color=color,
            linewidth=1.8,
            label=f"{design.name}  (best: {lam_lo:.0f}-{lam_hi:.0f} nm, R̄={mean_R:.3f})",
        )
        # Mark the best 300 nm window as a horizontal bar near the curve
        ax.hlines(
            mean_R,
            lam_lo,
            lam_hi,
            color=color,
            linewidth=3.5,
            alpha=0.45,
        )

    # Shade the fixed reference band
    ax.axvspan(
        REF_BAND_LO,
        REF_BAND_HI,
        color="gray",
        alpha=0.15,
        label=f"fixed target band {REF_BAND_LO:.0f}–{REF_BAND_HI:.0f} nm",
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance R")
    ax.set_title(
        f"R(λ) on 1000–2000 nm — best 300-nm window per lattice family (nG={NG})"
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    out_path = Path("results") / "nir_window_scan.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
