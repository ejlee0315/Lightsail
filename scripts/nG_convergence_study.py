"""RCWA Fourier-harmonic (nG) convergence study.

Evaluates a single representative triangular PhC reflector at 6 wavelengths
(3 in the NIR target band, 3 in the MIR target band) for a sweep of nG
values, then prints a comparison table and a recommendation for the
production ``nG`` setting.

Rule of thumb: a design is "converged" when the R, T values change by less
than ~1e-3 between the last two nG steps. If that threshold isn't hit, the
solver needs more harmonics — production should use at least the smallest
nG that passes.

Run:

    python3 scripts/nG_convergence_study.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver


# Representative PhC geometry — middle of the design space.
PHC = PhCReflector(
    lattice_family=LatticeFamily.TRIANGULAR,
    n_rings=6,
    thickness_nm=500.0,
    lattice_period_nm=1500.0,
    hole_a_rel=400.0 / 1500.0,   # 400 nm hole / 1500 nm period
    hole_b_rel=400.0 / 1500.0,
    hole_rotation_deg=0.0,
    corner_rounding=1.0,
    shape_parameter=6.0,
)

TEST_WAVELENGTHS_NM = np.array(
    [
        1350.0,   # NIR band low
        1500.0,   # NIR band center
        1650.0,   # NIR band high
        8000.0,   # MIR band low
        10500.0,  # MIR Si-N stretch peak
        14000.0,  # MIR band high
    ]
)

NG_SWEEP = [11, 21, 31, 41, 61, 81]

# Convergence thresholds (absolute differences in R or T vs the largest nG).
THRESHOLD_GOOD = 1e-3   # very well converged
THRESHOLD_OK = 5e-3    # acceptable for optimization


def _print_table(
    title: str,
    wavelengths: np.ndarray,
    results: dict,
    quantity: str,
) -> None:
    """Print a (wavelength × nG) table for R or T."""
    print(f"\n{title}")
    header_parts = [f"{'wl (nm)':>8}"]
    for nG in NG_SWEEP:
        header_parts.append(f"nG={nG:<3d}  ")
    header_parts.append("  Δ(last-prev)")
    header = "  ".join(header_parts)
    print(header)
    print("-" * len(header))

    for i, wl in enumerate(wavelengths):
        row_parts = [f"{int(wl):>8d}"]
        prev_val = None
        last_val = None
        for nG in NG_SWEEP:
            val = float(results[nG][quantity][i])
            row_parts.append(f"  {val:.5f} ")
            prev_val = last_val
            last_val = val
        diff = abs(last_val - prev_val) if prev_val is not None else 0.0
        row_parts.append(f"   {diff:.2e}")
        print("  ".join(row_parts))


def _recommend(results: dict, wavelengths: np.ndarray) -> int:
    """Pick the smallest nG whose R and T match the largest nG within THRESHOLD_OK.

    Returns the recommended nG. If none pass, returns the largest tested nG.
    """
    ref_nG = NG_SWEEP[-1]
    ref_R = results[ref_nG]["R"]
    ref_T = results[ref_nG]["T"]

    for nG in NG_SWEEP[:-1]:
        dR = np.max(np.abs(results[nG]["R"] - ref_R))
        dT = np.max(np.abs(results[nG]["T"] - ref_T))
        if max(dR, dT) < THRESHOLD_OK:
            return nG
    return ref_nG


def main() -> None:
    print("=" * 72)
    print("RCWA nG convergence study")
    print("=" * 72)
    print(f"Geometry : triangular lattice, period={PHC.lattice_period_nm:.0f} nm,")
    print(f"           hole a=b={PHC.hole_a_nm:.0f} nm, thickness={PHC.thickness_nm:.0f} nm,")
    print(f"           n_sides={PHC.n_sides}, corner_rounding={PHC.corner_rounding}")
    print(f"Grid     : 96 × 96 unit-cell raster, polarization averaged over TE/TM")
    print(f"nG sweep : {NG_SWEEP}")
    print(f"WL points: {[int(w) for w in TEST_WAVELENGTHS_NM]} (nm)")
    print()

    structure = PHC.to_structure()

    results: dict[int, dict] = {}
    for nG in NG_SWEEP:
        solver = RCWASolver(
            config=RCWAConfig(nG=nG, grid_nx=96, grid_ny=96, polarization="average")
        )
        t0 = time.time()
        R = solver.evaluate_reflectivity(structure, TEST_WAVELENGTHS_NM)
        T = solver.evaluate_transmission(structure, TEST_WAVELENGTHS_NM)
        dt = time.time() - t0
        results[nG] = {"R": R, "T": T, "time_s": dt}
        print(
            f"  nG={nG:3d}  evaluated {len(TEST_WAVELENGTHS_NM)} wavelengths "
            f"in {dt:5.2f} s  ({dt / len(TEST_WAVELENGTHS_NM) * 1000:.0f} ms/wl)"
        )

    # Tables
    _print_table(
        "R (reflectance)", TEST_WAVELENGTHS_NM, results, "R"
    )
    _print_table(
        "T (transmittance)", TEST_WAVELENGTHS_NM, results, "T"
    )

    # Energy-conservation sanity check
    print("\nEnergy conservation (R + T, should be ≤ 1)")
    header = f"{'wl (nm)':>8}  " + "  ".join(f"nG={nG:<3d}  " for nG in NG_SWEEP)
    print(header)
    print("-" * len(header))
    for i, wl in enumerate(TEST_WAVELENGTHS_NM):
        row = f"{int(wl):>8d}  "
        for nG in NG_SWEEP:
            total = float(results[nG]["R"][i] + results[nG]["T"][i])
            row += f"  {total:.5f} "
        print(row)

    # Convergence summary vs reference (largest nG)
    print("\nConvergence vs reference nG =", NG_SWEEP[-1])
    print(f"{'nG':>4}  {'max|ΔR|':>10}  {'max|ΔT|':>10}  status")
    print("-" * 46)
    ref_R = results[NG_SWEEP[-1]]["R"]
    ref_T = results[NG_SWEEP[-1]]["T"]
    for nG in NG_SWEEP[:-1]:
        dR = float(np.max(np.abs(results[nG]["R"] - ref_R)))
        dT = float(np.max(np.abs(results[nG]["T"] - ref_T)))
        worst = max(dR, dT)
        if worst < THRESHOLD_GOOD:
            tag = "✓ well converged"
        elif worst < THRESHOLD_OK:
            tag = "~ OK for optim"
        else:
            tag = "✗ under-converged"
        print(f"{nG:>4d}  {dR:>10.2e}  {dT:>10.2e}  {tag}")

    # Timing summary
    print("\nTiming (per nG, 6 wavelengths)")
    for nG in NG_SWEEP:
        print(f"  nG={nG:3d}  {results[nG]['time_s']:5.2f} s")

    # Recommendation
    recommended = _recommend(results, TEST_WAVELENGTHS_NM)
    print("\n" + "=" * 72)
    print(f"RECOMMENDATION: use nG = {recommended} for production")
    print("=" * 72)
    print(
        "  - Smallest nG whose R and T match nG="
        f"{NG_SWEEP[-1]} within {THRESHOLD_OK:.0e}"
    )
    production_time_per_trial = results[recommended]["time_s"] / 6 * 60
    print(
        f"  - Estimated cost: ~{production_time_per_trial:.1f} s per trial "
        "(60 wavelengths, 1 design)"
    )
    print(
        f"  - 100-trial production run ≈ {production_time_per_trial * 100 / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
