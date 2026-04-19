"""Phase 4-A2 — MIR patch backside absorber baseline scan.

Sweeps (material, patch size, thickness, fill) for a single-size
patterned backside layer behind Design A. Reports ε_MIR_avg over
8–14 µm, NIR R loss, and added mass per m².
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.materials import HBNDispersion, SiCDispersion
from lightsail.simulation import (
    LayeredRCWASolver,
    PatternedLayerSpec,
    RCWAConfig,
    RCWASolver,
)


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

NIR_BAND = (1550.0, 1898.0)
MIR_BAND = (8000.0, 14000.0)
PERIOD_UM = 1.58
SIN_DENSITY = 3100.0
SIC_DENSITY = 3210.0
HBN_DENSITY = 2100.0


def make_patch_grid(nx: int, patch_frac: float) -> np.ndarray:
    """Square patch centered in nx × nx grid; patch_frac in (0, 1) of period."""
    eps = np.zeros((nx, nx), dtype=float)
    nb = max(1, int(round(patch_frac * nx / 2)))
    cx = nx // 2
    eps[cx - nb:cx + nb, cx - nb:cx + nb] = 1.0
    return eps


def evaluate(material: str, patch_frac: float, thickness_nm: float):
    nx = 64
    eps_grid = make_patch_grid(nx, patch_frac)
    if material == "SiC":
        disp = SiCDispersion()
        rho = SIC_DENSITY
    elif material == "hBN":
        disp = HBNDispersion()
        rho = HBN_DENSITY
    else:
        raise ValueError(material)
    fill = float(np.mean(eps_grid > 0.5))
    extra_areal = rho * thickness_nm * 1e-9 * fill
    patched = PatternedLayerSpec(
        thickness_nm=thickness_nm, eps_grid=eps_grid,
        eps_callable=disp.epsilon_callable(), name=f"{material}_p{patch_frac}_t{thickness_nm:.0f}",
    )
    cfg = RCWAConfig(nG=21, grid_nx=nx, grid_ny=nx)
    solver = LayeredRCWASolver(config=cfg, layers_below=[patched])
    structure = PhCReflector(**DESIGN_A).to_structure()

    nir_wls = np.linspace(NIR_BAND[0], NIR_BAND[1], 7)
    mir_wls = np.linspace(MIR_BAND[0], MIR_BAND[1], 13)
    R_nir = solver.evaluate_reflectivity(structure, nir_wls)
    T_nir = solver.evaluate_transmission(structure, nir_wls)
    A_nir = np.clip(1 - R_nir - T_nir, 0.0, 1.0)
    R_mir = solver.evaluate_reflectivity(structure, mir_wls)
    T_mir = solver.evaluate_transmission(structure, mir_wls)
    eps_mir = np.clip(1 - R_mir - T_mir, 0.0, 1.0)

    return {
        "material": material,
        "patch_frac": patch_frac,
        "thickness_nm": thickness_nm,
        "fill": fill,
        "extra_g_per_m2": extra_areal * 1000.0,
        "mean_R_NIR": float(R_nir.mean()),
        "mean_alpha_NIR": float(A_nir.mean()),
        "mean_eps_MIR": float(eps_mir.mean()),
        "peak_eps_MIR": float(eps_mir.max()),
        "peak_lambda_um": float(mir_wls[int(np.argmax(eps_mir))] / 1000.0),
    }


def baseline():
    """Bare PhC (no back layer) for reference."""
    cfg = RCWAConfig(nG=21, grid_nx=64, grid_ny=64)
    solver = RCWASolver(config=cfg)
    structure = PhCReflector(**DESIGN_A).to_structure()
    nir_wls = np.linspace(NIR_BAND[0], NIR_BAND[1], 7)
    mir_wls = np.linspace(MIR_BAND[0], MIR_BAND[1], 13)
    R_nir = solver.evaluate_reflectivity(structure, nir_wls)
    T_nir = solver.evaluate_transmission(structure, nir_wls)
    R_mir = solver.evaluate_reflectivity(structure, mir_wls)
    T_mir = solver.evaluate_transmission(structure, mir_wls)
    eps_mir = np.clip(1 - R_mir - T_mir, 0.0, 1.0)
    return {
        "material": "none",
        "patch_frac": 0.0, "thickness_nm": 0.0, "fill": 0.0,
        "extra_g_per_m2": 0.0,
        "mean_R_NIR": float(R_nir.mean()),
        "mean_alpha_NIR": float(np.clip(1 - R_nir - T_nir, 0, 1).mean()),
        "mean_eps_MIR": float(eps_mir.mean()),
        "peak_eps_MIR": float(eps_mir.max()),
        "peak_lambda_um": float(mir_wls[int(np.argmax(eps_mir))] / 1000.0),
    }


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_mir_patch_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building baseline (bare PhC) ...")
    rows = [baseline()]
    print(
        f"  baseline: ε_MIR_avg={rows[0]['mean_eps_MIR']:.3f}  "
        f"peak {rows[0]['peak_eps_MIR']:.3f}@{rows[0]['peak_lambda_um']:.1f}µm  "
        f"R_NIR={rows[0]['mean_R_NIR']:.3f}"
    )
    print()
    print("=== MIR patch scan ===")
    for material in ("SiC", "hBN"):
        for patch_frac in (0.5, 0.7, 0.9):
            for thickness_nm in (50.0, 100.0, 200.0, 400.0):
                print(f"  {material} patch={patch_frac:.1f} t={thickness_nm:.0f}nm ...", end=" ", flush=True)
                r = evaluate(material, patch_frac, thickness_nm)
                rows.append(r)
                print(
                    f"ε_MIR={r['mean_eps_MIR']:.3f} (peak {r['peak_eps_MIR']:.3f}) "
                    f"R_NIR={r['mean_R_NIR']:.3f} mass=+{r['extra_g_per_m2']:.3f}g/m²"
                )

    csv_path = out_dir / "mir_patch_scan.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "mir_patch_scan.md"
    with open(md_path, "w") as f:
        f.write("# Phase 4-A2 — MIR patch absorber scan (Design A backside)\n\n")
        f.write("| Material | patch | t [nm] | ε_MIR avg | peak ε / λ | R_NIR | extra mass [g/m²] |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['material']} | {r['patch_frac']:.1f} | {r['thickness_nm']:.0f} "
                f"| {r['mean_eps_MIR']:.3f} | {r['peak_eps_MIR']:.3f} @ {r['peak_lambda_um']:.1f}µm "
                f"| {r['mean_R_NIR']:.3f} | {r['extra_g_per_m2']:.3f} |\n"
            )

    # Identify best
    rows_sorted = sorted(rows, key=lambda r: -r["mean_eps_MIR"])
    print(f"\nTop 3 by ε_MIR avg:")
    for r in rows_sorted[:3]:
        print(
            f"  {r['material']} patch={r['patch_frac']:.1f} t={r['thickness_nm']:.0f}nm: "
            f"ε_MIR={r['mean_eps_MIR']:.3f} R_NIR={r['mean_R_NIR']:.3f} mass=+{r['extra_g_per_m2']:.3f}g/m²"
        )
    print(f"\nCSV → {csv_path}")
    print(f"MD  → {md_path}")


if __name__ == "__main__":
    main()
