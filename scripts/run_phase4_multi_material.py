"""Phase 4-A3 — Multi-material backside MIR absorber.

SiC has Reststrahlen 10–12.6 µm, hBN has TWO bands at 6.2–7.3 µm and
12.1–13.2 µm. By stacking two patterned layers (SiC + hBN) we cover
more of the 8–14 µm window for higher band-averaged ε_MIR.

Configurations tested:
  - SiC patch only (best from A2)
  - hBN patch only
  - SiC patch + hBN patch (different layers, stacked)
  - SiC + hBN side-by-side in super-cell (single layer)
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
SIN_DENSITY = 3100.0
SIC_DENSITY = 3210.0
HBN_DENSITY = 2100.0


def make_centered_patch(nx: int, frac: float):
    eps = np.zeros((nx, nx))
    nb = max(1, int(round(frac * nx / 2)))
    cx = nx // 2
    eps[cx - nb:cx + nb, cx - nb:cx + nb] = 1.0
    return eps


def make_quadrant_patches(nx: int, frac_a: float, frac_b: float):
    """Two patches: a in left half (-x), b in right half (+x), both square."""
    eps = np.zeros((nx, nx))
    nb_a = max(1, int(round(frac_a * nx / 4)))
    nb_b = max(1, int(round(frac_b * nx / 4)))
    cx_left = nx // 4
    cx_right = 3 * nx // 4
    cy = nx // 2
    eps[cx_left - nb_a:cx_left + nb_a, cy - nb_a:cy + nb_a] = 1.0
    eps[cx_right - nb_b:cx_right + nb_b, cy - nb_b:cy + nb_b] = 2.0  # mark as material B
    return eps


def evaluate_stacked_two_layers(
    sic_frac: float, sic_thickness_nm: float,
    hbn_frac: float, hbn_thickness_nm: float,
):
    nx = 96
    sic_grid = make_centered_patch(nx, sic_frac)
    hbn_grid = make_centered_patch(nx, hbn_frac)
    sic = SiCDispersion()
    hbn = HBNDispersion()

    sic_layer = PatternedLayerSpec(
        thickness_nm=sic_thickness_nm, eps_grid=sic_grid,
        eps_callable=sic.epsilon_callable(), name="SiC_patch",
    )
    hbn_layer = PatternedLayerSpec(
        thickness_nm=hbn_thickness_nm, eps_grid=hbn_grid,
        eps_callable=hbn.epsilon_callable(), name="hBN_patch",
    )
    cfg = RCWAConfig(nG=21, grid_nx=nx, grid_ny=nx)
    # SiC closer to PhC, hBN further out
    solver = LayeredRCWASolver(config=cfg, layers_below=[sic_layer, hbn_layer])
    structure = PhCReflector(**DESIGN_A).to_structure()

    nir_wls = np.linspace(NIR_BAND[0], NIR_BAND[1], 5)
    mir_wls = np.linspace(MIR_BAND[0], MIR_BAND[1], 13)
    R_nir = solver.evaluate_reflectivity(structure, nir_wls)
    T_nir = solver.evaluate_transmission(structure, nir_wls)
    R_mir = solver.evaluate_reflectivity(structure, mir_wls)
    T_mir = solver.evaluate_transmission(structure, mir_wls)
    eps_mir = np.clip(1 - R_mir - T_mir, 0.0, 1.0)
    A_nir = np.clip(1 - R_nir - T_nir, 0.0, 1.0)

    sic_fill = float(np.mean(sic_grid > 0.5))
    hbn_fill = float(np.mean(hbn_grid > 0.5))
    mass_extra = (
        SIC_DENSITY * sic_thickness_nm * 1e-9 * sic_fill
        + HBN_DENSITY * hbn_thickness_nm * 1e-9 * hbn_fill
    )

    return {
        "config": f"SiC_{sic_frac:.1f}_{sic_thickness_nm:.0f}nm + hBN_{hbn_frac:.1f}_{hbn_thickness_nm:.0f}nm",
        "sic_frac": sic_frac, "sic_t": sic_thickness_nm,
        "hbn_frac": hbn_frac, "hbn_t": hbn_thickness_nm,
        "mean_R_NIR": float(R_nir.mean()),
        "mean_alpha_NIR": float(A_nir.mean()),
        "mean_eps_MIR": float(eps_mir.mean()),
        "peak_eps_MIR": float(eps_mir.max()),
        "extra_g_per_m2": mass_extra * 1000.0,
        "eps_band_8_10": float(np.mean(eps_mir[mir_wls <= 10000])),
        "eps_band_10_12": float(np.mean(eps_mir[(mir_wls > 10000) & (mir_wls <= 12000)])),
        "eps_band_12_14": float(np.mean(eps_mir[mir_wls > 12000])),
    }


def main():
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p4_multi_material"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4-A3 — SiC + hBN multi-material backside ===")
    rows = []
    # Best from A2: SiC 50% × 400nm. Add hBN at various sizes.
    configs = [
        # (sic_frac, sic_t, hbn_frac, hbn_t)
        (0.5, 400, 0.0, 0),     # SiC alone (A2 best)
        (0.0, 0,   0.5, 400),   # hBN alone
        (0.5, 400, 0.5, 200),   # combined
        (0.5, 400, 0.5, 400),
        (0.5, 200, 0.7, 400),
        (0.7, 200, 0.5, 400),
        (0.5, 400, 0.7, 200),
        (0.6, 300, 0.6, 300),
    ]
    for sf, st, hf, ht in configs:
        if sf == 0.0:
            # hBN alone
            print(f"  hBN_{hf:.1f}_{ht:.0f}nm alone ...", end=" ", flush=True)
        elif hf == 0.0:
            print(f"  SiC_{sf:.1f}_{st:.0f}nm alone ...", end=" ", flush=True)
        else:
            print(f"  SiC{sf:.1f}/{st:.0f} + hBN{hf:.1f}/{ht:.0f} ...", end=" ", flush=True)
        # Use single-material if other is zero
        if sf == 0.0:
            r = evaluate_stacked_two_layers(0.0, 1.0, hf, ht)  # tiny SiC dummy
        elif hf == 0.0:
            r = evaluate_stacked_two_layers(sf, st, 0.0, 1.0)
        else:
            r = evaluate_stacked_two_layers(sf, st, hf, ht)
        rows.append(r)
        print(
            f"ε_MIR={r['mean_eps_MIR']:.3f} (8-10:{r['eps_band_8_10']:.2f} "
            f"10-12:{r['eps_band_10_12']:.2f} 12-14:{r['eps_band_12_14']:.2f}) "
            f"R_NIR={r['mean_R_NIR']:.3f} mass=+{r['extra_g_per_m2']:.2f}g/m²"
        )

    csv_path = out_dir / "multi_material_scan.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "multi_material_scan.md"
    with open(md_path, "w") as f:
        f.write("# Phase 4-A3 — Multi-material backside MIR absorber\n\n")
        f.write("| Config | ε_MIR avg | 8-10µm | 10-12µm | 12-14µm | R_NIR | extra mass [g/m²] |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['config']} | {r['mean_eps_MIR']:.3f} "
                f"| {r['eps_band_8_10']:.3f} | {r['eps_band_10_12']:.3f} | {r['eps_band_12_14']:.3f} "
                f"| {r['mean_R_NIR']:.3f} | {r['extra_g_per_m2']:.3f} |\n"
            )

    rows_sorted = sorted(rows, key=lambda r: -r["mean_eps_MIR"])
    print(f"\nTop 3:")
    for r in rows_sorted[:3]:
        print(f"  {r['config']}: ε={r['mean_eps_MIR']:.3f} R={r['mean_R_NIR']:.3f} mass=+{r['extra_g_per_m2']:.2f}g/m²")

    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
