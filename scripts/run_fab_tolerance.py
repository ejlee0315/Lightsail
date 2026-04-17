"""Fabrication tolerance analysis for the best thin design.

Perturbs each parameter by +/- 5% around the best design
(t=280 nm, P=1580 nm, a_rel=0.38) and computes T to assess
robustness against fabrication errors.

Output: results/fab_tolerance_<timestamp>/tolerance.yaml with T for
each perturbation.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
from scipy.integrate import quad
from scipy.interpolate import interp1d

import lightsail.geometry.phc_reflector as _pr
_pr._THICKNESS_MIN_NM = 5.0
_pr._THICKNESS_MAX_NM = 500.0

from lightsail.geometry.base import LatticeFamily
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver

c = 299_792_458.0
I = 1.0e10
A = 10.0
M_PAY = 1.0e-3
RHO = 3100.0
LAM0 = 1550.0
BETA_F = 0.2


def compute_TD(phc, nG=81):
    solver = RCWASolver(config=RCWAConfig(
        nG=nG, grid_nx=96, grid_ny=96, polarization="average",
    ))
    wl_dop = np.linspace(LAM0, LAM0 * np.sqrt(1.2 / 0.8), 30)
    R = solver.evaluate_reflectivity(phc.to_structure(), wl_dop)
    R_interp = interp1d(wl_dop, R, kind="linear",
                        fill_value=(R[0], R[-1]), bounds_error=False)
    ucell = phc.unit_cell_area_nm2
    hole_area = phc.hole_shape().area_nm2()
    f_mat = max(1.0 - hole_area / ucell, 0.01)
    rho_l = RHO * phc.thickness_nm * 1e-9 * f_mat
    m_total = rho_l * A + M_PAY

    def T_f(beta):
        g = 1 / np.sqrt(1 - beta**2)
        lam = LAM0 * np.sqrt((1 + beta) / (1 - beta))
        return g**3 * (1 + beta) / (1 - beta) / max(float(R_interp(lam)), 0.001)

    def D_f(beta):
        lam = LAM0 * np.sqrt((1 + beta) / (1 - beta))
        return beta / (1 - beta)**2 * np.sqrt(1 - beta**2) / max(float(R_interp(lam)), 0.001)

    Ti, _ = quad(T_f, 0, BETA_F, limit=200)
    Di, _ = quad(D_f, 0, BETA_F, limit=200)
    T_min = m_total * c**2 / (2 * I * A) * Ti / 60
    D_gm = c**3 / (2 * I) * (m_total / A) * Di * 1e-9
    return {
        "T_min": float(T_min),
        "D_Gm": float(D_gm),
        "R_mean": float(R.mean()),
        "R_min": float(R.min()),
        "mass_g": float(m_total * 1e3),
        "fill": float(f_mat),
    }


def make_phc(t, P, a_rel, b_rel=None):
    if b_rel is None:
        b_rel = a_rel
    return PhCReflector(
        lattice_family=LatticeFamily.TRIANGULAR, n_rings=6,
        thickness_nm=float(t), lattice_period_nm=float(P),
        hole_a_rel=float(a_rel), hole_b_rel=float(b_rel),
        corner_rounding=1.0, shape_parameter=6.0,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Baseline: best design from grid scan
    t0, P0, a0 = 280.0, 1580.0, 0.38

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(f"results/{timestamp}_fab_tolerance")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Baseline
    logging.info("Evaluating baseline (t=%s, P=%s, a_rel=%s)...", t0, P0, a0)
    base = compute_TD(make_phc(t0, P0, a0))
    results["baseline"] = {**base, "t_nm": t0, "P_nm": P0, "a_rel": a0}
    logging.info("  baseline T = %.3f min", base["T_min"])

    # Single-parameter perturbations: ±5% and ±10%
    perturbations = [
        ("t", [-0.10, -0.05, 0.05, 0.10]),
        ("P", [-0.10, -0.05, 0.05, 0.10]),
        ("a_rel", [-0.10, -0.05, 0.05, 0.10]),
    ]

    for param, deltas in perturbations:
        results[param] = {}
        for d in deltas:
            if param == "t":
                t_p, P_p, a_p = t0 * (1 + d), P0, a0
            elif param == "P":
                t_p, P_p, a_p = t0, P0 * (1 + d), a0
            else:
                t_p, P_p, a_p = t0, P0, a0 * (1 + d)
            key = f"{d:+.2f}"
            logging.info("Perturbing %s by %s (t=%.1f, P=%.1f, a=%.3f)...",
                         param, key, t_p, P_p, a_p)
            r = compute_TD(make_phc(t_p, P_p, a_p))
            r.update({"t_nm": t_p, "P_nm": P_p, "a_rel": a_p})
            results[param][key] = r

    # Ellipse asymmetry: a != b
    results["ellipse_5pct"] = {}
    for d in [-0.05, 0.05]:
        a_p = a0 * (1 + d)
        b_p = a0 * (1 - d)
        logging.info("Ellipse a=%.3f, b=%.3f...", a_p, b_p)
        r = compute_TD(make_phc(t0, P0, a_p, b_p))
        r.update({"a_rel": a_p, "b_rel": b_p})
        results["ellipse_5pct"][f"{d:+.2f}"] = r

    # Summary
    logging.info("\n=== Fabrication Tolerance Summary ===")
    logging.info("Baseline: T=%.3f min", base["T_min"])
    for param in ["t", "P", "a_rel"]:
        logging.info("Parameter %s:", param)
        for k, v in results[param].items():
            dT = v["T_min"] - base["T_min"]
            logging.info("  %s: T=%.3f min (ΔT=%+.3f)", k, v["T_min"], dT)

    # Save
    with open(out_dir / "tolerance.yaml", "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    logging.info("Saved: %s", out_dir / "tolerance.yaml")


if __name__ == "__main__":
    main()
