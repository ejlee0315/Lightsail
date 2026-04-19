"""P1.3 — Compare analytic AsymmetryStabilizationProxy vs LocalPeriodFMMProxy.

For a sweep of metagrating designs we compute both proxy scores and
report rank correlation + a scatter plot. The motivation is the Phase 2B
finding (CLAUDE.md, 2026-04-18) that the analytic proxies saturate at
their parameter bounds — we now have a real diffraction calculation
to check whether the proxies actually track real stiffness.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from lightsail.geometry.metagrating import MetaGrating
from lightsail.optimization import (
    AsymmetryStabilizationProxy,
    LocalPeriodFMMProxy,
    ObjectiveContext,
    RadialMomentumProxy,
)
from lightsail.simulation import RCWAConfig, RCWASolver
from lightsail.simulation.grating_fmm import FMMGratingConfig


NIR_BAND = (1550.0, 1850.0)


def make_design(period_nm: float, duty: float, asym: float, curv: float) -> MetaGrating:
    return MetaGrating(
        inner_radius_nm=5_000_000.0,
        thickness_nm=240.0,
        grating_period_nm=period_nm,
        duty_cycle=duty,
        curvature=curv,
        asymmetry=asym,
        ring_width_um=2000.0,
    )


def main() -> None:
    out_dir = Path("results") / f"{datetime.now():%Y-%m-%d_%H%M%S}_p1_3_proxy_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rcwa_cfg = RCWAConfig(nG=15, grid_nx=48, grid_ny=48)
    solver = RCWASolver(config=rcwa_cfg)
    fmm_cfg = FMMGratingConfig(nG=15, nx=96, ny=4)

    asym_proxy = AsymmetryStabilizationProxy(nir_band_nm=NIR_BAND, nir_n_points=3)
    rad_proxy = RadialMomentumProxy(nir_band_nm=NIR_BAND, nir_n_points=3)
    fmm_proxy = LocalPeriodFMMProxy(
        nir_band_nm=NIR_BAND, nir_n_points=3,
        n_radial_bins=2, nG=15,
    )

    sweep = []
    for period in (1500.0, 1800.0, 2000.0, 2400.0):
        for duty in (0.3, 0.5, 0.7):
            for asym in (0.0, 0.1, 0.2):
                for curv in (-0.1, 0.0, 0.1):
                    sweep.append((period, duty, asym, curv))

    print(f"Running {len(sweep)} designs ...")
    rows = []
    for i, (p, d, a, c) in enumerate(sweep):
        mg = make_design(p, d, a, c)
        struct = mg.to_structure()
        ctx = ObjectiveContext(geometry=mg, structure=struct, solver=solver)
        s_asym, m_asym = asym_proxy.score(ctx)
        s_rad, m_rad = rad_proxy.score(ctx)
        s_fmm, m_fmm = fmm_proxy.score(ctx)
        rows.append(
            {
                "period_nm": p, "duty": d, "asymmetry": a, "curvature": c,
                "score_asymmetry": s_asym,
                "score_radial_momentum": s_rad,
                "score_fmm": s_fmm,
                "fmm_stiffness_per_rad": m_fmm.get("stiffness_raw_per_rad", 0.0),
                "fmm_damping_per_rad": m_fmm.get("damping_raw_per_rad", 0.0),
                "fmm_nir_R": m_fmm.get("nir_R", 0.0),
            }
        )
        if i % 10 == 0:
            print(
                f"  [{i+1:3d}/{len(sweep)}] p={p:.0f}, d={d:.2f}, "
                f"a={a:.2f}, c={c:+.2f} → s_asym={s_asym:.3f} "
                f"s_rad={s_rad:.3f} s_fmm={s_fmm:.3f}"
            )

    arr = np.array(
        [
            (r["score_asymmetry"], r["score_radial_momentum"], r["score_fmm"])
            for r in rows
        ]
    )

    print("\n=== Rank correlations (Spearman) ===")
    rho_asym = _spearman(arr[:, 0], arr[:, 2])
    rho_rad = _spearman(arr[:, 1], arr[:, 2])
    print(f"  asymmetry-proxy   ↔ FMM:  ρ = {rho_asym:+.3f}")
    print(f"  radial-momentum   ↔ FMM:  ρ = {rho_rad:+.3f}")

    json_path = out_dir / "proxy_vs_fmm.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "rows": rows,
                "spearman_asymmetry_vs_fmm": rho_asym,
                "spearman_radial_vs_fmm": rho_rad,
                "n_designs": len(rows),
                "nir_band_nm": list(NIR_BAND),
            },
            f, indent=2,
        )
    print(f"\nResults written to {json_path}")
    _maybe_plot(out_dir, arr)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    return float(1.0 - 6.0 * np.sum((rx - ry) ** 2) / (n * (n ** 2 - 1)))


def _maybe_plot(out_dir: Path, arr: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(arr[:, 0], arr[:, 2], alpha=0.6)
    axes[0].set_xlabel("AsymmetryStabilizationProxy score")
    axes[0].set_ylabel("LocalPeriodFMMProxy score")
    axes[0].set_title("Analytic vs FMM (asymmetry)")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)

    axes[1].scatter(arr[:, 1], arr[:, 2], alpha=0.6, c="C1")
    axes[1].set_xlabel("RadialMomentumProxy score")
    axes[1].set_ylabel("LocalPeriodFMMProxy score")
    axes[1].set_title("Analytic vs FMM (radial momentum)")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "proxy_vs_fmm_scatter.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
