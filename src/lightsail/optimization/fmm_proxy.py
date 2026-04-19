"""Full-wave (1D-FMM) Stage 2 stabilization proxy.

Replaces the analytic :class:`AsymmetryStabilizationProxy` and
:class:`RadialMomentumProxy` (which both saturate at their parameter
bounds — see CLAUDE.md, 2026-04-18 Phase 2B) with a real diffraction
calculation.

Score
-----
For each radial bin × NIR wavelength we compute the local restoring
stiffness proxy ``|∂C_pr,1/∂θ|`` (per-radian) via centered finite
difference. The score is

    s = NIR_R · sigmoid(stiffness_raw / (stiffness_raw + scale)) ·
        (1 + 0.5 · |asymmetry| / 0.2)

The first factor gates by available reflectance (no force without
reflection); the second sigmoid maps raw stiffness (rad⁻¹) into
[0, 1]; the third gives a mild asymmetry boost so blazing matters.

Metadata exposes the un-normalized stiffness and damping
(``mean_dC_pr_2_dtheta``) coefficients so downstream code (e.g.
DampingObjective in P3.x) can reuse the FMM output without
re-evaluating.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from lightsail.materials import SiNDispersion
from lightsail.optimization.objectives import (
    ObjectiveContext,
    StabilizationProxy,
)
from lightsail.simulation.grating_fmm import (
    FMMGratingConfig,
    aggregate_metagrating_response,
)


class LocalPeriodFMMProxy(StabilizationProxy):
    """Radial-bin 1D-FMM proxy for outer-metagrating stabilization."""

    def __init__(
        self,
        nir_band_nm: tuple[float, float] = (1550.0, 1850.0),
        nir_n_points: int = 5,
        n_radial_bins: int = 3,
        nG: int = 21,
        dtheta_deg: float = 1.0,
        stiffness_scale: float = 0.5,
        dispersion: Optional[SiNDispersion] = None,
    ):
        self.nir_band_nm = (float(nir_band_nm[0]), float(nir_band_nm[1]))
        self.nir_n_points = int(nir_n_points)
        self.n_radial_bins = int(n_radial_bins)
        self.dtheta_deg = float(dtheta_deg)
        self.stiffness_scale = float(stiffness_scale)
        self._cfg = FMMGratingConfig(nG=int(nG))
        self._disp = dispersion or SiNDispersion()

    def score(self, ctx: ObjectiveContext) -> tuple[float, dict]:
        structure = ctx.structure
        if not structure.has_metagrating:
            return 0.0, {"reason": "no rings"}

        period_nm = float(structure.metadata.get("grating_period_nm", 1500.0))
        duty = float(structure.metadata.get("duty_cycle", 0.5))
        thickness_nm = float(structure.thickness_nm)
        rings = structure.rings
        if not rings:
            return 0.0, {"reason": "no rings"}

        curvature = float(np.mean([r.curvature for r in rings]))
        asymmetry = float(np.mean([abs(r.asymmetry) for r in rings]))

        # NIR power gate (uses spectrum cache shared with other objectives).
        nir = ctx.spectrum(self.nir_band_nm, self.nir_n_points)
        nir_R = float(nir.reflectance.mean())

        # Radial-bin aggregate FMM.
        wls = np.linspace(
            self.nir_band_nm[0], self.nir_band_nm[1], self.nir_n_points
        )
        agg = aggregate_metagrating_response(
            grating_period_nm=period_nm,
            duty_cycle=duty,
            thickness_nm=thickness_nm,
            wavelengths_nm=wls,
            curvature=curvature,
            n_radial_bins=self.n_radial_bins,
            theta_center_deg=0.0,
            dtheta_deg=self.dtheta_deg,
            dispersion=self._disp,
            config=self._cfg,
        )
        if agg.get("n", 0) == 0:
            return 0.0, {"reason": "fmm aggregate empty"}

        # Use *diffracted-only* (m≠0) coefficients so the score isolates the
        # metagrating contribution from the trivial specular tilted-mirror
        # response. Without this subtraction the BO converges to designs
        # that suppress diffraction (small period) and exploit the
        # ∂[sin(θ)·R_0]/∂θ ≈ R+T baseline.
        stiffness_raw = abs(float(agg["mean_dC_pr_1_diff_dtheta"]))
        damping_raw = abs(float(agg["mean_dC_pr_2_diff_dtheta"]))

        stiffness_norm = stiffness_raw / (stiffness_raw + self.stiffness_scale)
        asym_boost = 1.0 + 0.5 * asymmetry / 0.2
        score = float(np.clip(nir_R * stiffness_norm * asym_boost, 0.0, 1.0))

        return score, {
            "nir_R": nir_R,
            "stiffness_raw_per_rad": stiffness_raw,
            "damping_raw_per_rad": damping_raw,
            "stiffness_norm": stiffness_norm,
            "asym_boost": asym_boost,
            "mean_R_total": float(agg["mean_R_total"]),
            "mean_T_total": float(agg["mean_T_total"]),
            "mean_C_pr_1": float(agg["mean_C_pr_1"]),
            "mean_C_pr_2": float(agg["mean_C_pr_2"]),
            "n_fmm_calls": int(agg["n"]) * 2,
        }
