"""1D-periodic grating FMM helpers for outer-metagrating analysis.

The Stage 2 metagrating zone is locally periodic in the radial direction.
Per radial bin, we approximate it by a 1D-periodic SiN/air grating and
solve it with grcwa to obtain per-Fourier-order R, T and the lateral
momentum coefficients

    C_pr,0 = Σ_m (R_m + T_m)            (total propagating power)
    C_pr,1 = Σ_m sin(θ_m) (R_m + T_m)   (net lateral momentum)
    C_pr,2 = Σ_m sin²(θ_m)(R_m + T_m)   (angular spread)

These feed the stiffness matrix (P1.2, ∂C_pr,1/∂θ_in) and the
relativistic damping enhancement (P3.x, ∂C_pr,2/∂θ_in) calculations.

Implementation notes
--------------------
* grcwa is a 2D Fourier-modal solver. We realize a 1D grating by using
  a rectangular cell ``L1 = (Λ, 0)`` and a small dummy perpendicular
  period ``L2 = (0, Λ_perp)``. The pattern is uniform along y so only
  the radial physics matters; nG can stay modest (~21) which keeps
  per-call cost ~50–150 ms.
* Per-Fourier-order R/T come from ``sim.RT_Solve(normalize=1, byorder=1)``.
* Per-order in-plane wavevectors from ``sim.kx``; sin θ_m = kx_m / (2π/λ).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from lightsail.materials import SiNDispersion


@dataclass
class GratingOrderResult:
    """Per-order diffraction result for one (θ_in, λ)."""

    theta_deg: float
    wavelength_nm: float
    R_per_order: np.ndarray
    T_per_order: np.ndarray
    sin_theta_m: np.ndarray
    propagating_mask: np.ndarray

    @property
    def R_total(self) -> float:
        return float(np.sum(self.R_per_order[self.propagating_mask]))

    @property
    def T_total(self) -> float:
        return float(np.sum(self.T_per_order[self.propagating_mask]))

    def C_pr(self, k: int) -> float:
        """k-th moment Σ_m sin^k(θ_m) (R_m + T_m) over propagating orders."""
        s = self.sin_theta_m[self.propagating_mask]
        w = (self.R_per_order + self.T_per_order)[self.propagating_mask]
        return float(np.sum((s ** k) * w))

    def C_pr_diffracted(self, k: int) -> float:
        """k-th moment over **diffracted (m ≠ 0) propagating orders only**.

        Removes the trivial specular contribution
        ``sin^k(θ_in) · (R_0 + T_0)`` which any reflective surface
        produces — what we actually want for "metagrating stabilization"
        is the excess from non-zero diffraction orders. Without this
        subtraction, ``C_pr,1`` is dominated by the specular term and
        the BO converges to designs that suppress diffraction (small
        period) rather than enhancing it.
        """
        # grcwa's circular G truncation puts the (0,0) order at index 0
        # (smallest |G|²). Mask it out.
        mask = self.propagating_mask.copy()
        mask[0] = False
        s = self.sin_theta_m[mask]
        w = (self.R_per_order + self.T_per_order)[mask]
        return float(np.sum((s ** k) * w))


@dataclass
class FMMGratingConfig:
    """Numerical knobs for the 1D-grating FMM."""

    nG: int = 21
    nx: int = 128
    ny: int = 8
    perp_period_um: float = 0.5
    polarization: str = "average"   # "te" / "tm" / "average"


def evaluate_1d_grating(
    period_nm: float,
    duty_cycle: float,
    thickness_nm: float,
    wavelength_nm: float,
    theta_deg: float = 0.0,
    dispersion: Optional[SiNDispersion] = None,
    config: Optional[FMMGratingConfig] = None,
) -> GratingOrderResult:
    """Solve one 1D SiN/air grating with grcwa and return per-order R, T."""
    import grcwa

    cfg = config or FMMGratingConfig()
    disp = dispersion or SiNDispersion()

    eps_sin = complex(disp.epsilon(wavelength_nm))
    eps_air = 1.0 + 0.0j

    nx, ny = cfg.nx, cfg.ny
    bar_end = max(1, min(nx - 1, int(round(float(duty_cycle) * nx))))
    eps_grid = np.zeros((nx, ny), dtype=float)
    eps_grid[:bar_end, :] = 1.0  # mark SiN bars

    L1 = [period_nm / 1000.0, 0.0]
    L2 = [0.0, cfg.perp_period_um]
    wl_um = wavelength_nm / 1000.0
    freq = 1.0 / wl_um
    thick_um = thickness_nm / 1000.0

    pol_amps = _polarization_amps(cfg.polarization)

    # NOTE: grcwa's theta/phi arguments are in RADIANS despite no docstring
    # to that effect. The existing RCWASolver code is theta=0-only so the
    # bug is latent there; we convert here.
    theta_rad = float(np.deg2rad(theta_deg))

    R_acc: Optional[np.ndarray] = None
    T_acc: Optional[np.ndarray] = None
    sim_kx: Optional[np.ndarray] = None
    for p_amp, s_amp in pol_amps:
        sim = grcwa.obj(cfg.nG, L1, L2, freq, theta_rad, 0.0, verbose=0)
        sim.Add_LayerUniform(0.0, eps_air)
        sim.Add_LayerGrid(thick_um, nx, ny)
        sim.Add_LayerUniform(0.0, eps_air)
        sim.Init_Setup()
        filled = np.where(eps_grid > 0.5, eps_sin, eps_air).astype(complex).flatten()
        sim.GridLayer_geteps(filled)
        sim.MakeExcitationPlanewave(p_amp, 0.0, s_amp, 0.0, order=0)
        try:
            R_arr, T_arr = sim.RT_Solve(normalize=1, byorder=1)
        except Exception:
            n_kept = cfg.nG
            R_arr = np.zeros(n_kept)
            T_arr = np.ones(n_kept) / n_kept
        R_arr = np.real(np.asarray(R_arr, dtype=complex))
        T_arr = np.real(np.asarray(T_arr, dtype=complex))
        R_acc = R_arr.copy() if R_acc is None else R_acc + R_arr
        T_acc = T_arr.copy() if T_acc is None else T_acc + T_arr
        if sim_kx is None:
            sim_kx = np.asarray(sim.kx, dtype=float).copy()

    n_pol = len(pol_amps)
    R_per_order = R_acc / n_pol
    T_per_order = T_acc / n_pol

    k0 = 2.0 * np.pi * freq    # 1/µm; sim.kx already includes 2π factor
    sin_theta_m = sim_kx / k0
    propagating = np.abs(sin_theta_m) <= 1.0

    return GratingOrderResult(
        theta_deg=float(theta_deg),
        wavelength_nm=float(wavelength_nm),
        R_per_order=R_per_order,
        T_per_order=T_per_order,
        sin_theta_m=sin_theta_m,
        propagating_mask=propagating,
    )


def compute_lateral_coefficients(
    period_nm: float,
    duty_cycle: float,
    thickness_nm: float,
    wavelength_nm: float,
    theta_deg: float = 0.0,
    dispersion: Optional[SiNDispersion] = None,
    config: Optional[FMMGratingConfig] = None,
) -> dict:
    """Bundle C_pr,k (full and diffracted-only) + R_total / T_total at one (θ, λ).

    The ``_diff`` variants exclude the m=0 specular order so that the
    "metagrating contribution" is isolated from the trivial tilted-mirror
    response.
    """
    res = evaluate_1d_grating(
        period_nm=period_nm,
        duty_cycle=duty_cycle,
        thickness_nm=thickness_nm,
        wavelength_nm=wavelength_nm,
        theta_deg=theta_deg,
        dispersion=dispersion,
        config=config,
    )
    return {
        "C_pr_0": res.C_pr(0),
        "C_pr_1": res.C_pr(1),
        "C_pr_2": res.C_pr(2),
        "C_pr_1_diff": res.C_pr_diffracted(1),
        "C_pr_2_diff": res.C_pr_diffracted(2),
        "R_total": res.R_total,
        "T_total": res.T_total,
        "result": res,
    }


def compute_dC_pr_dtheta(
    period_nm: float,
    duty_cycle: float,
    thickness_nm: float,
    wavelength_nm: float,
    theta_center_deg: float = 0.0,
    dtheta_deg: float = 1.0,
    dispersion: Optional[SiNDispersion] = None,
    config: Optional[FMMGratingConfig] = None,
) -> dict:
    """Centered finite difference of C_pr,k vs incidence angle (per radian).

    The ``dC_pr_1_dtheta`` value is the local restoring-stiffness proxy
    used in the stiffness matrix (P1.2). The ``dC_pr_2_dtheta`` value
    enters the relativistic damping enhancement of docx Eq. 4.8 (P3.x).
    """
    plus = compute_lateral_coefficients(
        period_nm, duty_cycle, thickness_nm, wavelength_nm,
        theta_deg=theta_center_deg + dtheta_deg,
        dispersion=dispersion, config=config,
    )
    minus = compute_lateral_coefficients(
        period_nm, duty_cycle, thickness_nm, wavelength_nm,
        theta_deg=theta_center_deg - dtheta_deg,
        dispersion=dispersion, config=config,
    )
    dtheta_rad = 2.0 * np.deg2rad(dtheta_deg)
    return {
        "dC_pr_0_dtheta": (plus["C_pr_0"] - minus["C_pr_0"]) / dtheta_rad,
        "dC_pr_1_dtheta": (plus["C_pr_1"] - minus["C_pr_1"]) / dtheta_rad,
        "dC_pr_2_dtheta": (plus["C_pr_2"] - minus["C_pr_2"]) / dtheta_rad,
        "dC_pr_1_diff_dtheta": (plus["C_pr_1_diff"] - minus["C_pr_1_diff"]) / dtheta_rad,
        "dC_pr_2_diff_dtheta": (plus["C_pr_2_diff"] - minus["C_pr_2_diff"]) / dtheta_rad,
        "C_pr_1_plus": plus["C_pr_1"],
        "C_pr_1_minus": minus["C_pr_1"],
        "C_pr_2_plus": plus["C_pr_2"],
        "C_pr_2_minus": minus["C_pr_2"],
    }


def aggregate_metagrating_response(
    grating_period_nm: float,
    duty_cycle: float,
    thickness_nm: float,
    wavelengths_nm: np.ndarray,
    curvature: float = 0.0,
    n_radial_bins: int = 5,
    theta_center_deg: float = 0.0,
    dtheta_deg: float = 1.0,
    dispersion: Optional[SiNDispersion] = None,
    config: Optional[FMMGratingConfig] = None,
) -> dict:
    """Aggregate FMM response across radial bins × wavelengths.

    The radial direction is approximated by varying the local period
    via the metagrating curvature::

        Λ_local(u) = Λ · (1 + curvature · u),     u ∈ [-0.5, 0.5]

    For each (bin, wavelength) we evaluate C_pr,k at θ = θ_center and
    its centered finite-difference angle derivative; the means are
    returned as the stiffness / damping-enhancement proxies.
    """
    wls = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    if n_radial_bins > 1:
        bin_us = np.linspace(-0.5, 0.5, n_radial_bins)
    else:
        bin_us = np.zeros(1)
    bin_periods = grating_period_nm * (1.0 + curvature * bin_us)

    sums = {
        "C_pr_0": 0.0, "C_pr_1": 0.0, "C_pr_2": 0.0,
        "C_pr_1_diff": 0.0, "C_pr_2_diff": 0.0,
        "dC_pr_1_dtheta": 0.0, "dC_pr_2_dtheta": 0.0,
        "dC_pr_1_diff_dtheta": 0.0, "dC_pr_2_diff_dtheta": 0.0,
        "R_total": 0.0, "T_total": 0.0,
    }
    n = 0
    for p_local in bin_periods:
        for wl in wls:
            base = compute_lateral_coefficients(
                float(p_local), duty_cycle, thickness_nm, float(wl),
                theta_deg=theta_center_deg,
                dispersion=dispersion, config=config,
            )
            d = compute_dC_pr_dtheta(
                float(p_local), duty_cycle, thickness_nm, float(wl),
                theta_center_deg=theta_center_deg, dtheta_deg=dtheta_deg,
                dispersion=dispersion, config=config,
            )
            sums["C_pr_0"] += base["C_pr_0"]
            sums["C_pr_1"] += base["C_pr_1"]
            sums["C_pr_2"] += base["C_pr_2"]
            sums["C_pr_1_diff"] += base["C_pr_1_diff"]
            sums["C_pr_2_diff"] += base["C_pr_2_diff"]
            sums["dC_pr_1_dtheta"] += d["dC_pr_1_dtheta"]
            sums["dC_pr_2_dtheta"] += d["dC_pr_2_dtheta"]
            sums["dC_pr_1_diff_dtheta"] += d["dC_pr_1_diff_dtheta"]
            sums["dC_pr_2_diff_dtheta"] += d["dC_pr_2_diff_dtheta"]
            sums["R_total"] += base["R_total"]
            sums["T_total"] += base["T_total"]
            n += 1

    if n == 0:
        return {"n": 0}

    return {
        "n": n,
        "mean_C_pr_0": sums["C_pr_0"] / n,
        "mean_C_pr_1": sums["C_pr_1"] / n,
        "mean_C_pr_2": sums["C_pr_2"] / n,
        "mean_C_pr_1_diff": sums["C_pr_1_diff"] / n,
        "mean_C_pr_2_diff": sums["C_pr_2_diff"] / n,
        "mean_dC_pr_1_dtheta": sums["dC_pr_1_dtheta"] / n,
        "mean_dC_pr_2_dtheta": sums["dC_pr_2_dtheta"] / n,
        "mean_dC_pr_1_diff_dtheta": sums["dC_pr_1_diff_dtheta"] / n,
        "mean_dC_pr_2_diff_dtheta": sums["dC_pr_2_diff_dtheta"] / n,
        "mean_R_total": sums["R_total"] / n,
        "mean_T_total": sums["T_total"] / n,
        "bin_periods_nm": tuple(float(p) for p in bin_periods),
        "wavelengths_nm": tuple(float(w) for w in wls),
    }


def _polarization_amps(mode: str) -> list[Tuple[float, float]]:
    if mode == "te":
        return [(0.0, 1.0)]
    if mode == "tm":
        return [(1.0, 0.0)]
    return [(1.0, 0.0), (0.0, 1.0)]
