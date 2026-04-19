"""Per-area force/torque Lookup Tables (LUT) for Center PhC + Outer Ring.

This is Step A1 of the paper-grade trajectory simulation (Gieseler 2024
adapted to our 2-zone architecture). For each zone we precompute the
optical force per unit area as a function of incidence angle, so that
the trajectory integrator (A3) can evaluate forces cheaply at arbitrary
sail position/orientation by looking up the LUT.

Physics (from photon-momentum balance on a 2D unit cell):

    F_x_per_area(θ_in) = (I/c) · [sin θ_in − Σ_m sin(θ_m) (R_m + T_m)]
    F_z_per_area(θ_in) = (I/c) · [cos θ_in − Σ_m (T_m − R_m) cos(θ_m)]

For the **central PhC** in our NIR Doppler band (1550–1898 nm), period
≈ 1580 nm forces only m = (0,0) to propagate (all higher G's are
evanescent). Then the LUT collapses to:

    F_x(θ) = (I/c) · A · sin θ                    (zero for lossless SiN!)
    F_z(θ) = (I/c) · (1 + R − T) · cos θ          (≈ 2R · cos θ for R≫T)

For the **outer ring** with period ≈ 2 µm, m = ±1 propagate so we need
the full per-order summation, computed via the existing 1D-FMM
(`grating_fmm.evaluate_1d_grating`).

Convention: incidence at angle θ_in from sail normal (+z). Positive
θ_in means in-plane wavevector along +x. Sail "front" faces the laser
(laser propagates in +z direction). F_z > 0 = forward thrust.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lightsail.geometry.base import Structure
from lightsail.geometry.phc_reflector import PhCReflector
from lightsail.materials import SiNDispersion
from lightsail.simulation.grating_fmm import (
    FMMGratingConfig,
    evaluate_1d_grating,
)
from lightsail.simulation.rcwa_solver import RCWAConfig, RCWASolver


C_LIGHT = 299_792_458.0


# ---------------------------------------------------------------------------
# Center PhC LUT
# ---------------------------------------------------------------------------


@dataclass
class CenterPhCLUT:
    """Per-area force LUT for the central PhC reflector zone.

    Stores R(θ, λ) and T(θ, λ) sweeps. The lossless case (k≈0 in NIR)
    means absorption A = 1−R−T ≈ 0 and so F_x ≈ 0 — the central PhC is
    purely propulsive and provides no lateral restoring force. We still
    expose F_x for completeness when small absorption is present.
    """

    theta_grid_deg: np.ndarray
    wavelengths_nm: np.ndarray
    R: np.ndarray              # (n_theta, n_wl)
    T: np.ndarray              # (n_theta, n_wl)

    @property
    def A(self) -> np.ndarray:
        return np.clip(1.0 - self.R - self.T, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Vectorized band-mean interpolators (fast for trajectory loops)
    # ------------------------------------------------------------------

    @property
    def _R_mean(self) -> np.ndarray:
        if not hasattr(self, "_R_mean_cache"):
            object.__setattr__(self, "_R_mean_cache", self.R.mean(axis=1))
        return self._R_mean_cache

    @property
    def _T_mean(self) -> np.ndarray:
        if not hasattr(self, "_T_mean_cache"):
            object.__setattr__(self, "_T_mean_cache", self.T.mean(axis=1))
        return self._T_mean_cache

    def force_per_area_vec(
        self,
        theta_deg_arr: np.ndarray,
        intensity_W_per_m2: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized band-mean F_z, F_x per area at array of θ values."""
        theta = np.clip(np.asarray(theta_deg_arr, dtype=float),
                        self.theta_grid_deg[0], self.theta_grid_deg[-1])
        R_th = np.interp(theta, self.theta_grid_deg, self._R_mean)
        T_th = np.interp(theta, self.theta_grid_deg, self._T_mean)
        A_th = np.clip(1.0 - R_th - T_th, 0.0, 1.0)
        theta_rad = np.deg2rad(theta)
        F_z = (intensity_W_per_m2 / C_LIGHT) * (1.0 + R_th - T_th) * np.cos(theta_rad)
        F_x = (intensity_W_per_m2 / C_LIGHT) * A_th * np.sin(theta_rad)
        return F_z, F_x

    def force_per_area(
        self,
        theta_deg: float,
        intensity_W_per_m2: float,
        wavelength_nm: Optional[float] = None,
    ) -> tuple[float, float]:
        """Linear-interpolated F_z, F_x per area at (θ, λ).

        If ``wavelength_nm`` is None, the LUT is averaged across the
        stored wavelengths (broadband mean) — useful when we don't want
        to track Doppler-shifted wavelength explicitly during a short
        trajectory simulation.
        """
        theta = float(np.clip(theta_deg, self.theta_grid_deg[0], self.theta_grid_deg[-1]))
        if wavelength_nm is None:
            R_th = float(np.interp(theta, self.theta_grid_deg, self.R.mean(axis=1)))
            T_th = float(np.interp(theta, self.theta_grid_deg, self.T.mean(axis=1)))
        else:
            wl = float(np.clip(
                wavelength_nm, self.wavelengths_nm[0], self.wavelengths_nm[-1]
            ))
            from scipy.interpolate import RectBivariateSpline
            R_th = float(RectBivariateSpline(
                self.theta_grid_deg, self.wavelengths_nm, self.R, kx=1, ky=1
            )(theta, wl)[0, 0])
            T_th = float(RectBivariateSpline(
                self.theta_grid_deg, self.wavelengths_nm, self.T, kx=1, ky=1
            )(theta, wl)[0, 0])
        A_th = max(0.0, 1.0 - R_th - T_th)

        theta_rad = np.deg2rad(theta)
        F_z = (intensity_W_per_m2 / C_LIGHT) * (1.0 + R_th - T_th) * np.cos(theta_rad)
        F_x = (intensity_W_per_m2 / C_LIGHT) * A_th * np.sin(theta_rad)
        return float(F_z), float(F_x)


def compute_center_lut(
    phc: PhCReflector,
    theta_grid_deg: np.ndarray,
    wavelengths_nm: np.ndarray,
    rcwa_config: Optional[RCWAConfig] = None,
    dispersion: Optional[SiNDispersion] = None,
) -> CenterPhCLUT:
    """Sweep RCWASolver over θ × λ for the central PhC and build LUT."""
    rcwa_config = rcwa_config or RCWAConfig(nG=41, grid_nx=64, grid_ny=64)
    structure = phc.to_structure()

    theta_arr = np.atleast_1d(np.asarray(theta_grid_deg, dtype=float))
    wl_arr = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    R = np.zeros((theta_arr.size, wl_arr.size), dtype=float)
    T = np.zeros_like(R)

    base_theta = float(rcwa_config.theta_deg)
    for i, th in enumerate(theta_arr):
        # New solver per angle (RCWAConfig is a dataclass; copy + modify)
        cfg = RCWAConfig(
            nG=rcwa_config.nG,
            grid_nx=rcwa_config.grid_nx,
            grid_ny=rcwa_config.grid_ny,
            polarization=rcwa_config.polarization,
            theta_deg=float(th),
            phi_deg=rcwa_config.phi_deg,
        )
        solver = RCWASolver(config=cfg, dispersion=dispersion)
        R[i, :] = solver.evaluate_reflectivity(structure, wl_arr)
        T[i, :] = solver.evaluate_transmission(structure, wl_arr)

    return CenterPhCLUT(
        theta_grid_deg=theta_arr,
        wavelengths_nm=wl_arr,
        R=R,
        T=T,
    )


# ---------------------------------------------------------------------------
# Ring LUT
# ---------------------------------------------------------------------------


@dataclass
class RingLUT:
    """Per-area force LUT for the concentric outer ring zone.

    Storage convention:
      * ``F_radial`` is positive = photons exit RADIALLY OUTWARD →
        recoil force on sail is INWARD (toward center) per Newton 3rd.
      * ``F_z`` is positive = forward (along +z) thrust.

    The integrator (A2) takes the radial direction at azimuthal angle φ
    and projects onto cartesian: F_x_pt = -F_radial · cos(φ),
    F_y_pt = -F_radial · sin(φ).
    """

    theta_grid_deg: np.ndarray
    wavelengths_nm: np.ndarray
    F_radial_per_area_norm: np.ndarray   # (n_theta, n_wl), in units of (1/c)
    F_z_per_area_norm: np.ndarray        # (n_theta, n_wl), in units of (1/c)

    @property
    def _Fz_mean(self) -> np.ndarray:
        if not hasattr(self, "_Fz_mean_cache"):
            object.__setattr__(self, "_Fz_mean_cache", self.F_z_per_area_norm.mean(axis=1))
        return self._Fz_mean_cache

    @property
    def _Fr_mean(self) -> np.ndarray:
        if not hasattr(self, "_Fr_mean_cache"):
            object.__setattr__(self, "_Fr_mean_cache", self.F_radial_per_area_norm.mean(axis=1))
        return self._Fr_mean_cache

    def force_per_area_vec(
        self,
        theta_deg_arr: np.ndarray,
        intensity_W_per_m2: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized band-mean (F_z, F_radial_inward) per area at θ array."""
        theta = np.clip(np.asarray(theta_deg_arr, dtype=float),
                        self.theta_grid_deg[0], self.theta_grid_deg[-1])
        Fz_n = np.interp(theta, self.theta_grid_deg, self._Fz_mean)
        Fr_n = np.interp(theta, self.theta_grid_deg, self._Fr_mean)
        return (
            intensity_W_per_m2 / C_LIGHT * Fz_n,
            intensity_W_per_m2 / C_LIGHT * Fr_n,
        )

    def force_per_area(
        self,
        theta_deg: float,
        intensity_W_per_m2: float,
        wavelength_nm: Optional[float] = None,
    ) -> tuple[float, float]:
        """Linear-interpolated (F_z, F_radial) per area at (θ, λ).

        Returns (F_z, F_radial) in N/m². ``F_radial`` here is the
        photon-momentum carried OUTWARD (positive = outward); the
        sail recoil is the negative.
        """
        theta = float(np.clip(theta_deg, self.theta_grid_deg[0], self.theta_grid_deg[-1]))
        if wavelength_nm is None:
            Fz_n = float(np.interp(theta, self.theta_grid_deg, self._Fz_mean))
            Fr_n = float(np.interp(theta, self.theta_grid_deg, self._Fr_mean))
        else:
            wl = float(np.clip(
                wavelength_nm, self.wavelengths_nm[0], self.wavelengths_nm[-1]
            ))
            from scipy.interpolate import RectBivariateSpline
            Fz_n = float(RectBivariateSpline(
                self.theta_grid_deg, self.wavelengths_nm, self.F_z_per_area_norm, kx=1, ky=1
            )(theta, wl)[0, 0])
            Fr_n = float(RectBivariateSpline(
                self.theta_grid_deg, self.wavelengths_nm, self.F_radial_per_area_norm, kx=1, ky=1
            )(theta, wl)[0, 0])
        F_z = intensity_W_per_m2 * Fz_n / C_LIGHT * 0  # placeholder, see below
        # The "_norm" arrays already contain the (1/c) factor's sibling:
        # we stored them as the dimensionless Σ-quantities, so:
        F_z = intensity_W_per_m2 / C_LIGHT * Fz_n
        F_r = intensity_W_per_m2 / C_LIGHT * Fr_n
        return float(F_z), float(F_r)


def compute_ring_lut(
    grating_period_nm: float,
    duty_cycle: float,
    thickness_nm: float,
    theta_grid_deg: np.ndarray,
    wavelengths_nm: np.ndarray,
    fmm_config: Optional[FMMGratingConfig] = None,
    dispersion: Optional[SiNDispersion] = None,
) -> RingLUT:
    """Sweep 1D-FMM over (θ, λ) for the ring grating; build per-area LUT.

    For each grid point we run the FMM, extract per-order R_m, T_m and
    sin(θ_m), and compute the dimensionless coefficients

        c_z(θ_in) = cos(θ_in) − Σ_m (T_m − R_m) cos(θ_m)
        c_r(θ_in) = Σ_m sin(θ_m) (R_m + T_m) − sin(θ_in)
                  = C_pr,1(θ_in) − sin(θ_in)        (radial outward)

    Sail recoil force per area in (radial, z) is then
    (1/c) × I × (−c_r, c_z).
    """
    cfg = fmm_config or FMMGratingConfig(nG=21, nx=128, ny=4)
    theta_arr = np.atleast_1d(np.asarray(theta_grid_deg, dtype=float))
    wl_arr = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))

    cz = np.zeros((theta_arr.size, wl_arr.size), dtype=float)
    cr = np.zeros_like(cz)

    for i, th in enumerate(theta_arr):
        for j, wl in enumerate(wl_arr):
            res = evaluate_1d_grating(
                period_nm=float(grating_period_nm),
                duty_cycle=float(duty_cycle),
                thickness_nm=float(thickness_nm),
                wavelength_nm=float(wl),
                theta_deg=float(th),
                dispersion=dispersion,
                config=cfg,
            )
            mask = res.propagating_mask
            R_arr = res.R_per_order[mask]
            T_arr = res.T_per_order[mask]
            sin_m = res.sin_theta_m[mask]
            cos_m = np.sqrt(np.clip(1.0 - sin_m ** 2, 0.0, 1.0))

            sin_in = float(np.sin(np.deg2rad(th)))
            cos_in = float(np.cos(np.deg2rad(th)))

            # Radial outward photon momentum carried away (per incoming photon):
            # Σ sin θ_m (R_m + T_m) − sin θ_in
            cr[i, j] = float(np.sum(sin_m * (R_arr + T_arr)) - sin_in)
            # Axial momentum: cos θ_in − Σ (T_m − R_m) cos θ_m
            cz[i, j] = float(cos_in - np.sum((T_arr - R_arr) * cos_m))

    # Sign convention:
    #   cr[i,j] = C_pr,1(θ) − sin θ = (photon outward momentum) − (incoming
    #   in-plane momentum). By Newton's 3rd, sail recoil (per area, normalized
    #   by I/c) in the INWARD radial direction equals +cr. (When photons gain
    #   net outward momentum, sail is pushed inward.)
    return RingLUT(
        theta_grid_deg=theta_arr,
        wavelengths_nm=wl_arr,
        F_radial_per_area_norm=cr,    # inward sail recoil (+ = pushed toward sail center)
        F_z_per_area_norm=cz,
    )
